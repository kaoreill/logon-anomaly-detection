#!/usr/bin/env python3
"""
Logon Time Anomaly Detection - Complete Pipeline
"""

import pandas as pd
import numpy as np
import os
from src.data_generator import generate_synthetic_logons
from src.feature_engineer import FeatureEngineer
from src.model import AdvancedAnomalyDetector
from src.visualiser import visualize_logon_metrics, visualize_anomalies
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


def _compute_psi(expected, actual, bins=10):
    """Compute population stability index between two numeric distributions."""
    expected = pd.Series(expected).dropna().astype(float).values
    actual = pd.Series(actual).dropna().astype(float).values

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(expected, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_ratio = expected_counts / max(expected_counts.sum(), 1)
    actual_ratio = actual_counts / max(actual_counts.sum(), 1)

    epsilon = 1e-6
    psi = np.sum((actual_ratio - expected_ratio) * np.log((actual_ratio + epsilon) / (expected_ratio + epsilon)))
    return float(psi)


def run_time_based_backtest(df, contamination=0.05, output_dir='visualizations/', n_splits=4):
    """Evaluate model using rolling, time-ordered train/test windows."""
    print("\n[STEP 5A] Running time-based backtesting...")
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    n_records = len(df_sorted)

    min_train_size = int(n_records * 0.5)
    test_size = int(n_records * 0.1)
    max_train_end = n_records - test_size

    if max_train_end <= min_train_size:
        print("  - Skipped: dataset too small for rolling backtest windows")
        return pd.DataFrame()

    train_end_points = np.linspace(min_train_size, max_train_end, n_splits, dtype=int)
    rows = []

    for split_number, train_end in enumerate(train_end_points, start=1):
        test_end = min(train_end + test_size, n_records)
        if test_end <= train_end:
            continue

        train_df = df_sorted.iloc[:train_end]
        test_df = df_sorted.iloc[train_end:test_end].copy()

        fold_detector = AdvancedAnomalyDetector(contamination=contamination)
        fold_detector.fit(train_df)
        fold_results = fold_detector.predict(test_df)

        y_true = fold_results['is_anomaly'].astype(int).values
        y_pred = fold_results['predicted_anomaly'].astype(int).values

        rows.append({
            'split': split_number,
            'train_end_timestamp': train_df['timestamp'].max(),
            'test_start_timestamp': test_df['timestamp'].min(),
            'test_end_timestamp': test_df['timestamp'].max(),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'test_records': len(test_df),
            'predicted_anomaly_rate': float(fold_results['predicted_anomaly'].mean())
        })

    backtest_df = pd.DataFrame(rows)
    if backtest_df.empty:
        print("  - Skipped: no valid backtest windows generated")
        return backtest_df

    backtest_df.to_csv(f'{output_dir}time_backtest_metrics.csv', index=False)
    print("✓ Saved: time_backtest_metrics.csv")

    plt.figure(figsize=(10, 5))
    plt.plot(backtest_df['split'], backtest_df['precision'], marker='o', label='Precision')
    plt.plot(backtest_df['split'], backtest_df['recall'], marker='o', label='Recall')
    plt.plot(backtest_df['split'], backtest_df['f1'], marker='o', label='F1')
    plt.ylim(0, 1.05)
    plt.xlabel('Backtest Split')
    plt.ylabel('Metric')
    plt.title('Time-Based Backtest Performance')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}time_backtest_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: time_backtest_performance.png")

    print(
        f"  - Average backtest metrics: "
        f"precision={backtest_df['precision'].mean():.3f}, "
        f"recall={backtest_df['recall'].mean():.3f}, "
        f"f1={backtest_df['f1'].mean():.3f}"
    )
    return backtest_df


def run_drift_monitoring(train_df, train_results, test_df, test_results, output_dir='visualizations/'):
    """Generate a simple drift report using PSI and key deltas."""
    print("\n[STEP 5B] Running drift monitoring...")

    features = [
        'logon_duration_sec', 'profile_load_time_sec', 'auth_time_sec',
        'desktop_init_time_sec', 'concurrent_logons',
        'machine_cpu_during_logon', 'machine_memory_during_logon'
    ]

    rows = []
    for feature in features:
        train_mean = float(train_df[feature].mean())
        test_mean = float(test_df[feature].mean())
        mean_delta_pct = ((test_mean - train_mean) / (train_mean + 1e-9)) * 100
        psi = _compute_psi(train_df[feature], test_df[feature])

        if np.isnan(psi):
            severity = 'unknown'
        elif psi >= 0.25:
            severity = 'high'
        elif psi >= 0.1:
            severity = 'moderate'
        else:
            severity = 'low'

        rows.append({
            'metric': feature,
            'train_mean': train_mean,
            'test_mean': test_mean,
            'mean_delta_pct': mean_delta_pct,
            'psi': psi,
            'drift_severity': severity
        })

    score_psi = _compute_psi(train_results['ensemble_anomaly_score'], test_results['ensemble_anomaly_score'])
    rows.append({
        'metric': 'ensemble_anomaly_score',
        'train_mean': float(train_results['ensemble_anomaly_score'].mean()),
        'test_mean': float(test_results['ensemble_anomaly_score'].mean()),
        'mean_delta_pct': ((float(test_results['ensemble_anomaly_score'].mean()) - float(train_results['ensemble_anomaly_score'].mean())) /
                           (float(train_results['ensemble_anomaly_score'].mean()) + 1e-9)) * 100,
        'psi': score_psi,
        'drift_severity': 'high' if score_psi >= 0.25 else 'moderate' if score_psi >= 0.1 else 'low'
    })

    drift_df = pd.DataFrame(rows).sort_values('psi', ascending=False)
    drift_df.to_csv(f'{output_dir}drift_report.csv', index=False)
    print("✓ Saved: drift_report.csv")

    top_drift = drift_df.head(8)
    plt.figure(figsize=(10, 5))
    plt.barh(top_drift['metric'][::-1], top_drift['psi'][::-1], color='teal')
    plt.axvline(0.1, color='orange', linestyle='--', linewidth=1, label='Moderate (PSI=0.1)')
    plt.axvline(0.25, color='red', linestyle='--', linewidth=1, label='High (PSI=0.25)')
    plt.xlabel('PSI')
    plt.title('Top Drifted Metrics (Train vs Test)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}drift_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: drift_summary.png")

    highest = drift_df.iloc[0]
    print(f"  - Highest drift metric: {highest['metric']} (PSI={highest['psi']:.3f}, severity={highest['drift_severity']})")
    return drift_df

def main():
    print("="*60)
    print("VDI LOGON TIME ANOMALY DETECTION")
    print("="*60)
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # STEP 1: Generate synthetic data
    print("\n[STEP 1] Generating synthetic logon data...")
    df = generate_synthetic_logons(n_records=10000, anomaly_rate=0.05)

    # STEP 1.5: Feature engineering
    print("\n[STEP 1.5] Engineering advanced features...")
    original_columns = list(df.columns)
    df = FeatureEngineer.engineer_features(df)
    engineered_columns = [col for col in df.columns if col not in original_columns]

    print(f"✓ Added {len(engineered_columns)} engineered features")
    if engineered_columns:
        sample_feature_cols = engineered_columns[:6]
        print("  - Sample engineered feature values:")
        print(df[sample_feature_cols].head(3).to_string(index=False))

    df.to_csv('data/synthetic_logons.csv', index=False)
    print(f"✓ Generated {len(df)} logon records")
    print(f"  - Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
    
    # STEP 2: Explore data
    print("\n[STEP 2] Creating exploratory visualizations...")
    visualize_logon_metrics(df)
    
    # STEP 3: Train anomaly detector
    print("\n[STEP 3] Training anomaly detector...")
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    test_df = df[split_idx:].copy()
    
    detector = AdvancedAnomalyDetector(contamination=0.05)
    detector.fit(train_df)
    print(f"  - Calibrated threshold: {detector.threshold:.3f} ({detector.threshold_source})")
    
    # STEP 4: Make predictions
    print("\n[STEP 4] Making predictions on test set...")
    results = detector.predict(test_df)
    train_results = detector.predict(train_df.copy())
    results.to_csv('data/latest_scored_logons.csv', index=False)
    print("✓ Saved: data/latest_scored_logons.csv")
    
    # STEP 5: Evaluate performance
    print("\n[STEP 5] Evaluating model performance...")
    true_labels = results['is_anomaly'].astype(int).values
    pred_labels = results['predicted_anomaly'].astype(int).values
    
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, 
                              target_names=['Normal', 'Anomaly']))

    run_time_based_backtest(df, contamination=0.05)
    run_drift_monitoring(train_df, train_results, test_df, results)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrix.png")
    plt.close()
    
    # STEP 6: Create result visualizations
    print("\n[STEP 6] Creating anomaly visualizations...")
    visualize_anomalies(results)

    # STEP 6.5: Save analyst explanation report
    print("\n[STEP 6.5] Saving anomaly explanation report...")
    explanation_report = results[results['predicted_anomaly']].nlargest(100, 'ensemble_anomaly_score')[[
        'timestamp', 'user_id', 'machine_id', 'logon_duration_sec',
        'concurrent_logons', 'ensemble_anomaly_score', 'anomaly_severity',
        'explanation_1', 'explanation_1_score',
        'explanation_2', 'explanation_2_score',
        'explanation_3', 'explanation_3_score',
        'explanation_summary'
    ]]
    explanation_report.to_csv('visualizations/anomaly_explanation_report.csv', index=False)
    print("✓ Saved: anomaly_explanation_report.csv")
    
    # STEP 7: Save model
    print("\n[STEP 7] Saving trained model...")
    detector.save('models/anomaly_detector.pkl')
    
    # STEP 8: Show interesting findings
    print("\n[STEP 8] Top detected anomalies:")
    top_anomalies = results[results['predicted_anomaly']].nlargest(5, 'ensemble_anomaly_score')
    print(top_anomalies[['timestamp', 'user_id', 'logon_duration_sec', 
                         'concurrent_logons', 'ensemble_anomaly_score', 'anomaly_severity',
                         'explanation_1', 'explanation_2', 'explanation_3']])
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE")
    print("="*60)
    print("\nOutput files:")
    print("  - data/synthetic_logons.csv")
    print("  - models/anomaly_detector.pkl")
    print("  - visualizations/logon_metrics_overview.png")
    print("  - visualizations/anomaly_scatter.html")
    print("  - visualizations/anomaly_timeline.html")
    print("  - visualizations/feature_deviation.html")
    print("  - visualizations/confusion_matrix.png")
    print("  - visualizations/anomaly_explanation_report.csv")
    print("  - visualizations/time_backtest_metrics.csv")
    print("  - visualizations/time_backtest_performance.png")
    print("  - visualizations/drift_report.csv")
    print("  - visualizations/drift_summary.png")
    print("  - data/latest_scored_logons.csv")

if __name__ == "__main__":
    main()