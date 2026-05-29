import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import precision_recall_curve
import pickle

class LogonAnomalyDetector:
    """
    Detects anomalous VDI logon patterns using multiple unsupervised methods.
    """
    
    def __init__(self, contamination=0.05):
        """
        Initialize anomaly detectors.
        
        contamination: expected proportion of anomalies in dataset (0-1)
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        
        # Two complementary anomaly detection methods
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.elliptic_envelope = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )

        self.base_feature_columns = [
            'logon_duration_sec', 'profile_load_time_sec', 'auth_time_sec',
            'desktop_init_time_sec', 'concurrent_logons',
            'machine_cpu_during_logon', 'machine_memory_during_logon'
        ]
        self.baseline_metrics = [
            'logon_duration_sec', 'profile_load_time_sec',
            'auth_time_sec', 'desktop_init_time_sec'
        ]

        self.feature_columns = None
        self.user_baseline_stats = None
        self.machine_baseline_stats = None
        self.global_means = None
        self.global_stds = None
        self.global_explain_stds = None
        self.threshold = 0.5
        self.medium_threshold = 0.65
        self.high_threshold = 0.8
        self.threshold_source = 'default_fixed'
        self.is_fitted = False

    @staticmethod
    def _flatten_stats_columns(stats_df):
        """Flatten multi-index columns from groupby aggregation."""
        stats_df.columns = [f"{metric}_{stat}" for metric, stat in stats_df.columns]
        return stats_df

    def _prepare_dataframe(self, df):
        """Create a safe working copy and normalize key dtypes."""
        work_df = df.copy()
        if 'timestamp' in work_df.columns:
            work_df['timestamp'] = pd.to_datetime(work_df['timestamp'])
        return work_df

    def _add_context_features(self, df):
        """Add user and machine baseline-aware z-score features."""
        work_df = self._prepare_dataframe(df)

        user_stats = self.user_baseline_stats.reindex(work_df['user_id']).reset_index(drop=True)
        machine_stats = self.machine_baseline_stats.reindex(work_df['machine_id']).reset_index(drop=True)

        for metric in self.baseline_metrics:
            global_mean = self.global_means[metric]
            global_std = self.global_stds[metric]

            user_mean = user_stats[f'{metric}_mean'].fillna(global_mean)
            user_std = user_stats[f'{metric}_std'].fillna(global_std).replace(0, global_std)

            machine_mean = machine_stats[f'{metric}_mean'].fillna(global_mean)
            machine_std = machine_stats[f'{metric}_std'].fillna(global_std).replace(0, global_std)

            work_df[f'user_{metric}_z'] = (work_df[metric] - user_mean) / user_std
            work_df[f'machine_{metric}_z'] = (work_df[metric] - machine_mean) / machine_std

        if 'timestamp' in work_df.columns:
            work_df['hour_sin'] = np.sin(2 * np.pi * work_df['timestamp'].dt.hour / 24)
            work_df['hour_cos'] = np.cos(2 * np.pi * work_df['timestamp'].dt.hour / 24)
            work_df['weekday_sin'] = np.sin(2 * np.pi * work_df['timestamp'].dt.dayofweek / 7)
            work_df['weekday_cos'] = np.cos(2 * np.pi * work_df['timestamp'].dt.dayofweek / 7)
        else:
            work_df['hour_sin'] = 0.0
            work_df['hour_cos'] = 0.0
            work_df['weekday_sin'] = 0.0
            work_df['weekday_cos'] = 0.0

        for metric in self.base_feature_columns:
            if metric in work_df.columns and metric in self.global_means.index:
                work_df[metric] = work_df[metric].fillna(self.global_means[metric])

        work_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        generated_feature_columns = [
            f'user_{metric}_z' for metric in self.baseline_metrics
        ] + [
            f'machine_{metric}_z' for metric in self.baseline_metrics
        ] + ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']

        for col in generated_feature_columns:
            work_df[col] = work_df[col].fillna(0.0)

        return work_df

    def _normalize_scores(self, if_scores, ee_scores):
        """Normalize model scores into comparable anomaly probabilities."""
        if_scores_norm = 1 / (1 + np.exp(if_scores))

        ee_range = ee_scores.max() - ee_scores.min()
        if ee_range == 0:
            ee_scores_norm = np.zeros_like(ee_scores, dtype=float)
        else:
            ee_scores_norm = (ee_scores - ee_scores.min()) / ee_range

        ensemble_score = (if_scores_norm + ee_scores_norm) / 2
        return if_scores_norm, ee_scores_norm, ensemble_score

    def _calibrate_threshold(self, y_true, scores):
        """Pick the threshold that maximizes F1 when labels are available."""
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        if len(thresholds) == 0:
            return float(np.quantile(scores, 1 - self.contamination)), 'quantile_fallback'

        f1_scores = (2 * precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + 1e-9)
        best_idx = int(np.argmax(f1_scores))
        return float(thresholds[best_idx]), 'f1_calibrated'

    def _build_explanations(self, feature_df):
        """Create top contributing signals for each prediction row."""
        metric_display = {
            'logon_duration_sec': 'Logon duration',
            'profile_load_time_sec': 'Profile load time',
            'auth_time_sec': 'Authentication time',
            'desktop_init_time_sec': 'Desktop initialization time',
            'concurrent_logons': 'Concurrent logons',
            'machine_cpu_during_logon': 'Machine CPU during logon',
            'machine_memory_during_logon': 'Machine memory during logon'
        }

        def format_reason(kind, metric, signed_score):
            direction = 'higher' if signed_score >= 0 else 'lower'
            metric_name = metric_display[metric]
            if kind == 'user':
                return f"{metric_name} is {direction} than this user's baseline"
            if kind == 'machine':
                return f"{metric_name} is {direction} than this machine's baseline"
            return f"{metric_name} is {direction} than the global baseline"

        contribution_arrays_signed = []
        contribution_metadata = []

        for metric in self.baseline_metrics:
            contribution_arrays_signed.append(feature_df[f'user_{metric}_z'].values)
            contribution_metadata.append(('user', metric))

            contribution_arrays_signed.append(feature_df[f'machine_{metric}_z'].values)
            contribution_metadata.append(('machine', metric))

        for metric in self.base_feature_columns:
            std = self.global_explain_stds[metric]
            global_z = (feature_df[metric].values - self.global_means[metric]) / std
            contribution_arrays_signed.append(global_z)
            contribution_metadata.append(('global', metric))

        contribution_matrix_signed = np.column_stack(contribution_arrays_signed)
        contribution_matrix_abs = np.abs(contribution_matrix_signed)
        top_n = 3
        top_indices = np.argsort(-contribution_matrix_abs, axis=1)[:, :top_n]

        top_reasons = []
        top_scores = []
        for rank in range(top_n):
            rank_indices = top_indices[:, rank]
            rank_reasons = []
            rank_scores = []
            for row_idx, col_idx in enumerate(rank_indices):
                kind, metric = contribution_metadata[col_idx]
                signed_score = contribution_matrix_signed[row_idx, col_idx]
                rank_reasons.append(format_reason(kind, metric, signed_score))
                rank_scores.append(abs(signed_score))

            top_reasons.append(np.array(rank_reasons))
            top_scores.append(np.array(rank_scores))

        return top_reasons, top_scores
    
    def fit(self, df):
        """
        Fit the anomaly detectors on training data.
        
        df: DataFrame with logon metrics
        """
        train_df = self._prepare_dataframe(df)

        self.global_means = train_df[self.base_feature_columns].mean()
        self.global_stds = train_df[self.baseline_metrics].std().replace(0, np.nan).fillna(1e-6)
        self.global_explain_stds = train_df[self.base_feature_columns].std().replace(0, np.nan).fillna(1e-6)

        self.user_baseline_stats = train_df.groupby('user_id')[self.baseline_metrics].agg(['mean', 'std'])
        self.user_baseline_stats = self._flatten_stats_columns(self.user_baseline_stats)

        self.machine_baseline_stats = train_df.groupby('machine_id')[self.baseline_metrics].agg(['mean', 'std'])
        self.machine_baseline_stats = self._flatten_stats_columns(self.machine_baseline_stats)

        train_with_features = self._add_context_features(train_df)
        self.feature_columns = self.base_feature_columns + [
            f'user_{metric}_z' for metric in self.baseline_metrics
        ] + [
            f'machine_{metric}_z' for metric in self.baseline_metrics
        ] + ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']

        X = train_with_features[self.feature_columns].values
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit both models
        self.isolation_forest.fit(X_scaled)
        self.elliptic_envelope.fit(X_scaled)

        if_scores = self.isolation_forest.score_samples(X_scaled)
        ee_scores = self.elliptic_envelope.mahalanobis(X_scaled)
        _, _, ensemble_scores = self._normalize_scores(if_scores, ee_scores)

        if 'is_anomaly' in train_with_features.columns and train_with_features['is_anomaly'].nunique() > 1:
            y_true = train_with_features['is_anomaly'].astype(int).values
            self.threshold, self.threshold_source = self._calibrate_threshold(y_true, ensemble_scores)
        else:
            self.threshold = float(np.quantile(ensemble_scores, 1 - self.contamination))
            self.threshold_source = 'contamination_quantile'

        self.medium_threshold = max(self.threshold, float(np.quantile(ensemble_scores, 0.9)))
        self.high_threshold = max(self.medium_threshold + 1e-6, float(np.quantile(ensemble_scores, 0.98)))
        
        self.is_fitted = True
        print(f"✓ Anomaly detectors trained successfully (threshold={self.threshold:.3f}, source={self.threshold_source})")
    
    def predict(self, df):
        """
        Predict anomalies in new data.
        
        Returns:
        - DataFrame with anomaly scores and predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        feature_df = self._add_context_features(df)
        X = feature_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        # Get anomaly scores (lower = more anomalous)
        if_scores = self.isolation_forest.score_samples(X_scaled)
        ee_scores = self.elliptic_envelope.mahalanobis(X_scaled)
        if_scores_norm, ee_scores_norm, ensemble_score = self._normalize_scores(if_scores, ee_scores)
        ensemble_pred = np.where(ensemble_score > self.threshold, -1, 1)

        severity = np.where(
            ensemble_score > self.high_threshold,
            'high',
            np.where(
                ensemble_score > self.medium_threshold,
                'medium',
                np.where(ensemble_score > self.threshold, 'low', 'normal')
            )
        )
        
        result_df = df.copy()
        result_df['if_anomaly_score'] = if_scores_norm
        result_df['ee_anomaly_score'] = ee_scores_norm
        result_df['ensemble_anomaly_score'] = ensemble_score
        result_df['anomaly_severity'] = severity
        result_df['predicted_anomaly'] = ensemble_pred == -1

        top_labels, top_scores = self._build_explanations(feature_df)
        result_df['explanation_1'] = top_labels[0]
        result_df['explanation_2'] = top_labels[1]
        result_df['explanation_3'] = top_labels[2]
        result_df['explanation_1_score'] = pd.Series(top_scores[0]).round(2)
        result_df['explanation_2_score'] = pd.Series(top_scores[1]).round(2)
        result_df['explanation_3_score'] = pd.Series(top_scores[2]).round(2)
        result_df['explanation_summary'] = (
            result_df['explanation_1'] + ' (strength=' + result_df['explanation_1_score'].astype(str) + '), ' +
            result_df['explanation_2'] + ' (strength=' + result_df['explanation_2_score'].astype(str) + '), ' +
            result_df['explanation_3'] + ' (strength=' + result_df['explanation_3_score'].astype(str) + ')'
        )
        
        return result_df
    
    def save(self, filepath):
        """Save trained model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    # Demo usage
    df = pd.read_csv('data/synthetic_logons.csv')
    
    # Split into train/test (use first 80% for training)
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    # Train detector
    detector = LogonAnomalyDetector(contamination=0.05)
    detector.fit(train_df)
    
    # Predict on test set
    results = detector.predict(test_df)
    
    # Evaluate (assuming test set has true labels)
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    if 'is_anomaly' in results.columns:
        precision = precision_score(results['is_anomaly'], results['predicted_anomaly'])
        recall = recall_score(results['is_anomaly'], results['predicted_anomaly'])
        f1 = f1_score(results['is_anomaly'], results['predicted_anomaly'])
        
        print(f"\nModel Performance:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")
    
    # Save model
    detector.save('models/anomaly_detector.pkl')