import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

def visualize_logon_metrics(df, output_dir='visualizations/'):
    """Create exploratory visualizations of logon metrics."""
    
    # 1. Logon duration distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(df['logon_duration_sec'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['logon_duration_sec'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].axvline(60, color='orange', linestyle='--', label='Warning (60s)')
    axes[0, 0].axvline(120, color='red', linestyle='--', label='Critical (120s)')
    axes[0, 0].set_xlabel('Logon Duration (seconds)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Logon Duration')
    axes[0, 0].legend()
    
    # 2. Machine CPU vs Memory during logon
    scatter = axes[0, 1].scatter(
        df['machine_cpu_during_logon'],
        df['machine_memory_during_logon'],
        alpha=0.5,
        s=30
    )
    axes[0, 1].set_xlabel('CPU Usage (%)')
    axes[0, 1].set_ylabel('Memory Usage (%)')
    axes[0, 1].set_title('Machine Resource Usage During Logon')
    axes[0, 1].axhline(80, color='orange', linestyle='--', alpha=0.5, label='Warning')
    axes[0, 1].axvline(80, color='orange', linestyle='--', alpha=0.5)
    axes[0, 1].legend()
    
    # 3. Concurrent logons over time
    df_sorted = df.sort_values('timestamp')
    axes[1, 0].plot(df_sorted['concurrent_logons'].rolling(50).mean(), linewidth=2)
    axes[1, 0].fill_between(
        range(len(df_sorted)),
        df_sorted['concurrent_logons'].rolling(50).min(),
        df_sorted['concurrent_logons'].rolling(50).max(),
        alpha=0.3
    )
    axes[1, 0].axhline(30, color='orange', linestyle='--', label='Peak threshold')
    axes[1, 0].axhline(60, color='red', linestyle='--', label='Critical')
    axes[1, 0].set_xlabel('Timeline')
    axes[1, 0].set_ylabel('Concurrent Logons')
    axes[1, 0].set_title('Concurrent Logons Over Time (Rolling Average)')
    axes[1, 0].legend()
    
    # 4. Logon component breakdown
    component_means = {
        'Auth': df['auth_time_sec'].mean(),
        'Profile Load': df['profile_load_time_sec'].mean(),
        'Desktop Init': df['desktop_init_time_sec'].mean()
    }
    axes[1, 1].bar(component_means.keys(), component_means.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_title('Average Logon Time by Component')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}logon_metrics_overview.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: logon_metrics_overview.png")
    plt.close()


def visualize_anomalies(results_df, output_dir='visualizations/'):
    """Create interactive Plotly visualizations for anomaly results."""
    
    # 1. Scatter plot: Logon Duration vs Ensemble Anomaly Score
    fig1 = px.scatter(
        results_df,
        x='logon_duration_sec',
        y='ensemble_anomaly_score',
        color='predicted_anomaly',
        hover_data=['user_id', 'machine_id', 'concurrent_logons', 'anomaly_severity',
                    'explanation_1', 'explanation_2', 'explanation_3'],
        title='Logon Duration vs Anomaly Score',
        labels={
            'logon_duration_sec': 'Logon Duration (sec)',
            'ensemble_anomaly_score': 'Anomaly Score (0=Normal, 1=Anomalous)',
            'predicted_anomaly': 'Anomaly'
        }
    )
    fig1.write_html(f'{output_dir}anomaly_scatter.html')
    print("✓ Saved: anomaly_scatter.html")
    
    # 2. Time series: Anomaly scores over time
    results_sorted = results_df.sort_values('timestamp')
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig2.add_trace(
        go.Scatter(
            x=results_sorted['timestamp'],
            y=results_sorted['ensemble_anomaly_score'],
            name='Anomaly Score',
            line=dict(color='steelblue'),
            fill='tozeroy'
        ),
        secondary_y=False
    )
    
    fig2.add_trace(
        go.Scatter(
            x=results_sorted['timestamp'],
            y=results_sorted['concurrent_logons'],
            name='Concurrent Logons',
            line=dict(color='orange'),
            mode='lines'
        ),
        secondary_y=True
    )
    
    fig2.update_layout(
        title='Anomaly Scores and Concurrent Logons Over Time',
        xaxis_title='Timestamp',
        yaxis_title='Anomaly Score',
        yaxis2_title='Concurrent Logons',
        hovermode='x unified'
    )
    fig2.write_html(f'{output_dir}anomaly_timeline.html')
    print("✓ Saved: anomaly_timeline.html")
    
    # 3. Feature importance: Which metrics most anomalous?
    anomaly_df = results_df[results_df['predicted_anomaly']]
    
    metrics = ['logon_duration_sec', 'profile_load_time_sec', 'auth_time_sec',
               'desktop_init_time_sec', 'concurrent_logons',
               'machine_cpu_during_logon', 'machine_memory_during_logon']
    
    metric_anomaly_pct = []
    for metric in metrics:
        normal_median = results_df[~results_df['predicted_anomaly']][metric].median()
        anomaly_median = anomaly_df[metric].median()
        pct_diff = ((anomaly_median - normal_median) / normal_median) * 100
        metric_anomaly_pct.append(pct_diff)
    
    fig3 = px.bar(
        x=metrics,
        y=metric_anomaly_pct,
        title='Feature Deviation in Anomalies vs Normal Logons',
        labels={'x': 'Metric', 'y': '% Difference from Normal Median'},
        color=metric_anomaly_pct,
        color_continuous_scale='RdYlGn_r'
    )
    fig3.update_layout(xaxis_tickangle=-45)
    fig3.write_html(f'{output_dir}feature_deviation.html')
    print("✓ Saved: feature_deviation.html")


if __name__ == "__main__":
    df = pd.read_csv('data/synthetic_logons.csv')
    visualize_logon_metrics(df)
    
    # For anomaly visualizations, you'd need results from the model
    # This is shown in main.py