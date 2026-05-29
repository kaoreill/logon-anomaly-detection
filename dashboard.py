#!/usr/bin/env python3
"""Streamlit dashboard for logon anomaly analysis."""

import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Logon Anomaly Dashboard', layout='wide')
st.title('VDI Logon Anomaly Dashboard')
st.caption('Operational view of detected anomalies, drift, and backtest performance')

DATA_PATH = 'data/latest_scored_logons.csv'
DRIFT_PATH = 'visualizations/drift_report.csv'
BACKTEST_PATH = 'visualizations/time_backtest_metrics.csv'
EXPLANATION_PATH = 'visualizations/anomaly_explanation_report.csv'

if not os.path.exists(DATA_PATH):
    st.error('Missing data/latest_scored_logons.csv. Run: python main.py')
    st.stop()

results_df = pd.read_csv(DATA_PATH)
results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])

for col in ['predicted_anomaly', 'is_anomaly']:
    if col in results_df.columns:
        results_df[col] = results_df[col].astype(bool)

st.sidebar.header('Filters')
severity_options = sorted(results_df['anomaly_severity'].dropna().unique().tolist())
selected_severity = st.sidebar.multiselect(
    'Anomaly Severity',
    options=severity_options,
    default=severity_options
)

min_score = float(results_df['ensemble_anomaly_score'].min())
max_score = float(results_df['ensemble_anomaly_score'].max())
score_threshold = st.sidebar.slider(
    'Min Anomaly Score',
    min_value=round(min_score, 3),
    max_value=round(max_score, 3),
    value=round(min_score, 3),
    step=0.001
)

filtered_df = results_df[
    (results_df['anomaly_severity'].isin(selected_severity))
    & (results_df['ensemble_anomaly_score'] >= score_threshold)
].copy()

if 'all_slow' in filtered_df.columns:
    all_slow_filter = st.sidebar.selectbox('All Slow Flag', options=['All', 'Yes', 'No'], index=0)
    if all_slow_filter == 'Yes':
        filtered_df = filtered_df[filtered_df['all_slow'].astype(int) == 1]
    elif all_slow_filter == 'No':
        filtered_df = filtered_df[filtered_df['all_slow'].astype(int) == 0]

if 'is_business_hours' in filtered_df.columns:
    business_hours_filter = st.sidebar.selectbox('Business Hours', options=['All', 'Yes', 'No'], index=0)
    if business_hours_filter == 'Yes':
        filtered_df = filtered_df[filtered_df['is_business_hours'].astype(bool)]
    elif business_hours_filter == 'No':
        filtered_df = filtered_df[~filtered_df['is_business_hours'].astype(bool)]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric('Scored Events', f"{len(filtered_df):,}")
kpi2.metric('Predicted Anomalies', f"{int(filtered_df['predicted_anomaly'].sum()):,}")
kpi3.metric('Anomaly Rate', f"{filtered_df['predicted_anomaly'].mean() * 100:.2f}%")
kpi4.metric('Avg Score', f"{filtered_df['ensemble_anomaly_score'].mean():.3f}")

st.subheader('Anomaly Timeline')
if filtered_df.empty:
    st.info('No records match the current filters.')
else:
    timeline = filtered_df.set_index('timestamp').resample('1h').agg(
        anomaly_rate=('predicted_anomaly', 'mean'),
        avg_score=('ensemble_anomaly_score', 'mean'),
        events=('predicted_anomaly', 'size')
    ).reset_index()
    fig_timeline = px.line(
        timeline,
        x='timestamp',
        y=['anomaly_rate', 'avg_score'],
        title='Hourly Anomaly Rate and Average Score'
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

st.subheader('Score vs Duration')
fig_scatter = px.scatter(
    filtered_df,
    x='logon_duration_sec',
    y='ensemble_anomaly_score',
    color='anomaly_severity',
    hover_data=[
        'timestamp', 'user_id', 'machine_id',
        'explanation_1', 'explanation_2', 'explanation_3'
    ],
    title='Logon Duration vs Anomaly Score'
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader('Model Score Comparison')
model_score_columns = ['if_score', 'lof_score', 'svm_score', 'ee_score', 'ensemble_score']
available_score_columns = [c for c in model_score_columns if c in filtered_df.columns]

if available_score_columns:
    score_summary = filtered_df[available_score_columns].mean().reset_index()
    score_summary.columns = ['model', 'avg_score']
    fig_score_summary = px.bar(
        score_summary,
        x='model',
        y='avg_score',
        color='model',
        title='Average Model Score by Algorithm'
    )
    st.plotly_chart(fig_score_summary, use_container_width=True)

    long_scores = filtered_df[available_score_columns].melt(
        var_name='model',
        value_name='score'
    )
    fig_score_distribution = px.box(
        long_scores,
        x='model',
        y='score',
        color='model',
        title='Model Score Distributions'
    )
    st.plotly_chart(fig_score_distribution, use_container_width=True)
else:
    st.warning('Model score columns not found in scored dataset.')

st.subheader('Engineered Feature Distributions')
engineered_features = [
    'profile_ratio', 'auth_ratio', 'init_ratio', 'resource_pressure',
    'load_severity', 'logon_zscore', 'all_slow', 'hour', 'is_business_hours'
]
available_engineered_features = [f for f in engineered_features if f in filtered_df.columns]

if available_engineered_features:
    selected_feature = st.selectbox(
        'Select engineered feature',
        options=available_engineered_features,
        index=0
    )
    feature_df = filtered_df.copy()
    if feature_df[selected_feature].dtype == bool:
        feature_df[selected_feature] = feature_df[selected_feature].astype(int)

    fig_feature_dist = px.histogram(
        feature_df,
        x=selected_feature,
        color='predicted_anomaly',
        barmode='overlay',
        nbins=40,
        title=f'Distribution of {selected_feature}'
    )
    st.plotly_chart(fig_feature_dist, use_container_width=True)
else:
    st.warning('No engineered feature columns found in scored dataset.')

st.subheader('Top Anomaly Explanations')
explanation_df = filtered_df[filtered_df['predicted_anomaly']].copy()
if explanation_df.empty:
    st.info('No anomalies under current filters.')
else:
    top_explanations = explanation_df[['explanation_1', 'explanation_2', 'explanation_3']]
    explanation_counts = pd.concat([
        top_explanations['explanation_1'],
        top_explanations['explanation_2'],
        top_explanations['explanation_3']
    ]).value_counts().head(10)

    fig_expl = px.bar(
        x=explanation_counts.values,
        y=explanation_counts.index,
        orientation='h',
        labels={'x': 'Frequency', 'y': 'Explanation'},
        title='Most Common Root-Cause Signals'
    )
    st.plotly_chart(fig_expl, use_container_width=True)

st.subheader('Top Alerts')
alert_columns = [
    'timestamp', 'user_id', 'machine_id', 'ensemble_anomaly_score',
    'anomaly_severity', 'explanation_1', 'explanation_2', 'explanation_3',
    'explanation_summary'
]
show_cols = [c for c in alert_columns if c in filtered_df.columns]
st.dataframe(
    filtered_df[filtered_df['predicted_anomaly']]
    .nlargest(200, 'ensemble_anomaly_score')[show_cols],
    use_container_width=True,
    hide_index=True
)

st.subheader('Backtest Metrics')
if os.path.exists(BACKTEST_PATH):
    backtest_df = pd.read_csv(BACKTEST_PATH)
    st.dataframe(backtest_df, use_container_width=True, hide_index=True)
    fig_backtest = px.line(
        backtest_df,
        x='split',
        y=['precision', 'recall', 'f1'],
        markers=True,
        title='Time-Based Backtest Metrics by Split'
    )
    st.plotly_chart(fig_backtest, use_container_width=True)
else:
    st.warning('Missing visualizations/time_backtest_metrics.csv. Run python main.py to generate it.')

st.subheader('Drift Report')
if os.path.exists(DRIFT_PATH):
    drift_df = pd.read_csv(DRIFT_PATH)
    st.dataframe(drift_df, use_container_width=True, hide_index=True)
    fig_drift = px.bar(
        drift_df.sort_values('psi', ascending=False).head(12),
        x='psi',
        y='metric',
        color='drift_severity',
        orientation='h',
        title='Top Drifted Metrics (PSI)'
    )
    st.plotly_chart(fig_drift, use_container_width=True)
else:
    st.warning('Missing visualizations/drift_report.csv. Run python main.py to generate it.')

if os.path.exists(EXPLANATION_PATH):
    st.subheader('Analyst Explanation Report (Top 100)')
    st.dataframe(pd.read_csv(EXPLANATION_PATH), use_container_width=True, hide_index=True)
