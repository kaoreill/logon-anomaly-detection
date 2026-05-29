# Logon Anomaly Detection

An end-to-end Python project for detecting anomalous VDI logon behavior using synthetic telemetry, ensemble anomaly detection, explainable alert outputs, time-based backtesting, drift monitoring, and a Streamlit analyst dashboard.

## Features

- Synthetic VDI logon data generation
- Ensemble anomaly detection using Isolation Forest and Elliptic Envelope
- Baseline-aware feature engineering (user and machine context)
- Calibrated anomaly thresholding
- Per-alert human-readable explanations
- Time-based rolling backtesting
- Drift monitoring with PSI reports
- Static and interactive visualizations
- Streamlit dashboard for analyst triage

## Project Structure

- main.py: Full pipeline entry point
- dashboard.py: Streamlit dashboard app
- src/data_generator.py: Synthetic logon telemetry generator
- src/model.py: Detection model and explanation logic
- src/visualiser.py: Matplotlib and Plotly visual outputs
- data/: Generated and sample datasets
- models/: Saved model artifacts
- visualizations/: Charts and report artifacts

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The Pipeline

```bash
python main.py
```

This generates:

- Synthetic dataset and scored output
- Model artifact
- Confusion matrix
- Interactive anomaly charts
- Explanation report CSV
- Backtest metrics and chart
- Drift report and drift chart

## Launch The Dashboard

```bash
streamlit run dashboard.py
```

If required files are missing, run the pipeline first with `python main.py`.

## Main Output Artifacts

- data/synthetic_logons.csv
- data/latest_scored_logons.csv
- models/anomaly_detector.pkl
- visualizations/anomaly_explanation_report.csv
- visualizations/time_backtest_metrics.csv
- visualizations/drift_report.csv

## Notes

- The project currently uses synthetic data generation for reproducible testing.
- Backtesting and drift monitoring are included to better approximate production readiness.
