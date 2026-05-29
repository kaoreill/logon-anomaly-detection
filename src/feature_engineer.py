import pandas as pd
import numpy as np


class FeatureEngineer:
    """Builds derived and time-based features for anomaly detection."""

    @staticmethod
    def engineer_features(df):
        """
        Add engineered features to a logon DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame with base logon telemetry columns.

        Returns
        -------
        pandas.DataFrame
            Enhanced DataFrame with original and engineered features.

        Raises
        ------
        ValueError
            If input DataFrame is empty.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty. Cannot engineer features.")

        enhanced_df = df.copy()

        required_columns = [
            'logon_duration_sec',
            'profile_load_time_sec',
            'auth_time_sec',
            'desktop_init_time_sec',
            'concurrent_logons',
            'machine_cpu_during_logon',
            'machine_memory_during_logon'
        ]

        missing_columns = [col for col in required_columns if col not in enhanced_df.columns]
        if missing_columns:
            raise ValueError(
                "Missing required columns for feature engineering: " + ", ".join(missing_columns)
            )

        denominator = enhanced_df['logon_duration_sec'].fillna(0).astype(float) + 1.0

        enhanced_df['profile_ratio'] = enhanced_df['profile_load_time_sec'].astype(float) / denominator
        enhanced_df['auth_ratio'] = enhanced_df['auth_time_sec'].astype(float) / denominator
        enhanced_df['init_ratio'] = enhanced_df['desktop_init_time_sec'].astype(float) / denominator

        enhanced_df['resource_pressure'] = (
            enhanced_df['machine_cpu_during_logon'].astype(float).clip(lower=0) / 100.0
        ) * (
            enhanced_df['machine_memory_during_logon'].astype(float).clip(lower=0) / 100.0
        )

        mean_concurrent_logons = float(enhanced_df['concurrent_logons'].astype(float).mean())
        enhanced_df['load_severity'] = enhanced_df['concurrent_logons'].astype(float) / (mean_concurrent_logons + 1.0)

        logon_mean = float(enhanced_df['logon_duration_sec'].astype(float).mean())
        logon_std = float(enhanced_df['logon_duration_sec'].astype(float).std())
        if np.isnan(logon_std) or logon_std == 0:
            enhanced_df['logon_zscore'] = 0.0
        else:
            enhanced_df['logon_zscore'] = np.abs(
                (enhanced_df['logon_duration_sec'].astype(float) - logon_mean) / logon_std
            )

        profile_q75 = enhanced_df['profile_load_time_sec'].astype(float).quantile(0.75)
        auth_q75 = enhanced_df['auth_time_sec'].astype(float).quantile(0.75)
        init_q75 = enhanced_df['desktop_init_time_sec'].astype(float).quantile(0.75)
        enhanced_df['all_slow'] = (
            (enhanced_df['profile_load_time_sec'].astype(float) > profile_q75)
            & (enhanced_df['auth_time_sec'].astype(float) > auth_q75)
            & (enhanced_df['desktop_init_time_sec'].astype(float) > init_q75)
        ).astype(int)

        if 'timestamp' in enhanced_df.columns:
            enhanced_df['timestamp'] = pd.to_datetime(enhanced_df['timestamp'], errors='coerce')
            enhanced_df['hour'] = enhanced_df['timestamp'].dt.hour.fillna(-1).astype(int)
            enhanced_df['is_business_hours'] = enhanced_df['hour'].between(8, 18, inclusive='both')

        enhanced_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        engineered_columns = [
            'profile_ratio', 'auth_ratio', 'init_ratio', 'resource_pressure',
            'load_severity', 'logon_zscore', 'all_slow'
        ]
        if 'hour' in enhanced_df.columns:
            engineered_columns.extend(['hour', 'is_business_hours'])

        for col in engineered_columns:
            if enhanced_df[col].dtype == bool:
                enhanced_df[col] = enhanced_df[col].fillna(False)
            else:
                enhanced_df[col] = enhanced_df[col].fillna(0)

        return enhanced_df
