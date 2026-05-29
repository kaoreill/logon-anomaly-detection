import pickle

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class BaseAnomalyDetector:
    """Base utilities for anomaly detectors in this project."""

    def __init__(self, contamination=0.05, feature_columns=None):
        """
        Initialize detector settings.

        Parameters
        ----------
        contamination : float
            Expected anomaly proportion in the dataset.
        feature_columns : list[str] | None
            Optional explicit feature list. If omitted, auto-detects supported columns.
        """
        self.contamination = contamination
        self.scaler = StandardScaler()

        self.default_feature_candidates = [
            'logon_duration_sec', 'profile_load_time_sec', 'auth_time_sec',
            'desktop_init_time_sec', 'concurrent_logons',
            'machine_cpu_during_logon', 'machine_memory_during_logon',
            'profile_ratio', 'auth_ratio', 'init_ratio', 'resource_pressure',
            'load_severity', 'logon_zscore', 'all_slow', 'hour', 'is_business_hours'
        ]

        self.feature_columns = feature_columns
        self.feature_fill_values = None

        self.threshold = 0.5
        self.medium_threshold = 0.65
        self.high_threshold = 0.8
        self.threshold_source = 'default_fixed'

        self.is_fitted = False

    def _prepare_dataframe(self, df):
        """Validate and normalize base DataFrame types."""
        if df is None or df.empty:
            raise ValueError('Input DataFrame is empty.')

        work_df = df.copy()
        if 'timestamp' in work_df.columns:
            work_df['timestamp'] = pd.to_datetime(work_df['timestamp'], errors='coerce')
        return work_df

    def _select_feature_columns(self, df):
        """Resolve feature columns from explicit list or supported defaults."""
        if self.feature_columns is not None:
            missing = [col for col in self.feature_columns if col not in df.columns]
            if missing:
                raise ValueError(
                    'Missing required model feature columns: ' + ', '.join(missing)
                )
            return list(self.feature_columns)

        selected = [col for col in self.default_feature_candidates if col in df.columns]
        if not selected:
            raise ValueError('No supported model feature columns found in DataFrame.')
        return selected

    def _prepare_feature_matrix(self, df, fit_mode=False):
        """Build a clean numeric feature matrix ready for scaling and model input."""
        work_df = self._prepare_dataframe(df)

        if fit_mode:
            self.feature_columns = self._select_feature_columns(work_df)
        elif self.feature_columns is None:
            raise ValueError('Model is not fitted. No feature columns are available.')

        for col in self.feature_columns:
            if col not in work_df.columns:
                work_df[col] = np.nan

        X = work_df[self.feature_columns].copy()

        for col in self.feature_columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
            elif X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors='coerce')

        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        if fit_mode:
            self.feature_fill_values = X.median(numeric_only=True).fillna(0.0)
        if self.feature_fill_values is None:
            raise ValueError('Model is not fitted. Missing feature fill values.')

        for col in self.feature_columns:
            if col not in self.feature_fill_values.index:
                self.feature_fill_values[col] = 0.0
            X[col] = X[col].fillna(float(self.feature_fill_values[col]))

        return work_df, X.astype(float)

    @staticmethod
    def _normalize_scores(raw_scores):
        """Min-max normalize a score vector into [0, 1]."""
        scores = np.asarray(raw_scores, dtype=float)
        score_range = float(scores.max() - scores.min())
        if score_range == 0:
            return np.zeros_like(scores)
        return (scores - scores.min()) / score_range

    def _calibrate_threshold(self, y_true, scores):
        """Pick threshold maximizing F1 on available labels."""
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        if len(thresholds) == 0:
            return float(np.quantile(scores, 1 - self.contamination)), 'quantile_fallback'

        f1_scores = (2 * precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + 1e-9)
        best_idx = int(np.argmax(f1_scores))
        return float(thresholds[best_idx]), 'f1_calibrated'

    def _build_explanations(self, X_scaled):
        """Create top feature-deviation reasons from standardized feature magnitudes."""
        abs_scaled = np.abs(X_scaled)
        top_n = min(3, abs_scaled.shape[1])
        top_indices = np.argsort(-abs_scaled, axis=1)[:, :top_n]

        explanations = []
        explanation_scores = []

        for rank in range(top_n):
            rank_idx = top_indices[:, rank]
            rank_explanations = []
            rank_scores = []

            for row_idx, col_idx in enumerate(rank_idx):
                feature_name = self.feature_columns[col_idx]
                rank_explanations.append(f'{feature_name} deviates from learned baseline')
                rank_scores.append(float(abs_scaled[row_idx, col_idx]))

            explanations.append(np.array(rank_explanations))
            explanation_scores.append(np.array(rank_scores))

        # Ensure exactly 3 outputs for backward compatibility.
        while len(explanations) < 3:
            explanations.append(np.array(['No strong deviation found'] * X_scaled.shape[0]))
            explanation_scores.append(np.array([0.0] * X_scaled.shape[0]))

        return explanations, explanation_scores

    def save(self, filepath):
        """Save model object to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f'✓ Model saved to {filepath}')

    @staticmethod
    def load(filepath):
        """Load model object from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f'✓ Model loaded from {filepath}')
        return model


class AdvancedAnomalyDetector(BaseAnomalyDetector):
    """Advanced 4-model weighted unsupervised anomaly detector."""

    def __init__(
        self,
        contamination=0.05,
        feature_columns=None,
        n_neighbors=20,
        kernel='rbf',
        gamma='auto',
        n_estimators=200
    ):
        """Initialize advanced ensemble with requested algorithm hyperparameters."""
        super().__init__(contamination=contamination, feature_columns=feature_columns)

        self.weights = {
            'if_score': 0.4,
            'lof_score': 0.2,
            'svm_score': 0.2,
            'ee_score': 0.2
        }

        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=n_estimators
        )
        self.local_outlier_factor = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            novelty=True
        )
        self.one_class_svm = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=contamination
        )
        self.elliptic_envelope = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )

    def fit(self, df):
        """Fit all four unsupervised models and calibrate ensemble threshold."""
        _, X = self._prepare_feature_matrix(df, fit_mode=True)
        X_scaled = self.scaler.fit_transform(X.values)

        self.isolation_forest.fit(X_scaled)
        self.local_outlier_factor.fit(X_scaled)
        self.one_class_svm.fit(X_scaled)
        self.elliptic_envelope.fit(X_scaled)

        if_score = self._normalize_scores(-self.isolation_forest.score_samples(X_scaled))
        lof_score = self._normalize_scores(-self.local_outlier_factor.score_samples(X_scaled))
        svm_score = self._normalize_scores(-self.one_class_svm.score_samples(X_scaled))
        ee_score = self._normalize_scores(self.elliptic_envelope.mahalanobis(X_scaled))

        ensemble_score = (
            self.weights['if_score'] * if_score +
            self.weights['lof_score'] * lof_score +
            self.weights['svm_score'] * svm_score +
            self.weights['ee_score'] * ee_score
        )

        work_df = self._prepare_dataframe(df)
        if 'is_anomaly' in work_df.columns and work_df['is_anomaly'].nunique() > 1:
            y_true = work_df['is_anomaly'].astype(int).values
            self.threshold, self.threshold_source = self._calibrate_threshold(y_true, ensemble_score)
        else:
            self.threshold = float(np.quantile(ensemble_score, 1 - self.contamination))
            self.threshold_source = 'contamination_quantile'

        self.medium_threshold = max(self.threshold, float(np.quantile(ensemble_score, 0.9)))
        self.high_threshold = max(self.medium_threshold + 1e-6, float(np.quantile(ensemble_score, 0.98)))

        self.is_fitted = True
        print(
            f'✓ Advanced anomaly detector trained '
            f'(threshold={self.threshold:.3f}, source={self.threshold_source}, '
            f'features={len(self.feature_columns)})'
        )

    def predict(self, df):
        """
        Score new records with all ensemble models.

        Returns DataFrame with required score columns and backward-compatible aliases.
        """
        if not self.is_fitted:
            raise ValueError('Model must be fitted first.')

        original_df, X = self._prepare_feature_matrix(df, fit_mode=False)
        X_scaled = self.scaler.transform(X.values)

        if_score = self._normalize_scores(-self.isolation_forest.score_samples(X_scaled))
        lof_score = self._normalize_scores(-self.local_outlier_factor.score_samples(X_scaled))
        svm_score = self._normalize_scores(-self.one_class_svm.score_samples(X_scaled))
        ee_score = self._normalize_scores(self.elliptic_envelope.mahalanobis(X_scaled))

        ensemble_score = (
            self.weights['if_score'] * if_score +
            self.weights['lof_score'] * lof_score +
            self.weights['svm_score'] * svm_score +
            self.weights['ee_score'] * ee_score
        )

        predicted_anomaly = ensemble_score > self.threshold

        anomaly_severity = np.where(
            ensemble_score > self.high_threshold,
            'high',
            np.where(
                ensemble_score > self.medium_threshold,
                'medium',
                np.where(ensemble_score > self.threshold, 'low', 'normal')
            )
        )

        result_df = original_df.copy()
        result_df['if_score'] = if_score
        result_df['lof_score'] = lof_score
        result_df['svm_score'] = svm_score
        result_df['ee_score'] = ee_score
        result_df['ensemble_score'] = ensemble_score
        result_df['predicted_anomaly'] = predicted_anomaly

        # Backward-compatible aliases used elsewhere in the project.
        result_df['if_anomaly_score'] = result_df['if_score']
        result_df['ee_anomaly_score'] = result_df['ee_score']
        result_df['ensemble_anomaly_score'] = result_df['ensemble_score']
        result_df['anomaly_severity'] = anomaly_severity

        top_labels, top_scores = self._build_explanations(X_scaled)
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


# Backward compatibility for older imports.
LogonAnomalyDetector = AdvancedAnomalyDetector


if __name__ == '__main__':
    df = pd.read_csv('data/synthetic_logons.csv')

    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    detector = AdvancedAnomalyDetector(contamination=0.05)
    detector.fit(train_df)

    results = detector.predict(test_df)

    from sklearn.metrics import f1_score, precision_score, recall_score

    if 'is_anomaly' in results.columns:
        precision = precision_score(results['is_anomaly'], results['predicted_anomaly'])
        recall = recall_score(results['is_anomaly'], results['predicted_anomaly'])
        f1 = f1_score(results['is_anomaly'], results['predicted_anomaly'])

        print('\nModel Performance:')
        print(f'Precision: {precision:.3f}')
        print(f'Recall:    {recall:.3f}')
        print(f'F1 Score:  {f1:.3f}')

    detector.save('models/anomaly_detector.pkl')
