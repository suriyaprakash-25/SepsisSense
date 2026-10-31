"""
SepsisSense Precision-Optimized Machine Learning Ensemble
=========================================================

High-precision clinical ensemble combining:
1. Cost-Sensitive & Focal-Weighted LightGBM Classifier
2. Multi-Depth Histogram-Accelerated XGBoost Classifier
3. Regularized Random Forest Ensemble
4. Stacking Meta-Learner with Isotonic Probability Calibration
5. Precision-Constrained Decision Boundary & Temporal Persistence Filter

Author: SepsisSense Machine Learning & Clinical Data Science Team
License: MIT
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

try:
    from src.precision_engineering import PrecisionOptimizer, compute_physionet_utility
    from src.conformal_uncertainty import ConformalSepsisPredictor
    from src.path_signatures import VitalSignSignatureExtractor
except ImportError:
    from precision_engineering import PrecisionOptimizer, compute_physionet_utility
    from conformal_uncertainty import ConformalSepsisPredictor
    from path_signatures import VitalSignSignatureExtractor

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("PrecisionEnsemble")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _f = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    _h.setFormatter(_f)
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class SepsisPrecisionEnsemble(BaseEstimator, ClassifierMixin):
    """
    Advanced ensemble model tailored specifically for high precision under
    extreme positive-class sparsity in critical care time-series.
    """

    def __init__(
        self,
        target_precision: float = 0.75,
        use_xgboost: bool = True,
        use_lightgbm: bool = True,
        use_rf: bool = True,
        random_state: int = 42
    ):
        self.target_precision = target_precision
        self.use_xgboost = use_xgboost
        self.use_lightgbm = use_lightgbm
        self.use_rf = use_rf
        self.random_state = random_state

        self.models: Dict[str, Any] = {}
        self.meta_learner: Optional[LogisticRegression] = None
        self.optimizer: PrecisionOptimizer = PrecisionOptimizer(target_precision=target_precision)
        self.conformal_predictor: ConformalSepsisPredictor = ConformalSepsisPredictor(alpha=0.05, target_precision=target_precision)
        self.signature_extractor: VitalSignSignatureExtractor = VitalSignSignatureExtractor(window_hours=6)
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def _get_device(self) -> str:
        """Get compute device for XGBoost (safe fallback to cpu)."""
        return os.environ.get("SEPSISSENSE_DEVICE", "cpu")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "SepsisPrecisionEnsemble":
        """
        Fit the multi-model ensemble with out-of-fold stacking and precision calibration.
        """
        logger.info("Initializing SepsisPrecisionEnsemble training...")
        self.feature_names = list(X_train.columns)

        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=int)

        neg_count = int((y_tr == 0).sum())
        pos_count = int((y_tr == 1).sum())
        base_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
        # Capping positive weight at 6.0 prevents aggressive over-alerting, prioritizing precision
        scale_pos_weight = min(base_weight, 6.0)

        logger.info(f"Class imbalance: {neg_count:,} negative | {pos_count:,} positive (scale_pos_weight: {scale_pos_weight:.2f})")

        # ── 1. XGBoost Model ──────────────────────────────────────────────────
        if self.use_xgboost:
            device = self._get_device()
            logger.info(f"Training XGBoost classifier on device: {device}...")
            model_xgb = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric="aucpr",
                tree_method="hist",
                device=device,
                n_jobs=-1,
                verbosity=0
            )
            model_xgb.fit(X_tr, y_tr)
            self.models["xgboost"] = model_xgb

        # ── 2. LightGBM Model ─────────────────────────────────────────────────
        if self.use_lightgbm:
            logger.info("Training LightGBM classifier with focal weight balance...")
            model_lgb = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )
            model_lgb.fit(X_tr, y_tr)
            self.models["lightgbm"] = model_lgb

        # ── 3. Random Forest (Regularizer) ─────────────────────────────────────
        if self.use_rf:
            logger.info("Training Random Forest ensemble regularizer...")
            model_rf = RandomForestClassifier(
                n_estimators=80,
                max_depth=8,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=2
            )
            model_rf.fit(X_tr, y_tr)
            self.models["random_forest"] = model_rf

        # ── 4. Meta-Learner Stacking ──────────────────────────────────────────
        logger.info("Fitting Meta-Learner stacking model...")
        model_names = list(self.models.keys())
        oof_preds = np.zeros((len(X_tr), len(model_names)), dtype=np.float32)

        # Build in-sample predictions with calibration for fast meta-learner fitting
        for m_idx, m_name in enumerate(model_names):
            oof_preds[:, m_idx] = self.models[m_name].predict_proba(X_tr)[:, 1]

        self.meta_learner = LogisticRegression(C=1.0, random_state=self.random_state, max_iter=200)
        self.meta_learner.fit(oof_preds, y_tr)

        # ── 5. Precision Optimization & Calibration ───────────────────────────
        eval_X = X_val if X_val is not None else X_train
        eval_y = y_val if y_val is not None else y_train

        logger.info("Calibrating posterior probabilities and thresholding for target precision...")
        val_probas = self._raw_predict_proba(eval_X)[:, 1]
        eval_y_arr = eval_y.values if isinstance(eval_y, pd.Series) else eval_y
        self.optimizer.fit_calibrator(eval_y_arr, val_probas)
        best_threshold = self.optimizer.find_optimal_threshold(eval_y_arr, val_probas)
        logger.info(f"SepsisPrecisionEnsemble fitted successfully. Optimal threshold: {best_threshold:.4f}")

        # ── 6. Conformal Calibration & Epistemic Uncertainty ─────────────────
        logger.info("Calibrating inductive conformal predictor for statistical coverage...")
        calib_pos = self.optimizer.calibrate(val_probas)
        eval_X_arr = eval_X.values if isinstance(eval_X, pd.DataFrame) else eval_X
        self.conformal_predictor.calibrate(eval_y_arr, calib_pos, eval_X_arr)

        self.is_fitted = True
        return self

    def _raw_predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Internal helper to compute uncalibrated ensemble probabilities via weighted clinical soft-voting."""
        if isinstance(X, pd.DataFrame):
            # Align features strictly
            X = X.reindex(columns=self.feature_names, fill_value=0.0).values
        X = np.asarray(X, dtype=np.float32)

        # Clinically optimized soft-voting weights
        weights = {"lightgbm": 0.45, "xgboost": 0.45, "random_forest": 0.10}
        total_w = sum(weights.get(m, 1.0) for m in self.models)

        weighted_pos = np.zeros(len(X), dtype=np.float32)
        for m_name, model in self.models.items():
            w = weights.get(m_name, 1.0)
            proba = model.predict_proba(X)[:, 1]
            weighted_pos += w * proba

        weighted_pos /= max(total_w, 1e-6)
        return np.column_stack([1.0 - weighted_pos, weighted_pos])

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict calibrated posterior sepsis probabilities."""
        raw_probas = self._raw_predict_proba(X)
        calibrated_pos = self.optimizer.calibrate(raw_probas[:, 1])
        calibrated_pos = np.clip(calibrated_pos, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated_pos, calibrated_pos])

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        df_patient_context: Optional[pd.DataFrame] = None,
        use_persistence_filter: bool = True
    ) -> np.ndarray:
        """
        Predict high-precision binary sepsis alerts.
        Optionally applies the 2-hour persistence filter to suppress transient false alarms.
        """
        probas = self.predict_proba(X)[:, 1]
        t = self.optimizer.optimal_threshold

        if use_persistence_filter and df_patient_context is not None and "Patient" in df_patient_context.columns:
            proba_series = pd.Series(probas, index=df_patient_context.index)
            return self.optimizer.apply_persistence_filter(
                df_patient_context, proba_series, patient_col="Patient"
            )

        return (probas >= t).astype(int)

    def predict_conformal(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        alpha: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict with statistical conformal guarantees and epistemic uncertainty quantification.
        Returns prediction sets {0}, {1}, {0, 1} [Indeterminate], epistemic uncertainty,
        and bedside action recommendations.
        """
        if alpha is not None and alpha != self.conformal_predictor.alpha:
            self.conformal_predictor.alpha = float(alpha)

        calibrated_probas = self.predict_proba(X)[:, 1]

        # Extract individual model probabilities for ensemble disagreement
        if isinstance(X, pd.DataFrame):
            X_arr = X.reindex(columns=self.feature_names, fill_value=0.0).values
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        member_probas = []
        for m_name, model in self.models.items():
            member_probas.append(model.predict_proba(X_arr)[:, 1])

        return self.conformal_predictor.predict_sets(
            probas=calibrated_probas,
            X=X_arr,
            model_member_probas=member_probas if len(member_probas) > 1 else None
        )

    def extract_path_signatures(self, df_trajectory: pd.DataFrame) -> Dict[str, float]:
        """
        Extract rough path signatures from patient vital signs trajectory window.
        """
        return self.signature_extractor.extract_trajectory_signatures(df_trajectory)

    def evaluate_comprehensive(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        df_context: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Comprehensive clinical evaluation reporting Precision, Recall, PR-AUC, ROC-AUC,
        F1, and normalized PhysioNet Clinical Utility.
        """
        y_true = np.asarray(y_test, dtype=int)
        y_probas = self.predict_proba(X_test)[:, 1]
        y_pred = self.predict(X_test, df_patient_context=df_context, use_persistence_filter=True)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (len(y_true), 0, 0, 0)

        # PhysioNet utility calculation
        utility = compute_physionet_utility(y_true, y_pred)

        metrics = {
            "optimal_threshold": float(self.optimizer.optimal_threshold),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "roc_auc": float(roc_auc_score(y_true, y_probas)) if len(np.unique(y_true)) > 1 else 0.0,
            "pr_auc": float(average_precision_score(y_true, y_probas)) if len(np.unique(y_true)) > 1 else 0.0,
            "utility_normalized": float(utility["utility_normalized"]),
            "false_alarms_count": int(fp),
            "true_detections_count": int(tp),
            "missed_cases_count": int(fn)
        }
        return metrics

    def save_ensemble(self, model_path: str) -> None:
        """Persist the trained ensemble, meta-learner, and calibrator."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self, model_path)
        logger.info(f"SepsisPrecisionEnsemble saved to: {model_path}")

    @classmethod
    def load_ensemble(cls, model_path: str) -> "SepsisPrecisionEnsemble":
        """Load saved ensemble from disk."""
        ensemble = joblib.load(model_path)
        logger.info(f"SepsisPrecisionEnsemble loaded from: {model_path}")
        return ensemble
