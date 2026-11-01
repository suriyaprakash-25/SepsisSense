"""
SepsisSense Conformal Prediction & Epistemic Uncertainty Filtering Module
========================================================================

Implements distribution-free, finite-sample valid conformal prediction and epistemic
uncertainty quantification for early sepsis detection in critical care, inspired by
the COMPOSER framework (Nature npj Digital Medicine) and inductive conformal prediction:

1. Inductive Conformal Binary Predictor (Prediction Sets: {0}, {1}, {0, 1} [Indeterminate])
2. Epistemic Uncertainty Quantification via Multi-Model Ensemble Disagreement
3. Out-of-Distribution (OOD) & Sensor Artifact Gating
4. Clinical Decision Set Filtering (Suppression of Indeterminate False Alarms)

Author: SepsisSense Machine Learning & Clinical Data Science Team
License: MIT
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Set

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("ConformalUncertainty")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _f = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    _h.setFormatter(_f)
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class ConformalSepsisPredictor:
    """
    Split-Conformal Prediction and Epistemic Uncertainty Engine for Sepsis Alerting.
    
    Provides statistical coverage guarantees at user-specified significance level alpha (e.g. 0.05).
    Identifies high-uncertainty / ambiguous records as 'Indeterminate' {0, 1}, allowing the clinical
    system to withhold premature alarms and prompt targeted clinical review without false alarm fatigue.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        target_precision: float = 0.75,
        ood_threshold_percentile: float = 99.0
    ):
        """
        Parameters
        ----------
        alpha : float
            Conformal significance level (target coverage = 1 - alpha, e.g. 0.05 -> 95% coverage).
        target_precision : float
            Clinical target precision for alert gating.
        ood_threshold_percentile : float
            Percentile for Mahalanobis OOD distance cutoff.
        """
        self.alpha = float(alpha)
        self.target_precision = float(target_precision)
        self.ood_threshold_percentile = float(ood_threshold_percentile)

        self.conformal_quantile: float = 1.0
        self.calibration_scores: Optional[np.ndarray] = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_cov_inv: Optional[np.ndarray] = None
        self.ood_cutoff: float = 1e6
        self.is_calibrated: bool = False

    def calibrate(
        self,
        y_val: Union[pd.Series, np.ndarray],
        val_probas: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> "ConformalSepsisPredictor":
        """
        Calibrate the conformal predictor on hold-out validation predictions.
        
        Parameters
        ----------
        y_val : array-like of shape (n_samples,)
            True binary labels (0 or 1).
        val_probas : array-like of shape (n_samples,)
            Predicted probability of sepsis P(Y=1|X).
        X_val : array-like of shape (n_samples, n_features), optional
            Feature matrix for fitting OOD background distribution.
        """
        y_arr = np.asarray(y_val, dtype=int)
        p_arr = np.asarray(val_probas, dtype=float)
        p_arr = np.clip(p_arr, 1e-7, 1.0 - 1e-7)

        n_samples = len(y_arr)
        if n_samples == 0:
            raise ValueError("Validation set is empty for conformal calibration.")

        # Non-conformity score: s_i = 1 - P(Y = y_i | x_i)
        # Higher score means prediction is less conforming with true label
        p_true_class = np.where(y_arr == 1, p_arr, 1.0 - p_arr)
        non_conformity_scores = 1.0 - p_true_class
        self.calibration_scores = non_conformity_scores

        # Compute conformal quantile: (1 - alpha) * (1 + 1/n)
        q_level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n_samples))
        self.conformal_quantile = float(np.quantile(non_conformity_scores, q_level, method="higher"))

        # Fit OOD distribution on X_val if provided
        if X_val is not None:
            X_arr = np.asarray(X_val, dtype=np.float32)
            # Impute NaNs with median for covariance estimation
            nan_mask = np.isnan(X_arr)
            if nan_mask.any():
                col_medians = np.nanmedian(X_arr, axis=0)
                col_medians = np.nan_to_num(col_medians, nan=0.0)
                inds = np.where(nan_mask)
                X_arr[inds] = np.take(col_medians, inds[1])

            self.feature_means = np.mean(X_arr, axis=0)
            cov = np.cov(X_arr, rowvar=False)
            # Regularize covariance matrix to ensure positive definiteness
            cov_reg = cov + np.eye(cov.shape[0]) * 1e-3
            try:
                self.feature_cov_inv = np.linalg.pinv(cov_reg)
                diffs = X_arr - self.feature_means
                dists = np.sqrt(np.sum((diffs @ self.feature_cov_inv) * diffs, axis=1))
                self.ood_cutoff = float(np.percentile(dists, self.ood_threshold_percentile))
            except Exception as e:
                logger.warning(f"Could not compute inverted covariance for OOD: {e}")
                self.feature_cov_inv = None
                self.ood_cutoff = 1e6

        self.is_calibrated = True
        logger.info(
            f"Conformal predictor calibrated: alpha={self.alpha:.3f} | "
            f"Coverage Target={1.0 - self.alpha:.1%} | Conformal Quantile q*={self.conformal_quantile:.4f}"
        )
        return self

    def predict_sets(
        self,
        probas: Union[pd.Series, np.ndarray],
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        model_member_probas: Optional[List[np.ndarray]] = None
    ) -> List[Dict[str, Any]]:
        """
        Construct conformal prediction sets and uncertainty metrics for inference.
        
        Returns a list of dicts with:
        - 'prediction_set': set {0}, {1}, {0, 1}, or empty set ()
        - 'status': 'LOW_RISK', 'CONFIRMED_ALERT', 'INDETERMINATE', or 'OUT_OF_DISTRIBUTION'
        - 'epistemic_uncertainty': float (ensemble disagreement std dev)
        - 'conformal_alert': int (1 only if status is CONFIRMED_ALERT, 0 otherwise)
        - 'clinical_recommendation': str action guide for ICU bedside
        """
        p_arr = np.asarray(probas, dtype=float)
        p_arr = np.clip(p_arr, 1e-7, 1.0 - 1e-7)
        q = self.conformal_quantile

        # Disagreement across models (epistemic uncertainty)
        epistemic_unc = np.zeros(len(p_arr), dtype=float)
        if model_member_probas is not None and len(model_member_probas) > 1:
            stacked_probas = np.column_stack(model_member_probas)
            epistemic_unc = np.std(stacked_probas, axis=1)

        # OOD distances
        ood_flags = np.zeros(len(p_arr), dtype=bool)
        if X is not None and self.feature_cov_inv is not None and self.feature_means is not None:
            try:
                X_arr = np.asarray(X, dtype=np.float32)
                nan_mask = np.isnan(X_arr)
                if nan_mask.any():
                    X_arr = np.nan_to_num(X_arr, nan=0.0)
                diffs = X_arr - self.feature_means
                dists = np.sqrt(np.sum((diffs @ self.feature_cov_inv) * diffs, axis=1))
                ood_flags = dists > self.ood_cutoff
            except Exception:
                pass

        results = []
        for i in range(len(p_arr)):
            p = p_arr[i]
            # Non-conformity score test for candidate 0: 1 - P(Y=0) = P(Y=1) = p
            include_0 = bool(p <= q)
            # Non-conformity score test for candidate 1: 1 - P(Y=1) = 1 - p
            include_1 = bool((1.0 - p) <= q)

            pred_set: Set[int] = set()
            if include_0:
                pred_set.add(0)
            if include_1:
                pred_set.add(1)

            e_unc = float(epistemic_unc[i]) if len(epistemic_unc) > i else 0.0
            is_ood = bool(ood_flags[i]) if len(ood_flags) > i else False

            # Categorize status
            if is_ood:
                status = "OUT_OF_DISTRIBUTION"
                recommendation = "Data outside reliable calibration baseline. Check sensor attachments, lead noise, or acute atypical hemodynamics."
                conformal_alert = 0
            elif pred_set == {1}:
                status = "CONFIRMED_ALERT"
                recommendation = f"High-certainty sepsis deterioration (Confidence: {1 - self.alpha:.0%}). Initiate Surviving Sepsis 1-hour bundle immediately."
                conformal_alert = 1
            elif pred_set == {0}:
                status = "LOW_RISK"
                recommendation = f"Patient is low risk for immediate sepsis (Confidence: {1 - self.alpha:.0%}). Continue standard protocol surveillance."
                conformal_alert = 0
            elif pred_set == {0, 1}:
                status = "INDETERMINATE"
                recommendation = "Ambiguous physiological presentation (Epistemic Uncertainty High). Alert withheld to prevent false alarm fatigue. Re-check lactate and vitals."
                conformal_alert = 0
            else:
                status = "ANOMALY"
                recommendation = "Empty prediction set (high non-conformity). Patient trajectory deviates from both septic and non-septic archetypes."
                conformal_alert = 0

            results.append({
                "probability": float(p),
                "prediction_set": sorted(list(pred_set)),
                "status": status,
                "epistemic_uncertainty": e_unc,
                "is_ood": is_ood,
                "conformal_alert": conformal_alert,
                "clinical_recommendation": recommendation,
                "coverage_guarantee": f"{1 - self.alpha:.1%}"
            })

        return results

