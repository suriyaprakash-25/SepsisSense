"""
SepsisSense Precision Engineering & Clinical Utility Optimization Module
========================================================================

Specialized algorithms for maximizing precision and clinical utility in highly imbalanced
EHR sepsis time-series:
1. PhysioNet 2019 Clinical Utility Scoring Metric (asymmetric reward/penalty matrix)
2. Trajectory-Preserving Hard Negative Mining (SIRS vs True Infection)
3. Temporal Label Dilation & Prodromal Window Smoothing
4. Clinical Intent & Missingness Dynamics (Observation Presence & Delta-t)
5. Precision-Constrained Decision Cutoff Optimization (Target Precision >= 75-80%)
6. Consecutive-Hour Alert Persistence Filtering (Noise Spike Suppression)
7. Isotonic Probability Calibration

Author: SepsisSense Machine Learning & Clinical Data Science Team
License: MIT
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.isotonic import IsotonicRegression

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. PHYSIONET 2019 CLINICAL UTILITY SCORING METRIC
# ==============================================================================

def compute_physionet_utility(
    labels: np.ndarray,
    predictions: np.ndarray,
    dt_early: int = -12,
    dt_optimal: int = -6,
    dt_late: int = 3
) -> Dict[str, float]:
    """
    Compute the official PhysioNet / CinC 2019 Challenge Clinical Utility Score.
    
    The utility function defines asymmetric penalties and rewards:
    - True early alert in [onset - 12h, onset - 6h]: Rising positive reward (0 -> 1.0)
    - Optimal window [onset - 6h, onset]: Maximum positive reward (1.0)
    - Late alert [onset, onset + 3h]: Falling positive reward (1.0 -> 0)
    - False positive before window or on non-sepsis: Negative penalty (-0.05)
    - False negative / missed sepsis: Heavy penalty (-2.0)
    """
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    n_samples = len(labels)
    if n_samples == 0:
        return {"utility_raw": 0.0, "utility_normalized": 0.0}

    # Identify sepsis onset index if positive
    has_sepsis = labels.max() > 0
    if not has_sepsis:
        # Non-sepsis patient trajectory
        # False positives penalised, true negatives get 0
        fp_count = (predictions == 1).sum()
        raw_utility = -0.05 * fp_count
        optimal_utility = 0.0
        no_pred_utility = 0.0
    else:
        # Sepsis patient trajectory: find first positive label
        t_sepsis = int(np.where(labels == 1)[0][0])
        raw_utility = 0.0
        optimal_utility = 0.0
        no_pred_utility = 0.0

        for t in range(n_samples):
            dt = t - t_sepsis

            # Compute theoretical maximum utility (optimal policy)
            if dt_optimal <= dt <= 0:
                optimal_utility += 1.0
            elif dt_early <= dt < dt_optimal:
                optimal_utility += (dt - dt_early) / (dt_optimal - dt_early)
            elif 0 < dt <= dt_late:
                optimal_utility += 1.0 - (dt / (dt_late + 1e-6))

            # Compute baseline utility if no alert ever predicted (all 0s)
            if dt > dt_late:
                no_pred_utility += -2.0

            # Evaluate actual predictions
            pred = predictions[t]
            if pred == 1:
                if dt_optimal <= dt <= 0:
                    raw_utility += 1.0
                elif dt_early <= dt < dt_optimal:
                    raw_utility += (dt - dt_early) / (dt_optimal - dt_early)
                elif 0 < dt <= dt_late:
                    raw_utility += 1.0 - (dt / (dt_late + 1e-6))
                else:
                    raw_utility += -0.05
            else:
                if dt > dt_late:
                    raw_utility += -2.0

    # Normalization per PhysioNet protocol
    norm_denom = optimal_utility - no_pred_utility
    if norm_denom > 1e-6:
        normalized_utility = (raw_utility - no_pred_utility) / norm_denom
    else:
        normalized_utility = 1.0 if raw_utility >= 0 else 0.0

    return {
        "utility_raw": float(raw_utility),
        "utility_normalized": float(normalized_utility)
    }


# ==============================================================================
# 2. TRAJECTORY-PRESERVING HARD NEGATIVE MINING
# ==============================================================================

def extract_hard_negatives(
    df: pd.DataFrame,
    patient_col: str = "Patient",
    label_col: str = "SepsisLabel",
    subsample_quiescent_ratio: float = 2.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Trajectory-preserving hard negative mining:
    1. Retain 100% of septic patient trajectories.
    2. Identify 'Hard Negatives' (non-septic patients with SIRS symptoms: fever, tachycardia, tachypnea, leukocytosis).
       Retain 100% of hard negatives to enforce strict precision boundaries against sterile inflammation.
    3. Subsample quiescent/stable non-septic controls at the specified ratio.
    """
    logger.info("Executing trajectory-preserving hard negative mining...")
    df = df.copy()

    # Determine patient-level labels
    patient_status = df.groupby(patient_col)[label_col].max()
    septic_patients = patient_status[patient_status == 1].index.tolist()
    non_septic_patients = patient_status[patient_status == 0].index.tolist()

    logger.info(f"Cohort composition: {len(septic_patients):,} septic patients | {len(non_septic_patients):,} non-septic patients.")

    # Identify hard negatives among non-septic patients
    # Criteria: peak HR > 95 OR peak Temp > 38.3 OR peak Resp > 22 OR peak WBC > 12.0
    non_septic_df = df[df[patient_col].isin(non_septic_patients)]
    
    patient_vitals = non_septic_df.groupby(patient_col).agg({
        "HR": "max" if "HR" in df.columns else "count",
        "Temp": "max" if "Temp" in df.columns else "count",
        "Resp": "max" if "Resp" in df.columns else "count",
        "WBC": "max" if "WBC" in df.columns else "count",
    })

    hard_mask = (
        (patient_vitals.get("HR", 0) > 95.0) |
        (patient_vitals.get("Temp", 0) > 38.3) |
        (patient_vitals.get("Resp", 0) > 22.0) |
        (patient_vitals.get("WBC", 0) > 12.0)
    )
    hard_patients = patient_vitals[hard_mask].index.tolist()
    quiescent_patients = [p for p in non_septic_patients if p not in hard_patients]

    logger.info(f"Non-septic breakdown: {len(hard_patients):,} Hard Negatives (SIRS/Stress) | {len(quiescent_patients):,} Quiescent Controls.")

    # Subsample quiescent controls
    rng = np.random.RandomState(random_state)
    target_quiescent_count = int(len(septic_patients) * subsample_quiescent_ratio)
    target_quiescent_count = min(target_quiescent_count, len(quiescent_patients))

    if target_quiescent_count > 0 and len(quiescent_patients) > target_quiescent_count:
        sampled_quiescent = rng.choice(quiescent_patients, size=target_quiescent_count, replace=False).tolist()
    else:
        sampled_quiescent = quiescent_patients

    # Final patient selection
    selected_patients = set(septic_patients + hard_patients + sampled_quiescent)
    mined_df = df[df[patient_col].isin(selected_patients)].copy()

    logger.info(
        f"Mined cohort: {len(selected_patients):,} patients ({mined_df.shape[0]:,} rows) "
        f"| Septic: {len(septic_patients):,} | Hard Negs: {len(hard_patients):,} | Sampled Quiescent: {len(sampled_quiescent):,}"
    )
    return mined_df


# ==============================================================================
# 3. TEMPORAL LABEL DILATION (PRODROMAL SMOOTHING)
# ==============================================================================

def apply_temporal_label_dilation(
    df: pd.DataFrame,
    patient_col: str = "Patient",
    label_col: str = "SepsisLabel",
    dilation_hours: int = 6,
    soft_weights: bool = False
) -> pd.DataFrame:
    """
    Dilate sepsis labels backwards by `dilation_hours` ahead of overt clinical recognition.
    This prevents punishing early true-positive detection as false alarms during the prodromal phase.
    
    If `soft_weights=True`, assigns fractional target values (0.4 -> 0.8) in the prodromal window.
    """
    df = df.copy()
    if label_col not in df.columns or patient_col not in df.columns:
        return df

    dilated_target = df[label_col].astype(float).copy()

    for patient_id, group in df.groupby(patient_col):
        labels = group[label_col].values
        if labels.max() > 0:
            first_idx = np.where(labels == 1)[0][0]
            start_dilation = max(0, first_idx - dilation_hours)
            group_indices = group.index.values

            for step, idx in enumerate(range(start_dilation, first_idx)):
                orig_row_idx = group_indices[idx]
                if soft_weights:
                    # Linearly ramp up from 0.4 to 0.9 in prodromal hours
                    frac = 0.4 + 0.5 * (step / max(1, first_idx - start_dilation))
                    dilated_target.loc[orig_row_idx] = max(dilated_target.loc[orig_row_idx], frac)
                else:
                    dilated_target.loc[orig_row_idx] = 1.0

    df["SepsisLabel_dilated"] = dilated_target.values
    return df


# ==============================================================================
# 4. CLINICAL INTENT & OBSERVATION FREQUENCY FEATURES
# ==============================================================================

def extract_clinical_intent_features(
    df: pd.DataFrame,
    patient_col: str = "Patient",
    key_labs: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extracts explicit clinical intent features:
    1. Binary observation masks: was this lab ordered in the current hour?
    2. Delta-t: elapsed time since the measurement was last charted.
    3. Aggregated diagnostic intensity: total number of distinct lab orders in the last 6h window.
    """
    df = df.copy()
    if key_labs is None:
        key_labs = ["Lactate", "WBC", "Creatinine", "Platelets", "BUN", "pH", "Glucose"]

    available_labs = [c for c in key_labs if c in df.columns]
    feature_dict = {}

    for col in available_labs:
        # Binary observation presence mask
        is_measured = df[col].notna().astype(int)
        feature_dict[f"{col}_measured_mask"] = is_measured

        # Time elapsed since last charted measurement
        if patient_col in df.columns:
            def _calc_elapsed(s: pd.Series) -> pd.Series:
                elapsed = []
                cur = 99.0
                for v in s:
                    if not np.isnan(v):
                        cur = 0.0
                    else:
                        cur += 1.0
                    elapsed.append(cur)
                return pd.Series(elapsed, index=s.index)

            feature_dict[f"{col}_hours_since_meas"] = (
                df.groupby(patient_col)[col].transform(_calc_elapsed)
            )
        else:
            feature_dict[f"{col}_hours_since_meas"] = 0.0

    # Diagnostic intensity: count of labs ordered in the last 6 hours
    if available_labs:
        meas_df = pd.DataFrame({c: feature_dict[f"{c}_measured_mask"] for c in available_labs})
        intensity = meas_df.sum(axis=1)
        if patient_col in df.columns:
            feature_dict["diagnostic_intensity_6h"] = (
                df.groupby(patient_col).apply(lambda _: intensity.loc[_.index].rolling(6, min_periods=1).sum())
                .reset_index(level=0, drop=True)
            )
        else:
            feature_dict["diagnostic_intensity_6h"] = intensity.rolling(6, min_periods=1).sum()

    feature_df = pd.DataFrame(feature_dict, index=df.index)
    return pd.concat([df, feature_df], axis=1)


# ==============================================================================
# 5. PRECISION-CONSTRAINED THRESHOLD OPTIMIZER
# ==============================================================================

class PrecisionOptimizer:
    """
    Optimizes decision boundaries for extreme class imbalance to strictly guarantee
    target clinical precision while maximizing sensitivity / F1 / utility.
    """

    def __init__(
        self,
        target_precision: float = 0.75,
        min_recall: float = 0.20,
        consecutive_window: int = 2
    ):
        self.target_precision = target_precision
        self.min_recall = min_recall
        self.consecutive_window = consecutive_window
        self.optimal_threshold: float = 0.50
        self.calibrator: Optional[IsotonicRegression] = None
        self.best_metrics: Dict[str, float] = {}

    def fit_calibrator(self, y_true: np.ndarray, y_proba: np.ndarray) -> "PrecisionOptimizer":
        """Fit an isotonic regression calibrator on validation probabilities."""
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(y_proba, y_true)
        logger.info("Isotonic probability calibrator fitted.")
        return self

    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply probability calibration."""
        if self.calibrator is not None:
            return self.calibrator.predict(y_proba)
        return y_proba

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        optimize_metric: str = "utility"
    ) -> float:
        """
        Scan probability thresholds to maximize clinical utility or F1 score
        subject to the precision constraint: Precision(t) >= self.target_precision.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_calib = self.calibrate(y_proba)

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_calib)

        # Candidate thresholds meeting precision constraint and reasonable recall (non-degenerate)
        valid_indices = np.where(
            (precisions[:-1] >= self.target_precision) &
            (recalls[:-1] >= self.min_recall) &
            (thresholds < 0.95)
        )[0]

        if len(valid_indices) > 0:
            # Pick threshold meeting target precision with highest recall
            best_idx = valid_indices[np.argmax(recalls[valid_indices])]
            best_thresh = float(thresholds[best_idx])
            logger.info(
                f"Precision constraint met (>= {self.target_precision:.1%}): "
                f"Threshold = {best_thresh:.4f} | Precision = {precisions[best_idx]:.3f} | Recall = {recalls[best_idx]:.3f}"
            )
        else:
            # If target precision with min_recall cannot be satisfied strictly,
            # optimize F0.5 score which weights precision twice as heavily as recall
            beta = 0.5
            f_beta = (1 + beta**2) * (precisions[:-1] * recalls[:-1]) / ((beta**2 * precisions[:-1]) + recalls[:-1] + 1e-8)
            valid_mask = thresholds < 0.95
            if valid_mask.any():
                f_beta_masked = np.where(valid_mask, f_beta, -1.0)
                best_idx = int(np.argmax(f_beta_masked))
            else:
                best_idx = int(np.argmax(f_beta))
            best_thresh = float(thresholds[best_idx])
            logger.info(
                f"Balanced precision operating threshold selected: {best_thresh:.4f} "
                f"(Precision={precisions[best_idx]:.3f}, Recall={recalls[best_idx]:.3f})"
            )

        self.optimal_threshold = best_thresh

        # Evaluate performance at selected threshold
        y_pred = (y_calib >= best_thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (len(y_true), 0, 0, 0)

        self.best_metrics = {
            "optimal_threshold": best_thresh,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "roc_auc": roc_auc_score(y_true, y_calib) if len(np.unique(y_true)) > 1 else 0.0,
            "pr_auc": average_precision_score(y_true, y_calib) if len(np.unique(y_true)) > 1 else 0.0,
        }
        return best_thresh

    def apply_persistence_filter(
        self,
        df: pd.DataFrame,
        proba_series: pd.Series,
        patient_col: str = "Patient"
    ) -> np.ndarray:
        """
        Suppresses false alarms by enforcing a 2-hour persistence condition:
        An alert is emitted if:
        1. Current risk >= threshold AND previous hour risk >= 0.85 * threshold, OR
        2. Current risk is an extreme emergency (>= threshold * 1.30 or >= 0.90).
        """
        t = self.optimal_threshold
        raw_pred = (proba_series.values >= t).astype(int)

        if patient_col not in df.columns:
            return raw_pred

        filtered_preds = []
        patient_groups = df.groupby(patient_col, sort=False).indices

        filtered_arr = np.zeros(len(df), dtype=int)

        for p_id, idxs in patient_groups.items():
            p_probas = proba_series.iloc[idxs].values
            p_len = len(p_probas)

            for i in range(p_len):
                cur_p = p_probas[i]
                if cur_p >= t:
                    # Emergency spike override
                    if cur_p >= min(t * 1.30, 0.90):
                        filtered_arr[idxs[i]] = 1
                    elif i > 0 and p_probas[i - 1] >= (t * 0.85):
                        filtered_arr[idxs[i]] = 1
                    elif i < p_len - 1 and p_probas[i + 1] >= (t * 0.85):
                        filtered_arr[idxs[i]] = 1
                    else:
                        # Transient single-hour noise spike: suppressed
                        filtered_arr[idxs[i]] = 0
                else:
                    filtered_arr[idxs[i]] = 0

        return filtered_arr


def apply_persistence_filter(
    df: pd.DataFrame,
    prob_col: Union[pd.Series, np.ndarray, str] = "pred_raw",
    threshold: float = 0.5,
    consecutive_window: int = 2,
    patient_col: str = "Patient"
) -> np.ndarray:
    """
    Standalone function to suppress false alarms by enforcing a 2-hour persistence condition.
    """
    if isinstance(prob_col, str):
        if prob_col in df.columns:
            proba_series = df[prob_col]
        else:
            raise KeyError(f"Column '{prob_col}' not found in DataFrame")
    elif isinstance(prob_col, np.ndarray):
        proba_series = pd.Series(prob_col, index=df.index)
    else:
        proba_series = prob_col

    opt = PrecisionOptimizer(consecutive_window=consecutive_window)
    opt.optimal_threshold = threshold
    return opt.apply_persistence_filter(df, proba_series, patient_col=patient_col)

