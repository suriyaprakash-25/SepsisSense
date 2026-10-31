"""
SepsisSense Multi-Dataset Retraining & Precision Engineering Pipeline
=====================================================================

Production-grade training orchestrator combining:
1. Multi-center data harmonization across Hospital Set A, Set B, and external cohorts.
2. Trajectory-preserving hard negative mining (100% sepsis + 100% SIRS patients retained).
3. Prodromal temporal label dilation (6-12 hour early alert transition smoothing).
4. Clinical intent missingness dynamics (observation masks, hours-since-measurement, diagnostic intensity).
5. Triple-model precision ensemble (LightGBM + XGBoost + Random Forest with Stacking).
6. Isotonic probability calibration and precision-constrained threshold scanning (Target >= 75%).
7. Official PhysioNet 2019 Clinical Utility Score evaluation.

Author: SepsisSense Machine Learning & Clinical Data Science Team
License: MIT
"""

import os
import sys
import glob
import time
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Setup project directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_harmonization import UniversalDataHarmonizer, CANONICAL_SCHEMA
from precision_engineering import (
    compute_physionet_utility,
    extract_hard_negatives,
    apply_temporal_label_dilation,
    extract_clinical_intent_features,
    PrecisionOptimizer,
    apply_persistence_filter,
)
from precision_ensemble import SepsisPrecisionEnsemble

logger = logging.getLogger("PrecisionRetrainer")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Core 52 production features for backward compatibility
CORE_52_FEATURES: List[str] = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3',
    'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
    'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
    'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS',
    'HR_roll6h', 'MAP_roll6h', 'Temp_roll6h', 'Resp_roll6h', 'O2Sat_roll6h',
    'shock_index', 'modified_shock_index', 'qsofa_score', 'hr_trend_3h', 'map_trend_3h',
    'lactate_creatinine_ratio', 'wbc_temp_interaction'
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract production rolling, clinical interaction, and intent features.
    Vectorized and executed efficiently across patient groups.
    """
    logger.info("Computing rolling, clinical interaction, and intent features...")
    df_out = df.copy()
    patient_col = "Patient" if "Patient" in df_out.columns else "patient_id"

    # Forward fill vitals and labs per patient to simulate bedside chart persistence
    numeric_cols = [c for c in df_out.columns if c not in [patient_col, "DatasetSource", "SepsisLabel"]]
    df_out[numeric_cols] = df_out.groupby(patient_col)[numeric_cols].ffill()

    # 1. Rolling 6-hour mean vitals
    for vital in ['HR', 'MAP', 'Temp', 'Resp', 'O2Sat']:
        col_name = f"{vital}_roll6h"
        if vital in df_out.columns:
            df_out[col_name] = (
                df_out.groupby(patient_col)[vital]
                .transform(lambda s: s.rolling(6, min_periods=1).mean())
            )
        else:
            df_out[col_name] = 0.0

    # 2. Clinical interaction features
    hr = df_out.get('HR', 0.0).fillna(80.0)
    sbp = df_out.get('SBP', 0.0).fillna(120.0)
    map_val = df_out.get('MAP', 0.0).fillna(85.0)
    resp = df_out.get('Resp', 0.0).fillna(18.0)
    lac = df_out.get('Lactate', 0.0).fillna(1.5)
    creat = df_out.get('Creatinine', 0.0).fillna(1.0)
    wbc = df_out.get('WBC', 0.0).fillna(8.0)
    temp = df_out.get('Temp', 0.0).fillna(37.0)

    df_out['shock_index'] = hr / (np.abs(sbp) + 1e-6)
    df_out['modified_shock_index'] = hr / (np.abs(map_val) + 1e-6)
    df_out['qsofa_score'] = ((resp >= 22).astype(float) + (sbp <= 100).astype(float))
    
    if 'HR' in df_out.columns:
        df_out['hr_trend_3h'] = df_out.groupby(patient_col)['HR'].diff(3).fillna(0.0)
    else:
        df_out['hr_trend_3h'] = 0.0

    if 'MAP' in df_out.columns:
        df_out['map_trend_3h'] = df_out.groupby(patient_col)['MAP'].diff(3).fillna(0.0)
    else:
        df_out['map_trend_3h'] = 0.0

    df_out['lactate_creatinine_ratio'] = lac / (np.abs(creat) + 1e-6)
    df_out['wbc_temp_interaction'] = wbc * temp

    if 'EtCO2' not in df_out.columns:
        df_out['EtCO2'] = 0.0

    # 3. Clinical intent features (diagnostic intensity, hours elapsed, observation masks)
    df_out = extract_clinical_intent_features(df_out, patient_col=patient_col)

    # Impute remaining missing values with median
    for col in df_out.columns:
        if col not in [patient_col, "DatasetSource", "SepsisLabel"]:
            median_val = df_out[col].median()
            df_out[col] = df_out[col].fillna(median_val if not pd.isna(median_val) else 0.0)

    return df_out


def train_precision_pipeline(
    data_sources: Optional[List[Dict[str, str]]] = None,
    max_patients_per_source: Optional[int] = None,
    target_precision: float = 0.75,
    negative_ratio: float = 6.0,
    save_models: bool = True
) -> Dict[str, Any]:
    """
    Execute the end-to-end multi-dataset precision retraining pipeline.
    """
    logger.info("=" * 80)
    logger.info("STARTING SEPSISSENSE MULTI-DATASET PRECISION RETRAINING PIPELINE")
    logger.info(f"Target Clinical Precision: {target_precision:.1%}")
    logger.info("=" * 80)

    harmonizer = UniversalDataHarmonizer()

    # Default data sources: Combined Hospital Set A + Set B
    if not data_sources:
        set_a_path = os.path.join(PROJECT_ROOT, "physionet.org", "files", "challenge-2019", "1.0.0", "training", "training_setA")
        set_b_path = os.path.join(PROJECT_ROOT, "physionet.org", "files", "challenge-2019", "1.0.0", "training", "training_setB")
        data_sources = []
        if os.path.isdir(set_a_path):
            data_sources.append({"source": "PhysioNet_SetA", "path": set_a_path})
        if os.path.isdir(set_b_path):
            data_sources.append({"source": "PhysioNet_SetB", "path": set_b_path})

    if not data_sources:
        raise FileNotFoundError("No valid training dataset paths found in project repository.")

    # ── Phase 1: Ingest & Harmonize Multi-Center Cohorts ───────────────────────
    logger.info(f"Ingesting {len(data_sources)} data source(s)...")
    source_dfs = []
    for cfg in data_sources:
        name = cfg["source"]
        path = cfg["path"]
        logger.info(f"Loading {name} from {path} (max_patients={max_patients_per_source})...")
        df_src = harmonizer.load_physionet_cohort(
            data_dir=path,
            max_patients=max_patients_per_source
        )
        source_dfs.append(df_src)

    raw_df = pd.concat(source_dfs, ignore_index=True)
    total_patients = raw_df["Patient"].nunique()
    total_records = len(raw_df)
    total_sepsis_rec = int(raw_df["SepsisLabel"].sum())
    sepsis_pt_count = raw_df.groupby("Patient")["SepsisLabel"].max().sum()

    logger.info(
        f"Multi-center cohort ingested: {total_patients:,} unique patients | "
        f"{total_records:,} hourly records | {sepsis_pt_count:,} septic patients "
        f"({sepsis_pt_count / total_patients:.2%}) | {total_sepsis_rec:,} septic hours "
        f"({total_sepsis_rec / total_records:.2%})"
    )

    # ── Phase 2: Feature Engineering & Clinical Intent ────────────────────────
    full_df = extract_features(raw_df)

    # ── Phase 3: Patient-Level Stratified Train/Val Split ─────────────────────
    logger.info("Partitioning patient-level stratified train / validation splits...")
    patient_sepsis = full_df.groupby("Patient")["SepsisLabel"].max().reset_index()
    train_pts, val_pts = train_test_split(
        patient_sepsis["Patient"],
        test_size=0.20,
        random_state=42,
        stratify=patient_sepsis["SepsisLabel"]
    )

    train_df = full_df[full_df["Patient"].isin(train_pts)].copy()
    val_df = full_df[full_df["Patient"].isin(val_pts)].copy()

    logger.info(
        f"Train cohort: {train_df['Patient'].nunique():,} patients, {len(train_df):,} records | "
        f"Val cohort: {val_df['Patient'].nunique():,} patients, {len(val_df):,} records"
    )

    # ── Phase 4: Temporal Label Dilation (Prodromal Transition) ───────────────
    logger.info("Applying 6-hour temporal label dilation to train cohort...")
    train_df = apply_temporal_label_dilation(train_df, dilation_hours=6)
    if "SepsisLabel_dilated" in train_df.columns:
        train_df["SepsisLabel"] = (train_df["SepsisLabel_dilated"] >= 0.5).astype(int)

    # ── Phase 5: Trajectory-Preserving Hard Negative Mining ────────────────────
    logger.info(f"Executing hard negative mining on train cohort (quiescent ratio {negative_ratio:.1f})...")
    train_df = extract_hard_negatives(train_df, subsample_quiescent_ratio=negative_ratio)

    # ── Phase 6: Prepare Feature Matrices ─────────────────────────────────────
    # Assemble feature column sets
    intent_cols = [c for c in full_df.columns if "_measured_mask" in c or "_hours_since_meas" in c or c == "diagnostic_intensity_6h"]
    ensemble_features = CORE_52_FEATURES + intent_cols

    # Ensure all features exist
    for f in ensemble_features:
        if f not in train_df.columns:
            train_df[f] = 0.0
        if f not in val_df.columns:
            val_df[f] = 0.0

    X_train_ens = train_df[ensemble_features]
    y_train_ens = train_df["SepsisLabel"]
    X_val_ens = val_df[ensemble_features]
    y_val_ens = val_df["SepsisLabel"]

    X_train_52 = train_df[CORE_52_FEATURES]
    X_val_52 = val_df[CORE_52_FEATURES]

    # ── Phase 7: Train Precision-Optimized Multi-Model Ensemble ───────────────
    logger.info("Training SepsisPrecisionEnsemble (LightGBM + XGBoost + RF + Stacking Meta-Learner)...")
    ensemble = SepsisPrecisionEnsemble(
        target_precision=target_precision,
        use_xgboost=True,
        use_lightgbm=True,
        use_rf=True,
        random_state=42
    )
    ensemble.fit(
        X_train=X_train_ens,
        y_train=y_train_ens,
        X_val=X_val_ens,
        y_val=y_val_ens
    )

    # Predict probabilities on validation set
    val_raw_proba = ensemble._raw_predict_proba(X_val_ens)[:, 1]
    val_calib_proba = ensemble.predict_proba(X_val_ens)[:, 1]

    # ── Phase 8: Optimize Precision Decision Threshold & Persistence ──────────
    logger.info(f"Scanning decision boundary for precision >= {target_precision:.1%}...")
    optimal_t = ensemble.optimizer.find_optimal_threshold(
        y_true=y_val_ens.values,
        y_proba=val_raw_proba
    )

    # Predictions before and after consecutive persistence filter
    val_preds_raw = (val_calib_proba >= optimal_t).astype(int)
    val_df_eval = val_df.copy()
    val_df_eval["pred_raw"] = val_preds_raw
    val_preds_filtered = apply_persistence_filter(
        val_df_eval, prob_col="pred_raw", threshold=optimal_t, consecutive_window=2
    )

    # ── Phase 9: Comprehensive Clinical Metrics & PhysioNet Utility ───────────
    y_val_true = y_val_ens.values
    roc_auc = float(roc_auc_score(y_val_true, val_calib_proba))
    pr_auc = float(average_precision_score(y_val_true, val_calib_proba))
    brier = float(brier_score_loss(y_val_true, val_calib_proba))

    # Metrics with optimal threshold
    prec_raw = float(precision_score(y_val_true, val_preds_raw, zero_division=0))
    rec_raw = float(recall_score(y_val_true, val_preds_raw, zero_division=0))
    f1_raw = float(f1_score(y_val_true, val_preds_raw, zero_division=0))

    # Metrics with 2-hour persistence filter
    prec_filt = float(precision_score(y_val_true, val_preds_filtered, zero_division=0))
    rec_filt = float(recall_score(y_val_true, val_preds_filtered, zero_division=0))
    f1_filt = float(f1_score(y_val_true, val_preds_filtered, zero_division=0))

    cm_raw = confusion_matrix(y_val_true, val_preds_raw)
    cm_filt = confusion_matrix(y_val_true, val_preds_filtered)
    tn_f, fp_f, fn_f, tp_f = cm_filt.ravel() if cm_filt.size == 4 else (len(y_val_true), 0, 0, 0)
    spec_filt = float(tn_f / (tn_f + fp_f)) if (tn_f + fp_f) > 0 else 1.0

    # PhysioNet clinical utility score
    val_df_eval["pred_filt"] = val_preds_filtered
    utility_raw_sum = 0.0
    for _, pt_data in val_df_eval.groupby("Patient"):
        u = compute_physionet_utility(pt_data["SepsisLabel"].values, pt_data["pred_filt"].values)
        utility_raw_sum += u["utility_raw"]
    avg_utility = float(utility_raw_sum / val_df_eval["Patient"].nunique())

    logger.info("=" * 80)
    logger.info("FINAL VALIDATION EVALUATION RESULTS")
    logger.info(f"ROC-AUC:               {roc_auc:.4f}")
    logger.info(f"PR-AUC:                {pr_auc:.4f}")
    logger.info(f"Brier Score:           {brier:.4f}")
    logger.info(f"Optimal Threshold t*:  {optimal_t:.4f}")
    logger.info(f"Precision (Raw):       {prec_raw:.2%}")
    logger.info(f"Precision (Filtered):  {prec_filt:.2%}")
    logger.info(f"Recall (Filtered):     {rec_filt:.2%}")
    logger.info(f"Specificity:           {spec_filt:.2%}")
    logger.info(f"F1 Score:              {f1_filt:.4f}")
    logger.info(f"PhysioNet Utility:     {avg_utility:.4f}")
    logger.info("=" * 80)

    # ── Phase 10: Model Artifact Serialization ────────────────────────────────
    if save_models:
        logger.info("Serializing model artifacts to models/ directory...")

        # 1. Full precision ensemble
        ens_path = os.path.join(MODELS_DIR, "sepsissense_precision_ensemble.pkl")
        ensemble.optimal_threshold = optimal_t
        joblib.dump(ensemble, ens_path, compress=3)
        logger.info(f"Saved precision ensemble to: {ens_path}")

        # 2. Standalone core XGBoost (for backward compatibility with 52 features)
        logger.info("Fitting standalone 52-feature XGBoost model for backward compatibility...")
        import xgboost as xgb
        pos_c = int((y_train_ens == 1).sum())
        neg_c = int((y_train_ens == 0).sum())
        spw = min(neg_c / max(pos_c, 1), 6.0)
        core_xgb_52 = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=spw,
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )
        core_xgb_52.fit(X_train_52, y_train_ens)
        xgb_path = os.path.join(MODELS_DIR, "sepsissense_xgb.pkl")
        joblib.dump(core_xgb_52, xgb_path, compress=3)
        logger.info(f"Saved backward-compatible XGBoost model to: {xgb_path}")

        # 3. Feature columns
        feat_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
        joblib.dump(CORE_52_FEATURES, feat_path)
        logger.info(f"Saved core 52 feature columns to: {feat_path}")

        ens_feat_path = os.path.join(MODELS_DIR, "precision_feature_columns.pkl")
        joblib.dump(ensemble_features, ens_feat_path)
        logger.info(f"Saved extended precision feature columns ({len(ensemble_features)}) to: {ens_feat_path}")

        # 4. JSON metrics report
        metrics = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data_sources": [s["source"] for s in data_sources],
            "total_patients": int(total_patients),
            "total_records": int(total_records),
            "sepsis_patients": int(sepsis_pt_count),
            "target_precision": float(target_precision),
            "optimal_threshold": float(optimal_t),
            "validation_metrics": {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_score": brier,
                "raw_precision": prec_raw,
                "filtered_precision": prec_filt,
                "filtered_recall": rec_filt,
                "filtered_f1": f1_filt,
                "specificity": spec_filt,
                "physionet_utility": avg_utility,
                "confusion_matrix": {
                    "tn": int(tn_f), "fp": int(fp_f),
                    "fn": int(fn_f), "tp": int(tp_f)
                }
            }
        }

        metrics_path = os.path.join(MODELS_DIR, "precision_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics summary to: {metrics_path}")

        # 5. Visual summary plot (PR curve & Confusion Matrix)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Subplot 1: Precision-Recall Curve
            precs, recs, _ = precision_recall_curve(y_val_true, val_calib_proba)
            axes[0].plot(recs, precs, color="#4f46e5", lw=2, label=f"Ensemble (PR-AUC = {pr_auc:.3f})")
            axes[0].axhline(y=target_precision, color="#ef4444", linestyle="--", label=f"Target Precision ({target_precision:.0%})")
            axes[0].scatter([rec_filt], [prec_filt], color="#10b981", s=100, zorder=5, label=f"Selected Op (P={prec_filt:.1%}, R={rec_filt:.1%})")
            axes[0].set_xlabel("Recall (Sensitivity)")
            axes[0].set_ylabel("Precision (Positive Predictive Value)")
            axes[0].set_title("Precision-Recall Operating Curve")
            axes[0].legend(loc="lower left")
            axes[0].grid(True, alpha=0.3)

            # Subplot 2: Confusion Matrix
            cm_matrix = np.array([[tn_f, fp_f], [fn_f, tp_f]])
            cax = axes[1].matshow(cm_matrix, cmap="Blues")
            fig.colorbar(cax, ax=axes[1])
            for i in range(2):
                for j in range(2):
                    axes[1].text(j, i, f"{cm_matrix[i, j]:,}", ha="center", va="center", color="black", fontsize=12, fontweight="bold")
            axes[1].set_xticks([0, 1])
            axes[1].set_yticks([0, 1])
            axes[1].set_xticklabels(["Predicted Non-Sepsis", "Predicted Sepsis"])
            axes[1].set_yticklabels(["True Non-Sepsis", "True Sepsis"])
            axes[1].set_title(f"Clinical Confusion Matrix (t* = {optimal_t:.2f})")

            plot_path = os.path.join(MODELS_DIR, "precision_curve_summary.png")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            logger.info(f"Saved visual evaluation summary to: {plot_path}")
        except Exception as e:
            logger.warning(f"Failed to generate evaluation plot: {e}")

    return {
        "ensemble": ensemble,
        "optimal_threshold": optimal_t,
        "metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "filtered_precision": prec_filt,
            "filtered_recall": rec_filt,
            "specificity": spec_filt,
            "physionet_utility": avg_utility
        }
    }


def main():
    parser = argparse.ArgumentParser(description="SepsisSense Precision Multi-Dataset Retraining")
    parser.add_argument("--max-patients", type=int, default=None, help="Cap patient count per cohort for fast testing")
    parser.add_argument("--target-precision", type=float, default=0.75, help="Target precision threshold (e.g., 0.75 or 0.80)")
    parser.add_argument("--neg-ratio", type=float, default=6.0, help="Negative to positive sampling ratio for hard negative mining")
    parser.add_argument("--set-a-only", action="store_true", help="Train on Set A only")
    args = parser.parse_args()

    data_sources = []
    set_a = os.path.join(PROJECT_ROOT, "physionet.org", "files", "challenge-2019", "1.0.0", "training", "training_setA")
    set_b = os.path.join(PROJECT_ROOT, "physionet.org", "files", "challenge-2019", "1.0.0", "training", "training_setB")

    if os.path.isdir(set_a):
        data_sources.append({"source": "PhysioNet_SetA", "path": set_a})
    if not args.set_a_only and os.path.isdir(set_b):
        data_sources.append({"source": "PhysioNet_SetB", "path": set_b})

    train_precision_pipeline(
        data_sources=data_sources,
        max_patients_per_source=args.max_patients,
        target_precision=args.target_precision,
        negative_ratio=args.neg_ratio,
        save_models=True
    )


if __name__ == "__main__":
    main()
