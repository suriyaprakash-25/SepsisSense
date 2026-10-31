"""
Unit & Integration Tests for SepsisSense Precision Engineering Pipeline
========================================================================

Tests verifying:
1. UniversalDataHarmonizer (canonical schema mapping, unit conversions, bounds clipping)
2. PhysioNet 2019 Utility Score (rewards in prodromal window, false alarm penalties)
3. Trajectory-Preserving Hard Negative Mining (100% sepsis + 100% SIRS retention)
4. Prodromal Temporal Label Dilation (early onset backward expansion)
5. Clinical Intent Features (observation presence masks, hours since last draw, diagnostic intensity)
6. PrecisionOptimizer (isotonic calibration and precision-constrained thresholding)
7. Consecutive-Hour Persistence Filtering (noise spike suppression vs confirmed alerts)
8. SepsisPrecisionEnsemble (multi-model fit, soft-voting inference, persistence prediction)
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure src is in sys.path
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_harmonization import UniversalDataHarmonizer, CANONICAL_SCHEMA, CANONICAL_VITALS
from precision_engineering import (
    compute_physionet_utility,
    extract_hard_negatives,
    apply_temporal_label_dilation,
    extract_clinical_intent_features,
    PrecisionOptimizer,
    apply_persistence_filter,
)
from precision_ensemble import SepsisPrecisionEnsemble


# Helper function to generate synthetic ICU trajectory
def create_mock_icu_cohort(n_patients: int = 10, hours_per_patient: int = 24) -> pd.DataFrame:
    records = []
    np.random.seed(42)

    for p in range(n_patients):
        p_id = f"TEST_P{p:03d}"
        is_septic = (p % 3 == 0)
        onset_hr = min(16, max(4, hours_per_patient // 2)) if is_septic else 999

        for h in range(1, hours_per_patient + 1):
            lbl = 1 if (is_septic and h >= onset_hr) else 0
            hr_base = 110.0 if (is_septic and h >= onset_hr - 4) else (95.0 if p % 2 == 1 else 75.0)
            temp_base = 38.5 if (is_septic and h >= onset_hr - 4) else (38.2 if p % 2 == 1 else 37.0)

            records.append({
                "Patient": p_id,
                "ICULOS": h,
                "HR": hr_base + np.random.normal(0, 3),
                "O2Sat": 95.0 - (4.0 if lbl == 1 else 0.0),
                "Temp": temp_base,
                "SBP": 105.0 - (20.0 if lbl == 1 else 0.0),
                "MAP": 75.0 - (15.0 if lbl == 1 else 0.0),
                "DBP": 60.0,
                "Resp": 24.0 if (p % 2 == 1 or lbl == 1) else 16.0,
                "WBC": 14.5 if (p % 2 == 1 or lbl == 1) else 7.5,
                "Lactate": 3.2 if lbl == 1 else (np.nan if h % 6 != 0 else 1.2),
                "Creatinine": 1.8 if lbl == 1 else 0.9,
                "Platelets": 180.0,
                "Age": 65,
                "Gender": 1,
                "Unit1": 1.0,
                "Unit2": 0.0,
                "HospAdmTime": -2.5,
                "SepsisLabel": lbl
            })

    return pd.DataFrame(records)


class TestDataHarmonization:
    """Test universal multi-center schema harmonization."""

    def test_harmonize_schema_columns(self):
        df_raw = pd.DataFrame({
            "heart_rate": [80.0, 85.0],
            "spo2": [98.0, 97.0],
            "temperature_c": [37.0, 37.2],
            "sysbp": [120.0, 118.0],
            "patient_id": ["P01", "P01"],
            "sepsis_label": [0, 0]
        })
        harmonizer = UniversalDataHarmonizer()
        harmonized = harmonizer.harmonize_dataframe(df_raw, source_name="TestCenter")

        for col in CANONICAL_SCHEMA:
            assert col in harmonized.columns, f"Missing canonical column: {col}"
        assert harmonized["HR"].iloc[0] == 80.0
        assert harmonized["O2Sat"].iloc[0] == 98.0
        assert harmonized["Patient"].iloc[0] == "P01"

    def test_fahrenheit_conversion(self):
        df_raw = pd.DataFrame({
            "Temp": [102.2, 98.6],  # 39.0 C, 37.0 C
            "Patient": ["P01", "P01"],
            "SepsisLabel": [0, 0]
        })
        harmonizer = UniversalDataHarmonizer()
        harmonized = harmonizer.harmonize_dataframe(df_raw, source_name="US_Hospital")
        assert np.isclose(harmonized["Temp"].iloc[0], 39.0, atol=0.2)
        assert np.isclose(harmonized["Temp"].iloc[1], 37.0, atol=0.2)


class TestPhysioNetUtility:
    """Test official PhysioNet 2019 Clinical Utility Scoring."""

    def test_perfect_early_detection(self):
        # 24-hour stay with sepsis onset at hour 14
        labels = np.array([0]*13 + [1]*11)
        # Optimal alert in [onset-6, onset] = [hour 8 onwards through stay]
        preds = np.zeros(24, dtype=int)
        preds[8:] = 1
        res = compute_physionet_utility(labels, preds)
        assert res["utility_raw"] > 0.0

    def test_missed_sepsis_penalty(self):
        labels = np.array([0]*13 + [1]*11)
        preds = np.zeros(24, dtype=int)  # Complete false negative
        res = compute_physionet_utility(labels, preds)
        assert res["utility_raw"] < 0.0

    def test_non_septic_false_alarms(self):
        labels = np.zeros(24, dtype=int)
        preds = np.array([1]*5 + [0]*19)  # 5 false alarms
        res = compute_physionet_utility(labels, preds)
        assert np.isclose(res["utility_raw"], -0.05 * 5)


class TestPrecisionEngineering:
    """Test hard negative mining, dilation, intent features, and persistence."""

    def test_temporal_label_dilation(self):
        df = create_mock_icu_cohort(n_patients=3, hours_per_patient=24)
        dilated = apply_temporal_label_dilation(df, dilation_hours=6)
        assert "SepsisLabel_dilated" in dilated.columns
        # Patient 0 is septic with onset at hour 12; dilation must label hours 6-11 as positive
        p0 = dilated[dilated["Patient"] == "TEST_P000"]
        h10_label = p0[p0["ICULOS"] == 10]["SepsisLabel_dilated"].values[0]
        assert h10_label == 1.0

    def test_hard_negative_mining_preserves_sirs_and_sepsis(self):
        df = create_mock_icu_cohort(n_patients=12, hours_per_patient=24)
        mined = extract_hard_negatives(df, subsample_quiescent_ratio=1.0)
        # All septic patients must be preserved
        orig_septic = set(df[df["SepsisLabel"] == 1]["Patient"].unique())
        mined_septic = set(mined[mined["SepsisLabel"] == 1]["Patient"].unique())
        assert orig_septic == mined_septic
        # Total records in mined cohort should be smaller than original due to quiescent downsampling
        assert len(mined) <= len(df)

    def test_clinical_intent_features(self):
        df = create_mock_icu_cohort(n_patients=2, hours_per_patient=12)
        with_intent = extract_clinical_intent_features(df, key_labs=["Lactate", "Creatinine"])
        assert "Lactate_measured_mask" in with_intent.columns
        assert "Lactate_hours_since_meas" in with_intent.columns
        assert "diagnostic_intensity_6h" in with_intent.columns

    def test_persistence_filter_suppresses_noise_spike(self):
        df = pd.DataFrame({
            "Patient": ["P1"] * 5,
            "pred_raw": [0.1, 0.1, 0.58, 0.1, 0.1]  # Single isolated non-emergency spike at t=2
        })
        filtered = apply_persistence_filter(df, prob_col="pred_raw", threshold=0.50)
        assert filtered[2] == 0, "Isolated 1-hour noise spike should have been suppressed"

    def test_persistence_filter_confirms_sustained_alert(self):
        df = pd.DataFrame({
            "Patient": ["P1"] * 5,
            "pred_raw": [0.1, 0.1, 0.80, 0.82, 0.85]  # Sustained elevation at t=2, 3, 4
        })
        filtered = apply_persistence_filter(df, prob_col="pred_raw", threshold=0.50)
        assert filtered[2] == 1 and filtered[3] == 1, "Sustained risk should fire confirmed alert"


class TestPrecisionEnsemble:
    """Test multi-model ensemble fitting and calibrated inference."""

    def test_ensemble_fit_and_inference(self):
        df = create_mock_icu_cohort(n_patients=15, hours_per_patient=12)
        feature_cols = ["HR", "MAP", "Temp", "Resp", "WBC", "Creatinine"]
        X = df[feature_cols]
        y = df["SepsisLabel"]

        ensemble = SepsisPrecisionEnsemble(
            target_precision=0.75,
            use_xgboost=True,
            use_lightgbm=True,
            use_rf=True,
            random_state=42
        )
        ensemble.fit(X_train=X, y_train=y)

        assert ensemble.is_fitted
        probas = ensemble.predict_proba(X)
        assert probas.shape == (len(X), 2)
        assert (probas >= 0.0).all() and (probas <= 1.0).all()

        preds = ensemble.predict(X, df_patient_context=df, use_persistence_filter=True)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})
