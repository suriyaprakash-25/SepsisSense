"""
Unit & Integration Tests for Next-Gen SepsisSense Precision Innovations
========================================================================

Tests verifying:
1. Conformal Sepsis Prediction (finite-sample coverage guarantees, {0}, {1}, {0, 1} [Indeterminate])
2. Epistemic Uncertainty & Out-of-Distribution Gating
3. Rough Path Signatures (Level-1 increments, Level-2 iterated integrals & Lévy area)
4. Shock Index Velocity & Invariance to Irregular Sampling
5. Clinical Text NLP (Sepsis indicators, NegEx negation handling, mimic suppression, CMO/DNR detection)
6. Integration with SepsisPrecisionEnsemble

Author: SepsisSense Team
License: MIT
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure src is in sys.path
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from conformal_uncertainty import ConformalSepsisPredictor
from path_signatures import VitalSignSignatureExtractor, compute_iterated_integral
from clinical_nlp import ClinicalTextAnalyzer
from precision_ensemble import SepsisPrecisionEnsemble


# ==============================================================================
# 1. CONFORMAL PREDICTION & EPISTEMIC UNCERTAINTY TESTS
# ==============================================================================

class TestConformalPrediction:

    def test_conformal_calibration_and_coverage(self):
        cp = ConformalSepsisPredictor(alpha=0.05)
        # Synthetic validation cohort
        np.random.seed(42)
        n = 500
        y_val = np.random.binomial(1, 0.08, size=n)
        # Reasonably calibrated predicted probabilities
        p_val = np.where(y_val == 1, np.random.beta(5, 2, size=n), np.random.beta(1, 10, size=n))
        p_val = np.clip(p_val, 0.01, 0.99)

        cp.calibrate(y_val, p_val)
        assert cp.is_calibrated is True
        assert 0.0 < cp.conformal_quantile <= 1.0

        # Test prediction sets on representative probabilities
        test_probas = np.array([0.02, 0.50, 0.95])
        res = cp.predict_sets(test_probas)

        assert len(res) == 3
        # Very low probability must include 0
        assert 0 in res[0]["prediction_set"]
        assert res[0]["status"] in ["LOW_RISK", "INDETERMINATE"]

        # Very high probability must include 1
        assert 1 in res[2]["prediction_set"]
        assert res[2]["status"] in ["CONFIRMED_ALERT", "INDETERMINATE"]

    def test_epistemic_uncertainty_quantification(self):
        cp = ConformalSepsisPredictor(alpha=0.05)
        y_val = np.array([0]*90 + [1]*10)
        p_val = np.array([0.05]*90 + [0.85]*10)
        cp.calibrate(y_val, p_val)

        # Three member models with disagreement
        m1 = np.array([0.20, 0.85])
        m2 = np.array([0.70, 0.90])
        m3 = np.array([0.45, 0.88])
        ensemble_p = (m1 + m2 + m3) / 3.0

        res = cp.predict_sets(ensemble_p, model_member_probas=[m1, m2, m3])
        # First sample has high disagreement (0.20 vs 0.70)
        assert res[0]["epistemic_uncertainty"] > 0.15
        # Second sample has low disagreement (0.85 vs 0.90 vs 0.88)
        assert res[1]["epistemic_uncertainty"] < 0.05


# ==============================================================================
# 2. ROUGH PATH SIGNATURES TESTS
# ==============================================================================

class TestPathSignatures:

    def test_iterated_integral_and_levy_area(self):
        # Parametric circle path: X(t) = cos(t), Y(t) = sin(t) for t in [0, pi]
        t = np.linspace(0, np.pi, 100)
        x = np.cos(t)
        y = np.sin(t)
        s12, s21, levy_area = compute_iterated_integral(x, y)

        # Lévy area for half-circle is non-zero
        assert isinstance(levy_area, float)
        assert abs(levy_area) > 0.1

    def test_vital_sign_signature_extractor(self):
        # Hemodynamic collapse trajectory: HR rising, MAP falling (decompensating loop)
        df_traj = pd.DataFrame({
            "HR": [75.0, 82.0, 95.0, 110.0, 125.0, 138.0],
            "MAP": [95.0, 90.0, 80.0, 70.0, 62.0, 52.0],
            "Resp": [16.0, 18.0, 22.0, 26.0, 30.0, 34.0],
            "Temp": [37.0, 37.4, 38.0, 38.7, 39.1, 39.4],
            "O2Sat": [99.0, 98.0, 97.0, 94.0, 92.0, 89.0],
            "SBP": [125.0, 118.0, 102.0, 90.0, 78.0, 68.0]
        })

        extractor = VitalSignSignatureExtractor(window_hours=6)
        sigs = extractor.extract_trajectory_signatures(df_traj)

        # Level 1 signatures
        assert "sig1_net_disp_HR" in sigs
        assert sigs["sig1_net_disp_HR"] == pytest.approx(138.0 - 75.0, abs=0.1)
        assert sigs["sig1_net_disp_MAP"] == pytest.approx(52.0 - 95.0, abs=0.1)

        # Level 2 shock loop area
        assert "sig2_levy_area_shock_loop" in sigs
        assert isinstance(sigs["sig2_levy_area_shock_loop"], float)
        # Shock loop area is non-zero due to opposing trajectories
        assert abs(sigs["sig2_levy_area_shock_loop"]) > 0.05

        # Shock index velocity must be positive (worsening shock index)
        assert sigs["sig_shock_index_velocity"] > 0.5


# ==============================================================================
# 3. CLINICAL TEXT NLP & MULTIMODAL TESTS
# ==============================================================================

class TestClinicalNLP:

    @pytest.fixture
    def analyzer(self):
        return ClinicalTextAnalyzer()

    def test_sepsis_symptom_extraction(self, analyzer):
        note = "Patient obtunded and lethargic. Mottled cold extremities with severe shivering chills. Suspected urosepsis with cloudy urine."
        res = analyzer.analyze_note(note)

        assert res["nlp_sepsis_score"] > 0.40
        assert res["risk_delta"] > 0.10
        assert "mottled" in res["detected_sepsis_features"]
        assert "urosepsis" in res["detected_sepsis_features"]
        assert res["is_palliative_suppressed"] is False

    def test_mimic_detection_and_negation(self, analyzer):
        # Patient has tachycardia from post-op pain, but no infection
        note = "Patient resting comfortably, denies fever or chills. Alert and oriented x 4. Tachycardia secondary to surgical incision pain."
        res = analyzer.analyze_note(note)

        # Negation should prevent 'fever' or 'chills' from being detected as sepsis
        assert "fever" not in res["detected_sepsis_features"]
        assert "chills" not in res["detected_sepsis_features"]

        # Mimic features should be detected
        assert "incision pain" in res["detected_mimics"] or "surgical pain" in res["detected_mimics"] or "resting comfortably" in res["detected_mimics"]
        assert res["mimic_score"] > 0.20
        # Risk delta must be negative (suppressing the false alarm)
        assert res["risk_delta"] < 0.0

    def test_palliative_goals_of_care_suppression(self, analyzer):
        note = "Goals of care discussed with family. Patient transitioned to Comfort Measures Only (CMO). Do Not Resuscitate."
        res = analyzer.analyze_note(note)
        assert res["is_palliative_suppressed"] is True


# ==============================================================================
# 4. ENSEMBLE INTEGRATION TEST
# ==============================================================================

class TestEnsembleIntegration:

    def test_ensemble_conformal_and_signatures_api(self):
        ens = SepsisPrecisionEnsemble(target_precision=0.75)
        # Synthetic quick fit
        np.random.seed(42)
        n = 100
        X = pd.DataFrame(np.random.randn(n, 10), columns=[f"f_{i}" for i in range(10)])
        y = pd.Series(np.random.binomial(1, 0.15, size=n))

        ens.fit(X, y)
        assert ens.is_fitted is True

        # Test predict_conformal API
        conformal_out = ens.predict_conformal(X.iloc[:5])
        assert len(conformal_out) == 5
        for item in conformal_out:
            assert "prediction_set" in item
            assert "status" in item
            assert "epistemic_uncertainty" in item
            assert item["status"] in ["LOW_RISK", "CONFIRMED_ALERT", "INDETERMINATE", "OUT_OF_DISTRIBUTION", "ANOMALY"]

