"""
Unit & Integration Tests for Next-Gen Clinical AI Enhancements
==============================================================

Tests verifying:
1. Continuous Waveform Analytics (QRS R-peak detection, HRV SDNN/RMSSD, Welch spectral LF/HF, Sample Entropy, Arterial line PPV)
2. Multi-Center Federated Learning (Differential Privacy DP-SGD, Gradient Clipping, FedAvg aggregation, Privacy Budget)
3. Safe Offline Reinforcement Learning ("The AI Clinician" treatment recommendations, Surviving Sepsis guardrails)

Author: SepsisSense Research Team
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

from waveform_analytics import ContinuousWaveformProcessor, compute_sample_entropy
from federated_learning import FederatedHospitalClient, FederatedCoordinator
from safe_reinforcement_learning import SafeOfflineSepsisRL, FLUID_DOSE_TIERS, VASOPRESSOR_TIERS


# ==============================================================================
# 1. CONTINUOUS WAVEFORM ANALYTICS TESTS
# ==============================================================================

class TestWaveformAnalytics:

    def test_sample_entropy_computation(self):
        # Deterministic periodic signal has low entropy
        periodic_sig = np.sin(np.linspace(0, 10 * np.pi, 100))
        # Random signal has high entropy
        np.random.seed(42)
        random_sig = np.random.randn(100)

        se_periodic = compute_sample_entropy(periodic_sig, m=2, r=0.2)
        se_random = compute_sample_entropy(random_sig, m=2, r=0.2)

        assert isinstance(se_periodic, float)
        assert isinstance(se_random, float)
        assert se_random > se_periodic

    def test_hrv_feature_extraction(self):
        cwp = ContinuousWaveformProcessor(sampling_rate_hz=100)
        # Synthetic regular R-R intervals of 800 ms with slight respiratory sinus arrhythmia
        t = np.linspace(0, 30, 30)
        rr_intervals = 800.0 + 40.0 * np.sin(2 * np.pi * 0.2 * t)

        hrv = cwp.extract_hrv_features(rr_intervals, is_rr_intervals=True)

        assert "hrv_mean_nn_ms" in hrv
        assert hrv["hrv_mean_nn_ms"] == pytest.approx(800.0, abs=15.0)
        assert hrv["hrv_sdnn_ms"] > 0.0
        assert hrv["hrv_rmssd_ms"] > 0.0
        assert "hrv_sample_entropy" in hrv
        assert "hrv_autonomic_blunting_index" in hrv

    def test_arterial_line_waveform_extraction(self):
        cwp = ContinuousWaveformProcessor(sampling_rate_hz=100)
        # Generate synthetic ABP waveform: 120/80 mmHg oscillating at 1.2 Hz
        fs = 100
        duration = 10  # seconds
        t = np.linspace(0, duration, duration * fs)
        # Asymmetric pulse wave
        pulse = 80.0 + 40.0 * (np.sin(2 * np.pi * 1.2 * t) ** 2)
        # Add respiratory variation in pulse pressure (PPV simulation)
        respiratory_mod = 1.0 + 0.15 * np.sin(2 * np.pi * 0.25 * t)
        abp_signal = 80.0 + (pulse - 80.0) * respiratory_mod

        features = cwp.extract_arterial_line_features(abp_signal)

        assert "abp_pulse_pressure_variation_pct" in features
        assert features["abp_pulse_pressure_variation_pct"] > 5.0
        assert "abp_dp_dt_max_mmhg_s" in features
        assert features["abp_dp_dt_max_mmhg_s"] > 100.0
        assert features["abp_waveform_mean_map"] > 70.0


# ==============================================================================
# 2. MULTI-CENTER FEDERATED LEARNING TESTS
# ==============================================================================

class TestFederatedLearning:

    def test_federated_client_gradient_clipping_and_dp(self):
        np.random.seed(42)
        n_samples = 80
        n_features = 6
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.binomial(1, 0.15, size=n_samples).astype(np.float32)

        client = FederatedHospitalClient(
            hospital_id="HOSP_01",
            hospital_name="Test General Hospital",
            X_local=X,
            y_local=y,
            clip_norm=1.0,
            noise_multiplier=0.5
        )

        global_w = np.zeros(n_features, dtype=np.float32)
        global_b = 0.0

        updated_w, updated_b, loss = client.compute_local_step(global_w, global_b, lr=0.1)

        assert updated_w.shape == global_w.shape
        assert isinstance(updated_b, float)
        assert client.cumulative_epsilon > 0.0
        assert len(client.round_history) == 1

    def test_federated_multi_node_aggregation(self):
        coord = FederatedCoordinator(n_features=4, initial_learning_rate=0.05)

        # Register 3 hospital clients
        np.random.seed(42)
        for i, name in enumerate(["US-MIMIC-Node", "EU-Amsterdam-Node", "CH-Bern-Node"]):
            X_h = np.random.randn(100 + i * 20, 4).astype(np.float32)
            y_h = np.random.binomial(1, 0.10, size=len(X_h)).astype(np.float32)
            client = FederatedHospitalClient(f"HOSP_{i}", name, X_h, y_h)
            coord.register_hospital(client)

        assert len(coord.clients) == 3

        # Run 2 communication rounds
        round_1 = coord.run_federated_round(1)
        round_2 = coord.run_federated_round(2)

        assert round_1["participating_hospitals"] == 3
        assert round_1["total_federated_records"] == 360
        assert round_2["round"] == 2

        # Test global prediction
        X_test = np.random.randn(5, 4).astype(np.float32)
        preds = coord.predict_global(X_test)
        assert len(preds) == 5
        assert np.all((preds >= 0.0) & (preds <= 1.0))


# ==============================================================================
# 3. SAFE REINFORCEMENT LEARNING ("THE AI CLINICIAN") TESTS
# ==============================================================================

class TestSafeReinforcementLearning:

    @pytest.fixture
    def rl_engine(self):
        return SafeOfflineSepsisRL()

    def test_state_vector_extraction(self, rl_engine):
        pt_data = {
            "MAP": 62.0, "Lactate": 3.2, "HR": 115.0, "SBP": 90.0,
            "SOFA": 4.0, "PaO2_FiO2_ratio": 240.0, "UrineOutput": 30.0,
            "CumulativeFluidBalance": 2500.0
        }
        state = rl_engine.extract_state_vector(pt_data)
        assert len(state) == 7
        assert np.all(np.isfinite(state))

    def test_guardrail_fluid_restriction_in_respiratory_failure(self, rl_engine):
        # Patient with acute lung injury / pulmonary edema (PaO2/FiO2 = 140)
        pt_ards = {
            "MAP": 70.0, "Lactate": 2.5, "HR": 105.0, "SBP": 105.0,
            "PaO2_FiO2_ratio": 140.0, "CumulativeFluidBalance": 4500.0
        }
        # Attempt an aggressive fluid dose (tier 4)
        safe_fluid, safe_vaso, alerts = rl_engine.apply_clinical_safety_guardrails(
            raw_fluid_tier=4, raw_vaso_tier=0, patient_data=pt_ards
        )

        # Guardrail must cap fluid tier to 1 to prevent drowning the lungs
        assert safe_fluid <= 1
        assert len(alerts) >= 1
        assert "pulmonary edema" in alerts[0].lower() or "fluid" in alerts[0].lower()

    def test_guardrail_refractory_shock_vasopressor_mandate(self, rl_engine):
        # Patient in refractory septic shock (MAP = 52 mmHg)
        pt_shock = {
            "MAP": 52.0, "Lactate": 4.8, "HR": 128.0, "SBP": 75.0,
            "PaO2_FiO2_ratio": 320.0, "CumulativeFluidBalance": 1000.0
        }
        # Attempt zero vasopressors (tier 0)
        safe_fluid, safe_vaso, alerts = rl_engine.apply_clinical_safety_guardrails(
            raw_fluid_tier=2, raw_vaso_tier=0, patient_data=pt_shock
        )

        # Guardrail must mandate vasopressor initiation (tier >= 1)
        assert safe_vaso >= 1
        assert any("perfusion target" in a or "vasopressor" in a.lower() for a in alerts)

    def test_treatment_recommendation_pipeline(self, rl_engine):
        pt_septic = {
            "MAP": 60.0, "Lactate": 3.8, "HR": 118.0, "SBP": 88.0,
            "SOFA": 5.0, "PaO2_FiO2_ratio": 280.0, "CumulativeFluidBalance": 1500.0
        }
        recommendation = rl_engine.recommend_treatment(pt_septic)

        assert "recommended_fluid_tier" in recommendation
        assert "recommended_vaso_tier" in recommendation
        assert "recommended_fluid_label" in recommendation
        assert "recommended_vaso_label" in recommendation
        assert "clinical_rationale" in recommendation
        assert isinstance(recommendation["guardrails_triggered"], bool)
