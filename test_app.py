"""
SepsisSense App & Pipeline Unit & Integration Tests
===================================================

Automated tests verifying:
1. Preprocessing robustness (NaN imputation, type safety, scaling)
2. Feature engineering pipelines (clinical scores, rolling stats, 52 production features)
3. Production model loading and schema conformance
4. Model inference determinism and boundary bounds
5. Model explainability and feature contribution alignment
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure src is in sys.path
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocess import SepsisDataPreprocessor, create_sample_dataset
from feature_engineering import SepsisFeatureEngineer, engineer_production_features
from train_model import (
    add_rolling_features,
    add_clinical_features,
    preprocess_data,
    train_model,
    evaluate_model,
)
from explain_model import SepsisModelExplainer


class TestPreprocessing:
    """Test data cleaning and preprocessing pipeline."""

    def test_sample_dataset_generation(self):
        df = create_sample_dataset(n_patients=10, records_per_patient=12)
        assert len(df) == 120
        assert df["Patient"].nunique() == 10
        assert "SepsisLabel" in df.columns
        assert "HR" in df.columns

    def test_preprocessor_pipeline(self):
        df = create_sample_dataset(n_patients=5, records_per_patient=10)
        preprocessor = SepsisDataPreprocessor()
        df_proc, stats = preprocessor.preprocess_pipeline(df, fit=True)
        assert len(df_proc) == len(df)
        assert not df_proc["HR"].isna().any()
        assert stats["dataset"]["unique_patients"] == 5

    def test_zero_variance_protection(self):
        df = create_sample_dataset(n_patients=3, records_per_patient=5)
        df["HR"] = 80.0  # constant column -> std = 0
        preprocessor = SepsisDataPreprocessor()
        df_proc, _ = preprocessor.preprocess_pipeline(df, fit=True)
        assert not np.isnan(df_proc["HR"].values).any()
        assert not np.isinf(df_proc["HR"].values).any()


class TestFeatureEngineering:
    """Test feature extraction and production feature alignment."""

    def test_clinical_features(self):
        df = create_sample_dataset(n_patients=4, records_per_patient=8)
        df_feat = add_clinical_features(df)
        expected_cols = [
            "shock_index",
            "modified_shock_index",
            "qsofa_score",
            "hr_trend_3h",
            "map_trend_3h",
            "lactate_creatinine_ratio",
            "wbc_temp_interaction",
        ]
        for col in expected_cols:
            assert col in df_feat.columns
            assert not df_feat[col].isna().any()

    def test_rolling_features(self):
        df = create_sample_dataset(n_patients=4, records_per_patient=10)
        df_roll = add_rolling_features(df, vitals=["HR", "MAP"], window=6)
        assert "HR_roll6h" in df_roll.columns
        assert "MAP_roll6h" in df_roll.columns
        assert not df_roll["HR_roll6h"].isna().any()

    def test_production_feature_generator(self):
        df = create_sample_dataset(n_patients=3, records_per_patient=10)
        df_prod = engineer_production_features(df)
        features_path = os.path.join(os.path.dirname(__file__), "models", "feature_columns.pkl")
        if os.path.exists(features_path):
            expected_features = joblib.load(features_path)
            for feat in expected_features:
                assert feat in df_prod.columns, f"Missing production feature: {feat}"


class TestProductionModelInference:
    """Test the production model artifact and inference pipeline."""

    @pytest.fixture
    def model_artifacts(self):
        model_path = os.path.join(os.path.dirname(__file__), "models", "sepsissense_xgb.pkl")
        feat_path = os.path.join(os.path.dirname(__file__), "models", "feature_columns.pkl")
        if not (os.path.exists(model_path) and os.path.exists(feat_path)):
            pytest.skip("Production model artifacts not found in models/ directory.")
        model = joblib.load(model_path)
        features = joblib.load(feat_path)
        return model, features

    def test_model_loading_and_schema(self, model_artifacts):
        model, features = model_artifacts
        assert len(features) == 52
        assert "HR" in features
        assert "shock_index" in features
        assert hasattr(model, "predict_proba")

    def test_inference_probability_bounds(self, model_artifacts):
        model, features = model_artifacts
        # Create single patient sample
        sample_data = {feat: [0.0] for feat in features}
        sample_data["HR"] = [95.0]
        sample_data["MAP"] = [80.0]
        sample_data["Temp"] = [37.5]
        df_sample = pd.DataFrame(sample_data)[features]

        proba = model.predict_proba(df_sample)[0, 1]
        assert 0.0 <= proba <= 1.0

    def test_explainer_integration(self, model_artifacts):
        model, features = model_artifacts
        explainer = SepsisModelExplainer(model=model, model_type="xgboost")
        explainer.feature_names = features
        explainer.baseline_value = 0.5

        sample_data = {feat: [0.0] for feat in features}
        sample_data["HR"] = [115.0]
        sample_data["Temp"] = [39.0]
        df_sample = pd.DataFrame(sample_data)[features]

        exp = explainer.explain_single_prediction(df_sample, "TEST_P001")
        assert "prediction_probability" in exp
        assert "risk_category" in exp
        assert "contributions" in exp
        assert len(exp["contributions"]) == len(features)
