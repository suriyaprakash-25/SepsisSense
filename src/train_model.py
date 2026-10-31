"""
SepsisSense Model Training Pipeline
=================================

Advanced machine learning pipeline for sepsis prediction using XGBoost and LightGBM
with comprehensive evaluation metrics and cross-validation.

Author: SepsisSense Team  
License: MIT
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    accuracy_score, precision_recall_curve, roc_curve, 
    classification_report, confusion_matrix, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
try:
    import lightgbm as lgb
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    _LIGHTGBM_AVAILABLE = False
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SepsisModelTrainer:
    """
    Advanced machine learning trainer for sepsis prediction.
    
    Features:
    - Temporal-aware cross-validation
    - Multiple model support (XGBoost, LightGBM)  
    - Comprehensive evaluation metrics
    - Class imbalance handling
    - Feature importance analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize trainer with configuration."""
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.models = {}
        self.feature_importances = {}
        self.evaluation_results = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for model training."""
        return {
            'random_state': 42,
            'test_size': 0.2,
            'val_size': 0.2,
            'cv_folds': 5,
            'probability_threshold': 0.5,
            'precision_threshold': 0.8,
            'models_to_train': ['xgboost'] + (['lightgbm'] if _LIGHTGBM_AVAILABLE else []),
            'xgb_params': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': False
            },
            'lgb_params': {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
                'n_estimators': 500
            }
        }
    
    def prepare_temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare temporal-aware, patient-stratified train/validation/test split.
        Splits by Patient (not individual records) to prevent data leakage,
        while ensuring proportional sepsis prevalence in all splits.
        
        Args:
            df: Input dataframe with ICULOS and Patient columns
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Preparing data with temporal-aware splitting...")
        
        patient_col = 'Patient' if 'Patient' in df.columns else df.columns[0]
        time_col = 'ICULOS' if 'ICULOS' in df.columns else df.columns[1]
        target_col = 'SepsisLabel' if 'SepsisLabel' in df.columns else df.columns[-1]
        
        # Sort by patient and time
        df_sorted = df.sort_values([patient_col, time_col]).copy()
        
        unique_patients = df_sorted[patient_col].unique()
        n_patients = len(unique_patients)
        
        # Ensure minimum patients for split
        if n_patients < 3:
            logger.warning(f"Only {n_patients} patients available. Using simple split.")
            train_df = df_sorted
            val_df = df_sorted.sample(frac=0.1, random_state=42) if len(df_sorted) > 10 else df_sorted.head(1)
            test_df = df_sorted.sample(frac=0.1, random_state=43) if len(df_sorted) > 10 else df_sorted.head(1)
        else:
            # Stratified split by patient-level sepsis outcome
            patient_labels = df_sorted.groupby(patient_col)[target_col].max()
            pos_patients = patient_labels[patient_labels == 1].index.tolist()
            neg_patients = patient_labels[patient_labels == 0].index.tolist()
            
            def split_items(items, test_frac, val_frac):
                n = len(items)
                if n == 0:
                    return [], [], []
                n_test = max(1 if (n >= 4 and test_frac > 0) else 0, int(round(n * test_frac)))
                n_val = max(1 if (n >= 4 and val_frac > 0) else 0, int(round(n * val_frac)))
                n_train = n - n_test - n_val
                if n_train < 1:
                    n_train = 1
                    if n_test > 0: n_test -= 1
                    elif n_val > 0: n_val -= 1
                return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]
            
            pos_train, pos_val, pos_test = split_items(pos_patients, self.config['test_size'], self.config['val_size'])
            neg_train, neg_val, neg_test = split_items(neg_patients, self.config['test_size'], self.config['val_size'])
            
            train_patients = pos_train + neg_train
            val_patients = pos_val + neg_val
            test_patients = pos_test + neg_test
            
            # Fallback if any set empty
            if len(val_patients) == 0: val_patients = train_patients[:1]
            if len(test_patients) == 0: test_patients = train_patients[-1:]
            
            train_df = df_sorted[df_sorted[patient_col].isin(train_patients)].copy()
            val_df = df_sorted[df_sorted[patient_col].isin(val_patients)].copy()
            test_df = df_sorted[df_sorted[patient_col].isin(test_patients)].copy()
        
        logger.info(f"Data split - Patients: {len(train_df[patient_col].unique())} train, {len(val_df[patient_col].unique())} val, {len(test_df[patient_col].unique())} test")
        logger.info(f"Data split - Records: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
        train_sepsis_rate = train_df[target_col].mean() if len(train_df) > 0 else 0
        val_sepsis_rate = val_df[target_col].mean() if len(val_df) > 0 else 0
        test_sepsis_rate = test_df[target_col].mean() if len(test_df) > 0 else 0
        
        logger.info(f"Sepsis rates - Train: {train_sepsis_rate:.3f}, Val: {val_sepsis_rate:.3f}, Test: {test_sepsis_rate:.3f}")
        return train_df, val_df, test_df
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable."""
        # Drop non-feature columns
        feature_columns = [col for col in df.columns if col not in ['Patient', 'SepsisLabel']]
        X = df[feature_columns].copy()
        y = df['SepsisLabel'].copy()
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution: {dict(y.value_counts())}")
        
        return X, y
    
    def calculate_class_weights(self, y: pd.Series) -> Dict[float, float]:
        """Calculate class weights for imbalanced dataset."""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        logger.info(f"Class weights: {class_weights}")
        return class_weights
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series) -> xgb.XGBClassifier:
        """Train XGBoost model with class balancing."""
        logger.info("Training XGBoost model...")
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y_train)
        scale_pos_weight = class_weights[0] / class_weights[1]
        
        # Update parameters with class weights
        xgb_params = self.config['xgb_params'].copy()
        xgb_params['scale_pos_weight'] = scale_pos_weight
        
        # Train model
        model = xgb.XGBClassifier(**xgb_params)
        
        # Fit model
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
        except Exception:
            # Fallback to simple fit if eval_set is not supported
            model.fit(X_train, y_train, verbose=False)
        
        logger.info("XGBoost training completed.")
        return model
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series):
        """Train LightGBM model with class balancing."""
        if not _LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

        logger.info("Training LightGBM model...")
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y_train)
        scale_pos_weight = class_weights[0] / class_weights[1]
        
        # Update parameters with class weights
        lgb_params = self.config['lgb_params'].copy()
        lgb_params['scale_pos_weight'] = scale_pos_weight
        
        # Train model
        model = lgb.LGBMClassifier(**lgb_params)
        
        # Fit model
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        except Exception:
            # Fallback to simple fit
            model.fit(X_train, y_train)
        
        logger.info("LightGBM training completed.")
        return model
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                      dataset_name: str = "Dataset") -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            dataset_name: Name for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_name}...")
        
        # Get predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= self.config['probability_threshold']).astype(int)
        
        # Check for NaN values and handle them
        if y.isna().any():
            logger.warning(f"Found {y.isna().sum()} NaN values in y_true, removing them")
            valid_mask = ~y.isna()
            y = y[valid_mask]
            y_pred_proba = y_pred_proba[valid_mask]
            y_pred = y_pred[valid_mask]
        
        # Calculate metrics
        if len(y) == 0:
            logger.error("No valid samples after NaN removal")
            return {'error': 'no_valid_samples'}
        
        try:
            metrics = {
                'roc_auc': roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0,
                'pr_auc': average_precision_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0,
                'f1_score': f1_score(y, y_pred, zero_division=0),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'accuracy': (y_pred == y).mean()
            }
            
            # Recall at high precision
            if len(np.unique(y)) > 1:
                precision_vals, recall_vals, thresholds = precision_recall_curve(y, y_pred_proba)
                high_precision_mask = precision_vals >= self.config['precision_threshold']
                if high_precision_mask.any():
                    metrics['recall_at_precision_80'] = recall_vals[high_precision_mask].max()
                else:
                    metrics['recall_at_precision_80'] = 0.0
            else:
                metrics['recall_at_precision_80'] = 0.0
            
            # Additional clinical metrics
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel() if len(np.unique(y)) > 1 else (len(y), 0, 0, 0)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            metrics = {'error': str(e)}
        
        # Log metrics
        if 'error' not in metrics:
            logger.info(f"{dataset_name} Metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def train_and_evaluate_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete training and evaluation pipeline.
        
        Args:
            df: Processed dataframe with features and target
            
        Returns:
            Dictionary containing trained models, metrics, and other results
        """
        logger.info("Starting complete training and evaluation pipeline...")
        
        # Temporal split
        train_df, val_df, test_df = self.prepare_temporal_split(df)
        
        # Prepare features and targets
        X_train, y_train = self.prepare_features_and_target(train_df)
        X_val, y_val = self.prepare_features_and_target(val_df)
        X_test, y_test = self.prepare_features_and_target(test_df)
        
        # Store feature names
        feature_names = X_train.columns.tolist()
        
        # Train models
        models = {}
        model_metrics = {}
        
        # XGBoost
        if 'xgboost' in self.config['models_to_train']:
            logger.info("Training xgboost model...")
            models['xgboost'] = self.train_xgboost(X_train, y_train, X_val, y_val)
            
            # Evaluate XGBoost
            train_metrics = self.evaluate_model(models['xgboost'], X_train, y_train, "Train")
            val_metrics = self.evaluate_model(models['xgboost'], X_val, y_val, "Validation")
            test_metrics = self.evaluate_model(models['xgboost'], X_test, y_test, "Test")
            
            model_metrics['xgboost'] = {
                'train': train_metrics,
                'validation': val_metrics,
                'test': test_metrics,
                **test_metrics,
            }
        
        # LightGBM
        if 'lightgbm' in self.config['models_to_train']:
            logger.info("Training lightgbm model...")
            models['lightgbm'] = self.train_lightgbm(X_train, y_train, X_val, y_val)
            
            # Evaluate LightGBM
            train_metrics = self.evaluate_model(models['lightgbm'], X_train, y_train, "Train")
            val_metrics = self.evaluate_model(models['lightgbm'], X_val, y_val, "Validation")
            test_metrics = self.evaluate_model(models['lightgbm'], X_test, y_test, "Test")
            
            model_metrics['lightgbm'] = {
                'train': train_metrics,
                'validation': val_metrics,
                'test': test_metrics,
                **test_metrics,
            }
        
        # Select best model based on validation AUC
        best_model_name = 'xgboost'  # Default
        best_val_auc = 0
        
        for model_name, metrics in model_metrics.items():
            if 'error' not in metrics['validation']:
                val_auc = metrics['validation'].get('roc_auc', 0)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_name = model_name
        
        logger.info(f"Best model: {best_model_name} (Validation AUC: {best_val_auc:.4f})")
        
        return {
            'models': models,
            'best_model': models[best_model_name],
            'best_model_name': best_model_name,
            'metrics': model_metrics,
            'feature_names': feature_names,
            'test_data': (X_test, y_test),
            'validation_data': (X_val, y_val),
            'train_data': (X_train, y_train)
        }
    
    def save_model(self, model, filepath: str) -> None:
        """Save trained model to disk."""
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model


# ==============================================================================
# STANDALONE PIPELINE FUNCTIONS (PhysioNet 2019 real-data training)
# ==============================================================================

def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load all .psv files from the given directory tree and combine into one DataFrame.
    Adds a 'Patient' column derived from the filename.
    """
    import glob
    print("Loading data...")
    logger.info(f"Scanning for .psv files under: {data_dir}")

    psv_files = glob.glob(os.path.join(data_dir, "**", "*.psv"), recursive=True)
    if not psv_files:
        raise FileNotFoundError(f"No .psv files found under {data_dir}")

    logger.info(f"Found {len(psv_files):,} patient files.")

    dfs = []
    for fp in psv_files:
        try:
            df_p = pd.read_csv(fp, sep="|")
            patient_id = os.path.splitext(os.path.basename(fp))[0]
            df_p.insert(0, "Patient", patient_id)
            dfs.append(df_p)
        except Exception as e:
            logger.warning(f"Could not read {fp}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataset shape: {combined.shape}")
    print(f"  -> Loaded {combined.shape[0]:,} rows from {len(psv_files):,} files.")
    return combined


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the combined PhysioNet DataFrame:
      1. Forward-fill per patient
      2. Backward-fill per patient
      3. Fill remaining NaN with 0
      4. Separate features (X) and target (y = SepsisLabel)
    Returns (X, y).
    """
    print("Preprocessing...")
    logger.info("Starting preprocessing...")

    df = df.copy()

    # Drop columns that are not useful as model features
    cols_to_drop = [c for c in ["Patient", "SepsisLabel"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    # Per-patient forward then backward fill
    df[feature_cols] = (
        df.groupby("Patient")[feature_cols]
        .transform(lambda s: s.ffill().bfill())
    )

    # Fill any remaining NaN with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    # Encode Gender if present and not numeric
    if "Gender" in df.columns and df["Gender"].dtype == object:
        df["Gender"] = df["Gender"].map({"M": 1, "F": 0, "m": 1, "f": 0}).fillna(0)

    # Ensure all feature columns are numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X = df[feature_cols].copy()
    y = df["SepsisLabel"].copy().astype(int)

    sepsis_rate = y.mean()
    logger.info(f"Preprocessing done. X shape: {X.shape}, Sepsis rate: {sepsis_rate:.3%}")
    print(f"  -> Features: {X.shape[1]} | Rows: {X.shape[0]:,} | Sepsis rate: {sepsis_rate:.3%}")
    return X, y


def add_rolling_features(df_full: pd.DataFrame,
                         vitals: Optional[List[str]] = None,
                         window: int = 6) -> pd.DataFrame:
    """
    Add rolling-mean features for key vitals (grouped by Patient, window hours).
    Mutates and returns df_full with new columns appended.
    """
    if vitals is None:
        vitals = ["HR", "MAP", "Temp", "Resp", "O2Sat"]

    print(f"Adding {window}h rolling-mean features for {vitals}...")
    logger.info(f"Computing {window}h rolling means for: {vitals}")

    available = [v for v in vitals if v in df_full.columns]
    if not available:
        logger.warning("None of the requested vitals found in DataFrame; skipping rolling features.")
        return df_full

    df_full = df_full.copy()
    for col in available:
        new_col = f"{col}_roll{window}h"
        df_full[new_col] = (
            df_full.groupby("Patient")[col]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
    return df_full


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7 clinically validated features directly:
    1. shock_index: HR / SBP
    2. modified_shock_index: HR / MAP
    3. qsofa_score: (Resp >= 22) + (SBP <= 100) + (binary GCS - approximated or default)
    4. hr_trend_3h: HR[t] - HR[t-3]
    5. map_trend_3h: MAP[t] - MAP[t-3]
    6. lactate_creatinine_ratio: Lactate / (Creatinine + 1e-6)
    7. wbc_temp_interaction: WBC * Temp
    """
    print("Adding 7 clinical features...")
    df = df.copy()

    # 1. Shock Index
    if "HR" in df.columns and "SBP" in df.columns:
        df["shock_index"] = (df["HR"] / (df["SBP"] + 1e-6)).fillna(0.0)
    else:
        df["shock_index"] = 0.0

    # 2. Modified Shock Index
    if "HR" in df.columns and "MAP" in df.columns:
        df["modified_shock_index"] = (df["HR"] / (df["MAP"] + 1e-6)).fillna(0.0)
    else:
        df["modified_shock_index"] = 0.0

    # 3. qSOFA score (approximate: Resp >= 22 (+1), SBP <= 100 (+1), GCS/altered mental state is not explicitly present, so we assume 0 or check if it exists)
    resp_term = (df["Resp"] >= 22).astype(int) if "Resp" in df.columns else 0
    sbp_term = (df["SBP"] <= 100).astype(int) if "SBP" in df.columns else 0
    df["qsofa_score"] = (resp_term + sbp_term).astype(float)

    # 4. HR Trend (3h)
    if "HR" in df.columns and "Patient" in df.columns:
        df["hr_trend_3h"] = df.groupby("Patient")["HR"].diff(periods=3).fillna(0.0)
    else:
        df["hr_trend_3h"] = 0.0

    # 5. MAP Trend (3h)
    if "MAP" in df.columns and "Patient" in df.columns:
        df["map_trend_3h"] = df.groupby("Patient")["MAP"].diff(periods=3).fillna(0.0)
    else:
        df["map_trend_3h"] = 0.0

    # 6. Lactate / Creatinine ratio
    if "Lactate" in df.columns and "Creatinine" in df.columns:
        df["lactate_creatinine_ratio"] = (df["Lactate"] / (df["Creatinine"] + 1e-6)).fillna(0.0)
    else:
        df["lactate_creatinine_ratio"] = 0.0

    # 7. WBC * Temp interaction
    if "WBC" in df.columns and "Temp" in df.columns:
        df["wbc_temp_interaction"] = (df["WBC"] * df["Temp"]).fillna(0.0)
    else:
        df["wbc_temp_interaction"] = 0.0

    return df


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> "xgb.XGBClassifier":
    """
    Train an XGBClassifier with settings from the user specification.
    scale_pos_weight is automatically derived from class imbalance but capped at 10.0 to reduce false alarms.
    Returns the fitted model.
    """
    print("Training model...")
    logger.info("Training XGBoost classifier...")

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    scale_pos_weight = min(scale_pos_weight, 10.0)
    logger.info(f"scale_pos_weight (capped) = {scale_pos_weight:.2f}  (neg={neg:,}, pos={pos:,})")

    device = "cpu"
    try:
        # Check if CUDA device can be utilized
        test_m = xgb.XGBClassifier(device="cuda", n_estimators=1)
        test_m.fit(np.zeros((2, 2)), np.array([0, 1]))
        device = "cuda"
    except Exception:
        device = "cpu"

    logger.info(f"Using XGBoost compute device: {device}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        tree_method="hist",
        device=device,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    logger.info(f"XGBoost training complete on {device}.")
    return model


def evaluate_model(model: "xgb.XGBClassifier",
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   plot: bool = True) -> Dict[str, float]:
    """
    Evaluate the model and print Accuracy, Precision, Recall, F1, ROC-AUC.
    Optionally plots top-20 feature importances.
    Returns dict of metrics.
    """
    print("Evaluating model...")
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1_score":  f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }

    print("\n  +-----------------------------+")
    print("  |   Model Evaluation Results  |")
    print("  +-----------------------------+")
    for name, val in metrics.items():
        print(f"  |  {name:<18} {val:.4f}  |")
    print("  +-----------------------------+\n")

    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend — no GUI window needed
            import matplotlib.pyplot as plt
            fi = pd.Series(model.feature_importances_, index=X_test.columns)
            fi_top = fi.nlargest(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            fi_top.sort_values().plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title("Top-20 Feature Importances — SepsisSense XGBoost")
            ax.set_xlabel("Importance Score")
            plt.tight_layout()
            plot_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "models", "feature_importance.png"
            )
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"  Feature importance plot saved → {plot_path}")
        except Exception as e:
            logger.warning(f"Could not plot feature importances: {e}")

    return metrics


def save_model(model: "xgb.XGBClassifier",
               feature_columns: List[str],
               model_path: str = None,
               features_path: str = None) -> None:
    """
    Save the trained model and feature column list using joblib.
    Defaults to models/sepsissense_xgb.pkl and models/feature_columns.pkl.
    """
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(base, exist_ok=True)

    if model_path is None:
        model_path = os.path.join(base, "sepsissense_xgb.pkl")
    if features_path is None:
        features_path = os.path.join(base, "feature_columns.pkl")

    print("Saving model...")
    joblib.dump(model, model_path)
    joblib.dump(feature_columns, features_path)
    logger.info(f"Model saved      → {model_path}")
    logger.info(f"Features saved   → {features_path}")
    print(f"  -> models/sepsissense_xgb.pkl  [OK]")
    print(f"  -> models/feature_columns.pkl  [OK]")


def run_pipeline(data_dir: str = None) -> None:
    """
    End-to-end training pipeline:
      load → rolling features → preprocess → split → train → evaluate → save
    """
    from sklearn.model_selection import train_test_split

    # ── resolve data directory ──────────────────────────────────────────────
    if data_dir is None:
        root = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(
            root, "physionet.org", "files", "challenge-2019", "1.0.0", "training"
        )

    print("\n" + "=" * 56)
    print("  [MD]  SepsisSense - XGBoost Training Pipeline")
    print("=" * 56)

    # STEP 1 — Load data
    df = load_data(data_dir)

    # STEP 3 — Rolling features (must run before preprocess strips 'Patient')
    df = add_rolling_features(df, vitals=["HR", "MAP", "Temp", "Resp", "O2Sat"], window=6)

    # STEP 3.5 — Clinical Features
    df = add_clinical_features(df)

    # STEP 5 — Train / test split (patient-level stratified split to prevent record leakage)
    if "Patient" in df.columns:
        patient_outcomes = df.groupby("Patient")["SepsisLabel"].max()
        train_pts, test_pts = train_test_split(
            patient_outcomes.index,
            test_size=0.2,
            random_state=42,
            stratify=patient_outcomes.values
        )
        df_train = df[df["Patient"].isin(train_pts)].copy()
        df_test = df[df["Patient"].isin(test_pts)].copy()
        X_train, y_train = preprocess_data(df_train)
        X_test, y_test = preprocess_data(df_test)
    else:
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    print(f"  -> Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")

    # STEP 4 — Train
    model = train_model(X_train, y_train)

    # STEP 6 — Evaluate
    evaluate_model(model, X_test, y_test, plot=True)

    # STEP 7 & 8 — Save model + feature list
    save_model(model, list(X_train.columns))

    print("\n" + "=" * 56)
    print("  [OK]  Pipeline complete!")
    print("=" * 56 + "\n")


def main():
    """Run the full PhysioNet 2019 XGBoost training pipeline."""
    run_pipeline()


if __name__ == "__main__":
    main()