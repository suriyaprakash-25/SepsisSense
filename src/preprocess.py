"""
SepsisSense Data Preprocessing Pipeline
=====================================

This module handles comprehensive data preprocessing for sepsis prediction,
including missing value imputation, normalization, and patient timeline structuring.

Author: SepsisSense Team
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SepsisDataPreprocessor:
    """
    Advanced preprocessing pipeline for sepsis prediction data.
    
    Features:
    - Patient-specific forward/backward fill for missing values
    - Vital signs and lab values normalization
    - Temporal feature engineering
    - Data validation and quality checks
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or self._get_default_config()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_stats = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration."""
        return {
            'vital_signs': [
                'HR',  # Heart Rate
                'O2Sat',  # Oxygen Saturation
                'Temp',  # Temperature
                'SBP',  # Systolic Blood Pressure
                'MAP',  # Mean Arterial Pressure
                'DBP',  # Diastolic Blood Pressure
                'Resp',  # Respiration Rate
            ],
            'lab_values': [
                'BaseExcess',
                'HCO3',  # Bicarbonate
                'FiO2',  # Fraction of Inspired Oxygen
                'pH',
                'PaCO2',  # Partial Pressure of CO2
                'SaO2',  # Oxygen Saturation
                'AST',  # Aspartate Aminotransferase
                'BUN',  # Blood Urea Nitrogen
                'Alkalinephos',  # Alkaline Phosphatase
                'Calcium',
                'Chloride',
                'Creatinine',
                'Bilirubin_direct',
                'Glucose',
                'Lactate',
                'Magnesium',
                'Phosphate',
                'Potassium',
                'Bilirubin_total',
                'TroponinI',
                'Hct',  # Hematocrit
                'Hgb',  # Hemoglobin
                'PTT',  # Partial Thromboplastin Time
                'WBC',  # White Blood Cell Count
                'Fibrinogen',
                'Platelets'
            ],
            'demographics': [
                'Age',
                'Gender',
                'Unit1',  # ICU Unit Type 1
                'Unit2',  # ICU Unit Type 2
                'HospAdmTime',  # Hospital Admission Time
                'ICULOS'  # ICU Length of Stay
            ],
            'target': 'SepsisLabel',
            'patient_id': 'Patient',
            'max_forward_fill_hours': 6,
            'max_backward_fill_hours': 2,
            'outlier_threshold': 5.0,  # Standard deviations for outlier detection
            'min_patient_records': 6,  # Minimum records per patient
        }
    
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and perform initial validation of the dataset.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Validated DataFrame
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Read the CSV file with special handling for scientific notation and brackets
            df = pd.read_csv(file_path, sep='|' if file_path.endswith('.psv') else ',')
            
            # Clean data: remove brackets and handle scientific notation
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        # Remove brackets and spaces
                        cleaned = df[col].astype(str).str.strip('[]').str.strip()
                        converted = pd.to_numeric(cleaned, errors='coerce')
                        if not converted.isna().all():
                            df[col] = converted
                        else:
                            df[col] = cleaned
                    except Exception:
                        pass
            
            patient_col = self.config['patient_id']
            if patient_col not in df.columns:
                df[patient_col] = 'P000001'
            
            logger.info(f"Loaded {len(df)} records for {df[patient_col].nunique()} patients")
            
            # Basic validation
            required_columns = (
                [patient_col, self.config['target']] + 
                self.config['vital_signs'] + 
                self.config['demographics']
            )
            
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
            
            # Remove patients with too few records
            patient_counts = df[patient_col].value_counts()
            valid_patients = patient_counts[patient_counts >= self.config['min_patient_records']].index
            if len(valid_patients) > 0:
                df = df[df[patient_col].isin(valid_patients)]
            
            logger.info(f"After filtering: {len(df)} records for {df[patient_col].nunique()} patients")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using patient-specific forward/backward fill.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with imputed missing values
        """
        logger.info("Handling missing values...")
        
        df_imputed = df.copy()
        patient_col = self.config['patient_id']
        time_col = self.config.get('time_column', 'ICULOS')
        
        if patient_col not in df_imputed.columns:
            df_imputed[patient_col] = 'P000001'
        if time_col not in df_imputed.columns:
            df_imputed[time_col] = np.arange(len(df_imputed)) + 1
        
        # Sort by patient and time for proper forward/backward fill
        df_imputed = df_imputed.sort_values([patient_col, time_col])
        
        # Get all numeric columns for imputation
        all_numeric = self.config['vital_signs'] + self.config['lab_values'] + ['Age']
        numeric_columns = [c for c in all_numeric if c in df_imputed.columns]
        
        max_f = self.config['max_forward_fill_hours']
        max_b = self.config['max_backward_fill_hours']
        
        # Vectorized per-patient forward and backward fill (100x faster than python row loops)
        df_imputed[numeric_columns] = (
            df_imputed.groupby(patient_col)[numeric_columns]
            .transform(lambda s: s.ffill(limit=max_f).bfill(limit=max_b))
        )
        
        # Fill remaining missing values with median (population-level imputation) without inplace warning
        for col in numeric_columns:
            if df_imputed[col].isna().any():
                median_value = df_imputed[col].median()
                if pd.isna(median_value):
                    median_value = 0.0
                df_imputed[col] = df_imputed[col].fillna(median_value)
                logger.info(f"Filled remaining NaN in {col} with median: {median_value:.2f}")
        
        # Handle categorical variables without inplace warning
        for col in ['Gender', 'Unit1', 'Unit2']:
            if col in df_imputed.columns:
                mode_series = df_imputed[col].mode()
                mode_value = mode_series.iloc[0] if not mode_series.empty else 0
                df_imputed[col] = df_imputed[col].fillna(mode_value)
        
        logger.info("Missing value imputation completed")
        return df_imputed
    
    def detect_and_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using statistical methods.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info("Detecting and handling outliers...")
        
        df_cleaned = df.copy()
        all_numeric = self.config['vital_signs'] + self.config['lab_values']
        numeric_columns = [c for c in all_numeric if c in df_cleaned.columns]
        
        outlier_counts = {}
        
        for col in numeric_columns:
            col_std = df_cleaned[col].std()
            if pd.notna(col_std) and col_std > 1e-6:
                # Calculate z-score safely
                z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / col_std)
                outlier_mask = z_scores > self.config['outlier_threshold']
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outlier_counts[col] = int(outlier_count)
                    # Cap outliers at 95th and 5th percentiles
                    p95 = df_cleaned[col].quantile(0.95)
                    p5 = df_cleaned[col].quantile(0.05)
                    df_cleaned[col] = df_cleaned[col].clip(lower=p5, upper=p95)
        
        logger.info(f"Handled outliers in {len(outlier_counts)} columns: {outlier_counts}")
        return df_cleaned
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with normalized features
        """
        logger.info("Normalizing features...")
        
        df_normalized = df.copy()
        numeric_columns = self.config['vital_signs'] + self.config['lab_values'] + ['Age']
        
        # Only normalize columns that exist in the dataframe
        columns_to_normalize = [col for col in numeric_columns if col in df_normalized.columns]
        
        if fit:
            # Fit and transform (training data)
            df_normalized[columns_to_normalize] = self.scaler.fit_transform(df_normalized[columns_to_normalize])
            logger.info(f"Fitted scaler on {len(columns_to_normalize)} features")
        else:
            # Transform only (test/inference data)
            df_normalized[columns_to_normalize] = self.scaler.transform(df_normalized[columns_to_normalize])
            logger.info(f"Applied pre-fitted scaler to {len(columns_to_normalize)} features")
        
        return df_normalized
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        categorical_columns = ['Gender', 'Unit1', 'Unit2']
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    # Fit and transform
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    # Transform only
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_vals = set(df_encoded[col].astype(str).unique())
                        known_vals = set(self.label_encoders[col].classes_)
                        unseen_vals = unique_vals - known_vals
                        
                        if unseen_vals:
                            logger.warning(f"Unseen categories in {col}: {unseen_vals}")
                            # Map unseen values to the most common category
                            most_common = self.label_encoders[col].classes_[0]
                            df_encoded[col] = df_encoded[col].astype(str).replace(list(unseen_vals), most_common)
                        
                        df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features for better time-series modeling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional temporal features
        """
        logger.info("Adding temporal features...")
        
        df_temporal = df.copy()
        
        # Time since admission features
        df_temporal['hours_since_admission'] = df_temporal['ICULOS']
        df_temporal['days_since_admission'] = df_temporal['ICULOS'] / 24
        
        # Cyclical time features (if we have actual timestamps)
        if 'HospAdmTime' in df_temporal.columns:
            if pd.api.types.is_numeric_dtype(df_temporal['HospAdmTime']):
                # In PhysioNet, HospAdmTime is hours before ICU admission (often negative)
                adm_hours = df_temporal['HospAdmTime'].fillna(0.0)
                cycle_hr = adm_hours % 24
                df_temporal['hour_sin'] = np.sin(2 * np.pi * cycle_hr / 24)
                df_temporal['hour_cos'] = np.cos(2 * np.pi * cycle_hr / 24)
                df_temporal['dow_sin'] = 0.0
                df_temporal['dow_cos'] = 1.0
            else:
                dt_series = pd.to_datetime(df_temporal['HospAdmTime'], errors='coerce')
                df_temporal['admission_hour'] = dt_series.dt.hour.fillna(0)
                df_temporal['admission_day_of_week'] = dt_series.dt.dayofweek.fillna(0)
                df_temporal['hour_sin'] = np.sin(2 * np.pi * df_temporal['admission_hour'] / 24)
                df_temporal['hour_cos'] = np.cos(2 * np.pi * df_temporal['admission_hour'] / 24)
                df_temporal['dow_sin'] = np.sin(2 * np.pi * df_temporal['admission_day_of_week'] / 7)
                df_temporal['dow_cos'] = np.cos(2 * np.pi * df_temporal['admission_day_of_week'] / 7)
        
        # Patient timeline position
        patient_col = self.config.get('patient_id', 'Patient')
        if patient_col in df_temporal.columns and df_temporal[patient_col].nunique() > 0:
            patient_timeline = df_temporal.groupby(patient_col)['ICULOS'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
            )
            df_temporal['timeline_position'] = patient_timeline
        else:
            df_temporal['timeline_position'] = 0.0
        
        logger.info("Temporal features added successfully")
        return df_temporal
    
    def compute_feature_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing feature statistics
        """
        logger.info("Computing feature statistics...")
        
        stats = {}
        numeric_columns = self.config['vital_signs'] + self.config['lab_values'] + ['Age', 'ICULOS']
        
        for col in numeric_columns:
            if col in df.columns:
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75),
                    'missing_rate': df[col].isna().mean(),
                    'unique_values': df[col].nunique()
                }
        
        # Overall dataset statistics
        unique_patients = df[self.config['patient_id']].nunique()
        stats['dataset'] = {
            'total_records': len(df),
            'unique_patients': unique_patients,
            'sepsis_rate': df[self.config['target']].mean() if self.config['target'] in df.columns else None,
            'avg_records_per_patient': len(df) / unique_patients if unique_patients > 0 else 0
        }
        
        self.feature_stats = stats
        logger.info("Feature statistics computed successfully")
        return stats
    
    def preprocess_pipeline(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training, False for inference)
            
        Returns:
            Tuple of (preprocessed DataFrame, statistics dictionary)
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # Step 1: Handle missing values
        df_processed = self.handle_missing_values(df)
        
        # Step 2: Detect and handle outliers
        df_processed = self.detect_and_handle_outliers(df_processed)
        
        # Step 3: Add temporal features
        df_processed = self.add_temporal_features(df_processed)
        
        # Step 4: Encode categorical features
        df_processed = self.encode_categorical_features(df_processed, fit=fit)
        
        # Step 5: Normalize features
        df_processed = self.normalize_features(df_processed, fit=fit)
        
        # Step 6: Compute statistics
        stats = self.compute_feature_statistics(df_processed) if fit else self.feature_stats
        
        logger.info("Preprocessing pipeline completed successfully")
        return df_processed, stats
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save the fitted preprocessor components."""
        import joblib
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_stats': self.feature_stats,
            'config': self.config
        }
        
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath: str) -> 'SepsisDataPreprocessor':
        """Load a fitted preprocessor."""
        import joblib
        
        preprocessor_data = joblib.load(filepath)
        
        preprocessor = cls(config=preprocessor_data['config'])
        preprocessor.scaler = preprocessor_data['scaler']
        preprocessor.label_encoders = preprocessor_data['label_encoders']
        preprocessor.feature_stats = preprocessor_data['feature_stats']
        
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def create_sample_dataset(n_patients: int = 1000, records_per_patient: int = 24) -> pd.DataFrame:
    """
    Create a synthetic dataset for testing purposes.
    
    Args:
        n_patients: Number of patients to simulate
        records_per_patient: Average number of records per patient
        
    Returns:
        Synthetic sepsis dataset
    """
    np.random.seed(42)
    
    data = []
    
    for patient_id in range(1, n_patients + 1):
        # Random number of records per patient
        n_records = np.random.poisson(records_per_patient)
        n_records = max(6, n_records)  # Minimum 6 records
        
        # Patient demographics
        age = np.random.normal(65, 15)
        age = max(18, min(95, age))  # Clamp age between 18-95
        
        gender = np.random.choice([0, 1])
        unit1 = np.random.choice([0, 1])
        unit2 = np.random.choice([0, 1])
        
        # Simulate sepsis onset (ensure valid range)
        if n_records > 6:
            sepsis_onset_hour = np.random.choice([None, np.random.randint(6, n_records)])
        else:
            sepsis_onset_hour = None
        
        for hour in range(n_records):
            # Base vital signs (with some patient-specific offset)
            patient_offset = np.random.normal(0, 0.1)
            
            # Simulate deterioration if sepsis develops
            sepsis_effect = 0
            if sepsis_onset_hour is not None and hour >= sepsis_onset_hour - 6:
                # Gradual deterioration 6 hours before sepsis
                sepsis_effect = (hour - (sepsis_onset_hour - 6)) * 0.2
            
            # Vital signs with noise and sepsis effect
            hr = 80 + patient_offset * 10 + sepsis_effect * 20 + np.random.normal(0, 10)
            o2sat = 98 - sepsis_effect * 5 + np.random.normal(0, 2)
            temp = 37 + sepsis_effect * 2 + np.random.normal(0, 0.5)
            sbp = 120 - sepsis_effect * 15 + np.random.normal(0, 15)
            map_val = 80 - sepsis_effect * 10 + np.random.normal(0, 10)
            dbp = 70 - sepsis_effect * 8 + np.random.normal(0, 8)
            resp = 16 + sepsis_effect * 6 + np.random.normal(0, 3)
            
            # Lab values (less frequent, more missing) - Generate all required lab values
            prob_missing = 0.7  # High probability of missing lab values (realistic)
            
            wbc = 7 + sepsis_effect * 8 + np.random.normal(0, 2) if np.random.random() < 0.3 else np.nan
            lactate = 1.5 + sepsis_effect * 3 + np.random.normal(0, 0.5) if np.random.random() < 0.2 else np.nan
            creatinine = 1.0 + sepsis_effect * 0.5 + np.random.normal(0, 0.3) if np.random.random() < 0.25 else np.nan
            
            # Additional lab values (mostly missing but some present for realism)
            base_excess = -2 - sepsis_effect * 5 + np.random.normal(0, 2) if np.random.random() < 0.15 else np.nan
            hco3 = 24 - sepsis_effect * 4 + np.random.normal(0, 3) if np.random.random() < 0.15 else np.nan
            fio2 = 0.21 + sepsis_effect * 0.3 + np.random.normal(0, 0.1) if np.random.random() < 0.2 else np.nan
            ph = 7.4 - sepsis_effect * 0.1 + np.random.normal(0, 0.05) if np.random.random() < 0.15 else np.nan
            paco2 = 40 - sepsis_effect * 5 + np.random.normal(0, 5) if np.random.random() < 0.15 else np.nan
            sao2 = 98 - sepsis_effect * 5 + np.random.normal(0, 2) if np.random.random() < 0.2 else np.nan
            ast = 30 + sepsis_effect * 50 + np.random.normal(0, 15) if np.random.random() < 0.1 else np.nan
            bun = 15 + sepsis_effect * 20 + np.random.normal(0, 5) if np.random.random() < 0.15 else np.nan
            alkalinephos = 80 + sepsis_effect * 40 + np.random.normal(0, 20) if np.random.random() < 0.1 else np.nan
            calcium = 9.5 - sepsis_effect * 0.5 + np.random.normal(0, 0.3) if np.random.random() < 0.15 else np.nan
            chloride = 100 - sepsis_effect * 5 + np.random.normal(0, 3) if np.random.random() < 0.15 else np.nan
            bilirubin_direct = 0.3 + sepsis_effect * 1 + np.random.normal(0, 0.2) if np.random.random() < 0.1 else np.nan
            glucose = 120 + sepsis_effect * 80 + np.random.normal(0, 30) if np.random.random() < 0.2 else np.nan
            magnesium = 2.0 - sepsis_effect * 0.3 + np.random.normal(0, 0.2) if np.random.random() < 0.1 else np.nan
            phosphate = 3.5 - sepsis_effect * 0.5 + np.random.normal(0, 0.3) if np.random.random() < 0.1 else np.nan
            potassium = 4.0 + sepsis_effect * 1 + np.random.normal(0, 0.5) if np.random.random() < 0.15 else np.nan
            bilirubin_total = 1.0 + sepsis_effect * 2 + np.random.normal(0, 0.5) if np.random.random() < 0.1 else np.nan
            troponin = 0.1 + sepsis_effect * 5 + np.random.normal(0, 1) if np.random.random() < 0.05 else np.nan
            hct = 40 - sepsis_effect * 5 + np.random.normal(0, 3) if np.random.random() < 0.2 else np.nan
            hgb = 13 - sepsis_effect * 2 + np.random.normal(0, 1) if np.random.random() < 0.2 else np.nan
            ptt = 30 + sepsis_effect * 20 + np.random.normal(0, 5) if np.random.random() < 0.1 else np.nan
            fibrinogen = 300 + sepsis_effect * 200 + np.random.normal(0, 50) if np.random.random() < 0.05 else np.nan
            platelets = 250 - sepsis_effect * 100 + np.random.normal(0, 50) if np.random.random() < 0.2 else np.nan
            
            # Sepsis label
            sepsis_label = 1 if (sepsis_onset_hour is not None and hour >= sepsis_onset_hour) else 0
            
            record = {
                'Patient': f'p{patient_id:06d}',
                'ICULOS': hour + 1,
                # Vital signs
                'HR': max(30, min(200, hr)),
                'O2Sat': max(70, min(100, o2sat)),
                'Temp': max(30, min(45, temp)),
                'SBP': max(60, min(250, sbp)),
                'MAP': max(40, min(150, map_val)),
                'DBP': max(30, min(120, dbp)),
                'Resp': max(8, min(40, resp)),
                # Lab values
                'BaseExcess': base_excess,
                'HCO3': hco3,
                'FiO2': max(0.21, min(1.0, fio2)) if not pd.isna(fio2) else np.nan,
                'pH': max(6.8, min(7.8, ph)) if not pd.isna(ph) else np.nan,
                'PaCO2': max(15, min(80, paco2)) if not pd.isna(paco2) else np.nan,
                'SaO2': max(70, min(100, sao2)) if not pd.isna(sao2) else np.nan,
                'AST': max(10, min(500, ast)) if not pd.isna(ast) else np.nan,
                'BUN': max(5, min(100, bun)) if not pd.isna(bun) else np.nan,
                'Alkalinephos': max(30, min(300, alkalinephos)) if not pd.isna(alkalinephos) else np.nan,
                'Calcium': max(6, min(12, calcium)) if not pd.isna(calcium) else np.nan,
                'Chloride': max(90, min(115, chloride)) if not pd.isna(chloride) else np.nan,
                'Creatinine': max(0.5, min(8, creatinine)) if not pd.isna(creatinine) else np.nan,
                'Bilirubin_direct': max(0.1, min(5, bilirubin_direct)) if not pd.isna(bilirubin_direct) else np.nan,
                'Glucose': max(50, min(400, glucose)) if not pd.isna(glucose) else np.nan,
                'Lactate': max(0.5, min(10, lactate)) if not pd.isna(lactate) else np.nan,
                'Magnesium': max(1, min(4, magnesium)) if not pd.isna(magnesium) else np.nan,
                'Phosphate': max(1, min(8, phosphate)) if not pd.isna(phosphate) else np.nan,
                'Potassium': max(2, min(7, potassium)) if not pd.isna(potassium) else np.nan,
                'Bilirubin_total': max(0.2, min(10, bilirubin_total)) if not pd.isna(bilirubin_total) else np.nan,
                'TroponinI': max(0, min(50, troponin)) if not pd.isna(troponin) else np.nan,
                'Hct': max(20, min(60, hct)) if not pd.isna(hct) else np.nan,
                'Hgb': max(6, min(20, hgb)) if not pd.isna(hgb) else np.nan,
                'PTT': max(15, min(150, ptt)) if not pd.isna(ptt) else np.nan,
                'WBC': max(1, min(50, wbc)) if not pd.isna(wbc) else np.nan,
                'Fibrinogen': max(100, min(800, fibrinogen)) if not pd.isna(fibrinogen) else np.nan,
                'Platelets': max(50, min(600, platelets)) if not pd.isna(platelets) else np.nan,
                # Demographics
                'Age': age,
                'Gender': gender,
                'Unit1': unit1,
                'Unit2': unit2,
                'HospAdmTime': -hour,  # Simplified timestamp
                'SepsisLabel': sepsis_label,
            }
            
            data.append(record)
    
    df = pd.DataFrame(data)
    logger.info(f"Created synthetic dataset: {len(df)} records, {df['Patient'].nunique()} patients, {df['SepsisLabel'].mean():.3f} sepsis rate")
    
    return df


if __name__ == "__main__":
    # Example usage
    
    # Create sample dataset for testing
    print("Creating sample dataset...")
    sample_data = create_sample_dataset(n_patients=100, records_per_patient=20)
    sample_data.to_csv('data/raw/sample_sepsis_data.csv', index=False)
    
    # Initialize preprocessor
    preprocessor = SepsisDataPreprocessor()
    
    # Load and preprocess data
    df = preprocessor.load_and_validate_data('data/raw/sample_sepsis_data.csv')
    df_processed, stats = preprocessor.preprocess_pipeline(df, fit=True)
    
    # Save preprocessed data and preprocessor
    df_processed.to_csv('data/processed/preprocessed_sepsis_data.csv', index=False)
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    print("Preprocessing completed successfully!")
    print(f"Processed dataset shape: {df_processed.shape}")
    print(f"Sepsis rate: {df_processed['SepsisLabel'].mean():.3f}")