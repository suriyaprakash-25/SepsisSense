"""
SepsisSense Feature Engineering Pipeline
======================================

Advanced feature engineering for sepsis prediction including rolling window statistics,
trend analysis, and patient-specific temporal features.

Author: SepsisSense Team
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SepsisFeatureEngineer:
    """
    Advanced feature engineering pipeline for sepsis prediction.
    
    Features:
    - Rolling window statistics (mean, std, min, max, slope)
    - Trend analysis and change point detection
    - Vital signs stability and variability measures
    - Patient-specific temporal patterns
    - Clinical scoring systems integration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: Configuration dictionary with feature engineering parameters
        """
        self.config = config or self._get_default_config()
        self.feature_columns = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature engineering configuration."""
        return {
            'vital_signs': [
                'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp'
            ],
            'lab_values': [
                'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
                'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
                'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
                'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
                'Fibrinogen', 'Platelets'
            ],
            'patient_id': 'Patient',
            'time_column': 'ICULOS',
            'rolling_windows': [2, 4, 6, 8, 12],  # Hours
            'trend_windows': [4, 6, 8],  # Hours for trend analysis
            'stability_threshold': 0.1,  # For stability detection
            'change_threshold': 2.0,  # Standard deviations for change detection
            'min_data_points': 3,  # Minimum points for calculations
        }
    
    def calculate_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling window statistics for vital signs and lab values.
        
        Args:
            df: Input DataFrame with patient data
            
        Returns:
            DataFrame with added rolling statistics features
        """
        logger.info("Calculating rolling window statistics...")
        
        df_features = df.copy()
        numeric_columns = [col for col in (self.config['vital_signs'] + self.config['lab_values']) 
                          if col in df.columns]
        
        # Sort by patient and time
        df_features = df_features.sort_values([self.config['patient_id'], self.config['time_column']])
        
        new_features = []
        
        for window in self.config['rolling_windows']:
            logger.info(f"Processing {window}-hour rolling window...")
            
            for col in numeric_columns:
                if col not in df_features.columns:
                    continue
                
                try:
                    # Calculate rolling statistics for each patient
                    for stat_name, stat_func in [
                        ('mean', 'mean'), ('std', 'std'), ('min', 'min'), 
                        ('max', 'max'), ('median', 'median'), ('count', 'count')
                    ]:
                        feature_name = f"{col}_rolling_{window}h_{stat_name}"
                        
                        # Group by patient and calculate rolling statistic
                        try:
                            rolled = (
                                df_features.groupby(self.config['patient_id'], group_keys=False)[col]
                                .rolling(window=window, min_periods=1)
                                .agg(stat_func)
                            )
                            
                            # Align the rolled values back to the dataframe
                            if len(rolled) == len(df_features):
                                df_features[feature_name] = rolled.values
                            else:
                                # Fallback: use the original column
                                df_features[feature_name] = df_features[col].fillna(0)
                                
                        except Exception as e:
                            logger.warning(f"Error calculating {feature_name}: {str(e)}")
                            # Fallback to original values
                            df_features[feature_name] = df_features[col].fillna(0)
                        
                        new_features.append(feature_name)
                    
                    # Range (max - min)
                    range_feature = f"{col}_rolling_{window}h_range"
                    max_col = f"{col}_rolling_{window}h_max"
                    min_col = f"{col}_rolling_{window}h_min"
                    
                    if max_col in df_features.columns and min_col in df_features.columns:
                        df_features[range_feature] = (
                            df_features[max_col] - df_features[min_col]
                        )
                    else:
                        df_features[range_feature] = 0
                    new_features.append(range_feature)
                    
                    # Coefficient of variation (std/mean)
                    cv_feature = f"{col}_rolling_{window}h_cv"
                    std_col = f"{col}_rolling_{window}h_std"
                    mean_col = f"{col}_rolling_{window}h_mean"
                    
                    if std_col in df_features.columns and mean_col in df_features.columns:
                        df_features[cv_feature] = (
                            df_features[std_col] / 
                            (df_features[mean_col] + 1e-6)
                        )
                    else:
                        df_features[cv_feature] = 0
                    new_features.append(cv_feature)
                    
                except Exception as e:
                    logger.warning(f"Error processing rolling statistics for {col}: {str(e)}")
                    continue
        
        logger.info(f"Added {len(new_features)} rolling statistics features")
        return df_features
    
    def calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend analysis features including slopes and change points.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with trend features
        """
        logger.info("Calculating trend analysis features...")
        
        df_trends = df.copy()
        numeric_columns = [col for col in (self.config['vital_signs'] + self.config['lab_values']) 
                          if col in df.columns]
        
        new_features = []
        
        for window in self.config['trend_windows']:
            logger.info(f"Processing {window}-hour trend window...")
            
            for col in numeric_columns:
                if col not in df_trends.columns:
                    continue
                
                # Calculate slope (trend) over the window
                slope_feature = f"{col}_trend_{window}h_slope"
                df_trends[slope_feature] = np.nan
                
                # Calculate acceleration (second derivative)
                accel_feature = f"{col}_trend_{window}h_acceleration"
                df_trends[slope_feature] = np.nan
                
                # Process each patient separately
                for patient_id in df_trends[self.config['patient_id']].unique():
                    patient_mask = df_trends[self.config['patient_id']] == patient_id
                    patient_data = df_trends[patient_mask].copy()
                    
                    if len(patient_data) < self.config['min_data_points']:
                        continue
                    
                    # Calculate rolling slopes
                    slopes = []
                    accelerations = []
                    
                    for i in range(len(patient_data)):
                        start_idx = max(0, i - window + 1)
                        end_idx = i + 1
                        
                        window_data = patient_data.iloc[start_idx:end_idx]
                        
                        if len(window_data) >= self.config['min_data_points']:
                            # Calculate slope using linear regression
                            x = np.array(range(len(window_data)))
                            y = window_data[col].values
                            
                            # Remove NaN values
                            valid_mask = ~np.isnan(y)
                            if valid_mask.sum() >= self.config['min_data_points']:
                                x_valid = x[valid_mask]
                                y_valid = y[valid_mask]
                                
                                if len(x_valid) > 1:
                                    slope, _, _, _, _ = stats.linregress(x_valid, y_valid)
                                    slopes.append(slope)
                                    
                                    # Calculate acceleration (change in slope)
                                    if len(slopes) >= 2:
                                        acceleration = slopes[-1] - slopes[-2]
                                        accelerations.append(acceleration)
                                    else:
                                        accelerations.append(0)
                                else:
                                    slopes.append(0)
                                    accelerations.append(0)
                            else:
                                slopes.append(np.nan)
                                accelerations.append(np.nan)
                        else:
                            slopes.append(np.nan)
                            accelerations.append(np.nan)
                    
                    # Update dataframe
                    df_trends.loc[patient_mask, slope_feature] = slopes
                    df_trends.loc[patient_mask, accel_feature] = accelerations
                
                new_features.extend([slope_feature, accel_feature])
        
        logger.info(f"Added {len(new_features)} trend analysis features")
        return df_trends
    
    def calculate_stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate stability and variability features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with stability features
        """
        logger.info("Calculating stability features...")
        
        df_stability = df.copy()
        numeric_columns = [col for col in (self.config['vital_signs'] + self.config['lab_values']) 
                          if col in df.columns]
        
        new_features = []
        
        for col in numeric_columns:
            if col not in df_stability.columns:
                continue
            
            # Stability score (inverse of coefficient of variation)
            stability_feature = f"{col}_stability_score"
            cv_6h = df_stability.get(f"{col}_rolling_6h_cv", 1.0)
            df_stability[stability_feature] = 1 / (1 + cv_6h)
            new_features.append(stability_feature)
            
            # Variability trend (change in std over time)
            variability_trend_feature = f"{col}_variability_trend"
            std_2h = df_stability.get(f"{col}_rolling_2h_std", 0)
            std_6h = df_stability.get(f"{col}_rolling_6h_std", 0)
            df_stability[variability_trend_feature] = std_6h - std_2h
            new_features.append(variability_trend_feature)
            
            # Peak detection features
            peaks_feature = f"{col}_peaks_6h"
            df_stability[peaks_feature] = 0
            
            # Process each patient separately for peak detection
            for patient_id in df_stability[self.config['patient_id']].unique():
                patient_mask = df_stability[self.config['patient_id']] == patient_id
                patient_data = df_stability[patient_mask][col].values
                
                if len(patient_data) >= 6:  # Need at least 6 points
                    # Find peaks in the last 6 hours
                    for i in range(5, len(patient_data)):
                        window_data = patient_data[max(0, i-5):i+1]
                        if not np.isnan(window_data).all():
                            peaks, _ = find_peaks(window_data, height=np.nanmean(window_data))
                            df_stability.loc[patient_mask.iloc[i], peaks_feature] = len(peaks)
            
            new_features.append(peaks_feature)
        
        logger.info(f"Added {len(new_features)} stability features")
        return df_stability
    
    def calculate_change_point_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect significant change points in vital signs.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with change point features
        """
        logger.info("Calculating change point features...")
        
        df_changes = df.copy()
        numeric_columns = [col for col in (self.config['vital_signs'] + self.config['lab_values']) 
                          if col in df.columns]
        
        new_features = []
        
        for col in numeric_columns:
            if col not in df_changes.columns:
                continue
            
            # Sudden change detection
            change_feature = f"{col}_sudden_change"
            df_changes[change_feature] = 0
            
            # Rate of change (first derivative approximation)
            rate_change_feature = f"{col}_rate_of_change"
            df_changes[rate_change_feature] = 0
            
            # Process each patient separately
            for patient_id in df_changes[self.config['patient_id']].unique():
                patient_mask = df_changes[self.config['patient_id']] == patient_id
                patient_data = df_changes[patient_mask][col].values
                
                if len(patient_data) >= 3:
                    # Calculate differences
                    diffs = np.diff(patient_data)
                    
                    # Rate of change (smoothed differences)
                    rate_changes = np.concatenate([[0], diffs])
                    df_changes.loc[patient_mask, rate_change_feature] = rate_changes
                    
                    # Detect sudden changes (outliers in differences)
                    if len(diffs) > 3:
                        diff_mean = np.nanmean(diffs)
                        diff_std = np.nanstd(diffs)
                        
                        sudden_changes = np.abs(diffs - diff_mean) > (self.config['change_threshold'] * diff_std)
                        sudden_changes = np.concatenate([[False], sudden_changes])
                        
                        df_changes.loc[patient_mask, change_feature] = sudden_changes.astype(int)
            
            new_features.extend([change_feature, rate_change_feature])
        
        logger.info(f"Added {len(new_features)} change point features")
        return df_changes
    
    def calculate_clinical_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate clinical scoring systems (SOFA-like scores).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with clinical score features
        """
        logger.info("Calculating clinical scores...")
        
        df_scores = df.copy()
        new_features = []
        
        # Simplified SOFA-like respiratory score
        if 'O2Sat' in df_scores.columns:
            sofa_resp_feature = 'sofa_respiratory_score'
            df_scores[sofa_resp_feature] = 0
            
            # Score based on O2Sat levels
            df_scores.loc[df_scores['O2Sat'] < 88, sofa_resp_feature] = 4
            df_scores.loc[(df_scores['O2Sat'] >= 88) & (df_scores['O2Sat'] < 92), sofa_resp_feature] = 3
            df_scores.loc[(df_scores['O2Sat'] >= 92) & (df_scores['O2Sat'] < 96), sofa_resp_feature] = 2
            df_scores.loc[(df_scores['O2Sat'] >= 96) & (df_scores['O2Sat'] < 98), sofa_resp_feature] = 1
            
            new_features.append(sofa_resp_feature)
        
        # Cardiovascular score (based on MAP)
        if 'MAP' in df_scores.columns:
            sofa_cardio_feature = 'sofa_cardiovascular_score'
            df_scores[sofa_cardio_feature] = 0
            
            df_scores.loc[df_scores['MAP'] < 50, sofa_cardio_feature] = 4
            df_scores.loc[(df_scores['MAP'] >= 50) & (df_scores['MAP'] < 60), sofa_cardio_feature] = 3
            df_scores.loc[(df_scores['MAP'] >= 60) & (df_scores['MAP'] < 70), sofa_cardio_feature] = 2
            df_scores.loc[(df_scores['MAP'] >= 70) & (df_scores['MAP'] < 80), sofa_cardio_feature] = 1
            
            new_features.append(sofa_cardio_feature)
        
        # Renal score (based on Creatinine)
        if 'Creatinine' in df_scores.columns:
            sofa_renal_feature = 'sofa_renal_score'
            df_scores[sofa_renal_feature] = 0
            
            df_scores.loc[df_scores['Creatinine'] > 5.0, sofa_renal_feature] = 4
            df_scores.loc[(df_scores['Creatinine'] > 3.5) & (df_scores['Creatinine'] <= 5.0), sofa_renal_feature] = 3
            df_scores.loc[(df_scores['Creatinine'] > 2.0) & (df_scores['Creatinine'] <= 3.5), sofa_renal_feature] = 2
            df_scores.loc[(df_scores['Creatinine'] > 1.2) & (df_scores['Creatinine'] <= 2.0), sofa_renal_feature] = 1
            
            new_features.append(sofa_renal_feature)
        
        # Hepatic score (based on Bilirubin)
        if 'Bilirubin_total' in df_scores.columns:
            sofa_hepatic_feature = 'sofa_hepatic_score'
            df_scores[sofa_hepatic_feature] = 0
            
            df_scores.loc[df_scores['Bilirubin_total'] > 12.0, sofa_hepatic_feature] = 4
            df_scores.loc[(df_scores['Bilirubin_total'] > 6.0) & (df_scores['Bilirubin_total'] <= 12.0), sofa_hepatic_feature] = 3
            df_scores.loc[(df_scores['Bilirubin_total'] > 2.0) & (df_scores['Bilirubin_total'] <= 6.0), sofa_hepatic_feature] = 2
            df_scores.loc[(df_scores['Bilirubin_total'] > 1.2) & (df_scores['Bilirubin_total'] <= 2.0), sofa_hepatic_feature] = 1
            
            new_features.append(sofa_hepatic_feature)
        
        # Coagulation score (based on Platelets)
        if 'Platelets' in df_scores.columns:
            sofa_coag_feature = 'sofa_coagulation_score'
            df_scores[sofa_coag_feature] = 0
            
            df_scores.loc[df_scores['Platelets'] < 20, sofa_coag_feature] = 4
            df_scores.loc[(df_scores['Platelets'] >= 20) & (df_scores['Platelets'] < 50), sofa_coag_feature] = 3
            df_scores.loc[(df_scores['Platelets'] >= 50) & (df_scores['Platelets'] < 100), sofa_coag_feature] = 2
            df_scores.loc[(df_scores['Platelets'] >= 100) & (df_scores['Platelets'] < 150), sofa_coag_feature] = 1
            
            new_features.append(sofa_coag_feature)
        
        # Total SOFA-like score
        sofa_columns = [col for col in new_features if col.startswith('sofa_')]
        if sofa_columns:
            total_sofa_feature = 'total_sofa_score'
            df_scores[total_sofa_feature] = df_scores[sofa_columns].sum(axis=1)
            new_features.append(total_sofa_feature)
        
        logger.info(f"Added {len(new_features)} clinical score features")
        return df_scores
    
    def calculate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate interaction features between important vital signs.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Calculating interaction features...")
        
        df_interactions = df.copy()
        new_features = []
        
        # Vital sign ratios and interactions
        interactions = [
            ('HR', 'MAP', 'hr_map_ratio'),  # Heart rate to blood pressure ratio
            ('HR', 'O2Sat', 'hr_o2sat_ratio'),  # Heart rate to oxygen saturation
            ('Temp', 'WBC', 'temp_wbc_interaction'),  # Temperature and white blood cells
            ('Lactate', 'MAP', 'lactate_map_interaction'),  # Lactate and blood pressure
            ('Resp', 'O2Sat', 'resp_o2sat_ratio'),  # Respiratory rate to oxygen saturation
        ]
        
        for var1, var2, feature_name in interactions:
            if var1 in df_interactions.columns and var2 in df_interactions.columns:
                # Simple ratio
                df_interactions[feature_name] = df_interactions[var1] / (df_interactions[var2] + 1e-6)
                new_features.append(feature_name)
                
                # Multiplicative interaction
                mult_feature = f"{var1}_{var2}_product"
                df_interactions[mult_feature] = df_interactions[var1] * df_interactions[var2]
                new_features.append(mult_feature)
        
        # Shock index (HR/SBP)
        if 'HR' in df_interactions.columns and 'SBP' in df_interactions.columns:
            shock_index_feature = 'shock_index'
            df_interactions[shock_index_feature] = df_interactions['HR'] / (df_interactions['SBP'] + 1e-6)
            new_features.append(shock_index_feature)
        
        # Modified shock index (HR/MAP)
        if 'HR' in df_interactions.columns and 'MAP' in df_interactions.columns:
            modified_shock_index_feature = 'modified_shock_index'
            df_interactions[modified_shock_index_feature] = df_interactions['HR'] / (df_interactions['MAP'] + 1e-6)
            new_features.append(modified_shock_index_feature)
        
        logger.info(f"Added {len(new_features)} interaction features")
        return df_interactions
    
    def calculate_patient_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate patient-specific baseline and deviation features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with patient-specific features
        """
        logger.info("Calculating patient-specific features...")
        
        df_patient = df.copy()
        numeric_columns = [col for col in (self.config['vital_signs'] + self.config['lab_values']) 
                          if col in df.columns]
        
        new_features = []
        
        # Calculate patient baselines (first few hours)
        baseline_hours = 4
        
        for col in numeric_columns:
            if col not in df_patient.columns:
                continue
            
            # Patient baseline (mean of first few hours)
            baseline_feature = f"{col}_patient_baseline"
            df_patient[baseline_feature] = np.nan
            
            # Deviation from baseline
            deviation_feature = f"{col}_baseline_deviation"
            df_patient[deviation_feature] = np.nan
            
            # Relative change from baseline
            relative_change_feature = f"{col}_relative_change"
            df_patient[relative_change_feature] = np.nan
            
            for patient_id in df_patient[self.config['patient_id']].unique():
                patient_mask = df_patient[self.config['patient_id']] == patient_id
                patient_data = df_patient[patient_mask].copy()
                
                if len(patient_data) >= baseline_hours:
                    # Calculate baseline from first few hours
                    baseline_data = patient_data.head(baseline_hours)[col]
                    baseline_value = baseline_data.mean()
                    
                    # Fill baseline value for all records of this patient
                    df_patient.loc[patient_mask, baseline_feature] = baseline_value
                    
                    # Calculate deviations
                    current_values = patient_data[col]
                    deviations = current_values - baseline_value
                    relative_changes = (current_values - baseline_value) / (baseline_value + 1e-6)
                    
                    df_patient.loc[patient_mask, deviation_feature] = deviations.values
                    df_patient.loc[patient_mask, relative_change_feature] = relative_changes.values
            
            new_features.extend([baseline_feature, deviation_feature, relative_change_feature])
        
        logger.info(f"Added {len(new_features)} patient-specific features")
        return df_patient
    
    def engineer_features_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input preprocessed DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting complete feature engineering pipeline...")
        
        # Step 1: Rolling statistics
        df_engineered = self.calculate_rolling_statistics(df)
        
        # Step 2: Trend analysis
        df_engineered = self.calculate_trend_features(df_engineered)
        
        # Step 3: Stability features
        df_engineered = self.calculate_stability_features(df_engineered)
        
        # Step 4: Change point detection
        df_engineered = self.calculate_change_point_features(df_engineered)
        
        # Step 5: Clinical scores
        df_engineered = self.calculate_clinical_scores(df_engineered)
        
        # Step 6: Interaction features
        df_engineered = self.calculate_interaction_features(df_engineered)
        
        # Step 7: Patient-specific features
        df_engineered = self.calculate_patient_specific_features(df_engineered)
        
        # Store feature column names
        original_columns = set(df.columns)
        new_columns = set(df_engineered.columns) - original_columns
        self.feature_columns = list(new_columns)
        
        logger.info(f"Feature engineering completed. Added {len(self.feature_columns)} new features")
        logger.info(f"Total features: {df_engineered.shape[1]}")
        
        return df_engineered
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Group features by category for interpretability.
        
        Returns:
            Dictionary mapping feature categories to feature lists
        """
        feature_groups = {
            'rolling_stats': [],
            'trends': [],
            'stability': [],
            'changes': [],
            'clinical_scores': [],
            'interactions': [],
            'patient_specific': [],
            'original': []
        }
        
        for feature in self.feature_columns:
            if 'rolling' in feature:
                feature_groups['rolling_stats'].append(feature)
            elif 'trend' in feature or 'slope' in feature or 'acceleration' in feature:
                feature_groups['trends'].append(feature)
            elif 'stability' in feature or 'variability' in feature or 'peaks' in feature:
                feature_groups['stability'].append(feature)
            elif 'change' in feature or 'rate_of_change' in feature:
                feature_groups['changes'].append(feature)
            elif 'sofa' in feature:
                feature_groups['clinical_scores'].append(feature)
            elif any(x in feature for x in ['ratio', 'interaction', 'product', 'shock_index']):
                feature_groups['interactions'].append(feature)
            elif any(x in feature for x in ['baseline', 'deviation', 'relative_change']):
                feature_groups['patient_specific'].append(feature)
            else:
                feature_groups['original'].append(feature)
        
        return feature_groups
    
    def save_feature_engineer(self, filepath: str) -> None:
        """Save the feature engineering configuration."""
        import joblib
        
        engineer_data = {
            'config': self.config,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(engineer_data, filepath)
        logger.info(f"Feature engineer saved to {filepath}")
    
    @classmethod
    def load_feature_engineer(cls, filepath: str) -> 'SepsisFeatureEngineer':
        """Load a feature engineering configuration."""
        import joblib
        
        engineer_data = joblib.load(filepath)
        
        engineer = cls(config=engineer_data['config'])
        engineer.feature_columns = engineer_data['feature_columns']
        
        logger.info(f"Feature engineer loaded from {filepath}")
        return engineer


if __name__ == "__main__":
    # Example usage
    from preprocess import SepsisDataPreprocessor, create_sample_dataset
    
    # Create sample data
    print("Creating sample dataset...")
    sample_data = create_sample_dataset(n_patients=50, records_per_patient=20)
    
    # Preprocess data
    preprocessor = SepsisDataPreprocessor()
    df_processed, _ = preprocessor.preprocess_pipeline(sample_data, fit=True)
    
    # Engineer features
    feature_engineer = SepsisFeatureEngineer()
    df_features = feature_engineer.engineer_features_pipeline(df_processed)
    
    # Save results
    df_features.to_csv('data/processed/features_sepsis_data.csv', index=False)
    feature_engineer.save_feature_engineer('models/feature_engineer.pkl')
    
    print("Feature engineering completed successfully!")
    print(f"Final dataset shape: {df_features.shape}")
    print(f"New features added: {len(feature_engineer.feature_columns)}")
    
    # Display feature groups
    feature_groups = feature_engineer.get_feature_importance_groups()
    for group, features in feature_groups.items():
        if features:
            print(f"\n{group.upper()} ({len(features)} features):")
            print(", ".join(features[:5]) + ("..." if len(features) > 5 else ""))