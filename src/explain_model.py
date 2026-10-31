"""
SepsisSense Model Explainability System
======================================

Advanced AI explainability pipeline using SHAP (SHapley Additive exPlanations)
for interpreting sepsis prediction models and providing clinical insights.

Author: SepsisSense Team
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SepsisModelExplainer:
    """
    Advanced explainability system for sepsis prediction models.
    
    Features:
    - SHAP value calculation for individual predictions
    - Global feature importance analysis
    - Clinical decision support insights
    - Interactive visualizations
    - Patient-specific explanations
    - Risk factor identification
    """
    
    def __init__(self, model=None, model_type: str = 'xgboost', config: Optional[Dict] = None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained model (XGBoost, LightGBM, etc.)
            model_type: Type of model for SHAP explainer selection
            config: Configuration dictionary
        """
        self.model = model
        self.model_type = model_type
        self.config = config or self._get_default_config()
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.baseline_value = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default explainer configuration."""
        return {
            'max_evals': 1000,  # For SHAP explainer
            'feature_perturbation': 'interventional',
            'top_features': 20,  # Number of top features to display
            'shap_sample_size': 500,  # Sample size for SHAP calculation
            'clinical_thresholds': {
                # Normal ranges for clinical interpretation
                'HR': (60, 100),
                'MAP': (70, 105),
                'O2Sat': (95, 100),
                'Temp': (36.1, 37.2),
                'Resp': (12, 20),
                'WBC': (4, 11),
                'Lactate': (0.5, 2.2),
                'Creatinine': (0.7, 1.3),
                'Platelets': (150, 450)
            },
            'risk_categories': {
                'very_low': (0, 0.2),
                'low': (0.2, 0.4),
                'moderate': (0.4, 0.6),
                'high': (0.6, 0.8),
                'very_high': (0.8, 1.01)
            }
        }
    
    def initialize_explainer(self, X_background: pd.DataFrame) -> None:
        """
        Initialize SHAP explainer with background data.
        
        Args:
            X_background: Background dataset for SHAP baseline calculation
        """
        logger.info(f"Initializing SHAP explainer for {self.model_type}...")
        
        # Clean background data - ensure all numeric types
        X_background = X_background.copy()
        
        # Convert all columns to float64 and remove any string data
        for col in X_background.columns:
            if X_background[col].dtype == 'object':
                # Try to convert string to numeric
                X_background[col] = pd.to_numeric(X_background[col], errors='coerce')
            
            # Ensure float64 type
            X_background[col] = X_background[col].astype(np.float64)
        
        # Replace inf and nan values
        X_background = X_background.replace([np.inf, -np.inf], np.nan)
        X_background = X_background.fillna(0)
        
        self.feature_names = X_background.columns.tolist()
        
        # Sample background data if too large
        if len(X_background) > self.config['shap_sample_size']:
            X_background = X_background.sample(
                n=self.config['shap_sample_size'], 
                random_state=42
            )
        
        try:
            # Initialize appropriate SHAP explainer based on model type
            if self.model_type.lower() in ['xgboost', 'xgb']:
                self.explainer = shap.TreeExplainer(
                    self.model,
                    X_background,
                    feature_perturbation=self.config['feature_perturbation']
                )
            elif self.model_type.lower() in ['lightgbm', 'lgb']:
                self.explainer = shap.TreeExplainer(
                    self.model,
                    X_background,
                    feature_perturbation=self.config['feature_perturbation']
                )
            else:
                # Fallback to model-agnostic explainer
                self.explainer = shap.Explainer(
                    self.model.predict_proba,
                    X_background,
                    max_evals=self.config['max_evals']
                )
            
            # Calculate baseline value
            self.baseline_value = self.explainer.expected_value
            if isinstance(self.baseline_value, np.ndarray):
                self.baseline_value = float(self.baseline_value[1])  # For binary classification
            else:
                self.baseline_value = float(self.baseline_value)
            
            logger.info(f"SHAP explainer initialized. Baseline value: {self.baseline_value:.4f}")
            
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {str(e)}")
            logger.warning("Using simplified explainer without SHAP")
            # Set to None to handle gracefully in other methods
            self.explainer = None
            self.baseline_value = 0.5  # Default baseline
    
    def calculate_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for given data.
        
        Args:
            X: Input features
            
        Returns:
            SHAP values array
        """
        logger.info(f"Calculating SHAP values for {len(X)} samples...")
        
        if self.explainer is None:
            logger.warning("SHAP explainer not available. Returning zero values.")
            # Return zero SHAP values as fallback
            return np.zeros((len(X), len(self.feature_names)))
        
        try:
            # Clean input data
            X_clean = X.copy()
            for col in X_clean.columns:
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce').fillna(0).astype(np.float64)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_clean)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification with list output
                shap_values = shap_values[1]  # Positive class
            elif shap_values.ndim == 3:
                # Multi-output format
                shap_values = shap_values[:, :, 1]  # Positive class
            
            self.shap_values = shap_values
            logger.info(f"SHAP values calculated. Shape: {shap_values.shape}")
            
            return shap_values
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
            # Return zero values as fallback
            return np.zeros((len(X), len(self.feature_names)))
    
    def get_global_feature_importance(self, shap_values: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate global feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values array (uses stored values if None)
            
        Returns:
            DataFrame with global feature importance
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("No SHAP values available. Calculate SHAP values first.")
        
        # Calculate mean absolute SHAP values
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance,
            'abs_importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Add normalized importance
        importance_df['importance_normalized'] = (
            importance_df['importance'] / importance_df['importance'].sum()
        )
        
        # Add cumulative importance
        importance_df['cumulative_importance'] = importance_df['importance_normalized'].cumsum()
        
        logger.info(f"Global feature importance calculated for {len(importance_df)} features")
        return importance_df
    
    def explain_single_prediction(self, X_single: pd.DataFrame, 
                                patient_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Provide detailed explanation for a single prediction.
        
        Args:
            X_single: Single sample features (DataFrame with 1 row)
            patient_id: Optional patient identifier
            
        Returns:
            Dictionary with explanation details
        """
        if len(X_single) != 1:
            raise ValueError("X_single must contain exactly one sample")
        
        logger.info(f"Explaining prediction for patient {patient_id or 'Unknown'}...")
        
        # Ensure features are aligned with expected feature names
        if self.feature_names is not None:
            X_eval = X_single.reindex(columns=self.feature_names, fill_value=0.0)
        else:
            X_eval = X_single.copy()
            self.feature_names = X_eval.columns.tolist()

        # Get prediction
        prediction_proba = float(self.model.predict_proba(X_eval)[0, 1])
        prediction_class = int(prediction_proba >= 0.5)
        
        # Get SHAP values for this sample or use feature importance
        if self.explainer is not None:
            try:
                shap_values_single = self.calculate_shap_values(X_eval)[0]
            except Exception:
                logger.warning("SHAP calculation failed, using model feature importance")
                # Use model's native feature importance as fallback
                if hasattr(self.model, 'feature_importances_'):
                    shap_values_single = self.model.feature_importances_ * (prediction_proba - 0.5)
                else:
                    shap_values_single = np.zeros(len(self.feature_names))
        else:
            # Use model's native feature importance as fallback
            if hasattr(self.model, 'feature_importances_'):
                # Scale importance by prediction deviation from baseline
                shap_values_single = self.model.feature_importances_ * (prediction_proba - self.baseline_value)
            else:
                shap_values_single = np.zeros(len(self.feature_names))

        if len(shap_values_single) != len(self.feature_names):
            shap_values_single = np.resize(shap_values_single, len(self.feature_names))
        
        # Create feature contribution DataFrame
        contributions_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_eval.iloc[0].values,
            'shap_value': shap_values_single,
            'abs_shap_value': np.abs(shap_values_single)
        })
        
        # Sort by absolute SHAP value
        contributions_df = contributions_df.sort_values('abs_shap_value', ascending=False)
        
        # Add clinical interpretations
        contributions_df['clinical_interpretation'] = contributions_df.apply(
            self._get_clinical_interpretation, axis=1
        )
        
        # Get risk category
        risk_category = self._get_risk_category(prediction_proba)
        
        # Calculate feature groups contribution
        feature_groups = self._group_features_by_type(contributions_df)
        
        explanation = {
            'patient_id': patient_id,
            'prediction_probability': prediction_proba,
            'prediction_class': prediction_class,
            'risk_category': risk_category,
            'baseline_risk': self.baseline_value,
            'contributions': contributions_df,
            'top_positive_features': contributions_df[contributions_df['shap_value'] > 0].head(10),
            'top_negative_features': contributions_df[contributions_df['shap_value'] < 0].head(10),
            'feature_groups': feature_groups,
            'clinical_summary': self._generate_clinical_summary(contributions_df, prediction_proba)
        }
        
        return explanation
    
    def _get_clinical_interpretation(self, row: pd.Series) -> str:
        """Get clinical interpretation for a feature value."""
        feature = row['feature']
        value = row['value']
        shap_value = row['shap_value']
        
        # Extract base feature name (remove prefixes/suffixes)
        base_feature = feature.split('_')[0]
        
        if base_feature in self.config['clinical_thresholds']:
            normal_range = self.config['clinical_thresholds'][base_feature]
            
            if value < normal_range[0]:
                status = "LOW"
            elif value > normal_range[1]:
                status = "HIGH"
            else:
                status = "NORMAL"
            
            direction = "increases" if shap_value > 0 else "decreases"
            
            return f"{status} value {direction} sepsis risk"
        
        return "Contributes to prediction"
    
    def _get_risk_category(self, probability: float) -> str:
        """Categorize risk based on probability."""
        for category, (low, high) in self.config['risk_categories'].items():
            if low <= probability < high:
                return category
        return 'very_high'  # For probability = 1.0
    
    def _group_features_by_type(self, contributions_df: pd.DataFrame) -> Dict[str, float]:
        """Group feature contributions by clinical category."""
        groups = {
            'vital_signs': 0,
            'lab_values': 0,
            'demographics': 0,
            'temporal': 0,
            'derived': 0
        }
        
        vital_keywords = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
        lab_keywords = ['WBC', 'Lactate', 'Creatinine', 'Platelets', 'pH', 'BUN']
        demo_keywords = ['Age', 'Gender', 'Unit']
        temporal_keywords = ['rolling', 'trend', 'slope', 'hours']
        
        for _, row in contributions_df.iterrows():
            feature = row['feature']
            shap_val = row['shap_value']
            
            if any(keyword in feature for keyword in vital_keywords):
                groups['vital_signs'] += shap_val
            elif any(keyword in feature for keyword in lab_keywords):
                groups['lab_values'] += shap_val
            elif any(keyword in feature for keyword in demo_keywords):
                groups['demographics'] += shap_val
            elif any(keyword in feature for keyword in temporal_keywords):
                groups['temporal'] += shap_val
            else:
                groups['derived'] += shap_val
        
        return groups
    
    def _generate_clinical_summary(self, contributions_df: pd.DataFrame, 
                                 probability: float) -> str:
        """Generate clinical summary for the prediction."""
        risk_category = self._get_risk_category(probability)
        
        # Get top contributing factors
        top_positive = contributions_df[contributions_df['shap_value'] > 0].head(3)
        top_negative = contributions_df[contributions_df['shap_value'] < 0].head(3)
        
        summary = f"Patient has {risk_category.replace('_', ' ')} sepsis risk ({probability:.1%}). "
        
        if len(top_positive) > 0:
            summary += f"Main risk factors: {', '.join(top_positive['feature'].tolist())}. "
        
        if len(top_negative) > 0:
            summary += f"Protective factors: {', '.join(top_negative['feature'].tolist())}."
        
        return summary
    
    def create_explanation_dashboard(self, explanation: Dict[str, Any]) -> go.Figure:
        """
        Create comprehensive explanation dashboard.
        
        Args:
            explanation: Explanation dictionary from explain_single_prediction
            
        Returns:
            Plotly figure with explanation dashboard
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Risk Gauge', 'Top Contributing Features',
                'Feature Groups Impact', 'SHAP Waterfall',
                'Clinical Values vs Normal Ranges', 'Risk Timeline'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "waterfall"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # Risk Gauge
        probability = explanation['prediction_probability']
        risk_category = explanation['risk_category']
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Sepsis Risk - {risk_category.replace('_', ' ').title()}"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self._get_risk_color(probability)},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 40], 'color': "yellow"},
                        {'range': [40, 60], 'color': "orange"},
                        {'range': [60, 80], 'color': "red"},
                        {'range': [80, 100], 'color': "darkred"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ),
            row=1, col=1
        )
        
        # Top Contributing Features
        top_features = explanation['contributions'].head(10)
        colors = ['red' if x > 0 else 'blue' for x in top_features['shap_value']]
        
        fig.add_trace(
            go.Bar(
                x=top_features['shap_value'],
                y=top_features['feature'],
                orientation='h',
                marker_color=colors,
                name='SHAP Values'
            ),
            row=1, col=2
        )
        
        # Feature Groups Impact
        groups = explanation['feature_groups']
        fig.add_trace(
            go.Bar(
                x=list(groups.keys()),
                y=list(groups.values()),
                marker_color=['red' if x > 0 else 'blue' for x in groups.values()],
                name='Group Impact'
            ),
            row=2, col=1
        )
        
        # SHAP Waterfall (simplified)
        baseline = explanation['baseline_risk']
        contributions = explanation['contributions'].head(5)
        
        waterfall_values = [baseline] + contributions['shap_value'].tolist() + [probability]
        waterfall_labels = ['Baseline'] + contributions['feature'].tolist() + ['Final']
        
        fig.add_trace(
            go.Waterfall(
                x=waterfall_labels,
                y=waterfall_values,
                name="SHAP Waterfall"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text=f"🔍 Sepsis Prediction Explanation - Patient {explanation['patient_id']}",
            showlegend=False
        )
        
        return fig
    
    def _get_risk_color(self, probability: float) -> str:
        """Get color for risk level."""
        if probability < 0.2:
            return "green"
        elif probability < 0.4:
            return "yellow"
        elif probability < 0.6:
            return "orange"
        elif probability < 0.8:
            return "red"
        else:
            return "darkred"
    
    def create_global_explanation_report(self, X: pd.DataFrame, 
                                       sample_size: int = 100) -> Dict[str, Any]:
        """
        Create global model explanation report.
        
        Args:
            X: Feature dataset
            sample_size: Number of samples for analysis
            
        Returns:
            Global explanation report
        """
        logger.info("Creating global explanation report...")
        
        # Sample data if needed
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X.copy()
        
        # Calculate SHAP values
        shap_values = self.calculate_shap_values(X_sample)
        
        # Global feature importance
        global_importance = self.get_global_feature_importance(shap_values)
        
        # Feature interactions (top features only)
        top_features = global_importance.head(20)['feature'].tolist()
        X_top = X_sample[top_features]
        shap_top = shap_values[:, [self.feature_names.index(f) for f in top_features]]
        
        # Calculate feature interactions
        interaction_scores = self._calculate_feature_interactions(X_top, shap_top)
        
        # Model behavior analysis
        predictions = self.model.predict_proba(X_sample)[:, 1]
        behavior_analysis = self._analyze_model_behavior(X_sample, predictions, shap_values)
        
        report = {
            'global_importance': global_importance,
            'interaction_scores': interaction_scores,
            'behavior_analysis': behavior_analysis,
            'sample_size': len(X_sample),
            'top_features': top_features[:10],
            'model_statistics': {
                'mean_prediction': predictions.mean(),
                'prediction_std': predictions.std(),
                'high_risk_percentage': (predictions > 0.8).mean(),
                'low_risk_percentage': (predictions < 0.2).mean()
            }
        }
        
        return report
    
    def _calculate_feature_interactions(self, X: pd.DataFrame, 
                                      shap_values: np.ndarray) -> pd.DataFrame:
        """Calculate pairwise feature interactions."""
        interactions = []
        features = X.columns.tolist()
        
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                # Simple interaction metric: correlation of SHAP values
                corr = np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]
                if not np.isnan(corr):
                    interactions.append({
                        'feature_1': features[i],
                        'feature_2': features[j],
                        'interaction_strength': abs(corr)
                    })
        
        interactions_df = pd.DataFrame(interactions).sort_values(
            'interaction_strength', ascending=False
        )
        
        return interactions_df.head(20)  # Top 20 interactions
    
    def _analyze_model_behavior(self, X: pd.DataFrame, predictions: np.ndarray,
                              shap_values: np.ndarray) -> Dict[str, Any]:
        """Analyze overall model behavior patterns."""
        
        # Prediction distribution
        pred_bins = np.histogram(predictions, bins=10)
        
        # Feature importance by prediction level
        high_risk_mask = predictions > 0.8
        low_risk_mask = predictions < 0.2
        
        high_risk_importance = np.abs(shap_values[high_risk_mask]).mean(axis=0) if high_risk_mask.any() else np.zeros(shap_values.shape[1])
        low_risk_importance = np.abs(shap_values[low_risk_mask]).mean(axis=0) if low_risk_mask.any() else np.zeros(shap_values.shape[1])
        
        analysis = {
            'prediction_distribution': {
                'bins': pred_bins[1].tolist(),
                'counts': pred_bins[0].tolist()
            },
            'high_risk_patterns': {
                'count': high_risk_mask.sum(),
                'top_features': [self.feature_names[i] for i in high_risk_importance.argsort()[-10:][::-1]]
            },
            'low_risk_patterns': {
                'count': low_risk_mask.sum(),
                'top_features': [self.feature_names[i] for i in low_risk_importance.argsort()[-10:][::-1]]
            },
            'model_confidence': {
                'high_confidence_predictions': ((predictions > 0.8) | (predictions < 0.2)).mean(),
                'uncertain_predictions': ((predictions >= 0.4) & (predictions <= 0.6)).mean()
            }
        }
        
        return analysis
    
    def save_explainer(self, filepath: str) -> None:
        """Save the explainer and its components."""
        explainer_data = {
            'model': self.model,
            'model_type': self.model_type,
            'config': self.config,
            'feature_names': self.feature_names,
            'baseline_value': self.baseline_value
        }
        
        joblib.dump(explainer_data, filepath)
        logger.info(f"Explainer saved to {filepath}")
    
    @classmethod
    def load_explainer(cls, filepath: str) -> 'SepsisModelExplainer':
        """Load a saved explainer."""
        explainer_data = joblib.load(filepath)
        
        explainer = cls(
            model=explainer_data['model'],
            model_type=explainer_data['model_type'],
            config=explainer_data['config']
        )
        
        explainer.feature_names = explainer_data['feature_names']
        explainer.baseline_value = explainer_data['baseline_value']
        
        logger.info(f"Explainer loaded from {filepath}")
        return explainer


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('.')
    
    from src.preprocess import SepsisDataPreprocessor, create_sample_dataset
    from src.feature_engineering import SepsisFeatureEngineer
    from src.train_model import SepsisModelTrainer
    
    # Create and prepare data
    print("Creating sample dataset...")
    sample_data = create_sample_dataset(n_patients=100, records_per_patient=15)
    
    # Preprocess and engineer features
    preprocessor = SepsisDataPreprocessor()
    df_processed, _ = preprocessor.preprocess_pipeline(sample_data, fit=True)
    
    feature_engineer = SepsisFeatureEngineer()
    df_features = feature_engineer.engineer_features_pipeline(df_processed)
    
    # Train model
    trainer = SepsisModelTrainer()
    results = trainer.train_and_evaluate_pipeline(df_features)
    
    # Initialize explainer
    best_model = results['best_model']
    X_test, y_test = results['test_data']
    
    explainer = SepsisModelExplainer(
        model=best_model,
        model_type=results['best_model_name']
    )
    
    # Initialize with background data
    explainer.initialize_explainer(X_test.head(50))
    
    # Create global explanation
    global_report = explainer.create_global_explanation_report(X_test)
    
    # Explain single prediction
    single_sample = X_test.head(1)
    explanation = explainer.explain_single_prediction(single_sample, "test_patient_001")
    
    # Create dashboard
    dashboard = explainer.create_explanation_dashboard(explanation)
    try:
        dashboard.write_html('models/explanation_dashboard.html')
        print("Explanation dashboard saved -> models/explanation_dashboard.html")
    except Exception as e:
        logger.warning(f"Could not save dashboard HTML: {e}")
    
    # Save explainer
    explainer.save_explainer('models/explainer.pkl')
    
    print("Explainability analysis completed!")
    print(f"Patient risk: {explanation['prediction_probability']:.3f}")
    print(f"Risk category: {explanation['risk_category']}")
    print(f"Top risk factor: {explanation['top_positive_features'].iloc[0]['feature']}")