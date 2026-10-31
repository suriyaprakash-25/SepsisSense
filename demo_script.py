#!/usr/bin/env python3
"""
SepsisSense Demo Script
=====================

Quick demonstration of the SepsisSense AI-powered sepsis prediction system.
This script showcases the complete pipeline from data preprocessing to prediction.

Usage:
    python demo_script.py

Author: SepsisSense Team
License: MIT
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_header():
    """Print demo header."""
    print("\n" + "="*60)
    print("🩺 SepsisSense - AI-Powered Early Sepsis Detection")
    print("="*60)
    print("🚀 Running Complete Demo Pipeline...")
    print()

def print_step(step_num, description):
    """Print step header."""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 50)

def main():
    """Run complete demo."""
    
    print_header()
    
    # Import modules
    try:
        from preprocess import SepsisDataPreprocessor, create_sample_dataset
        from feature_engineering import SepsisFeatureEngineer
        from train_model import SepsisModelTrainer
        from explain_model import SepsisModelExplainer
        import pandas as pd
        import numpy as np
        
        print("✅ All modules imported successfully!")
        
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("Please ensure you're running from the SepsisSense root directory.")
        return
    
    # Step 1: Generate Sample Data
    print_step(1, "Generating Sample Patient Data")
    
    print("Creating synthetic ICU patient dataset...")
    sample_data = create_sample_dataset(n_patients=100, records_per_patient=20)
    
    print(f"✅ Generated {len(sample_data):,} records for {sample_data['Patient'].nunique()} patients")
    print(f"   Sepsis rate: {sample_data['SepsisLabel'].mean():.2%}")
    print(f"   Time range: {sample_data['ICULOS'].min()}-{sample_data['ICULOS'].max()} hours")
    
    # Step 2: Data Preprocessing
    print_step(2, "Data Preprocessing & Cleaning")
    
    print("Preprocessing patient data...")
    preprocessor = SepsisDataPreprocessor()
    df_processed, stats = preprocessor.preprocess_pipeline(sample_data, fit=True)
    
    print(f"✅ Preprocessing completed")
    print(f"   Original shape: {sample_data.shape}")
    print(f"   Processed shape: {df_processed.shape}")
    print(f"   Missing values handled: ✅")
    print(f"   Features normalized: ✅")
    
    # Step 3: Feature Engineering
    print_step(3, "Advanced Feature Engineering")
    
    print("Engineering predictive features...")
    feature_engineer = SepsisFeatureEngineer()
    df_features = feature_engineer.engineer_features_pipeline(df_processed)
    
    print(f"✅ Feature engineering completed")
    print(f"   Original features: {df_processed.shape[1]}")
    print(f"   Engineered features: {df_features.shape[1]}")
    print(f"   New features added: {len(feature_engineer.feature_columns)}")
    
    feature_groups = feature_engineer.get_feature_importance_groups()
    for group, features in feature_groups.items():
        if features:
            print(f"   {group}: {len(features)} features")
    
    # Step 4: Model Training
    print_step(4, "AI Model Training & Validation")
    
    print("Training machine learning models...")
    trainer = SepsisModelTrainer()
    results = trainer.train_and_evaluate_pipeline(df_features)
    
    best_model_name = results['best_model_name']
    best_metrics = results['metrics'][best_model_name]
    
    print(f"✅ Model training completed")
    print(f"   Best model: {best_model_name}")
    print(f"   ROC-AUC: {best_metrics['roc_auc']:.3f}")
    print(f"   Precision-Recall AUC: {best_metrics['pr_auc']:.3f}")
    print(f"   F1 Score: {best_metrics['f1_score']:.3f}")
    print(f"   Recall@Precision≥80%: {best_metrics['recall_at_precision_80']:.3f}")
    
    # Step 5: Model Explainability
    print_step(5, "AI Explainability & Interpretability")
    
    print("Initializing explainability system...")
    explainer = SepsisModelExplainer(
        model=results['best_model'],
        model_type=best_model_name
    )
    
    X_test, y_test = results['test_data']
    explainer.initialize_explainer(X_test.head(50))
    
    print(f"✅ Explainability system ready")
    print(f"   SHAP explainer initialized")
    print(f"   Background dataset: {len(X_test)} samples")
    
    # Step 6: Sample Prediction
    print_step(6, "Sample Patient Prediction & Explanation")
    
    print("Analyzing sample patient...")
    
    # Get a sample patient
    sample_patient = X_test.head(1)
    sample_patient_id = "DEMO_PATIENT_001"
    
    # Make prediction
    prediction_proba = results['best_model'].predict_proba(sample_patient)[0, 1]
    
    # Get explanation
    explanation = explainer.explain_single_prediction(sample_patient, sample_patient_id)
    
    print(f"✅ Prediction completed for {sample_patient_id}")
    print(f"   Sepsis Risk: {prediction_proba:.1%}")
    print(f"   Risk Category: {explanation['risk_category']}")
    print(f"   Clinical Summary: {explanation['clinical_summary'][:100]}...")
    
    # Show top contributing features
    top_features = explanation['top_positive_features'].head(5)
    print(f"\n   🔍 Top Risk Factors:")
    for _, feature in top_features.iterrows():
        print(f"      • {feature['feature']}: {feature['shap_value']:+.3f}")
    
    # Step 7: Demo Summary
    print_step(7, "Demo Results Summary")
    
    print("📊 SepsisSense System Performance:")
    print(f"   🎯 Accuracy: {best_metrics['accuracy']:.1%}")
    print(f"   🎯 Sensitivity: {best_metrics['sensitivity']:.1%}")
    print(f"   🎯 Specificity: {best_metrics['specificity']:.1%}")
    print(f"   🎯 Early Warning: 6 hours before sepsis onset")
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   👥 Patients analyzed: {sample_data['Patient'].nunique():,}")
    print(f"   📊 Total observations: {len(sample_data):,}")
    print(f"   ⚕️ Sepsis cases detected: {sample_data['SepsisLabel'].sum():,}")
    print(f"   🔬 Features engineered: {len(feature_engineer.feature_columns):,}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Run 'streamlit run app/streamlit_app.py' for the web interface")
    print(f"   2. Explore Jupyter notebooks in the 'notebooks/' folder")
    print(f"   3. Check out the comprehensive README.md documentation")
    print(f"   4. Customize the system for your specific use case")
    
    print("\n" + "="*60)
    print("✨ SepsisSense Demo Completed Successfully! ✨")
    print("="*60)
    print()
    
    return {
        'model_performance': best_metrics,
        'sample_prediction': {
            'patient_id': sample_patient_id,
            'risk_probability': prediction_proba,
            'risk_category': explanation['risk_category']
        },
        'dataset_stats': {
            'patients': sample_data['Patient'].nunique(),
            'records': len(sample_data),
            'sepsis_rate': sample_data['SepsisLabel'].mean()
        }
    }

if __name__ == "__main__":
    try:
        demo_results = main()
        
        # Optionally save results
        import json
        os.makedirs('demo_output', exist_ok=True)
        with open('demo_output/demo_results.json', 'w') as f:
            json.dump(demo_results, f, indent=2, default=str)
        
        print("📁 Demo results saved to 'demo_output/demo_results.json'")
        
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("Please check the installation and try again.")