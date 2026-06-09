#!/usr/bin/env python3
"""
Debug script to identify the model loading issue.
"""

import sys
import traceback
sys.path.append('.')

print("Testing model loading components...")

try:
    print("1. Testing imports...")
    from src.preprocess import SepsisDataPreprocessor
    from src.feature_engineering import SepsisFeatureEngineer
    from src.train_model import SepsisModelTrainer
    from src.explain_model import SepsisModelExplainer
    print("✓ All imports successful")
    
    print("\n2. Testing data loading...")
    from src.preprocess import create_sample_dataset
    
    # Use the actual sample data generation function with more patients
    test_data = create_sample_dataset(n_patients=20, records_per_patient=12)
    print("✓ Test data created")
    
    print("\n3. Testing preprocessor...")
    preprocessor = SepsisDataPreprocessor()
    df_processed, _ = preprocessor.preprocess_pipeline(test_data, fit=True)
    print("✓ Preprocessing successful")
    
    print("\n4. Testing feature engineering...")
    feature_engineer = SepsisFeatureEngineer()
    df_features = feature_engineer.engineer_features_pipeline(df_processed)
    print("✓ Feature engineering successful")
    
    print("\n5. Testing model training...")
    trainer = SepsisModelTrainer()
    results = trainer.train_and_evaluate_pipeline(df_features)
    print("✓ Model training successful")
    
    print("\n6. Testing explainer initialization...")
    explainer = SepsisModelExplainer(
        model=results['best_model'],
        model_type=results['best_model_name']
    )
    print("✓ Explainer created")
    
    X_test, y_test = results['test_data']
    print(f"Test data shape: {X_test.shape}")
    explainer.initialize_explainer(X_test.head(10))  # Use smaller sample
    print("✓ Explainer initialization successful")
    
    print("\n7. Testing risk categorization...")
    test_probs = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for prob in test_probs:
        category = explainer._get_risk_category(prob)
        print(f"  Probability {prob:.1f} -> {category}")
    print("✓ Risk categorization successful")
    
    print("\n🎉 All tests passed! The model loading should work.")
    
except Exception as e:
    print(f"\n❌ Error occurred: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()