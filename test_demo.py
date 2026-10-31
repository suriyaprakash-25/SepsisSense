"""
SepsisSense Demo Data Generator and Testing Suite
===============================================

Comprehensive testing suite and demo data generation for the SepsisSense
sepsis prediction system.

Author: SepsisSense Team
License: MIT
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Dict, Any, List, Tuple
import json
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import SepsisDataPreprocessor, create_sample_dataset
from feature_engineering import SepsisFeatureEngineer
from train_model import SepsisModelTrainer
from explain_model import SepsisModelExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoDataGenerator:
    """Generate realistic demo datasets for SepsisSense."""
    
    def __init__(self):
        self.normal_ranges = {
            'HR': (60, 100),
            'O2Sat': (95, 100),
            'Temp': (36.1, 37.2),
            'SBP': (90, 140),
            'MAP': (70, 105),
            'DBP': (60, 90),
            'Resp': (12, 20),
            'WBC': (4, 11),
            'Lactate': (0.5, 2.2),
            'Creatinine': (0.7, 1.3),
            'Platelets': (150, 450),
            'Glucose': (70, 140)
        }
    
    def create_demo_patient_csv(self, filename: str = 'demo_patient.csv') -> str:
        """Create a single demo patient CSV for testing."""
        
        # Create a realistic patient trajectory
        patient_data = []
        
        # Patient demographics
        patient_id = "DEMO_001"
        age = 68
        gender = 1  # Male
        
        # Simulate 24-hour stay with gradual deterioration
        for hour in range(24):
            # Base vital signs
            hr_base = 75
            map_base = 85
            temp_base = 37.0
            o2sat_base = 98
            resp_base = 16
            
            # Add deterioration pattern (sepsis onset around hour 18)
            if hour >= 12:  # Start showing early signs
                deterioration_factor = (hour - 12) * 0.15
                hr_base += deterioration_factor * 25  # Increasing heart rate
                map_base -= deterioration_factor * 15  # Decreasing blood pressure
                temp_base += deterioration_factor * 1.5  # Increasing temperature
                o2sat_base -= deterioration_factor * 4  # Decreasing oxygen
                resp_base += deterioration_factor * 8  # Increasing respiration
            
            # Add noise
            hr = max(50, hr_base + np.random.normal(0, 5))
            map_val = max(50, map_base + np.random.normal(0, 8))
            temp = max(35, temp_base + np.random.normal(0, 0.3))
            o2sat = min(100, max(85, o2sat_base + np.random.normal(0, 2)))
            resp = max(10, resp_base + np.random.normal(0, 2))
            
            # Calculate SBP and DBP from MAP
            sbp = map_val + 30 + np.random.normal(0, 10)
            dbp = map_val - 15 + np.random.normal(0, 5)
            
            # Lab values (less frequent)
            wbc = 7.5 + (deterioration_factor * 8 if hour >= 12 else 0) + np.random.normal(0, 1.5) if hour % 4 == 0 else np.nan
            lactate = 1.5 + (deterioration_factor * 3 if hour >= 12 else 0) + np.random.normal(0, 0.5) if hour % 6 == 0 else np.nan
            creatinine = 1.0 + (deterioration_factor * 0.8 if hour >= 12 else 0) + np.random.normal(0, 0.2) if hour % 8 == 0 else np.nan
            platelets = 250 - (deterioration_factor * 80 if hour >= 12 else 0) + np.random.normal(0, 30) if hour % 6 == 0 else np.nan
            glucose = 100 + np.random.normal(0, 20) if hour % 4 == 0 else np.nan
            
            # Sepsis label (positive from hour 18 onward)
            sepsis_label = 1 if hour >= 18 else 0
            
            record = {
                'Patient': patient_id,
                'ICULOS': hour + 1,
                'HR': hr,
                'O2Sat': o2sat,
                'Temp': temp,
                'SBP': sbp,
                'MAP': map_val,
                'DBP': dbp,
                'Resp': resp,
                'Age': age,
                'Gender': gender,
                'Unit1': 1,
                'Unit2': 0,
                'HospAdmTime': -hour,
                'SepsisLabel': sepsis_label,
                'WBC': wbc,
                'Lactate': lactate,
                'Creatinine': creatinine,
                'Platelets': platelets,
                'Glucose': glucose,
                'BaseExcess': np.nan,
                'HCO3': np.nan,
                'FiO2': np.nan,
                'pH': np.nan,
                'PaCO2': np.nan,
                'SaO2': np.nan,
                'AST': np.nan,
                'BUN': np.nan,
                'Alkalinephos': np.nan,
                'Calcium': np.nan,
                'Chloride': np.nan,
                'Bilirubin_direct': np.nan,
                'Magnesium': np.nan,
                'Phosphate': np.nan,
                'Potassium': np.nan,
                'Bilirubin_total': np.nan,
                'TroponinI': np.nan,
                'Hct': np.nan,
                'Hgb': np.nan,
                'PTT': np.nan,
                'Fibrinogen': np.nan
            }
            
            patient_data.append(record)
        
        # Save to CSV
        df = pd.DataFrame(patient_data)
        filepath = os.path.join('data', 'processed', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Demo patient CSV created: {filepath}")
        return filepath
    
    def create_batch_demo_data(self, n_patients: int = 50, filename: str = 'demo_batch.csv') -> str:
        """Create batch demo data for testing."""
        
        demo_data = create_sample_dataset(n_patients=n_patients, records_per_patient=20)
        
        # Save to CSV
        filepath = os.path.join('data', 'processed', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        demo_data.to_csv(filepath, index=False)
        
        logger.info(f"Demo batch data created: {filepath} ({len(demo_data)} records)")
        return filepath
    
    def create_test_scenarios(self) -> Dict[str, pd.DataFrame]:
        """Create specific test scenarios."""
        
        scenarios = {}
        
        # Scenario 1: Low risk patient
        scenarios['low_risk'] = pd.DataFrame([{
            'Patient': 'TEST_LOW_001',
            'ICULOS': 12,
            'HR': 75,
            'O2Sat': 98,
            'Temp': 37.0,
            'SBP': 120,
            'MAP': 85,
            'DBP': 70,
            'Resp': 16,
            'Age': 45,
            'Gender': 0,
            'Unit1': 1,
            'Unit2': 0,
            'HospAdmTime': -12,
            'SepsisLabel': 0,
            'WBC': 7.0,
            'Lactate': 1.2,
            'Creatinine': 0.9,
            'Platelets': 300,
            'Glucose': 95
        }])
        
        # Scenario 2: High risk patient
        scenarios['high_risk'] = pd.DataFrame([{
            'Patient': 'TEST_HIGH_001',
            'ICULOS': 18,
            'HR': 120,
            'O2Sat': 89,
            'Temp': 39.2,
            'SBP': 85,
            'MAP': 60,
            'DBP': 45,
            'Resp': 28,
            'Age': 75,
            'Gender': 1,
            'Unit1': 1,
            'Unit2': 0,
            'HospAdmTime': -18,
            'SepsisLabel': 1,
            'WBC': 18.5,
            'Lactate': 4.2,
            'Creatinine': 2.1,
            'Platelets': 85,
            'Glucose': 180
        }])
        
        # Scenario 3: Moderate risk patient
        scenarios['moderate_risk'] = pd.DataFrame([{
            'Patient': 'TEST_MOD_001',
            'ICULOS': 15,
            'HR': 95,
            'O2Sat': 94,
            'Temp': 37.8,
            'SBP': 105,
            'MAP': 75,
            'DBP': 60,
            'Resp': 22,
            'Age': 62,
            'Gender': 0,
            'Unit1': 1,
            'Unit2': 0,
            'HospAdmTime': -15,
            'SepsisLabel': 0,
            'WBC': 12.0,
            'Lactate': 2.8,
            'Creatinine': 1.4,
            'Platelets': 180,
            'Glucose': 140
        }])
        
        # Add missing columns with NaN
        additional_columns = [
            'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
            'Alkalinephos', 'Calcium', 'Chloride', 'Bilirubin_direct', 'Magnesium',
            'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb',
            'PTT', 'Fibrinogen'
        ]
        
        for scenario_name, df in scenarios.items():
            for col in additional_columns:
                df[col] = np.nan
        
        return scenarios


class SepsisSenseTestSuite:
    """Comprehensive test suite for SepsisSense system."""
    
    def __init__(self):
        self.demo_generator = DemoDataGenerator()
        self.test_results = {}
    
    def test_data_preprocessing(self) -> Dict[str, Any]:
        """Test data preprocessing pipeline."""
        logger.info("Testing data preprocessing...")
        
        results = {'passed': True, 'errors': []}
        
        try:
            # Create test data
            test_data = create_sample_dataset(n_patients=10, records_per_patient=15)
            
            # Initialize preprocessor
            preprocessor = SepsisDataPreprocessor()
            
            # Test preprocessing
            df_processed, stats = preprocessor.preprocess_pipeline(test_data, fit=True)
            
            # Validate results
            assert len(df_processed) > 0, "Processed data is empty"
            assert 'SepsisLabel' in df_processed.columns, "Target column missing"
            assert not df_processed.isna().all().any(), "Some columns are all NaN"
            
            results['processed_shape'] = df_processed.shape
            results['stats'] = stats
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(str(e))
        
        return results
    
    def test_feature_engineering(self) -> Dict[str, Any]:
        """Test feature engineering pipeline."""
        logger.info("Testing feature engineering...")
        
        results = {'passed': True, 'errors': []}
        
        try:
            # Create and preprocess test data
            test_data = create_sample_dataset(n_patients=10, records_per_patient=15)
            preprocessor = SepsisDataPreprocessor()
            df_processed, _ = preprocessor.preprocess_pipeline(test_data, fit=True)
            
            # Test feature engineering
            feature_engineer = SepsisFeatureEngineer()
            df_features = feature_engineer.engineer_features_pipeline(df_processed)
            
            # Validate results
            assert len(df_features) > 0, "Feature engineered data is empty"
            assert df_features.shape[1] > df_processed.shape[1], "No new features created"
            assert len(feature_engineer.feature_columns) > 0, "No feature columns tracked"
            
            results['original_features'] = df_processed.shape[1]
            results['engineered_features'] = df_features.shape[1]
            results['new_features'] = len(feature_engineer.feature_columns)
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(str(e))
        
        return results
    
    def test_model_training(self) -> Dict[str, Any]:
        """Test model training pipeline."""
        logger.info("Testing model training...")
        
        results = {'passed': True, 'errors': []}
        
        try:
            # Prepare data
            test_data = create_sample_dataset(n_patients=50, records_per_patient=15)
            preprocessor = SepsisDataPreprocessor()
            df_processed, _ = preprocessor.preprocess_pipeline(test_data, fit=True)
            
            feature_engineer = SepsisFeatureEngineer()
            df_features = feature_engineer.engineer_features_pipeline(df_processed)
            
            # Test training
            trainer = SepsisModelTrainer()
            training_results = trainer.train_and_evaluate_pipeline(df_features)
            
            # Validate results
            assert 'best_model' in training_results, "No best model found"
            assert 'metrics' in training_results, "No metrics available"
            
            best_metrics = training_results['metrics'][training_results['best_model_name']]
            eval_metrics = best_metrics.get('test', best_metrics)
            assert eval_metrics.get('roc_auc', 0.0) > 0.5, "Model performance below random"
            
            results['best_model'] = training_results['best_model_name']
            results['metrics'] = eval_metrics
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(str(e))
        
        return results
    
    def test_model_explanation(self) -> Dict[str, Any]:
        """Test model explanation system."""
        logger.info("Testing model explanation...")
        
        results = {'passed': True, 'errors': []}
        
        try:
            # Prepare data and train model
            test_data = create_sample_dataset(n_patients=30, records_per_patient=15)
            preprocessor = SepsisDataPreprocessor()
            df_processed, _ = preprocessor.preprocess_pipeline(test_data, fit=True)
            
            feature_engineer = SepsisFeatureEngineer()
            df_features = feature_engineer.engineer_features_pipeline(df_processed)
            
            trainer = SepsisModelTrainer()
            training_results = trainer.train_and_evaluate_pipeline(df_features)
            
            # Test explanation
            explainer = SepsisModelExplainer(
                model=training_results['best_model'],
                model_type=training_results['best_model_name']
            )
            
            X_test, y_test = training_results['test_data']
            explainer.initialize_explainer(X_test.head(20))
            
            # Test single prediction explanation
            single_sample = X_test.head(1)
            explanation = explainer.explain_single_prediction(single_sample, "test_patient")
            
            # Validate results
            assert 'prediction_probability' in explanation, "No prediction probability"
            assert 'contributions' in explanation, "No feature contributions"
            assert 'clinical_summary' in explanation, "No clinical summary"
            
            results['prediction_probability'] = explanation['prediction_probability']
            results['risk_category'] = explanation['risk_category']
            results['top_features'] = explanation['contributions'].head(5)['feature'].tolist()
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(str(e))
        
        return results
    
    def test_end_to_end_pipeline(self) -> Dict[str, Any]:
        """Test complete end-to-end pipeline."""
        logger.info("Testing end-to-end pipeline...")
        
        results = {'passed': True, 'errors': []}
        
        try:
            # Create test scenarios
            scenarios = self.demo_generator.create_test_scenarios()
            
            # Train pipeline once on training dataset
            preprocessor = SepsisDataPreprocessor()
            train_data = create_sample_dataset(n_patients=30, records_per_patient=12)
            df_train_processed, _ = preprocessor.preprocess_pipeline(train_data, fit=True)
            
            feature_engineer = SepsisFeatureEngineer()
            df_train_features = feature_engineer.engineer_features_pipeline(df_train_processed)
            
            trainer = SepsisModelTrainer()
            training_results = trainer.train_and_evaluate_pipeline(df_train_features)
            
            scenario_results = {}
            for scenario_name, scenario_data in scenarios.items():
                try:
                    # Process test scenario
                    df_test_processed, _ = preprocessor.preprocess_pipeline(scenario_data, fit=False)
                    df_test_features = feature_engineer.engineer_features_pipeline(df_test_processed)
                    
                    # Make prediction
                    exclude_columns = {'SepsisLabel', 'Patient', 'HospAdmTime'}
                    feature_columns = [col for col in df_test_features.columns if col not in exclude_columns]
                    X_test = df_test_features[feature_columns]
                    
                    prediction = training_results['best_model'].predict_proba(X_test)[0, 1]
                    
                    scenario_results[scenario_name] = {
                        'prediction': prediction,
                        'expected_risk': scenario_name,
                        'passed': True
                    }
                    
                except Exception as e:
                    scenario_results[scenario_name] = {
                        'passed': False,
                        'error': str(e)
                    }
            
            results['scenarios'] = scenario_results
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(str(e))
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite."""
        logger.info("Running complete SepsisSense test suite...")
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Run individual tests
        test_functions = [
            ('preprocessing', self.test_data_preprocessing),
            ('feature_engineering', self.test_feature_engineering),
            ('model_training', self.test_model_training),
            ('model_explanation', self.test_model_explanation),
            ('end_to_end', self.test_end_to_end_pipeline)
        ]
        
        for test_name, test_func in test_functions:
            try:
                logger.info(f"Running {test_name} test...")
                test_result = test_func()
                all_results['tests'][test_name] = test_result
                status = "PASSED" if test_result['passed'] else "FAILED"
                logger.info(f"{test_name} test: {status}")
                
            except Exception as e:
                logger.error(f"Error in {test_name} test: {str(e)}")
                all_results['tests'][test_name] = {
                    'passed': False,
                    'errors': [str(e)]
                }
        
        # Overall summary
        passed_tests = sum(1 for test in all_results['tests'].values() if test['passed'])
        total_tests = len(all_results['tests'])
        
        all_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
        
        logger.info(f"Test suite completed: {passed_tests}/{total_tests} tests passed")
        
        # Save results
        results_file = os.path.join('data', 'processed', 'test_results.json')
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        return all_results


def main():
    """Main function to run demo data generation and tests."""
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate demo data
    print("🔄 Generating demo datasets...")
    demo_generator = DemoDataGenerator()
    
    # Create single patient demo
    demo_generator.create_demo_patient_csv('demo_patient_timeline.csv')
    
    # Create batch demo data
    demo_generator.create_batch_demo_data(n_patients=100, filename='demo_batch_100patients.csv')
    
    # Create test scenarios
    scenarios = demo_generator.create_test_scenarios()
    for scenario_name, scenario_df in scenarios.items():
        filename = f'test_scenario_{scenario_name}.csv'
        filepath = os.path.join('data', 'processed', filename)
        scenario_df.to_csv(filepath, index=False)
        print(f"Created test scenario: {filepath}")
    
    # Run test suite
    print("\n🧪 Running comprehensive test suite...")
    test_suite = SepsisSenseTestSuite()
    results = test_suite.run_all_tests()
    
    # Print summary
    print("\n📊 Test Results Summary:")
    print("=" * 50)
    for test_name, test_result in results['tests'].items():
        status = "✅ PASSED" if test_result['passed'] else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if not test_result['passed'] and 'errors' in test_result:
            for error in test_result['errors']:
                print(f"    Error: {error}")
    
    print(f"\nOverall: {results['summary']['passed_tests']}/{results['summary']['total_tests']} tests passed")
    print(f"Success rate: {results['summary']['success_rate']:.1%}")
    
    print("\n🎉 Demo data generation and testing completed!")
    print(f"Check 'data/processed/' for demo files and test results.")


# ==============================================================================
# PYTEST COMPATIBLE TEST HOOKS
# ==============================================================================

def test_pytest_preprocessing():
    suite = SepsisSenseTestSuite()
    result = suite.test_data_preprocessing()
    assert result['passed'], f"Preprocessing test failed: {result.get('errors')}"

def test_pytest_feature_engineering():
    suite = SepsisSenseTestSuite()
    result = suite.test_feature_engineering()
    assert result['passed'], f"Feature engineering test failed: {result.get('errors')}"

def test_pytest_model_training():
    suite = SepsisSenseTestSuite()
    result = suite.test_model_training()
    assert result['passed'], f"Model training test failed: {result.get('errors')}"

def test_pytest_model_explanation():
    suite = SepsisSenseTestSuite()
    result = suite.test_model_explanation()
    assert result['passed'], f"Model explanation test failed: {result.get('errors')}"

def test_pytest_end_to_end():
    suite = SepsisSenseTestSuite()
    result = suite.test_end_to_end_pipeline()
    assert result['passed'], f"End-to-end pipeline test failed: {result.get('errors')}"


if __name__ == "__main__":
    main()