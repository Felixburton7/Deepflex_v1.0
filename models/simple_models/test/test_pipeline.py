# test/test_pipeline.py
"""
Test the ensembleflex pipeline end-to-end using aggregated data.

Tests loading aggregated data, training the single model, evaluation,
and prediction (requiring temperature input).
"""

import os
import sys
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
import subprocess # To run the aggregation script
from pathlib import Path

# Add project root to path to import ensembleflex
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Check if imports work before proceeding
try:
    from ensembleflex.config import load_config, get_enabled_models
    from ensembleflex.pipeline import Pipeline
    from ensembleflex.data.processor import load_and_process_data, prepare_data_for_model # For direct check
    from test.generate_test_data import generate_data as generate_source_data # Import generator
except ImportError as e:
    print(f"Import Error during test setup: {e}")
    print("Ensure ensembleflex is installed correctly (`pip install -e .`) and tests are run from the project root.")
    sys.exit(1)

class TestPipeline(unittest.TestCase):
    """Test the ensembleflex pipeline with aggregated data."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once."""
        cls.temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_dir = cls.temp_dir_obj.name
        cls.output_dir = Path(cls.temp_dir) / 'output' / 'ensembleflex' # Match default structure
        cls.models_dir = Path(cls.temp_dir) / 'models' / 'ensembleflex'
        cls.test_data_dir = Path(cls.temp_dir) / 'test_data' # Source data goes here
        cls.aggregated_data_path = cls.test_data_dir / 'aggregated_test_data.csv'
        cls.scripts_dir = project_root / 'scripts'
        cls.aggregate_script_path = cls.scripts_dir / 'aggregate_data.py'

        print(f"\nSetting up TestPipeline in temporary directory: {cls.temp_dir}")

        # 1. Generate source data
        print("Generating source test data files...")
        original_cwd = Path.cwd()
        os.chdir(cls.temp_dir)
        try: generate_source_data();
        finally: os.chdir(original_cwd)
        if not list(cls.test_data_dir.glob("temperature_*.csv")):
             raise FileNotFoundError(f"Failed to generate source test data in {cls.test_data_dir}")

        # 2. Run aggregation script
        print("Running data aggregation script...")
        if not cls.aggregate_script_path.exists():
            raise FileNotFoundError(f"Aggregation script not found: {cls.aggregate_script_path}")
        cmd = [sys.executable, str(cls.aggregate_script_path),
               "--input-dir", str(cls.test_data_dir),
               "--output-file", str(cls.aggregated_data_path),
               "--exclude-average"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if not cls.aggregated_data_path.exists():
                 raise FileNotFoundError(f"Aggregated data file not created: {cls.aggregated_data_path}")
            print(f"Aggregated test data created: {cls.aggregated_data_path}")
        except subprocess.CalledProcessError as e:
             print("Aggregation script stderr:\n", e.stderr)
             raise RuntimeError("Failed to create aggregated test data.") from e

        # 3. Load and configure
        print("Loading config and preparing pipeline...")
        cls.config = load_config() # Load default
        # Override paths and data file
        cls.config['paths']['output_dir'] = str(cls.output_dir.parent) # Pipeline expects parent of ensembleflex subdir? Check Pipeline init. Let's assume it expects the base output dir.
        cls.config['paths']['models_dir'] = str(cls.models_dir.parent)
        cls.config['paths']['data_dir'] = str(cls.test_data_dir.parent) # Point to parent of test_data
        cls.config['dataset']['file_pattern'] = str(cls.aggregated_data_path.relative_to(cls.test_data_dir.parent)) # Relative path from data_dir

        # Use minimal settings for speed
        cls.config['models']['random_forest']['enabled'] = True # Enable only RF
        cls.config['models']['random_forest']['n_estimators'] = 5
        cls.config['models']['neural_network']['enabled'] = False
        cls.config['dataset']['features']['use_features']['temperature'] = True # Ensure temp feature is on

        # Create directories expected by pipeline
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.models_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate pipeline for use in tests (optional, tests can create their own)
        # cls.pipeline = Pipeline(cls.config)
        print("Test setup complete.")


    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        print(f"\nTearing down TestPipeline, removing temporary directory: {cls.temp_dir}")
        cls.temp_dir_obj.cleanup()

    def test_01_data_loading_aggregated(self):
        """Test loading and processing the aggregated data file."""
        pipeline = Pipeline(self.config)
        df = pipeline.load_data() # Should load aggregated file based on config

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        # Check for unified/new columns
        self.assertIn('rmsf', df.columns)
        self.assertIn('temperature', df.columns)
        self.assertNotIn('rmsf_320', df.columns) # Check old column is gone
        # Check window features were created if enabled
        if self.config['dataset']['features']['window']['enabled']:
            window_cols = [col for col in df.columns if '_offset_' in col]
            self.assertGreater(len(window_cols), 0)
            self.assertTrue(all('temperature_offset' not in col for col in window_cols)) # Ensure temp wasn't windowed

    # test_temperature_templating is removed

    def test_02_pipeline_train_single_model(self):
        """Test training the single unified model (RF)."""
        pipeline = Pipeline(self.config)
        models = pipeline.train() # Train the enabled model (RF)

        self.assertIn('random_forest', models)
        self.assertEqual(len(models), 1, "Should only train one model")
        self.assertIsNotNone(models['random_forest'].model)

        # Check that the single model file was saved to the non-templated path
        model_path = self.models_dir / 'random_forest.pkl' # Use Path object
        self.assertTrue(model_path.exists(), f"Model file not saved at {model_path}")

    def test_03_pipeline_predict_requires_temperature(self):
        """Test that pipeline.predict requires the temperature argument."""
        pipeline = Pipeline(self.config)
        # Ensure model is trained first (depends on test_02 order or train here)
        if not (self.models_dir / 'random_forest.pkl').exists():
             print("Training model for predict test...")
             pipeline.train()

        # Prepare minimal input data (just features needed)
        # Load a small piece of the aggregated data for input features
        input_df = pd.read_csv(self.aggregated_data_path, nrows=5)

        # Expect TypeError if temperature is missing
        with self.assertRaises(TypeError):
             pipeline.predict(data=input_df)

    def test_04_pipeline_predict_with_temperature(self):
        """Test model prediction using the required temperature argument."""
        pipeline = Pipeline(self.config)
        test_temp = 370.0 # An arbitrary temperature

        # Ensure model is trained
        if not (self.models_dir / 'random_forest.pkl').exists():
             print("Training model for predict test...")
             pipeline.train()

        # Load a small piece for prediction input
        input_df_raw = pd.read_csv(self.aggregated_data_path, nrows=10)
        # Drop target if present, keep features
        input_df_features = input_df_raw.drop(columns=['rmsf'], errors='ignore')

        # Make predictions
        predictions_df = pipeline.predict(data=input_df_features, temperature=test_temp)

        self.assertIsInstance(predictions_df, pd.DataFrame)
        self.assertIn('rmsf_predicted', predictions_df.columns) # Check for the unified prediction column
        self.assertIn('prediction_temperature', predictions_df.columns) # Check for added column
        self.assertEqual(len(predictions_df), len(input_df_features))
        self.assertTrue(np.allclose(predictions_df['prediction_temperature'], test_temp))
        # Check predictions are reasonable (non-negative)
        self.assertTrue(np.all(predictions_df['rmsf_predicted'] >= 0))

        # Predict at a different temperature and check if predictions differ
        predictions_df_high_temp = pipeline.predict(data=input_df_features, temperature=test_temp + 50.0)
        # Simple check: are the prediction arrays different? (Assumes model learned *some* temp dependence)
        self.assertFalse(np.allclose(predictions_df['rmsf_predicted'], predictions_df_high_temp['rmsf_predicted']),
                         "Predictions did not change with temperature input.")


    def test_05_pipeline_evaluate_aggregated(self):
        """Test model evaluation on the aggregated test set."""
        pipeline = Pipeline(self.config)

        # Ensure model is trained
        if not (self.models_dir / 'random_forest.pkl').exists():
             print("Training model for evaluate test...")
             pipeline.train()

        # Evaluate model (uses test split from aggregated data)
        results = pipeline.evaluate() # Evaluate the enabled model (RF)

        self.assertIn('random_forest', results)
        self.assertIn('rmse', results['random_forest'])
        self.assertIn('r2', results['random_forest'])
        # Check metrics are reasonable
        self.assertGreater(results['random_forest']['r2'], -1.0)
        self.assertLess(results['random_forest']['r2'], 1.1)

        # Check output files exist in the non-templated output dir
        eval_metrics_path = self.output_dir / 'evaluation_results.csv'
        all_results_path = self.output_dir / 'all_results.csv'
        self.assertTrue(eval_metrics_path.exists())
        self.assertTrue(all_results_path.exists())

        # Check content of all_results.csv
        all_results_df = pd.read_csv(all_results_path)
        self.assertIn('temperature', all_results_df.columns) # Should contain original temp feature
        self.assertIn('rmsf', all_results_df.columns)       # Should contain actual target
        self.assertIn('random_forest_predicted', all_results_df.columns) # Should contain predictions
        self.assertIn('random_forest_abs_error', all_results_df.columns) # Should contain errors


    def test_06_omniflex_mode_aggregated(self):
        """Test OmniFlex mode with aggregated data."""
        local_config = self.config.copy() # Modify config locally for this test
        local_config['mode']['active'] = 'omniflex'
        local_config['mode']['omniflex']['use_esm'] = True
        local_config['mode']['omniflex']['use_voxel'] = True
        # Ensure features are enabled in the config for the test
        local_config['dataset']['features']['use_features']['esm_rmsf'] = True
        local_config['dataset']['features']['use_features']['voxel_rmsf'] = True

        # Need specific output dirs for this test run? Or reuse? Reuse for simplicity.
        pipeline = Pipeline(local_config)

        # Check data loading includes omniflex features
        df = pipeline.load_data()
        self.assertIn('esm_rmsf', df.columns)
        self.assertIn('voxel_rmsf', df.columns)

        # Train model
        models = pipeline.train() # Train RF in OmniFlex mode
        self.assertIn('random_forest', models)

        # Check feature importances include ESM and voxel (and temperature)
        X_test, y_test, feature_names = prepare_data_for_model(
            pipeline.load_data().iloc[:100], # Use small sample of test data
            local_config
        )
        importance_values = models['random_forest'].get_feature_importance(X_test, y_test)
        self.assertIsNotNone(importance_values)
        importance_dict = dict(zip(feature_names, importance_values))

        self.assertIn('esm_rmsf', importance_dict)
        self.assertIn('voxel_rmsf', importance_dict)
        self.assertIn('temperature', importance_dict) # Should still be there


    def test_07_pipeline_run_full(self):
        """Test running the full pipeline (train, evaluate, analyze)."""
        pipeline = Pipeline(self.config)
        # Run the full pipeline (should train RF, evaluate, analyze)
        results = pipeline.run_pipeline(skip_analysis=False) # Ensure analysis runs

        # Check final evaluation results are returned
        self.assertIn('random_forest', results)
        self.assertIn('rmse', results['random_forest'])

        # Check analysis output files exist
        importance_path = self.output_dir / 'feature_importance' / 'random_forest_feature_importance.csv'
        temp_analysis_path = self.output_dir / 'temperature_analysis' / 'random_forest_error_vs_temp_binned.csv'
        self.assertTrue(importance_path.exists(), f"Feature importance file missing: {importance_path}")
        self.assertTrue(temp_analysis_path.exists(), f"Temperature analysis file missing: {temp_analysis_path}")


if __name__ == '__main__':
     # Run tests sequentially based on naming convention (test_01_, test_02_, etc.)
     # This helps ensure training happens before prediction/evaluation tests.
     # You might want a more robust dependency management using pytest fixtures eventually.
    suite = unittest.TestSuite()
    tests = unittest.defaultTestLoader.getTestCaseNames(TestPipeline)
    for test in sorted(tests):
         suite.addTest(TestPipeline(test))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    # unittest.main() # Standard execution if order doesn't matter