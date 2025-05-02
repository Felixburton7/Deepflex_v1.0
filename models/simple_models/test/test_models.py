# test/test_models.py
"""
Test model implementations for the ensembleflex pipeline.

Tests Random Forest and Neural Network models using aggregated test data.
Verifies initialization, training, prediction, saving/loading, uncertainty,
and feature importance (including temperature).
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
import subprocess # To run the aggregation script
from pathlib import Path

# Add project root to path to import ensembleflex
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Check if imports work before proceeding
try:
    from ensembleflex.models.random_forest import RandomForestModel
    from ensembleflex.models.neural_network import NeuralNetworkModel
    from ensembleflex.config import load_config
    from ensembleflex.data.processor import load_and_process_data, prepare_data_for_model
    from test.generate_test_data import generate_data as generate_source_data # Import generator
except ImportError as e:
    print(f"Import Error during test setup: {e}")
    print("Ensure ensembleflex is installed correctly (`pip install -e .`) and tests are run from the project root.")
    sys.exit(1)


class TestModels(unittest.TestCase):
    """Test the ensembleflex model implementations with aggregated data."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests in this class."""
        cls.temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_dir = cls.temp_dir_obj.name
        cls.test_data_dir = Path(cls.temp_dir) / 'test_data'
        cls.aggregated_data_path = cls.test_data_dir / 'aggregated_test_data.csv'
        cls.scripts_dir = project_root / 'scripts'
        cls.aggregate_script_path = cls.scripts_dir / 'aggregate_data.py'

        print(f"\nSetting up TestModels in temporary directory: {cls.temp_dir}")

        # 1. Generate the source temperature_*.csv files if they don't exist
        #    (We'll generate them into the temp dir for isolation)
        print("Generating source test data files...")
        # Temporarily change CWD for generate_data which uses relative paths
        original_cwd = Path.cwd()
        os.chdir(cls.temp_dir)
        try:
             generate_source_data() # This creates test/test_data inside temp_dir
        finally:
             os.chdir(original_cwd) # Change back

        # Check if source files were created
        temp_files_exist = list((cls.test_data_dir).glob("temperature_*.csv"))
        if not temp_files_exist:
             raise FileNotFoundError(f"Failed to generate source test data in {cls.test_data_dir}")
        print(f"Found {len(temp_files_exist)} source data files.")

        # 2. Run the aggregation script
        print("Running data aggregation script...")
        if not cls.aggregate_script_path.exists():
             raise FileNotFoundError(f"Aggregation script not found at {cls.aggregate_script_path}")

        cmd = [
            sys.executable, # Use the same python interpreter running the tests
            str(cls.aggregate_script_path),
            "--input-dir", str(cls.test_data_dir),
            "--output-file", str(cls.aggregated_data_path),
            "--exclude-average" # Exclude average for standard tests
        ]
        try:
            # Run with check=True to raise an error if the script fails
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Aggregation script output:\n", result.stdout)
            if result.stderr:
                 print("Aggregation script errors:\n", result.stderr)

            if not cls.aggregated_data_path.exists():
                 raise FileNotFoundError(f"Aggregation script ran but output file not found: {cls.aggregated_data_path}")
            print(f"Aggregated test data created: {cls.aggregated_data_path}")
        except subprocess.CalledProcessError as e:
             print(f"Aggregation script failed with exit code {e.returncode}")
             print("stdout:\n", e.stdout)
             print("stderr:\n", e.stderr)
             raise RuntimeError("Failed to create aggregated test data.") from e
        except FileNotFoundError as e:
             print(f"Error running aggregation script (python executable not found?): {e}")
             raise

        # 3. Load configuration and aggregated data
        print("Loading config and aggregated data for tests...")
        try:
            # Load default config, assuming default_config.yaml is at project root
            cls.config = load_config()
            # Override paths and data file to use temp aggregated data
            cls.config['paths']['data_dir'] = str(cls.test_data_dir)
            cls.config['dataset']['file_pattern'] = cls.aggregated_data_path.name # Just the filename
            # Ensure temperature feature is USED for most tests
            cls.config['dataset']['features']['use_features']['temperature'] = True
            # Use minimal settings for speed
            cls.config['models']['random_forest']['n_estimators'] = 5
            cls.config['models']['neural_network']['training']['epochs'] = 2
            cls.config['models']['neural_network']['training']['batch_size'] = 16

            # Load and process the aggregated data ONCE
            cls.df = load_and_process_data(config=cls.config) # Uses config path now
            cls.assertIsNotNone(cls.df, "Failed to load aggregated data in setUpClass")
            cls.assertFalse(cls.df.empty, "Loaded aggregated data is empty in setUpClass")
            cls.assertIn('temperature', cls.df.columns, "'temperature' column missing in loaded data")
            cls.assertIn('rmsf', cls.df.columns, "'rmsf' column missing in loaded data")

            # Prepare data for models ONCE (including temperature feature)
            cls.X, cls.y, cls.feature_names = prepare_data_for_model(cls.df, cls.config)
            cls.assertIn('temperature', cls.feature_names, "'temperature' feature missing from prepared features")

            # Split into small train/test sets
            cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
                cls.X, cls.y, test_size=0.2, random_state=42
            )
            print("Test data preparation complete.")

        except Exception as e:
             print(f"Error during setUpClass data loading/processing: {e}")
             cls.tearDownClass() # Attempt cleanup
             raise # Re-raise the exception

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests in this class."""
        print(f"\nTearing down TestModels, removing temporary directory: {cls.temp_dir}")
        cls.temp_dir_obj.cleanup()

    # --- Individual Test Cases ---

    def test_random_forest_initialization(self):
        """Test Random Forest model initialization."""
        model = RandomForestModel()
        self.assertIsNotNone(model)
        model = RandomForestModel(n_estimators=20, max_depth=5, min_samples_split=3)
        self.assertEqual(model.n_estimators, 20)
        self.assertEqual(model.max_depth, 5)

    def test_neural_network_initialization(self):
        """Test Neural Network model initialization."""
        model = NeuralNetworkModel()
        self.assertIsNotNone(model)
        model = NeuralNetworkModel(architecture={"hidden_layers": [32, 16]}, training={"epochs": 5})
        self.assertEqual(model.architecture["hidden_layers"], [32, 16])
        self.assertEqual(model.training["epochs"], 5)

    def test_random_forest_train_predict(self):
        """Test Random Forest training and prediction with temperature feature."""
        model = RandomForestModel(n_estimators=5, random_state=42) # Use minimal settings
        model.fit(self.X_train, self.y_train, feature_names=self.feature_names) # Pass feature names
        self.assertIsNotNone(model.model)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_neural_network_train_predict(self):
        """Test Neural Network training and prediction with temperature feature."""
        model = NeuralNetworkModel(
            architecture={"hidden_layers": [16, 8]},
            training={"epochs": 2, "batch_size": 16}, # Minimal epochs
            random_state=42
        )
        model.fit(self.X_train, self.y_train, feature_names=self.feature_names) # Pass feature names
        self.assertIsNotNone(model.model)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        # Check history exists (basic check)
        self.assertIsNotNone(model.get_training_history())


    def test_random_forest_feature_importance(self):
        """Test RF feature importance, including temperature."""
        model = RandomForestModel(n_estimators=5, random_state=42)
        model.fit(self.X_train, self.y_train, feature_names=self.feature_names)
        # Use permutation importance on test set for better validation
        importance_values = model.get_feature_importance(self.X_test, self.y_test) # Uses permutation by default if data provided
        self.assertIsNotNone(importance_values, "get_feature_importance returned None")
        self.assertIsInstance(importance_values, np.ndarray, "Importance should be numpy array from permutation")
        self.assertEqual(len(importance_values), len(self.feature_names), "Importance array length mismatch")
        importance_dict = dict(zip(self.feature_names, importance_values))
        self.assertIn('temperature', importance_dict, "Temperature feature missing from importance results")
        print(f"\nRF Temp Importance: {importance_dict.get('temperature', 'N/A'):.4f}")


    def test_neural_network_feature_importance(self):
        """Test NN feature importance, including temperature."""
        model = NeuralNetworkModel(
            architecture={"hidden_layers": [16, 8]},
            training={"epochs": 2},
            random_state=42
        )
        model.fit(self.X_train, self.y_train, feature_names=self.feature_names)
        # Use permutation importance on test set
        importance_values = model.get_feature_importance(self.X_test, self.y_test) # Uses permutation
        self.assertIsNotNone(importance_values, "get_feature_importance returned None")
        self.assertIsInstance(importance_values, np.ndarray, "Importance should be numpy array from permutation")
        self.assertEqual(len(importance_values), len(self.feature_names), "Importance array length mismatch")
        importance_dict = dict(zip(self.feature_names, importance_values))
        self.assertIn('temperature', importance_dict, "Temperature feature missing from importance results")
        print(f"\nNN Temp Importance: {importance_dict.get('temperature', 'N/A'):.4f}")


    def test_random_forest_save_load(self):
        """Test Random Forest model saving and loading."""
        model = RandomForestModel(n_estimators=5, random_state=42)
        model.fit(self.X_train, self.y_train, feature_names=self.feature_names)
        original_preds = model.predict(self.X_test)
        model_path = Path(self.temp_dir) / "test_rf_model.pkl"
        model.save(str(model_path))
        self.assertTrue(model_path.exists())
        loaded_model = RandomForestModel.load(str(model_path))
        self.assertIsNotNone(loaded_model.model)
        # Verify feature names were saved/loaded
        self.assertListEqual(model.feature_names_, loaded_model.feature_names_)
        loaded_preds = loaded_model.predict(self.X_test)
        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)

    def test_neural_network_save_load(self):
        """Test Neural Network model saving and loading."""
        model = NeuralNetworkModel(
            architecture={"hidden_layers": [16, 8]},
            training={"epochs": 2}, random_state=42
        )
        model.fit(self.X_train, self.y_train, feature_names=self.feature_names)
        original_preds = model.predict(self.X_test)
        model_path = Path(self.temp_dir) / "test_nn_model.pt" # Use .pt extension convention
        model.save(str(model_path))
        self.assertTrue(model_path.exists())
        loaded_model = NeuralNetworkModel.load(str(model_path))
        self.assertIsNotNone(loaded_model.model)
        # Verify feature names and scaler were loaded
        self.assertListEqual(model.feature_names_, loaded_model.feature_names_)
        self.assertIsNotNone(loaded_model.scaler)
        loaded_preds = loaded_model.predict(self.X_test)
        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5, atol=1e-5) # NN might have slightly larger diffs

    def test_random_forest_uncertainty(self):
        """Test Random Forest uncertainty estimation."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions, uncertainties = model.predict_with_std(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(len(uncertainties), len(self.X_test))
        self.assertTrue(np.all(uncertainties >= 0))

    def test_neural_network_uncertainty(self):
        """Test Neural Network uncertainty estimation (MC Dropout)."""
        model = NeuralNetworkModel(
            architecture={"hidden_layers": [16, 8], "dropout": 0.2}, # Need dropout > 0
            training={"epochs": 2}, random_state=42
        )
        model.fit(self.X_train, self.y_train)
        predictions, uncertainties = model.predict_with_std(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(len(uncertainties), len(self.X_test))
        self.assertTrue(np.all(uncertainties >= 0))
        # Check if std is non-zero (dropout should introduce variance)
        self.assertTrue(np.any(uncertainties > 1e-9))


    # HPO test can remain similar, just ensure it runs with aggregated data structure
    def test_model_hyperparameter_optimization_runs(self):
        """Test that model hyperparameter optimization runs without crashing."""
        if len(self.X_train) < 30: # Need enough data for CV splits
            self.skipTest("Not enough data for HPO test")

        # RF HPO
        rf_model = RandomForestModel(n_estimators=5, random_state=42)
        rf_param_grid = {'n_estimators': [5, 8], 'max_depth': [3, 5]}
        try:
             # Assume RF uses RandomizedSearchCV internally via fit if params are passed
             # Modify config temporarily or pass specific params
             # For simplicity, test the dedicated method if available
             rf_model.hyperparameter_optimize(
                 self.X_train[:30], self.y_train[:30], rf_param_grid, # Use tiny subset
                 method="random", n_trials=1, cv=2
             )
        except NotImplementedError:
             pass # Some models might not implement this standalone
        except Exception as e:
             self.fail(f"RF hyperparameter optimization failed: {e}")

        # NN HPO
        nn_model = NeuralNetworkModel(architecture={"hidden_layers": [8]}, training={"epochs": 1}, random_state=42)
        nn_param_grid = {'hidden_layers': [[8], [4]], 'learning_rate': [0.01]}
        try:
             nn_model.hyperparameter_optimize(
                 self.X_train[:30], self.y_train[:30], nn_param_grid, # Use tiny subset
                 method="random", n_trials=1, cv=2
             )
        except NotImplementedError:
             pass # Expected if not implemented
        except Exception as e:
             # Allow ImportErrors for optuna as it's optional
             if not isinstance(e, ImportError):
                  self.fail(f"NN hyperparameter optimization failed: {e}")


if __name__ == '__main__':
    unittest.main()