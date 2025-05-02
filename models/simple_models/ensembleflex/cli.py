# /home/s_felix/ensembleflex/ensembleflex/cli.py

"""
Command-line interface for the EnsembleFlex ML pipeline.

Handles training, evaluation, prediction, and analysis for the
single, unified, temperature-aware model.
"""

import os
import sys
import logging
from typing import List, Optional, Tuple, Dict, Any # Removed Union

import click
import pandas as pd
import numpy as np
from ensembleflex.utils.helpers import ensure_dir



# Updated imports for ensembleflex structure
from ensembleflex.config import (
    load_config,
    get_enabled_models,
    # get_model_config, # Potentially needed by specific commands if not using Pipeline methods directly
    # get_available_temperatures, # No longer needed for driving runs
    get_output_dir,  # UPDATED
    get_models_dir,  # UPDATED
    # get_comparison_output_dir # Can use get_output_dir + "/comparison" if needed
)
from ensembleflex.pipeline import Pipeline
from ensembleflex.models import get_available_models # Keep this for listing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_model_list(model_arg: Optional[str]) -> List[str]:
    """
    Parse comma-separated list of models.

    Args:
        model_arg: Comma-separated model names or None

    Returns:
        List of model names
    """
    if not model_arg:
        return []

    return [m.strip() for m in model_arg.split(",")]

@click.group()
@click.version_option(package_name="ensembleflex", version="0.1.0") # Updated package name
def cli():
    """
    ensembleflex: ML pipeline for temperature-aware protein flexibility prediction.

    Trains a single, unified model on aggregated data across temperatures.
    Requires temperature input for predictions.
    """
    pass

# --- Modified `train` Command ---
@cli.command()
@click.option("--model",
              help="Model to train (e.g., random_forest, neural_network). Defaults to enabled models in config.")
@click.option("--config",
              type=click.Path(exists=True, dir_okay=False),
              help="Path to custom YAML config file.")
@click.option("--param",
              multiple=True,
              help="Override config parameter (e.g. models.random_forest.n_estimators=200)")
# --input option might be less relevant if config specifies the aggregated file, but keep for flexibility
@click.option("--input",
              type=click.Path(exists=True, dir_okay=False),
              help="Override input data file (CSV). Defaults to config dataset.file_pattern.")
# REMOVED: --temperature option
@click.option("--mode",
              type=click.Choice(["standard", "omniflex"], case_sensitive=False),
              help="Override operation mode (standard or omniflex).")
def train(
    model, config, param, input, mode
):
    """
    Train the unified flexibility prediction model on aggregated data.
    Uses settings from default_config.yaml, overridden by --config, env vars, and --param.

    Examples:
        ensembleflex train
        ensembleflex train --model random_forest
        ensembleflex train --mode standard
        ensembleflex train --input path/to/my_aggregated_data.csv
        ensembleflex train --param models.neural_network.training.epochs=50
    """
    # Load configuration (No temperature override needed here)
    cfg = load_config(config_path=config, param_overrides=param)

    # Set mode if specified via CLI
    if mode:
        cfg["mode"]["active"] = mode.lower()
        logger.info(f"Mode overridden by CLI: {cfg['mode']['active']}")
        # Re-apply mode logic after potential override
        try:
            if cfg["mode"]["active"] == "omniflex":
                omniflex_cfg = cfg.get("mode", {}).get("omniflex", {})
                use_features = cfg.setdefault("dataset", {}).setdefault("features", {}).setdefault("use_features", {})
                use_features["esm_rmsf"] = omniflex_cfg.get("use_esm", False)
                use_features["voxel_rmsf"] = omniflex_cfg.get("use_voxel", False)
            else: # Standard mode
                 use_features = cfg.setdefault("dataset", {}).setdefault("features", {}).setdefault("use_features", {})
                 use_features["esm_rmsf"] = False
                 use_features["voxel_rmsf"] = False
        except Exception as e:
            logger.error(f"Error applying mode override settings: {e}", exc_info=True)


    # Determine which models to train (usually just one based on config)
    model_list = parse_model_list(model)
    if not model_list:
        model_list = get_enabled_models(cfg)

    if not model_list:
        click.echo(click.style("Error: No models specified via --model or enabled in config.", fg="red"))
        sys.exit(1)
    if len(model_list) > 1:
         click.echo(click.style(f"Warning: Multiple models enabled/specified ({model_list}). ensembleflex trains one unified model. Using first: '{model_list[0]}'", fg="yellow"))
         model_list = [model_list[0]]


    # Get unified output/models directories from config
    output_dir = get_output_dir(cfg)
    models_dir = get_models_dir(cfg)

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    click.echo(f"Using Output Directory: {os.path.abspath(output_dir)}")
    click.echo(f"Using Models Directory: {os.path.abspath(models_dir)}")

    # Determine input data path
    data_path = input # Use CLI input if provided
    if not data_path:
        # Construct path from config if CLI input not given
        data_dir = cfg["paths"]["data_dir"]
        file_pattern = cfg["dataset"]["file_pattern"]
        data_path = os.path.join(data_dir, file_pattern)
        click.echo(f"Using Input Data from config: {os.path.abspath(data_path)}")
    else:
        click.echo(f"Using Input Data from CLI: {os.path.abspath(data_path)}")


    # Create pipeline and train the single model
    try:
        pipeline = Pipeline(cfg) # Config now uses static paths
        click.echo(f"Training model: {model_list[0]}...")
        trained_models = pipeline.train(model_names=model_list, data_path=data_path) # Pass data_path explicitly

        if trained_models:
            click.echo(click.style(f"Successfully trained model: {list(trained_models.keys())[0]}", fg="green"))
        else:
             click.echo(click.style("Training finished, but no models were returned by the pipeline.", fg="yellow"))

    except FileNotFoundError as e:
         click.echo(click.style(f"Error: Input data file not found. {e}", fg="red"))
         sys.exit(1)
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        click.echo(click.style(f"An error occurred during training: {e}", fg="red"))
        sys.exit(1)


# --- Modified `evaluate` Command ---
@cli.command()
@click.option("--model",
              help="Model to evaluate (e.g., random_forest). Defaults to first enabled model.")
@click.option("--config",
              type=click.Path(exists=True, dir_okay=False),
              help="Path to custom YAML config file.")
@click.option("--param",
              multiple=True,
              help="Override config parameter.")
@click.option("--input",
              type=click.Path(exists=True, dir_okay=False),
              help="Override input data file (CSV) for evaluation. Defaults to config.")
# REMOVED: --temperature option
@click.option("--mode",
              type=click.Choice(["standard", "omniflex"], case_sensitive=False),
              help="Override operation mode (standard or omniflex).")
def evaluate(model, config, param, input, mode):
    """
    Evaluate the trained unified model on the test split of aggregated data.
    """
     # Load configuration
    cfg = load_config(config_path=config, param_overrides=param)

    # Set mode if specified via CLI
    if mode:
        cfg["mode"]["active"] = mode.lower()
        logger.info(f"Mode overridden by CLI: {cfg['mode']['active']}")
        # Re-apply mode logic (copied from train)
        try:
            if cfg["mode"]["active"] == "omniflex":
                omniflex_cfg = cfg.get("mode", {}).get("omniflex", {})
                use_features = cfg.setdefault("dataset", {}).setdefault("features", {}).setdefault("use_features", {})
                use_features["esm_rmsf"] = omniflex_cfg.get("use_esm", False)
                use_features["voxel_rmsf"] = omniflex_cfg.get("use_voxel", False)
            else: # Standard mode
                 use_features = cfg.setdefault("dataset", {}).setdefault("features", {}).setdefault("use_features", {})
                 use_features["esm_rmsf"] = False
                 use_features["voxel_rmsf"] = False
        except Exception as e:
            logger.error(f"Error applying mode override settings: {e}", exc_info=True)

    # Determine which model to evaluate
    model_list = parse_model_list(model)
    if not model_list:
        model_list = get_enabled_models(cfg)
        if model_list:
             model_to_eval = model_list[0]
             logger.info(f"No model specified via --model, using first enabled model: {model_to_eval}")
        else:
             click.echo(click.style("Error: No model specified or enabled in config to evaluate.", fg="red"))
             sys.exit(1)
    else:
         model_to_eval = model_list[0]
         if len(model_list) > 1:
              click.echo(click.style(f"Warning: Multiple models specified ({model_list}). Evaluating only the first: '{model_to_eval}'", fg="yellow"))

    # Get unified output/models directories
    output_dir = get_output_dir(cfg)
    models_dir = get_models_dir(cfg)
    click.echo(f"Using Output Directory: {os.path.abspath(output_dir)}")
    click.echo(f"Using Models Directory: {os.path.abspath(models_dir)}")

    # Determine input data path
    data_path = input # Use CLI input if provided
    if not data_path:
        data_dir = cfg["paths"]["data_dir"]
        file_pattern = cfg["dataset"]["file_pattern"]
        data_path = os.path.join(data_dir, file_pattern)
        click.echo(f"Using Input Data from config: {os.path.abspath(data_path)}")
    else:
        click.echo(f"Using Input Data from CLI: {os.path.abspath(data_path)}")

    # Create pipeline and evaluate the model
    try:
        pipeline = Pipeline(cfg)
        click.echo(f"Evaluating model: {model_to_eval}...")
        # Evaluate expects a list, even if it's just one model
        results = pipeline.evaluate(model_names=[model_to_eval], data_path=data_path)

        if results and model_to_eval in results:
            click.echo(click.style("\nEvaluation Results:", bold=True))
            metrics = results[model_to_eval]
            for metric, value in metrics.items():
                click.echo(f"  {metric}: {value:.4f}")
            results_file = os.path.join(output_dir, "evaluation_results.csv")
            click.echo(f"\nDetailed results saved in: {output_dir}")
            click.echo(f"Metrics summary saved to: {results_file}")
        else:
            click.echo(click.style(f"Evaluation completed, but no metrics found for model '{model_to_eval}'.", fg="yellow"))


    except FileNotFoundError as e:
         click.echo(click.style(f"Error: Required file not found. {e}", fg="red"))
         sys.exit(1)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        click.echo(click.style(f"An error occurred during evaluation: {e}", fg="red"))
        sys.exit(1)

# Place this corrected command function in ensembleflex/cli.py

@cli.command()
@click.option("--model",
              help="Model to use for prediction (e.g., random_forest). Defaults to best/first enabled model.")
@click.option("--config",
              type=click.Path(exists=True, dir_okay=False),
              help="Path to custom YAML config file.")
@click.option("--param",
              multiple=True,
              help="Override config parameter.")
@click.option("--input",
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help="Input data file (CSV) with features for prediction. May optionally contain true 'rmsf' target column.")
@click.option("--output",
              type=click.Path(dir_okay=False),
              help="Output file path for predictions (CSV). Defaults to <input_base>_predictions_<temp>K.csv in output dir.")
@click.option("--metrics-output",
              type=click.Path(dir_okay=False),
              help="Output file path for prediction metrics (CSV), if target is present in input. Defaults to <output>_metrics.csv.")
@click.option("--temperature", "--temp",
              type=float,
              required=True,
              help="REQUIRED: Target temperature (K) for which to generate predictions.")
@click.option("--mode",
              type=click.Choice(["standard", "omniflex"], case_sensitive=False),
              help="Override operation mode. Should match the mode the model was trained in.")
@click.option("--uncertainty",
              is_flag=True,
              help="Include uncertainty estimates if the model supports it.")
def predict(model, config, param, input, output, metrics_output, temperature, mode, uncertainty):
    """
    Generate RMSF predictions for new data at a SPECIFIED TEMPERATURE.

    If the input file contains the true 'rmsf' target column, evaluation
    metrics (PCC, R2, etc.) will be calculated and saved.
    """
    # --- Config loading and mode handling ---
    cfg = load_config(config_path=config, param_overrides=param)
    if mode:
        cfg["mode"]["active"] = mode.lower()
        logger.info(f"Mode overridden by CLI: {cfg['mode']['active']}")
        try: # Re-apply mode logic
            use_features = cfg.setdefault("dataset", {}).setdefault("features", {}).setdefault("use_features", {})
            if cfg["mode"]["active"] == "omniflex":
                omniflex_cfg = cfg.get("mode", {}).get("omniflex", {})
                use_features["esm_rmsf"] = omniflex_cfg.get("use_esm", False)
                use_features["voxel_rmsf"] = omniflex_cfg.get("use_voxel", False)
            else:
                 use_features["esm_rmsf"] = False
                 use_features["voxel_rmsf"] = False
        except Exception as e: logger.error(f"Error applying mode override: {e}", exc_info=True)

    # --- Get paths ---
    output_cfg_dir = get_output_dir(cfg) # Base output dir from config
    models_dir = get_models_dir(cfg)
    logger.info(f"Using Models Directory: {os.path.abspath(models_dir)}")

    try:
        # Initialize pipeline
        pipeline = Pipeline(cfg)

        click.echo(f"Generating predictions for temperature: {temperature} K")
        click.echo(f"Using input data: {os.path.abspath(input)}")

        # --- Call predict method ---
        # It returns predictions DataFrame and metrics dictionary (or None)
        predictions_df, calculated_metrics = pipeline.predict(
            data=input,
            model_name=model, # Pass CLI arg for model name
            temperature=temperature,
            with_uncertainty=uncertainty
        )

        # --- Determine output paths ---
        if not output:
            base = os.path.splitext(os.path.basename(input))[0]
            # Use K for Kelvin in filename for clarity
            output_filename = f"{base}_predictions_{int(temperature)}K.csv"
            output = os.path.join(output_cfg_dir, output_filename)
        click.echo(f"Prediction output path: {os.path.abspath(output)}")

        if calculated_metrics and not metrics_output:
             metrics_output = os.path.splitext(output)[0] + "_metrics.csv"
        elif calculated_metrics:
             click.echo(f"Metrics output path: {os.path.abspath(metrics_output)}")
        elif metrics_output:
             click.echo(click.style("Warning: --metrics-output specified, but no metrics calculated.", fg="yellow"))

        # --- Save Predictions ---
        output_save_dir = os.path.dirname(os.path.abspath(output))
        ensure_dir(output_save_dir) # Uses the imported function
        predictions_df.to_csv(output, index=False)
        click.echo(click.style(f"Saved predictions to {output}", fg="green"))

        # --- Save Metrics (if calculated) ---
        if calculated_metrics and metrics_output:
             metrics_save_dir = os.path.dirname(os.path.abspath(metrics_output))
             ensure_dir(metrics_save_dir)
             metrics_df_to_save = pd.DataFrame([calculated_metrics])

             # --- CORRECTED MODEL NAME FALLBACK ---
             # Determine the model name used (either from CLI or determined by pipeline)
             model_name_used = model
             if not model_name_used: # If --model wasn't specified
                # Get the name from the loaded model in the pipeline cache
                if pipeline.models:
                     model_name_used = list(pipeline.models.keys())[0] # Convert keys to list
                else: # Fallback if model wasn't cached (shouldn't happen if predict worked)
                     model_name_used = get_enabled_models(cfg)[0] if get_enabled_models(cfg) else 'unknown'

             metrics_df_to_save.insert(0, 'model', model_name_used)
             # --- END CORRECTION ---

             metrics_df_to_save.insert(1, 'predicted_temperature', temperature)
             metrics_df_to_save.to_csv(metrics_output, index=False)
             click.echo(click.style(f"Saved prediction metrics to {metrics_output}", fg="green"))
             click.echo(click.style("Prediction Metrics:", bold=True))
             for key in ['r2', 'pearson_correlation', 'rmse', 'mae']:
                  if key in calculated_metrics:
                       click.echo(f"  {key}: {calculated_metrics[key]:.4f}")

    except FileNotFoundError as e:
         click.echo(click.style(f"Error: Required file not found. {e}", fg="red"))
         sys.exit(1)
    except Exception as e:
        logger.error(f"Error generating predictions: {e}", exc_info=True)
        click.echo(click.style(f"An error occurred during prediction: {e}", fg="red"))
        sys.exit(1)


# --- REMOVED `train-all-temps` Command ---
# (Delete the entire function and its decorator)


# --- Redesigned `compare-temperatures` Command ---
@cli.command()
@click.option("--model",
              help="Model to use for comparison (e.g., random_forest). Defaults to best/first enabled model.")
@click.option("--config",
              type=click.Path(exists=True, dir_okay=False),
              help="Path to custom YAML config file.")
@click.option("--param",
              multiple=True,
              help="Override config parameter.")
@click.option("--input",
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help="Input data file (CSV) containing features (e.g., the test split).")
@click.option("--temp-list",
              help="Comma-separated list of temperatures to predict (e.g., '320,350,400').")
@click.option("--temp-range",
              help="Temperature range MIN,MAX,STEP (e.g., '300,450,10' for 300K to 450K in 10K steps).")
@click.option("--output-dir",
              required=True,
              type=click.Path(file_okay=False),
              help="Directory to save the comparison results (plots and CSVs).")
@click.option("--mode",
              type=click.Choice(["standard", "omniflex"], case_sensitive=False),
              help="Override operation mode. Should match the mode the model was trained in.")
def compare_temperatures(model, config, param, input, temp_list, temp_range, output_dir, mode):
    """
    Compare the single trained model's predictions across a range of temperatures.

    Uses the trained model specified (or default) and the feature data from
    --input. Predicts RMSF for each temperature in the specified list or range.
    Saves aggregated predictions and analysis plots to --output-dir.
    """
    if not temp_list and not temp_range:
        click.echo(click.style("Error: Must provide either --temp-list or --temp-range.", fg="red"))
        sys.exit(1)
    if temp_list and temp_range:
        click.echo(click.style("Error: Cannot use both --temp-list and --temp-range.", fg="red"))
        sys.exit(1)

    # Parse temperatures
    temperatures_to_predict = []
    if temp_list:
        try:
            temperatures_to_predict = [float(t.strip()) for t in temp_list.split(',')]
        except ValueError:
            click.echo(click.style("Error: Invalid format in --temp-list. Use comma-separated numbers.", fg="red"))
            sys.exit(1)
    elif temp_range:
        try:
            min_t, max_t, step_t = map(float, temp_range.split(','))
            if step_t <= 0: raise ValueError("Step must be positive")
            # Use numpy.arange for float steps, include endpoint carefully
            temperatures_to_predict = np.arange(min_t, max_t + step_t / 2, step_t).tolist() # Add small buffer to include max_t
        except (ValueError, IndexError):
            click.echo(click.style("Error: Invalid format in --temp-range. Use MIN,MAX,STEP.", fg="red"))
            sys.exit(1)

    if not temperatures_to_predict:
         click.echo(click.style("Error: No valid temperatures specified for comparison.", fg="red"))
         sys.exit(1)

    click.echo(f"Comparing model predictions across {len(temperatures_to_predict)} temperatures: {temperatures_to_predict}")

    # Load configuration
    cfg = load_config(config_path=config, param_overrides=param)

    # Set mode if specified via CLI
    if mode:
        cfg["mode"]["active"] = mode.lower()
        logger.info(f"Mode overridden by CLI: {cfg['mode']['active']}")
        # Re-apply mode logic (copied from train)
        try:
            if cfg["mode"]["active"] == "omniflex":
                omniflex_cfg = cfg.get("mode", {}).get("omniflex", {})
                use_features = cfg.setdefault("dataset", {}).setdefault("features", {}).setdefault("use_features", {})
                use_features["esm_rmsf"] = omniflex_cfg.get("use_esm", False)
                use_features["voxel_rmsf"] = omniflex_cfg.get("use_voxel", False)
            else: # Standard mode
                 use_features = cfg.setdefault("dataset", {}).setdefault("features", {}).setdefault("use_features", {})
                 use_features["esm_rmsf"] = False
                 use_features["voxel_rmsf"] = False
        except Exception as e:
            logger.error(f"Error applying mode override settings: {e}", exc_info=True)


    # Get models directory for loading
    models_dir = get_models_dir(cfg)
    click.echo(f"Using Models Directory: {os.path.abspath(models_dir)}")
    os.makedirs(output_dir, exist_ok=True)
    click.echo(f"Saving comparison results to: {os.path.abspath(output_dir)}")


    try:
        # Initialize pipeline (needed for predict method structure)
        pipeline = Pipeline(cfg)

        # Load the input data ONCE (features only)
        click.echo(f"Loading input data for features: {input}")
        from ensembleflex.data.processor import load_and_process_data, prepare_data_for_model
        # Load the raw data, process features, but don't necessarily need target
        input_df_processed = load_and_process_data(data_path=input, config=cfg)
        # Prepare just the features, target isn't strictly needed for prediction input prep
        X_input, _, feature_names = prepare_data_for_model(input_df_processed, cfg, include_target=False)
        input_ids_df = input_df_processed[['domain_id', 'resid', 'resname']].reset_index(drop=True) # Keep IDs for merging

        # Load the single model ONCE
        # Determine model name (use specified or find best/first enabled)
        if not model:
            # Logic to find best model (e.g., from evaluation results) could go here
            # Fallback to first enabled model
            enabled_models = get_enabled_models(cfg)
            if not enabled_models:
                 click.echo(click.style("Error: No model specified and no models enabled in config.", fg="red"))
                 sys.exit(1)
            model_name_to_load = enabled_models[0]
            click.echo(f"No model specified, using first enabled model: {model_name_to_load}")
        else:
            model_name_to_load = model

        click.echo(f"Loading model: {model_name_to_load}...")
        loaded_model = pipeline.load_model(model_name_to_load) # Assumes load_model works correctly

        # Store predictions per temperature
        all_predictions = []

        # Loop through temperatures and predict
        for temp in temperatures_to_predict:
            click.echo(f"  Predicting for T = {temp:.1f} K...")

            # --- Manual Feature Augmentation for Temperature ---
            # Find index of 'temperature' feature if it was used during training
            temp_feature_index = -1
            if cfg['dataset']['features']['use_features'].get('temperature', False):
                try:
                    temp_feature_index = feature_names.index('temperature')
                except ValueError:
                     click.echo(click.style("Error: Model trained with temperature feature, but 'temperature' not in prepared feature names.", fg="red"))
                     sys.exit(1)
                 # Create augmented feature matrix X_augmented
                X_augmented = X_input.copy()
                X_augmented[:, temp_feature_index] = temp # Set the temperature column
            else:
                # If model wasn't trained with temperature, we shouldn't be here.
                # Log warning or error, potentially skip?
                logger.warning(f"Running compare-temperatures, but model {model_name_to_load} was not trained with temperature as a feature. Predictions might be constant.")
                X_augmented = X_input # Use original features

            # Predict using the loaded model and augmented features
            temp_preds = loaded_model.predict(X_augmented)

            # Store results with identifiers and temperature
            temp_df = input_ids_df.copy()
            temp_df['temperature'] = temp
            temp_df['predicted_rmsf'] = temp_preds
            all_predictions.append(temp_df)

        # Combine all predictions
        if not all_predictions:
             click.echo(click.style("Error: No predictions were generated.", fg="red"))
             sys.exit(1)

        combined_results_df = pd.concat(all_predictions, ignore_index=True)

        # Save combined predictions
        output_csv_path = os.path.join(output_dir, f"{model_name_to_load}_predictions_vs_temp.csv")
        combined_results_df.to_csv(output_csv_path, index=False)
        click.echo(f"Saved combined predictions to: {output_csv_path}")

        # --- Perform Analysis and Visualization ---
        click.echo("Generating analysis plots...")

        # 1. Plot Predicted RMSF vs. Input Temperature for a few residues/average
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from ensembleflex.utils.visualization import save_plot # Assume this exists

            # Select a few domains/residues for plotting
            ids_to_plot = combined_results_df[['domain_id', 'resid']].drop_duplicates().sample(min(5, len(combined_results_df)), random_state=cfg['system']['random_state'])

            plt.figure(figsize=(10, 6))
            for _, row in ids_to_plot.iterrows():
                domain, resid = row['domain_id'], row['resid']
                subset = combined_results_df[(combined_results_df['domain_id'] == domain) & (combined_results_df['resid'] == resid)]
                plt.plot(subset['temperature'], subset['predicted_rmsf'], marker='o', linestyle='-', label=f"{domain}-{resid}")

            plt.xlabel("Input Temperature (K)")
            plt.ylabel("Predicted RMSF")
            plt.title(f"Predicted RMSF vs. Temperature ({model_name_to_load})")
            plt.legend(title="Domain-ResID", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plot_path = os.path.join(output_dir, f"{model_name_to_load}_pred_vs_temp_samples.png")
            save_plot(plt, plot_path)
            click.echo(f"Saved sample prediction vs. temperature plot to: {plot_path}")

            # 2. Plot average predicted RMSF vs Temperature
            avg_rmsf_vs_temp = combined_results_df.groupby('temperature')['predicted_rmsf'].mean().reset_index()
            plt.figure(figsize=(10, 6))
            plt.plot(avg_rmsf_vs_temp['temperature'], avg_rmsf_vs_temp['predicted_rmsf'], marker='o', linestyle='-')
            plt.xlabel("Input Temperature (K)")
            plt.ylabel("Average Predicted RMSF")
            plt.title(f"Average Predicted RMSF vs. Temperature ({model_name_to_load})")
            plt.grid(True, linestyle='--', alpha=0.6)
            plot_path = os.path.join(output_dir, f"{model_name_to_load}_avg_pred_vs_temp.png")
            save_plot(plt, plot_path)
            click.echo(f"Saved average prediction vs. temperature plot to: {plot_path}")


        except ImportError:
            click.echo(click.style("Warning: Matplotlib/Seaborn not found. Skipping plot generation.", fg="yellow"))
        except Exception as e:
            logger.error(f"Error during plot generation: {e}", exc_info=True)
            click.echo(click.style(f"Warning: Failed to generate plots: {e}", fg="yellow"))

        click.echo(click.style("Temperature comparison finished.", fg="green"))


    except FileNotFoundError as e:
         click.echo(click.style(f"Error: Required file not found. {e}", fg="red"))
         sys.exit(1)
    except Exception as e:
        logger.error(f"Error during temperature comparison: {e}", exc_info=True)
        click.echo(click.style(f"An error occurred during temperature comparison: {e}", fg="red"))
        sys.exit(1)


# --- `preprocess` Command (Largely unchanged, but remove temp templating) ---
@cli.command()
@click.option("--input",
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help="Input data file (CSV) to preprocess.")
@click.option("--config",
              type=click.Path(exists=True, dir_okay=False),
              help="Path to custom YAML config file.")
@click.option("--param",
              multiple=True,
              help="Override config parameter.")
@click.option("--output",
              type=click.Path(dir_okay=False),
              help="Output file path for processed data (CSV). Defaults to input_processed.csv.")
# REMOVED: --temperature option
@click.option("--mode",
              type=click.Choice(["standard", "omniflex"], case_sensitive=False),
              help="Operation mode (influences feature processing).")
def preprocess(input, config, param, output, mode):
    """
    Preprocess data only (load, clean, engineer features) without training.
    Uses the aggregated data structure.
    """
    from ensembleflex.data.processor import load_and_process_data # Import here

    # Load configuration
    cfg = load_config(config_path=config, param_overrides=param)

    # Set mode if specified
    if mode:
        cfg["mode"]["active"] = mode.lower()
        logger.info(f"Mode overridden by CLI: {cfg['mode']['active']}")
        # Re-apply mode logic (copied from train)
        try:
            if cfg["mode"]["active"] == "omniflex":
                omniflex_cfg = cfg.get("mode", {}).get("omniflex", {})
                use_features = cfg.setdefault("dataset", {}).setdefault("features", {}).setdefault("use_features", {})
                use_features["esm_rmsf"] = omniflex_cfg.get("use_esm", False)
                use_features["voxel_rmsf"] = omniflex_cfg.get("use_voxel", False)
            else: # Standard mode
                 use_features = cfg.setdefault("dataset", {}).setdefault("features", {}).setdefault("use_features", {})
                 use_features["esm_rmsf"] = False
                 use_features["voxel_rmsf"] = False
        except Exception as e:
            logger.error(f"Error applying mode override settings: {e}", exc_info=True)


    try:
        # Process data using the main function
        click.echo(f"Preprocessing data from: {os.path.abspath(input)}")
        # We pass the input path directly to load_and_process_data
        processed_df = load_and_process_data(data_path=input, config=cfg)

        # Determine output path
        if not output:
            base = os.path.splitext(os.path.basename(input))[0]
            output_filename = f"{base}_processed.csv"
             # Save relative to input file's directory or current dir? Let's use current dir.
            output = os.path.join(".", output_filename) # Save in current directory
            click.echo(f"Output path not specified, saving to: {output}")


        # Ensure output directory exists
        output_save_dir = os.path.dirname(os.path.abspath(output))
        os.makedirs(output_save_dir, exist_ok=True)

        # Save processed data
        processed_df.to_csv(output, index=False)
        click.echo(click.style(f"Saved processed data to {output}", fg="green"))

    except FileNotFoundError as e:
         click.echo(click.style(f"Error: Input file not found. {e}", fg="red"))
         sys.exit(1)
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}", exc_info=True)
        click.echo(click.style(f"An error occurred during preprocessing: {e}", fg="red"))
        sys.exit(1)

# --- Modified `run` Command ---
@cli.command()
@click.option("--model",
              help="Model to run (train, evaluate, analyze). Defaults to first enabled model.")
@click.option("--config",
              type=click.Path(exists=True, dir_okay=False),
              help="Path to custom YAML config file.")
@click.option("--param",
              multiple=True,
              help="Override config parameter.")
@click.option("--input",
              type=click.Path(exists=True, dir_okay=False),
              help="Override input data file (CSV). Defaults to config.")
# REMOVED: --temperature option
@click.option("--mode",
              type=click.Choice(["standard", "omniflex"], case_sensitive=False),
              help="Override operation mode.")
@click.option("--skip-analysis", # Changed from skip-visualization for clarity
              is_flag=True,
              help="Skip the analysis step after training and evaluation.")
def run(model, config, param, input, mode, skip_analysis):
    """
    Run the complete pipeline (train, evaluate, analyze) for the unified model.
    """
     # Load configuration
    cfg = load_config(config_path=config, param_overrides=param)

    # Set mode if specified
    if mode:
        cfg["mode"]["active"] = mode.lower()
        logger.info(f"Mode overridden by CLI: {cfg['mode']['active']}")
        # Re-apply mode logic (copied from train)
        try:
            if cfg["mode"]["active"] == "omniflex":
                omniflex_cfg = cfg.get("mode", {}).get("omniflex", {})
                use_features = cfg.setdefault("dataset", {}).setdefault("features", {}).setdefault("use_features", {})
                use_features["esm_rmsf"] = omniflex_cfg.get("use_esm", False)
                use_features["voxel_rmsf"] = omniflex_cfg.get("use_voxel", False)
            else: # Standard mode
                 use_features = cfg.setdefault("dataset", {}).setdefault("features", {}).setdefault("use_features", {})
                 use_features["esm_rmsf"] = False
                 use_features["voxel_rmsf"] = False
        except Exception as e:
            logger.error(f"Error applying mode override settings: {e}", exc_info=True)


    # Determine which models to use (usually just one)
    model_list = parse_model_list(model)
    if not model_list:
        model_list = get_enabled_models(cfg)

    if not model_list:
        click.echo(click.style("Error: No models specified or enabled in config.", fg="red"))
        sys.exit(1)
    if len(model_list) > 1:
         click.echo(click.style(f"Warning: Multiple models enabled/specified ({model_list}). ensembleflex runs one unified model. Using first: '{model_list[0]}'", fg="yellow"))
         model_list = [model_list[0]]

    # Get unified output/models directories
    output_dir = get_output_dir(cfg)
    models_dir = get_models_dir(cfg)

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    click.echo(f"Using Output Directory: {os.path.abspath(output_dir)}")
    click.echo(f"Using Models Directory: {os.path.abspath(models_dir)}")

    # Determine input data path
    data_path = input # Use CLI input if provided
    if not data_path:
        data_dir = cfg["paths"]["data_dir"]
        file_pattern = cfg["dataset"]["file_pattern"]
        data_path = os.path.join(data_dir, file_pattern)
        click.echo(f"Using Input Data from config: {os.path.abspath(data_path)}")
    else:
        click.echo(f"Using Input Data from CLI: {os.path.abspath(data_path)}")

    # Create pipeline and run
    try:
        pipeline = Pipeline(cfg)
        click.echo(f"Running full pipeline for model: {model_list[0]}...")

        # Pass skip_analysis flag correctly
        results = pipeline.run_pipeline(
            model_names=model_list,
            data_path=data_path,
            skip_analysis=skip_analysis # Use the flag directly
        )

        click.echo(click.style("\nPipeline completed successfully!", fg="green"))
        click.echo(f"Results saved to: {output_dir}")

        if results and model_list[0] in results:
            click.echo(click.style("\nFinal Evaluation Metrics:", bold=True))
            metrics = results[model_list[0]]
            for metric, value in metrics.items():
                click.echo(f"  {metric}: {value:.4f}")
        else:
            click.echo(click.style("Evaluation completed, but no final metrics were returned.", fg="yellow"))


    except FileNotFoundError as e:
         click.echo(click.style(f"Error: Required file not found. {e}", fg="red"))
         sys.exit(1)
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        click.echo(click.style(f"An error occurred running the pipeline: {e}", fg="red"))
        sys.exit(1)

# --- `list-models` Command (Unchanged) ---
@cli.command()
def list_models():
    """
    List available models registered in the application.

    Examples:
        ensembleflex list-models
    """
    models = get_available_models() # Function imported from ensembleflex.models

    click.echo("Available models:")
    if models:
        for model in sorted(models):
            click.echo(f"  - {model}")
    else:
        click.echo("  No models found or registered.")

# --- REMOVED `list-temperatures` Command ---
# (Its purpose is informational and less relevant now)


if __name__ == "__main__":
    cli()