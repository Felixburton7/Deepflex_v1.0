# ESM-Flex (ESM-3 Version)

This project predicts protein flexibility (RMSF) from amino acid sequences using pre-trained ESM-3 models via the Hugging Face `transformers` library.

## Project Structure

*   `config.yaml`: Configuration file for model, training, data paths.
*   `data/`: Contains raw and processed data.
    *   `data/raw/`: Place for initial RMSF data (e.g., `rmsf_replica_average_temperature320.csv`).
    *   `data/raw/fix_data_.py`: Script to standardize residue names (e.g., HIS variants).
    *   `data/processed/`: Output location for split data (FASTA, NPY, TXT).
*   `models/`: Output location for saved model checkpoints, logs, and plots.
*   `utils/`: Placeholder for utility scripts (currently empty).
*   `data_processor.py`: Script to process raw CSV data into usable splits.
*   `dataset.py`: Defines the PyTorch Dataset and DataLoader creation logic (with length-batching).
*   `model.py`: Defines the `ESMRegressionModel` (frozen ESM-3 + trainable head).
*   `train.py`: Script for training the model.
*   `predict.py`: Script for generating RMSF predictions using a trained model.
*   `main.py`: Main command-line interface (CLI) to run processing, training, or prediction.
*   `requirements.txt`: Python dependencies.

## Setup

1.  **Create Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure your `torch` version is compatible with your CUDA toolkit.*

## Usage

The pipeline is controlled via `main.py`:

```bash
python main.py --help
```

**1. Process Data:**

*   Place your raw RMSF data CSV (e.g., `rmsf_replica_average_temperature320.csv`) in `data/raw/`.
*   *Optional but recommended:* Standardize residue names (especially Histidine variants) if needed:
    ```bash
    python data/raw/fix_data_.py --input data/raw/your_data.csv --output data/raw/your_data_fixed.csv
    ```
*   Run the main processing script (update `--csv` path if you fixed the data):
    ```bash
    python main.py process --csv data/raw/your_data_fixed.csv --output data/processed
    ```
    This creates `train_*`, `val_*`, `test_*` files in `data/processed/`.

**2. Train Model:**

*   Adjust parameters in `config.yaml` (especially `model.esm_version`, `training.batch_size`, `training.learning_rate`).
*   Start training:
    ```bash
    python main.py train --config config.yaml
    ```
    Checkpoints, logs, and plots will be saved in `models/` (or the directory specified in `config.yaml`).

**3. Predict RMSF:**

*   Use a trained checkpoint (`.pt` file, e.g., `models/best_model.pt`).
*   Provide a FASTA file (`input.fasta`) with sequences to predict.
*   Run prediction:
    ```bash
    python main.py predict --model_checkpoint models/best_model.pt --fasta_path input.fasta --output_dir prediction_results
    ```
    Results (CSV, plots, log) will be in `prediction_results/`.

## Configuration (`config.yaml`)

*   `model.esm_version`: Set the Hugging Face identifier for the desired ESM-3 model (e.g., "facebook/esm3_medium_150M_1160").
*   `training.batch_size` / `training.accumulation_steps`: Adjust based on GPU memory. Effective batch size = `batch_size * accumulation_steps`.
*   `training.learning_rate`: Tune for the regression head (can often be higher than full fine-tuning rates).
*   `output.model_dir`: Where training outputs are saved.

