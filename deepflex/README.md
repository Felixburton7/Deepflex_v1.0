
# 🧬 DeepFlex: Temperature-Aware Protein Flexibility Prediction

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![ESM](https://img.shields.io/badge/ESM-Evolutionary%20Scale%20Modeling-green.svg)](https://github.com/evolutionaryscale/esm)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

*Predicting protein dynamics across temperatures using ESM embeddings and attention.*

</div>

---

## 📋 Table of Contents

> 1. [🔍 Overview](#-overview)
> 2. [✨ Features](#-features)
> 3. [📦 Installation](#-installation)
> 4. [🧬 ESM Integration](#-esm-integration)
> 5. [📂 Project Structure](#-project-structure)
> 6. [📊 Data Format](#-data-format)
> 7. [🚀 Usage Pipeline](#-usage-pipeline)
>    - [1. Process Data 🔍](#1-process-data-)
>    - [2. Train Model 🏋️](#2-train-model-)
>    - [3. Predict RMSF 🔮](#3-predict-rmsf-)
>    - [4. Evaluate Predictions 📏](#4-evaluate-predictions-)
> 8. [🧠 Model Architecture](#-model-architecture)
> 9. [⚙️ Configuration](#-configuration)
> 10. [📝 Example Workflows](#-example-workflows)
> 11. [🔧 Extending the Model](#-extending-the-model)
> 12. [🛠️ Troubleshooting](#troubleshooting)
> 13. [❓ FAQs](#-faqs)

---

## 🔍 Overview

DeepFlex predicts per-residue protein flexibility (RMSF) at specific temperatures. It leverages the power of the ESM-C protein language model, structural information (like secondary structure, solvent accessibility), and explicit temperature conditioning to understand how a protein's dynamics change with temperature.

**What is RMSF?** 🤔
Imagine a protein constantly wiggling and jiggling. RMSF measures how much each atom in a residue moves around its average position.
-   **High RMSF:** More movement, greater flexibility (like loops).
-   **Low RMSF:** Less movement, more rigid (like alpha-helices or beta-sheets).
Temperature increases the jiggling, generally leading to higher RMSF. DeepFlex aims to predict this temperature-dependent behavior.

---

## ✨ Features

-   🌡️ **Temperature-aware prediction**: Accurately predict how proteins behave at different temperatures.
-   🧬 **ESM-C Embeddings**: Leverages powerful pre-trained protein language model representations.
-   📊 **Multi-feature integration**: Combines sequence embeddings with structural info (secondary structure, accessibility, angles, B-factors, etc.).
-   👁️ **Self-attention**: Uses attention mechanisms (`AttentionWithTemperature`) to capture residue interactions.
-   ⚠️ **Uncertainty Quantification**: Estimates prediction confidence via Monte Carlo Dropout (`--mc_samples`).
-   🧩 **Topology-based Splitting**: Ensures robust evaluation by keeping similar protein structures together during train/val/test splits (`split_by_topology`).
-   🔄 **End-to-End Pipeline**: Provides scripts for the complete workflow from raw data processing to prediction and evaluation (`main.py`).

---

## 📦 Installation

### Prerequisites

-   🐍 Python 3.8+
-   🖥️ CUDA-compatible GPU (strongly recommended for performance)

### Step 1: Clone the repository 📥

```bash
git clone https://github.com/[yourusername]/deepflex.git
cd deepflex
```
*(Replace `[yourusername]` with the actual GitHub username or organization)*

### Step 2: Create a virtual environment (recommended) 🔒

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies 📚

```bash
pip install -r requirements.txt
```

### Step 4: Install ESM ⚡

DeepFlex requires the `esm` library.

```bash
pip install "fair-esm[esmfold]"
# or simply: pip install esm
```

> 💡 **Note**
> The specific ESM-C model weights (e.g., `esmc_600m` defined in `config.yaml`) will be downloaded automatically by the `esm` library when the DeepFlex model is first loaded (`model.py`). See the [ESM GitHub repository](https://github.com/evolutionaryscale/esm) for more details.

### Step 5: Verify installation ✅

```bash
python main.py --help
```
You should see the help message for the main script.

---

## 🧬 ESM Integration

DeepFlex uses **ESM-C** models to generate powerful numerical representations (embeddings) for each amino acid in a protein sequence.

-   🧠 **Model Used:** Configured via `model.esm_version` in `config.yaml` (default: `esmc_600m`). Ensure your hardware meets the VRAM requirements for the chosen model.
-   ⚙️ **How it's used (`model.py`):**
    1.  The `EnhancedTemperatureAwareESMModel` class loads the specified ESM-C model using `ESMC.from_pretrained()`.
    2.  **Crucially, the pre-trained weights of this ESM-C model are frozen** (`param.requires_grad = False`). DeepFlex *only* trains the layers added *after* the ESM model.
    3.  During the `forward` pass, input sequences are fed to the frozen `esm_model` to get per-residue embeddings. Special start/end tokens added by ESM are removed.
    4.  These fixed embeddings serve as the primary sequence input to the trainable parts of DeepFlex (feature fusion, attention, regression head).
-   🚀 **Benefits:** By using these pre-trained embeddings, DeepFlex leverages knowledge learned from billions of protein sequences, capturing complex patterns related to evolution, structure, and function, which boosts predictive performance.

---

## 📂 Project Structure

```plaintext
deepflex/
├── main.py                   # Main CLI entry point (process, train, predict)
├── config.yaml               # Central configuration file
├── data_processor.py         # Data loading, cleaning, feature extraction, splitting
├── dataset.py                # PyTorch Dataset & DataLoader (EnhancedRMSFDataset)
├── model.py                  # Model architecture (EnhancedTemperatureAwareESMModel, layers)
├── train.py                  # Training & validation loops, checkpointing
├── predict.py                # Prediction script
├── evaluate_predictions.py   # Calculates prediction metrics
├── merge_predictions.py      # Utility: Merge predictions back into original data
├── concatenate_fastas.py     # Utility: Combine split FASTAs for prediction
├── data/
│   ├── raw/                  # Input CSV data folder
│   │   └── fix_data_.py      # (Optional) Script stub for raw data preprocessing
│   ├── processed/            # Output folder for processed data splits
│   └── utils/               # Utility scripts (e.g., filter_by_temperature.py)
├── models/                   # Default output folder for trained models (checkpoints)
├── predictions/              # Default output folder for prediction results
├── requirements.txt          # Python dependencies
├── LICENSE                   # Project license
└── README.md                 # This file
```

---

## 📊 Data Format

### Input Data (`process` command) 📥

-   **Format:** CSV file.
-   **Source:** Path specified via `--csv` or in `config.yaml` (`data.raw_csv_path`).
-   **Required Columns:**

| Column        | Description                             | Example       |
| :------------ | :-------------------------------------- | :------------ |
| `domain_id`   | Unique protein domain identifier        | `1xyz_A`      |
| `resid`       | Residue position number                 | `42`          |
| `resname`     | Residue name (3-letter code)            | `ALA`         |
| `temperature` | Temperature (Kelvin)                    | `320.0`       |
| `rmsf`        | Target RMSF value (ground truth)        | `0.156`       |

-   **Optional Feature Columns:** (Control via `data.features` in `config.yaml`)

| Feature                       | Example Column Name           |
| :---------------------------- | :---------------------------- |
| Normalized Residue Position   | `normalized_resid`            |
| Core/Exterior Encoding        | `core_exterior_encoded`       |
| Secondary Structure Encoding  | `secondary_structure_encoded` |
| Relative Solvent Accessibility| `relative_accessibility`      |
| Normalized Phi Angle          | `phi_norm`                    |
| Normalized Psi Angle          | `psi_norm`                    |
| Protein Size                  | `protein_size`                |
| Normalized B-factor           | `bfactor_norm`                |
| Voxel RMSF (Alternative)      | `voxel_rmsf`                  |

### Processed Data (Output of `process` command) 🔄

-   **Location:** Directory specified via `--output` or in `config.yaml` (`data.data_dir`).
-   **Files Generated per Split (`train`, `val`, `test`):**

| File Type                       | Example Filename                | Format         | Content                                                    |
| :------------------------------ | :------------------------------ | :------------- | :--------------------------------------------------------- |
| Instance Keys                   | `train_instances.txt`           | Text           | List of `instance_key` strings (`domain_id@temp.1f`)       |
| Sequences                       | `train_sequences.fasta`         | FASTA          | Sequences with `instance_key` as header                    |
| Target RMSF                     | `train_rmsf.npy`                | NumPy Dict     | `instance_key` -> `np.ndarray[L]` (float32)                |
| Temperatures (Raw)              | `train_temperatures.npy`        | NumPy Dict     | `instance_key` -> `float` (Kelvin)                         |
| Feature Data (Per-Residue)    | `train_normalized_resid.npy`    | NumPy Dict     | `instance_key` -> `np.ndarray[L]` (float32)                |
| Feature Data (Global)         | `train_protein_size.npy`        | NumPy Dict     | `instance_key` -> `float`                                  |
| **Normalization Params:**       |                                 |                | **Calculated from Training Split ONLY**                    |
| Feature Normalization           | `feature_normalization.json`    | JSON           | Mean/Std/Min/Max/Is_Global per feature                     |
| Temperature Scaling             | `temp_scaling_params.json`      | JSON           | Min/Max temperature                                        |

*(L = sequence length)*

### Prediction Output (`predict` command) 📤

-   **Format:** CSV file.
-   **Location:** Subdirectory within `--output_dir`.
-   **Columns:**

| Column        | Description                                             | Example          |
| :------------ | :------------------------------------------------------ | :--------------- |
| `instance_key`| Unique identifier (`domain_id@temperature.1f`)          | `1xyz_A@320.0`   |
| `resid`       | Residue position (1-based integer)                      | `42`             |
| `rmsf_pred`   | Predicted RMSF value (float)                            | `0.183`          |
| `uncertainty` | Prediction uncertainty (StdDev from MC Dropout, float)  | `0.021`          |
| `temperature` | Raw temperature (Kelvin, float) used for prediction     | `320.0`          |

---

## 🚀 Usage Pipeline

The workflow is managed by `main.py` using subcommands: `process`, `train`, `predict`, `evaluate`.

### 1. Process Data 🔍

**Script:** `data_processor.py` (via `main.py process`)
**Goal:** Transform raw CSV into standardized, split datasets for the model.

**Inputs & Outputs:**

| Type               | Item                       | Source / Location                           |
| :----------------- | :------------------------- | :------------------------------------------ |
| 📄 **Input**       | Raw Data CSV               | `--csv` / `data.raw_csv_path`             |
| ⚙️ **Input**       | Config File                | `--config config.yaml`                      |
| 📊 **Input**       | Split Ratios               | `--train_ratio`, `--val_ratio`            |
| 🎲 **Input**       | Random Seed                | `--seed`                                    |
| 📤 **Output**      | Processed Files            | `--output` dir / `data.data_dir`            |
|                    | (`{split}_*.{txt,fasta,npy}`) |                                           |
| ⚖️ **Output**      | Normalization/Scaling Files| `--output` dir / `data.data_dir`            |
|                    | (`*.json`)                 |                                           |

**Command:**
```bash
python main.py process \
  --config config.yaml \
  --csv data/raw/your_data.csv \
  --output data/processed \
  --train_ratio 0.85 \
  --val_ratio 0.075 \
  --seed 42
```

### 2. Train Model 🏋️

**Script:** `train.py` (via `main.py train`)
**Goal:** Train the DeepFlex model using the processed data splits.

**Inputs & Outputs:**

| Type                | Item                     | Source / Location                    |
| :------------------ | :----------------------- | :----------------------------------- |
| 📊 **Input**        | Processed Data           | `data.data_dir` in `config.yaml`     |
| ⚙️ **Input**        | Config File              | `--config config.yaml`               |
| 💾 **Output**       | Model Checkpoints (`.pt`)| `output.model_dir` in `config.yaml`  |
| 📝 **Output**       | Log File (`.log`)        | `output.model_dir`/logs              |
| 📈 **Output**       | Metrics Plot (`.png`)    | `output.model_dir`                   |

**Command:**
```bash
# Ensure config.yaml is correctly set up
python main.py train --config config.yaml
```

### 3. Predict RMSF 🔮

**Script:** `predict.py` (via `main.py predict`)
**Goal:** Use a trained model to predict RMSF on new or existing sequences.

**Inputs & Outputs:**

| Type               | Item                        | Source / Location                    |
| :----------------- | :-------------------------- | :----------------------------------- |
| 💾 **Input**       | Model Checkpoint (`.pt`)    | `--model_checkpoint`                 |
| 🧬 **Input**       | Input FASTA                 | `--fasta_path`                       |
| 🌡️ **Input**       | Temperature Source          | `--temperature` OR `--temperature_npy` |
| 📊 **Input** (Opt.)| Feature `.npy` files        | Implicitly loaded                    |
| ⚖️ **Input** (Opt.)| Scaling/Norm `.json` files  | Implicitly loaded                    |
| ⚙️ **Input** (Opt.)| Config File (defaults)      | `--config config.yaml`               |
| ⚠️ **Input** (Opt.)| MC Samples                  | `--mc_samples`                       |
| 📄 **Output**      | Prediction CSV              | Subdirectory in `--output_dir`       |
| 📝 **Output**      | Log File                    | Output subdirectory                  |
| 📊 **Output** (Opt.)| Plot Files (`.png`)         | Output subdirectory /plots           |

**Command:**
```bash
# Example with NPY temperatures and uncertainty
python main.py predict \
  --config config.yaml \
  --model_checkpoint models/best_model.pt \
  --fasta_path data/processed/test_sequences.fasta \
  --temperature_npy data/processed/test_temperatures.npy \
  --mc_samples 10 \
  --output_dir predictions/test_set_uncertainty
```

### 4. Evaluate Predictions 📏

**Script:** `evaluate_predictions.py`
**Goal:** Assess prediction accuracy against ground truth for a *specific temperature*.

**Inputs & Outputs:**

| Type              | Item                     | Source / Location               |
| :---------------- | :----------------------- | :------------------------------ |
| 📄 **Input**      | Prediction CSV           | `--predictions_csv`             |
| ✅ **Input**      | Ground Truth RMSF (`.npy`)| `--ground_truth_npy`          |
| 🎯 **Input**      | Target Temperature       | `--temperature`                 |
| 📈 **Output**     | Metrics JSON             | `--output_json`                 |

**Command:**
```bash
python evaluate_predictions.py \
  --predictions_csv predictions/test_set_379K/predictions_test_set_379K.csv \
  --ground_truth_npy data/processed/test_rmsf.npy \
  --temperature 379 \
  --output_json predictions/test_set_379K/evaluation_metrics_379K.json
```

---

## 🧠 Model Architecture

DeepFlex utilizes a multi-component architecture designed specifically for temperature-aware flexibility prediction:

```text
┌────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│   ESM-C        │     │  Temperature      │     │  Structural       │
│   Embeddings   │     │  Encoding         │     │  Features         │
└───────┬────────┘     └─────────┬─────────┘     └──────────┬────────┘
        │                        │                          │
        ▼                        │                          ▼
┌────────────────┐               │              ┌───────────────────┐
│  Protein       │               │              │  Feature          │
│  Representation│               │              │  Processing       │
└───────┬────────┘               │              └──────────┬────────┘
        │                        │                         │
        └─────────────┐          │          ┌─────────────┘
                      ▼          ▼          ▼
                 ┌────────────────────────────────┐
                 │  Temperature-Aware Attention   │
                 └─────────────┬─────────────────┘
                               │
                               ▼
                 ┌────────────────────────────────┐
                 │  Regression Head               │
                 └─────────────┬─────────────────┘
                               │
                               ▼
                 ┌────────────────────────────────┐
                 │  RMSF Predictions              │
                 └────────────────────────────────┘
```

### Component Details (`model.py`): 🧩

1.  🧠 **ESM-C Embeddings**: Extracts residue representations using a frozen pre-trained ESM-C model (`esm_model`). Special tokens removed.
2.  🌡️ **Temperature Encoding (`TemperatureEncoding`)**: Converts *scaled* temperature into a vector embedding that influences the model.
3.  📊 **Structural Feature Processing (`FeatureProcessor`)**: Normalizes and projects enabled features into a consistent dimension.
4.  🔗 **Feature Fusion & Temperature-Aware Attention**: Combines ESM embeddings, processed features, and temperature embeddings. Uses attention mechanisms (`AttentionWithTemperature`) where temperature conditions the attention scores.
5.  📉 **Regression Head**: An MLP or linear layer projecting the final representations to per-residue RMSF predictions.

---

## ⚙️ Configuration

The `config.yaml` file controls the `process` and `train` stages.

```yaml
# config.yaml (Key Sections)

data:
  data_dir: data/processed       # Processed data location (Input for train)
  raw_csv_path: data/raw/...csv  # Raw data location (Input for process)
  temp_scaling_filename: temp_scaling_params.json # Output name for temp scaler params
  features:
    # --- Enable/disable structural features ---
    use_accessibility: true
    use_bfactor: true
    # ... other features ...
    normalization_params_file: feature_normalization.json # Output name for feature norm params

model:
  esm_version: "esmc_600m"      # ESM-C model variant
  architecture:
    use_enhanced_features: true # Use structural features?
    use_attention: true         # Use attention layers?
    attention_heads: 8
    # ... other architecture settings ...
  regression:
    hidden_dim: 128             # Hidden layer size in MLP head (0=Linear)
    dropout: 0.1

training:
  num_epochs: 50
  batch_size: 8                 # Adjust based on GPU memory
  learning_rate: 1.0e-4
  weight_decay: 0.01
  accumulation_steps: 4         # Simulates larger batch size
  # ... other training settings (scheduler, early stopping, seed) ...
  length_bucket_size: 50        # For efficient batching

output:
  model_dir: models             # Where to save trained models/logs/plots

prediction: # Default settings for predict.py (can be overridden by CLI)
  batch_size: 8
  plot_predictions: true
  # ... other prediction settings ...
```
*(See the actual `config.yaml` for all options and detailed comments.)*

---

## 📝 Example Workflows

### Complete Training & Evaluation Pipeline 🔄

```bash
# 1️⃣ Process Data
python main.py process --config config.yaml --csv path/to/your/data.csv --output data/processed

# 2️⃣ Configure (Ensure config.yaml points to data/processed & desired model output dir)

# 3️⃣ Train Model
python main.py train --config config.yaml

# 4️⃣ Predict on Test Set
python main.py predict \
  --config config.yaml \
  --model_checkpoint models/best_model.pt \
  --fasta_path data/processed/test_sequences.fasta \
  --temperature_npy data/processed/test_temperatures.npy \
  --output_dir predictions/test_set_results

# 5️⃣ Evaluate @ Specific Temperature
TARGET_TEMP=379
python evaluate_predictions.py \
  --predictions_csv predictions/test_set_results/predictions_*.csv \
  --ground_truth_npy data/processed/test_rmsf.npy \
  --temperature $TARGET_TEMP \
  --output_json predictions/test_set_results/evaluation_${TARGET_TEMP}K.json
```

### Predicting for a Single Protein FASTA at Multiple Temperatures 🌡️

```bash
# FASTA: my_protein.fasta (e.g., >1xyz_A\nSEQUENCE...)
MODEL_CKPT="models/best_model.pt"
INPUT_FASTA="my_protein.fasta"

for temp in 300 320 350 380; do
  python main.py predict \
    --config config.yaml \
    --model_checkpoint $MODEL_CKPT \
    --fasta_path $INPUT_FASTA \
    --temperature $temp \
    --output_dir predictions/1xyz_A_${temp}K
done
```

### Predicting with Uncertainty Estimation ⚠️

```bash
python main.py predict \
  --config config.yaml \
  --model_checkpoint models/best_model.pt \
  --fasta_path data/processed/test_sequences.fasta \
  --temperature_npy data/processed/test_temperatures.npy \
  --mc_samples 20 \
  --output_dir predictions/test_uncertainty
```
*(Output CSV will include an `uncertainty` column)*

---

## 🔧 Extending the Model

### Training on New Data 📚

1.  **Prepare CSV**: Format your data according to [Input Data Format](#-data-format).
2.  **Configure**: Update `data.raw_csv_path` in `config.yaml`. Enable only features present in your CSV under `data.features`. Set output dirs (`data.data_dir`, `output.model_dir`).
3.  **Process**: Run `python main.py process --config config.yaml`.
4.  **Train**: Run `python main.py train --config config.yaml`.

### Modifying the Architecture 🏗️

1.  **Edit `model.py`**: Change layers like `EnhancedTemperatureAwareESMModel`, `AttentionWithTemperature`, etc.
2.  **Configure**: Update `config.yaml` (`model.architecture`, `model.regression`) if you add new hyperparameters.
3.  **Train**: Run `python main.py train --config config.yaml`.

### Adding New Features ➕

1.  **CSV**: Add the new feature column.
2.  **Config**: Add `use_my_new_feature: true` under `data.features`.
3.  **`data_processor.py`**:
    -   Add feature name to `STRUCTURAL_FEATURES`.
    -   Update `extract_sequence_rmsf_temp_features` to read/impute it.
    -   Update `calculate_feature_normalization_params` & `save_split_data`.
4.  **`dataset.py`**: Ensure `EnhancedRMSFDataset` & `enhanced_collate_fn` handle it.
5.  **`model.py`**: Update `FeatureProcessor` or `forward` method to use the new feature tensor. Adjust layer dimensions if needed.

---

## 🛠️ Troubleshooting

### Common Issues

-   💥 **CUDA Out of Memory (OOM)**: Decrease `batch_size`, increase `accumulation_steps` (in `config.yaml`), use a smaller `esm_version`, filter long sequences (`max_length`).
-   🐢 **Slow Training**: Check GPU usage (`nvidia-smi`), consider a smaller `esm_version`, enable mixed-precision (default on CUDA), increase `num_workers` in DataLoader (more RAM needed).
-   📉 **Poor Performance**: Verify data processing (check `.npy` files, `.json` normalization values), tune hyperparameters (`learning_rate`, `weight_decay`, `hidden_dim`), check `extract_topology` function, try enabling/disabling features. Check plots (`training_metrics.png`).
-   ❌ **Prediction Failures / KeyErrors**: Check FASTA headers (need to be `instance_key` format if using `--temperature_npy`), ensure correct `.json` scaling/normalization files are found relative to the checkpoint, check feature `.npy` files exist and correspond to input FASTA keys.

---

## ❓ FAQs

**Q: How does the code use the temperature input?** 🌡️
A: Raw temperatures (Kelvin) are scaled (0-1) using the training range (`temp_scaling_params.json`). This scaled value becomes an input feature, often embedded into a vector (`TemperatureEncoding` in `model.py`) that influences the attention mechanism (`AttentionWithTemperature`).

**Q: How are the extra structural features handled?** 📊
A: Enabled in `config.yaml`, read from CSV (`data_processor.py`), NaNs filled using median, saved to `.npy` files, normalized using training stats (`feature_normalization.json`), then processed (`FeatureProcessor`) and combined with ESM embeddings in `model.py`.

**Q: How does the train/val/test split work?** 🧩
A: It's "topology-based" (`split_by_topology` in `data_processor.py`). It groups all data points (different temps) by protein structure ID. It then shuffles and splits these *groups* to prevent similar structures from appearing in different sets.

**Q: What's inside the `.npy` files in `data/processed/`?** 💾
A: Python dictionaries (keys = `instance_key`). Values are: RMSF arrays (`_rmsf.npy`), raw temperatures (`_temperatures.npy`), or feature arrays/values (`_{feature}.npy`).

**Q: How does batching handle different sequence lengths?** 📏
A: `create_enhanced_dataloader` uses "length bucketing." It groups similar-length sequences (`length_bucket_size` config) and creates batches within these groups, minimizing padding and speeding up training.

**Q: Can I predict without feature files?** 🤔
A: Only if the model checkpoint was trained *without* features (`use_enhanced_features: false` in its config). If trained *with* features, `predict.py` expects corresponding `.npy` files; missing ones might use defaults (zeros), potentially reducing accuracy.

**Q: How is 'uncertainty' calculated?** ⚠️
A: Via Monte Carlo Dropout (`--mc_samples > 1` in `predict`). The model runs multiple times with dropout active. `rmsf_pred` is the *average*, `uncertainty` is the *standard deviation* across these runs.

**Q: `domain_id` vs `instance_key`?** 🏷️
A: **`domain_id`**: Base protein identifier (e.g., `1xyz_A`) from raw CSV. **`instance_key`**: Unique key (`domain_id@temp.1f`, e.g., `1xyz_A@320.0`) created by `data_processor.py` linking domain + temperature; used in processed files and outputs.

---

<p align="center">
  <em>Developed with ❤️ by the DeepFlex Team</em><br>
  🔬 Questions? Contact: Felixburton2002@gmail.com
</p>
