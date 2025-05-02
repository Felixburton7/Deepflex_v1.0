# Temperature-Aware Protein Dynamics Prediction Suite

<div align="center">
  <!-- Optional: Add relevant high-level badges here -->
  <!-- [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) -->
  <!-- [![Build Status](...)](...) -->
</div>

---

## 🌟 Overview

This repository contains a collection of packages and tools focused on **predicting protein flexibility (Root Mean Square Fluctuation - RMSF)**, with a strong emphasis on **temperature awareness**. The goal is to understand how protein dynamics change under different thermal conditions using various machine learning and deep learning approaches.

The suite includes models that leverage different input data types and methodologies:

*   **Sequence-based models** using powerful language models like ESM.
*   **Feature-based models** combining sequence information with structural properties using traditional ML algorithms.
*   **Structure-based models** using 3D Convolutional Neural Networks (CNNs) on voxelized protein representations.

Associated tools for processing specific datasets (like mdCATH) and creating robust data splits are also included.

Each major component resides in its own directory and contains a detailed `README.md` file explaining its specific functionality, architecture, setup, and usage. This main README provides a high-level map to navigate the repository.

## 📂 Repository Structure & Component Overview

Below is a summary of the main packages and folders within this repository:

*   **[`deepflex/`](./deepflex/)**
    *   **Description:** The full **DeepFlex** model. Predicts temperature-dependent RMSF using **ESM-C embeddings**, combined with **structural features** (secondary structure, accessibility, etc.) and **temperature-aware attention mechanisms**. Uses a custom PyTorch implementation.
    *   **Input:** Processed CSV data containing sequence info, structural features, temperature, and target RMSF.
    *   ➡️ **See the [`deepflex/README.md`](./deepflex/README.md) for full details on its architecture, pipeline, and usage.**

*   **[`ensembleflex/`](./ensembleflex/)**
    *   **Description:** **EnsembleFlex** predicts temperature-dependent RMSF using **standard machine learning models** (Random Forest, Neural Network, LightGBM) trained on **aggregated tabular data (CSV)** where temperature is an explicit feature. Offers "Standard" and "OmniFlex" modes (OmniFlex incorporates external RMSF predictions like `esm_rmsf` or `voxel_rmsf` as features).
    *   **Input:** A single aggregated CSV file containing features, temperature, and target RMSF across multiple conditions.
    *   ➡️ **See the [`ensembleflex/README.md`](./simple_models/ensembleflex/README.md) for details on features, models, configuration, and the CLI.**

*   **[`voxelflex/`](./voxelflex/)** *(Note: Contains implementations of 3D CNN approaches)*
    *   **Description:** **VoxelFlex** predicts temperature-dependent RMSF directly from **3D voxelized protein structures (HDF5 format)** using **Convolutional Neural Networks (CNNs)**. Implementations include architectures like `MultipathRMSFNet`, `DenseNet3D`, and `DilatedResNet3D`. Features memory-efficient loading for large HDF5 datasets and incorporates temperature as a feature.
    *   **Input:** Voxel data (HDF5), RMSF data (CSV), and domain split files (TXT).
    *   ➡️ **See the detailed READMEs within the [`voxelflex/`](./voxelflex/) directory (or specific sub-folders if structured that way) for architecture specifics, the two-stage pipeline (preprocess/train), and configuration.**

*   **[`esm-flex/`](./esm-flex/)**
    *   **Description:** **ESM-Flex** predicts RMSF primarily using **ESM-3 embeddings** obtained via the Hugging Face `transformers` library. It features a simpler architecture consisting of a frozen ESM-3 model followed by a trainable regression head. *Temperature awareness might be implicit or require specific handling based on its implementation details.*
    *   **Input:** Sequence data (FASTA), potentially with corresponding target RMSF values from processed data.
    *   ➡️ **See the [`esm-flex/README.md`](./esm-flex/README.md) for setup and usage with ESM-3 models.**

*   **[`mdcath-data-loader-and-processor/`](./mdcath-data-loader-and-processor/)**
    *   **Description:** A utility package specifically designed for processing the **mdCATH** protein dynamics dataset. It likely handles extraction from simulations, structure cleaning, calculation of metrics (RMSF, accessibility, secondary structure, etc.), feature engineering, and potentially **voxelization** needed for models like VoxelFlex.
    *   **Purpose:** To convert raw mdCATH simulation data into ML-ready formats (CSV, HDF5) suitable for the predictive models in this repository.
    *   ➡️ **See the [`mdcath-data-loader-and-processor/README.md`](./mdcath-data-loader-and-processor/README.md) for its capabilities and usage.**

*   **[`mdcath-holdout-set-creator/`](./mdcath-holdout-set-creator/)**
    *   **Description:** Contains scripts and tools specifically designed to create robust holdout (test) or validation datasets, likely from the data processed by `mdcath-data-loader-and-processor`.
    *   **Purpose:** To ensure proper dataset splitting (potentially topology-aware, separating similar structures) for reliable model evaluation, preventing data leakage between training and testing phases.
    *   ➡️ A dedicated README within this folder likely provides instructions on generating these splits.

*   **[`models/`](./models/)**
    *   **Description:** A common default directory used by various pipelines (e.g., `deepflex`, `ensembleflex`) to store trained model checkpoints (e.g., `.pt`, `.pkl` files), logs, and potentially evaluation results or plots.

*   **[`trained_models/`](./trained_models/)**
    *   **Description:** Another directory intended for storing trained models. As it's listed in `.gitignore`, it's likely meant for local storage of potentially large model files that shouldn't be committed to the repository. This might hold specific experimental results or final production models.

---

## 🚀 Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
2.  **Set up Environment (Recommended):**
    Create a virtual environment (e.g., using `venv` or `conda`).
    ```bash
    python -m venv venv
    source venv/bin/activate # or venv\Scripts\activate on Windows
    # OR for conda: conda create -n proteinflex python=3.9 && conda activate proteinflex
    ```
3.  **Install Base Dependencies:**
    The root `requirements.txt` file likely contains common dependencies shared across packages.
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install Package-Specific Dependencies:**
    **Crucially**, each package (`deepflex/`, `ensembleflex/`, `voxelflex/`, `esm-flex/`) may have its own specific dependencies (and potentially its own `requirements.txt` or `setup.py`). **Navigate into the respective package directory and follow the installation instructions in its `README.md`**. This might involve steps like `pip install -e .` if they are set up as installable packages. Pay close attention to PyTorch/CUDA version compatibility if required.

5.  **Explore Packages:** Consult the individual `README.md` files within each package directory for detailed setup, configuration, data format requirements, and specific usage examples.

## 🤝 Contributing

[Optional: Add contribution guidelines here or link to a CONTRIBUTING.md file.]
We welcome contributions! Please read our `CONTRIBUTING.md` (if available) for details on our code of conduct and the process for submitting pull requests.

## 📜 License

[Optional: Specify your license here or link to a LICENSE file.]
This project is licensed under the MIT License - see the `LICENSE` file for details.

---

<p align="center">
  <em>A comprehensive suite for exploring temperature-dependent protein dynamics.</em><br>
  🔬 Questions? Contact: Felixburton2002@gmail.com
</p>

4.  Clarifies the likely purpose of `models/` vs. `trained_models/`.
5.  Emphasizes the critical step of checking package-specific READMEs for dependencies and detailed setup.
