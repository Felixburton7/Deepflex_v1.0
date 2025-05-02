## 📂 Repository Structure & Component Overview

Below is a summary of the main packages and folders within this repository:

*   **[`deepflex/`](./deepflex/)**
    *   **Description:** The current top **DeepFlex** model. Predicts temperature-dependent RMSF using **ESM-C embeddings**, combined with **structural features** (secondary structure, accessibility, etc.) and **temperature-aware attention mechanisms**. Uses a custom PyTorch implementation.
    *   **Input:** Processed CSV data containing sequence info, structural features, temperature, and target RMSF.
    *   ➡️ **See the [`deepflex/README.md`](./deepflex/README.md) for full details on its architecture, pipeline, and usage.**

*   **[`models/simple_models/`](./models/simple_models/)** *(Contains EnsembleFlex (AKA Random Forest, Light Gradient boost, AND a Classifier Model)*
    *   **Description:** **EnsembleFlex** predicts temperature-dependent RMSF using **standard machine learning models** (Random Forest, Neural Network, LightGBM) trained on **aggregated tabular data (CSV)** where temperature is an explicit feature. Offers "Standard" and "OmniFlex" modes (OmniFlex incorporates external RMSF predictions like `esm_rmsf` or `voxel_rmsf` as features). The code resides within the `ensembleflex/` sub-directory here.
    *   **Input:** A single aggregated CSV file containing features, temperature, and target RMSF across multiple conditions.
    *   ➡️ **See the [`models/simple_models/README.md`](./models/simple_models/README.md) for details on features, models, configuration, and the CLI.**

*   **[`models/voxel_models/`](./models/voxel_models/)** *(Contains VoxelFlex implementations)*
    *   **Description:** **VoxelFlex** predicts temperature-dependent RMSF directly from **3D voxelized protein structures (HDF5 format)** using **Convolutional Neural Networks (CNNs)**. Contains different implementations (e.g., `VoxelFlex_Multi-Temperature`, `VoxelFlex-Single-Temperature`) with architectures like `MultipathRMSFNet`, `DenseNet3D`, and `DilatedResNet3D`. Features memory-efficient loading for large HDF5 datasets and incorporates temperature as a feature.
    *   **Input:** Voxel data (HDF5), RMSF data (CSV), and domain split files (TXT).
    *   ➡️ **Navigate into [`models/voxel_models/`](./models/voxel_models/) and see the README files within specific implementation folders (e.g., `VoxelFlex_Multi-Temperature/README.md`) for architecture specifics, pipeline details, and configuration.**

*   **[`models/esm_models/`](./models/esm_models/)** *(Contains ESM-Flex-ONLY implementations)*
    *   **Description:** **ESM-Flex** predicts RMSF primarily using **ESM embeddings** (e.g., ESM-3) obtained via the Hugging Face `transformers` library. Contains different fine-tuning approaches like using a simple MLP head (`esm-flex-MLP/`) or LoRA (`esm-flex-LoRA/`).
    *   **Input:** Sequence data (FASTA), potentially with corresponding target RMSF values from processed data.
    *   ➡️ **Navigate into [`models/esm_models/`](./models/esm_models/) and see the README files within specific implementation folders (e.g., `esm-flex-MLP/README.md`) for setup and usage.**

*   **[`mdcath-data-loader-and-processor/`](./mdcath-data-loader-and-processor/)**
    *   **Description:** A utility package specifically designed for processing the **mdCATH** protein dynamics dataset. It handles extraction from simulations, structure cleaning, calculation of metrics (RMSF, accessibility, secondary structure, etc.), feature engineering, and  **voxelization** needed for models like VoxelFlex.
    *   **Purpose:** To convert raw mdCATH simulation data into ML-ready formats (CSV, HDF5) suitable for the predictive models in this repository.
    *   ➡️ **See the [`mdcath-data-loader-and-processor/README.md`](./mdcath-data-loader-and-processor/README.md) for its capabilities and usage.**

*   **[`mdcath-holdout-set-creator/`](./mdcath-holdout-set-creator/)**
    *   **Description:** Contains scripts and tools specifically designed to create robust holdout (test).
    *   **Purpose:** To ensure proper dataset splitting ( topology-aware, separating similar structures) for reliable model evaluation, preventing data leakage between training and testing phases.
    *   ➡️ **See the [`mdcath-holdout-set-creator/README.md`](./mdcath-holdout-set-creator/README.md) for instructions on generating these splits.**

*   **[`models/`](./models/)**
    *   **Description:** The primary directory containing various model implementations (`simple_models`, `voxel_models`, `esm_models`). 

---
