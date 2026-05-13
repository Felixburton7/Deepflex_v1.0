<!-- Badges (Pocketeer-style) -->
<p align="left">

  <!-- Core stack -->
  <img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">

  <!-- Scientific Python -->
  <img src="https://img.shields.io/badge/Numerics-NumPy-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Data-Pandas-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Scientific-SciPy-8CAAE6?logo=scipy&logoColor=white" alt="SciPy">

  <!-- Bio / structure ecosystem -->
  <img src="https://img.shields.io/badge/BioPython-1f8f5f" alt="BioPython">
  <img src="https://img.shields.io/badge/Visualisation-PyMOL-0b7285" alt="PyMOL">

  <!-- ESM / Transformers -->
  <a href="https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12">
    <img src="https://img.shields.io/badge/pLM-ESM--C-6f42c1" alt="ESM-C">
  </a>
  <img src="https://img.shields.io/badge/NLP-Transformers-ffd21e?logo=huggingface&logoColor=black" alt="Transformers">
  <img src="https://img.shields.io/badge/Hub-Hugging%20Face-ffd21e?logo=huggingface&logoColor=black" alt="Hugging Face">

  <!-- Dataset -->
  <img src="https://img.shields.io/badge/Dataset-mdCATH-0b7285" alt="mdCATH">
</p>

# DeepFlex: Deep Learning for Protein Flexibility Prediction

<!-- DeepFlex logo (now wired up) -->
<!-- <p align="center">
  <img src="https://github.com/Felixburton7/Deepflex_v1.0/blob/main/DeepFlex%20logo.png?raw=1" alt="DeepFlex Logo" width="550">
</p> -->


<div style="background-color: #f1f8ff; border: 1px solid #c8e1ff; border-radius: 6px; padding: 15px; margin-bottom: 25px;">
  <strong>📢 Project Status:</strong> This repository contains the complete codebase DeepFlex. This project won the <a href="https://www.biochemistry.org/grants-and-awards/grants-and-bursaries/undergraduate-recognition-awards/current-awardees/" target="_blank">UK Undergraduate Biochemical Society Award</a> and achieved the highest grade of the year (2025). The core architecture is currently being refactored into a final, publication-ready package. 
</div>


### Key Contributions:
This project introduces **DeepFlex** a **temperature-aware Deep Learning framework** capable of predicting per-residue Root Mean Square Fluctuation (RMSF) profiles several orders of magnitude than traditional simulations. It was trained on [mdCATH dataset](https://huggingface.co/datasets/compsciencelab/mdCATH). 

*   **Flexibility Prediction:** Replaces expensive MD simulations with rapid inference, achieving high correlation with ground-truth trajectory data.
*   **Temperature as an input:** Unlike standard B-factor predictors, DeepFlex explicitly models **temperature dependence**, allowing users to probe protein flexibility across different thermal conditions.
*   **Architecture:** Takes **Protein Language Models (<a href="https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12" target="_blank">ESM-C</a>)** with geometric structural features and attention mechanisms to capture long-range allosteric effects.

<!-- RMSF formula (rendered as an image so it ALWAYS works on GitHub) -->
<div style="background-color: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px; padding: 12px 15px; margin: 0 0 25px 0;">
  <strong>RMSF definition</strong> (per residue <em>i</em>):<br><br>

  <p align="center" style="margin: 0;">
    <img
      src="https://latex.codecogs.com/svg.image?\mathrm{RMSF}_i=\sqrt{\left\langle\left\|\mathbf{r}_i(t)-\left\langle\mathbf{r}_i\right\rangle\right\|^2\right\rangle_t}"
      alt="RMSF_i = sqrt( < || r_i(t) - <r_i> ||^2 >_t )"
    />
  </p>

  <div style="color:#586069; font-size: 0.9em; margin-top: 10px;">
    where <b>r</b><sub>i</sub>(t) is the position vector of residue <i>i</i> at time <i>t</i>, and ⟨<b>r</b><sub>i</sub>⟩ is its time-averaged position.
  </div>
</div>

## 📂 Repository Architecture & Component Overview

This codebase is structured into three primary domains: **Core Architecture** (DeepFlex Model), **Data Engineering** (Processing & Validation), and **Comparative Benchmarks** (Baseline Models).

<div style="display: grid; grid-template-columns: 1fr; gap: 20px; margin-bottom: 30px;">

<!-- SECTION 1: MAIN MODEL -->
<div style="border-left: 5px solid #6236FF; background-color: #f8f9fa; padding: 15px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
  <h3 style="margin-top: 0; display: flex; align-items: center;">
    <a href="./deepflex/" style="text-decoration: none; color: #0366d6;"><code>deepflex/</code></a> 
    <span style="font-size: 0.8em; color: #586069; font-weight: normal; margin-left: auto;">(Architecture)</span>
  </h3>

  <div style="margin: 15px 0; border: 1px solid #e1e4e8; border-radius: 4px; overflow: hidden;">
    <img src="https://raw.githubusercontent.com/Felixburton7/Deepflex_v1.0/main/DeepFlex.png" alt="DeepFlex Model Architecture" style="width: 100%; display: block;">
  </div>

  <ul style="margin-bottom: 5px; padding-left: 20px; color: #444;">
    <li><strong>The Core Model:</strong> Integrates <strong>ESM-C embeddings</strong> with geometric features using a novel <strong>temperature-aware attention mechanism</strong>.</li>
    <li><em>Input:</em> Pre-processed feature vectors (CSV).</li>
    <li><em>Documentation:</em> <strong><a href="./deepflex/README.md">View Architecture Details</a></strong></li>
  </ul>
</div>

<!-- SECTION 2: DATA PIPELINES -->
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
  
  <div style="border-left: 5px solid #28a745; background-color: #fff; border: 1px solid #e1e4e8; border-left-width: 5px; padding: 15px; border-radius: 4px;">
    <h4 style="margin-top: 0;">
      <a href="./mdcath-data-loader-and-processor/" style="color: #24292e;"><code>mdcath-processor/</code></a>
      <span style="float: right;">🔧</span>
    </h4>
    <p style="font-size: 0.9em; margin-bottom: 5px; color: #586069;"><strong>Data Engineering</strong></p>
    <ul style="font-size: 0.9em; padding-left: 15px; margin-bottom: 0;">
      <li>Processes raw <strong>mdCATH</strong> simulations.</li>
      <li>Handles voxelization and HDF5 conversion.</li>
      <li><a href="./mdcath-data-loader-and-processor/README.md">View Docs</a></li>
    </ul>
  </div>

  <div style="border-left: 5px solid #28a745; background-color: #fff; border: 1px solid #e1e4e8; border-left-width: 5px; padding: 15px; border-radius: 4px;">
    <h4 style="margin-top: 0;">
      <a href="./mdcath-holdout-set-creator/" style="color: #24292e;"><code>holdout-creator/</code></a>
      <span style="float: right;">🔧</span>
    </h4>
    <p style="font-size: 0.9em; margin-bottom: 5px; color: #586069;"><strong>Validation Strategy</strong></p>
    <ul style="font-size: 0.9em; padding-left: 15px; margin-bottom: 0;">
      <li>Creates rigorous train/test splits.</li>
      <li>Ensures topological distinctness (CATH separation).</li>
      <li><a href="./mdcath-holdout-set-creator/README.md">View Docs</a></li>
    </ul>
  </div>

</div>

<!-- SECTION 3: EXPERIMENTAL MODELS -->
<div style="border-left: 5px solid #fd7e14; background-color: #fff8f0; padding: 15px; border-radius: 4px; border: 1px solid #fae3cc; border-left-width: 5px;">
  <h3 style="margin-top: 0;">
    <a href="./models/" style="text-decoration: none; color: #cf5a02;"><code>models/</code></a> 
    <span style="font-size: 0.8em; color: #586069; font-weight: normal;">(Benchmarks & Ablations)</span>
  </h3>
  
  <p style="font-size: 0.95em; color: #444;">Comprehensive library of baseline architectures used for comparative analysis.</p>
  
  <!-- Sub-Model 1 -->
  <div style="background-color: white; border: 1px solid #eee; padding: 15px; margin-bottom: 10px; border-radius: 4px;">
    <h5 style="margin: 0 0 8px 0;">
      <a href="./models/simple_models/">📊 <code>simple_models/</code> (EnsembleFlex)</a>
    </h5>
    <p style="font-size: 0.85em; margin: 0 0 5px 0; color: #666;">
      A suite of classical machine learning regressors operating on aggregated biophysical features:
    </p>
    <ul style="font-size: 0.85em; color: #444; padding-left: 20px; margin-bottom: 0;">
       <li><strong>Random Forest Regressors:</strong> High-dimensional ensemble learning.</li>
       <li><strong>LightGBM (Gradient Boosting):</strong> Efficient, tree-based gradient boosting frameworks.</li>
       <li><strong>Tabular Neural Networks:</strong> Deep feed-forward networks for structured data.</li>
       <li><strong>OmniFlex:</strong> A meta-learning architecture aggregating predictions from the above.</li>
    </ul>
  </div>
  
  <!-- Sub-Model 2 -->
<div style="background-color: white; border: 1px solid #eee; padding: 15px; margin-bottom: 10px; border-radius: 4px;">
    <h5 style="margin: 0 0 8px 0;">
      <a href="./models/voxel_models/">🧊 <code>voxel_models/</code> (VoxelFlex)</a>
    </h5>

<div style="margin: 10px 0; border: 1px solid #eee; border-radius: 3px; overflow: hidden;">
<img src="https://raw.githubusercontent.com/Felixburton7/Deepflex_v1.0/main/VoxelFlex.png" alt="VoxelFlex Input Representation" style="width: 100%; display: block;">
</div>

<p style="font-size: 0.85em; margin: 0 0 5px 0; color: #666;">
End-to-end Deep Learning using geometric protein representations:
</p>
<ul style="font-size: 0.85em; color: #444; padding-left: 20px; margin-bottom: 0;">
<li><strong>3D Convolutional Neural Networks (3D-CNNs):</strong> Learning spatial flexibility features directly from HDF5 voxel grids.</li>
<li><strong>Multi-Temperature Architectures:</strong> Handling thermodynamic variability in inputs.</li>
</ul>

  </div>

  
  <!-- Sub-Model 3 -->
  <div style="background-color: white; border: 1px solid #eee; padding: 15px; margin-bottom: 0; border-radius: 4px;">
    <h5 style="margin: 0 0 8px 0;">
      <a href="./models/esm_models/">🧬 <code>esm_models/</code> (ESM-Flex)</a>
    </h5>
    <p style="font-size: 0.85em; margin: 0 0 5px 0; color: #666;">
      Pure sequence-based approaches leveraging Large Protein Language Models (pLMs):
    </p>
     <ul style="font-size: 0.85em; color: #444; padding-left: 20px; margin-bottom: 0;">
       <li><strong>ESM-3 Embeddings:</strong> Utilizing state-of-the-art transformer representations.</li>
       <li><strong>MLP Projection Heads:</strong> Direct dimensionality reduction and regression.</li>
       <li><strong>LoRA (Low-Rank Adaptation):</strong> Parameter-efficient fine-tuning of large transformer weights.</li>
    </ul>
  </div>
</div>

<!-- SECTION 4: STORAGE -->
<div style="border-left: 5px solid #6c757d; background-color: #f6f8fa; padding: 10px; border-radius: 4px; opacity: 0.8;">
  <h4 style="margin: 0; font-size: 0.9em;">
    <a href="./trained_models/" style="color: #24292e; text-decoration: none;">💾 <code>trained_models/</code></a> <span style="font-weight: normal; font-size: 0.9em;">(Local Storage)</span>
  </h4>
  <p style="font-size: 0.8em; margin: 0; padding-left: 20px; color: #586069;">
    Directory for large model checkpoints and experimental artifacts (excluded from git tracking).
  </p>
</div>

</div>

<hr style="border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.)); margin: 30px 0;">
