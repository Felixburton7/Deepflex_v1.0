# 🧬 FlexTemp: Multi-Model Temperature-Aware Protein Flexibility Prediction

## 📂 Repository Structure & Component Overview

<div style="background-color: #f8f9fa; border-radius: 8px; padding: 20px; margin-bottom: 25px;">
This repository brings together several approaches for predicting protein flexibility (RMSF) with temperature awareness. Here's a quick guide to the main components:
</div>

<div style="display: grid; grid-template-columns: 1fr; gap: 20px; margin-bottom: 30px;">

<div style="border-left: 4px solid #6236FF; padding-left: 15px; background-color: #f5f5ff; padding: 15px; border-radius: 5px;">
  <h3 style="margin-top: 0;">
    <a href="./deepflex/"><code>deepflex/</code></a> 
    <span style="vertical-align: middle; margin-left: 5px;">🧠</span> 
    <span style="font-size: 0.9em; color: #6236FF;">(Attention + ESM + Features)</span>
  </h3>
  <ul style="margin-bottom: 5px;">
    <li>The original <strong>DeepFlex</strong> model using <strong>ESM-C embeddings</strong>, structural features, and <strong>temperature-aware attention</strong>.</li>
    <li><em>Input:</em> Processed CSV data.</li>
    <li><em>Details:</em> <strong>See the <a href="./deepflex/README.md"><code>deepflex/README.md</code></a></strong></li>
  </ul>
</div>

<div style="border-left: 4px solid #28a745; padding-left: 15px; background-color: #f0fff4; padding: 15px; border-radius: 5px;">
  <h3 style="margin-top: 0;">
    <a href="./mdcath-data-loader-and-processor/"><code>mdcath-data-loader-and-processor/</code></a> 
    <span style="vertical-align: middle; margin-left: 5px;">🔧</span> 
    <span style="font-size: 0.9em; color: #28a745;">(Data Utility)</span>
  </h3>
  <ul style="margin-bottom: 5px;">
    <li>A utility package for processing the <strong>mdCATH dataset</strong> – handles simulations, structures, feature calculation, and potentially <strong>voxelization</strong>.</li>
    <li><em>Purpose:</em> Prepares mdCATH data into formats usable by the models (CSV, HDF5).</li>
    <li><em>Details:</em> <strong>See the <a href="./mdcath-data-loader-and-processor/README.md"><code>mdcath-data-loader-and-processor/README.md</code></a></strong></li>
  </ul>
</div>

<div style="border-left: 4px solid #28a745; padding-left: 15px; background-color: #f0fff4; padding: 15px; border-radius: 5px;">
  <h3 style="margin-top: 0;">
    <a href="./mdcath-holdout-set-creator/"><code>mdcath-holdout-set-creator/</code></a> 
    <span style="vertical-align: middle; margin-left: 5px;">🔧</span> 
    <span style="font-size: 0.9em; color: #28a745;">(Data Utility)</span>
  </h3>
  <ul style="margin-bottom: 5px;">
    <li>Tools for creating robust <strong>holdout/validation sets</strong> from processed data, likely ensuring proper splitting (e.g., by topology).</li>
    <li><em>Purpose:</em> Ensures reliable model evaluation.</li>
    <li><em>Details:</em> <strong>See the <a href="./mdcath-holdout-set-creator/README.md"><code>mdcath-holdout-set-creator/README.md</code></a></strong></li>
  </ul>
</div>

<div style="border-left: 4px solid #fd7e14; padding-left: 15px; background-color: #fff8f0; padding: 15px; border-radius: 5px;">
  <h3 style="margin-top: 0;">
    <a href="./models/"><code>models/</code></a> 
    <span style="vertical-align: middle; margin-left: 5px;">📂</span> 
    <span style="font-size: 0.9em; color: #fd7e14;">(Model Implementations)</span>
  </h3>
  
  <p>The central directory containing all model code, categorized as follows:</p>
  
  <div style="margin-left: 15px; border-left: 3px solid #6236FF; padding-left: 15px; margin-bottom: 15px; background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
    <h4 style="margin-top: 0;">
      <a href="./models/simple_models/"><code>simple_models/</code></a> 
      <span style="vertical-align: middle; margin-left: 5px;">🧠</span> 
      <span style="font-size: 0.9em; color: #6236FF;">(EnsembleFlex: RF, LGBM, NN)</span>
    </h4>
    <ul style="margin-bottom: 5px;">
      <li>Uses <strong>standard ML models</strong> (Random Forest, LightGBM, Neural Net) on <strong>aggregated tabular features</strong> (CSV). Temperature is an input feature. Includes "OmniFlex" mode using external predictions.</li>
      <li><em>Input:</em> Aggregated CSV.</li>
      <li><em>Details:</em> <strong>See the <a href="./models/simple_models/README.md"><code>models/simple_models/README.md</code></a></strong></li>
    </ul>
  </div>
  
  <div style="margin-left: 15px; border-left: 3px solid #6236FF; padding-left: 15px; margin-bottom: 15px; background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
    <h4 style="margin-top: 0;">
      <a href="./models/voxel_models/"><code>voxel_models/</code></a> 
      <span style="vertical-align: middle; margin-left: 5px;">🧠</span> 
      <span style="font-size: 0.9em; color: #6236FF;">(VoxelFlex: 3D CNNs)</span>
    </h4>
    <ul style="margin-bottom: 5px;">
      <li>Uses <strong>3D Convolutional Neural Networks</strong> to predict RMSF directly from <strong>voxelized protein structures (HDF5)</strong>. Includes multi-temperature and single-temperature implementations.</li>
      <li><em>Input:</em> HDF5 Voxels, CSV RMSF data, TXT domain splits.</li>
      <li><em>Details:</em> <strong>Explore <a href="./models/voxel_models/"><code>models/voxel_models/</code></a></strong> - Check READMEs inside specific folders (e.g., <code>VoxelFlex_Multi-Temperature/README.md</code>).</li>
    </ul>
  </div>
  
  <div style="margin-left: 15px; border-left: 3px solid #6236FF; padding-left: 15px; margin-bottom: 0; background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
    <h4 style="margin-top: 0;">
      <a href="./models/esm_models/"><code>esm_models/</code></a> 
      <span style="vertical-align: middle; margin-left: 5px;">🧠</span> 
      <span style="font-size: 0.9em; color: #6236FF;">(ESM-Flex: ESM Embeddings)</span>
    </h4>
    <ul style="margin-bottom: 5px;">
      <li>Uses <strong>ESM embeddings (e.g., ESM-3)</strong> via Hugging Face <code>transformers</code> as the primary input. Includes MLP and LoRA fine-tuning approaches.</li>
      <li><em>Input:</em> FASTA sequences.</li>
      <li><em>Details:</em> <strong>Explore <a href="./models/esm_models/"><code>models/esm_models/</code></a></strong> - Check READMEs inside specific folders (e.g., <code>esm-flex-MLP/README.md</code>).</li>
    </ul>
  </div>
</div>

<div style="border-left: 4px solid #17a2b8; padding-left: 15px; background-color: #f0f8ff; padding: 15px; border-radius: 5px;">
  <h3 style="margin-top: 0;">
    <a href="./trained_models/"><code>trained_models/</code></a> 
    <span style="vertical-align: middle; margin-left: 5px;">💾</span> 
    <span style="font-size: 0.9em; color: #17a2b8;">(Local Model Storage)</span>
  </h3>
  <ul style="margin-bottom: 5px;">
    <li>Directory for storing <strong>trained model files locally</strong> (ignored by Git). Suitable for large checkpoints or specific experimental results not intended for version control.</li>
  </ul>
</div>

</div>

<hr style="height: 3px; background: linear-gradient(to right, #6236FF, #28a745, #fd7e14, #17a2b8); border: none; border-radius: 2px; margin: 30px 0;">
