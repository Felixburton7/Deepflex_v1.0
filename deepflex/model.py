import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import LogitsConfig, ESMProtein
except ImportError:
    logger.error("Failed to import from 'esm' library. Please install `pip install esm`.", exc_info=True)
    raise

class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequences.
    Based on the sine/cosine encoding from "Attention Is All You Need".
    """
    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but part of the module's state)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model] or [seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        if x.dim() == 3:  # [seq_len, batch_size, d_model]
            seq_len = x.size(0)
            x = x + self.pe[:seq_len, :]
        elif x.dim() == 2:  # [seq_len, d_model]
            seq_len = x.size(0)
            x = x + self.pe[:seq_len, :]
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D")
            
        return self.dropout(x)

class TemperatureEncoding(nn.Module):
    """
    Advanced temperature encoding module that creates a richer
    representation of temperature for integration with protein features.
    """
    def __init__(self, embedding_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(p=dropout)
        
        # Project scalar temperature to a higher dimensional space
        self.temp_projection = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        Convert scalar temperature values to rich embeddings.
        
        Args:
            temperature: Tensor of shape [batch_size] containing scaled temperature values
            
        Returns:
            Tensor of shape [batch_size, embedding_dim]
        """
        # Reshape to [batch_size, 1]
        temp = temperature.unsqueeze(1)
        
        # Project to higher dimension
        temp_embedding = self.temp_projection(temp)
        
        return self.dropout(temp_embedding)

class FeatureProcessor(nn.Module):
    """
    Process structural features before combining with ESM embeddings.
    """
    def __init__(self, feature_dims: Dict[str, int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # Create linear projections for each feature
        self.projections = nn.ModuleDict()
        
        # Process each feature type
        for feature_name, feature_dim in feature_dims.items():
            # Create projection for this feature
            self.projections[feature_name] = nn.Sequential(
                nn.Linear(feature_dim, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout)
            )
        
        # Final feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(output_dim * len(feature_dims), output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process and combine structural features.
        
        Args:
            features: Dictionary of feature tensors or lists of tensors
            
        Returns:
            Tensor of processed features [batch_size, seq_len, output_dim]
        """
        # Process each feature and collect
        processed_features = []
        
        for feature_name, projection in self.projections.items():
            if feature_name in features:
                feature_data = features[feature_name]
                
                # Process this feature
                processed = projection(feature_data)
                processed_features.append(processed)
            else:
                logger.warning(f"Feature {feature_name} not found in input features")
        
        # Combine all features
        if not processed_features:
            raise ValueError("No valid features were processed")
            
        # Concatenate feature tensors along the feature dimension
        combined = torch.cat(processed_features, dim=-1)
        
        # Final fusion
        return self.feature_fusion(combined)

class AttentionWithTemperature(nn.Module):
    """
    Multi-head self-attention module with temperature conditioning.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, temp_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.temp_dim = temp_dim
        
        # Temperature conditioning for attention
        self.temp_to_qkv = nn.Linear(temp_dim, 3 * embed_dim)
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, 
                x: torch.Tensor, 
                temp_embedding: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply temperature-conditioned self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            temp_embedding: Temperature embedding [batch_size, temp_dim]
            key_padding_mask: Optional mask for padding positions
            
        Returns:
            Attended tensor [batch_size, seq_len, embed_dim]
        """
        # Apply layer norm first (pre-norm formulation)
        residual = x
        x = self.norm(x)
        
        # Temperature conditioning
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Generate QKV biases from temperature
        temp_qkv_bias = self.temp_to_qkv(temp_embedding)  # [batch_size, 3*embed_dim]
        temp_q, temp_k, temp_v = torch.chunk(temp_qkv_bias, 3, dim=-1)
        
        # Apply temperature biases to the input before attention
        # This effectively conditions attention on temperature
        q_bias = temp_q.unsqueeze(1).expand(-1, seq_len, -1)
        k_bias = temp_k.unsqueeze(1).expand(-1, seq_len, -1)
        v_bias = temp_v.unsqueeze(1).expand(-1, seq_len, -1)
        
        q = x + q_bias
        k = x + k_bias
        v = x + v_bias
        
        # Multi-head attention
        attn_output, _ = self.self_attn(q, k, v, key_padding_mask=key_padding_mask)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Residual connection
        return output + residual

class EnhancedTemperatureAwareESMModel(nn.Module):
    """
    Enhanced ESM-C based model for Temperature-Aware RMSF prediction.

    Uses ESM-C embeddings with structural features, attention mechanism,
    and improved temperature integration.
    """
    def __init__(self,
                 esm_model_name: str = "esmc_600m",
                 regression_hidden_dim: int = 128,
                 regression_dropout: float = 0.1,
                 temp_embedding_dim: int = 16,
                 use_attention: bool = True,
                 attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 use_positional_encoding: bool = True,
                 use_enhanced_features: bool = True,
                 improved_temp_integration: bool = True):
        super().__init__()

        logger.info(f"Initializing EnhancedTemperatureAwareESMModel...")
        logger.info(f"Loading base ESM-C Model: {esm_model_name}")
        try:
            # Load the base ESMC model object
            self.esm_model = ESMC.from_pretrained(esm_model_name)
        except Exception as e:
            logger.error(f"Failed to load ESM-C model '{esm_model_name}'. Error: {e}")
            raise

        self.esm_model.eval()  # Set base model to evaluation mode
        self.esm_model_name = esm_model_name

        # --- Freeze ESM-C parameters ---
        logger.info("Freezing ESM-C model parameters...")
        for param in self.esm_model.parameters():
            param.requires_grad = False

        # --- Detect embedding dimension ---
        # Do a dummy forward pass on CPU first to avoid moving large model prematurely
        temp_cpu_model = ESMC.from_pretrained(esm_model_name)
        embedding_dim = -1
        try:
            with torch.no_grad():
                test_protein = ESMProtein(sequence="A")  # Minimal sequence
                encoded = temp_cpu_model.encode(test_protein)
                logits_output = temp_cpu_model.logits(
                    encoded, LogitsConfig(sequence=True, return_embeddings=True)
                )
                embedding_dim = logits_output.embeddings.size(-1)
                logger.info(f"Detected ESM embedding dimension: {embedding_dim}")
        except Exception as e:
            logger.error(f"Error during embedding dimension detection: {e}")
            raise ValueError(f"Could not determine embedding dimension for {esm_model_name}.")
        finally:
            del temp_cpu_model  # Clean up temporary model

        if embedding_dim <= 0:
            raise ValueError("Failed to detect a valid embedding dimension.")

        self.esm_hidden_dim = embedding_dim
        
        # Store configuration
        self.use_attention = use_attention
        self.use_positional_encoding = use_positional_encoding
        self.use_enhanced_features = use_enhanced_features
        self.improved_temp_integration = improved_temp_integration
        
        # --- Advanced Temperature Integration ---
        if improved_temp_integration:
            self.temp_encoder = TemperatureEncoding(
                embedding_dim=temp_embedding_dim,
                dropout=regression_dropout
            )
            temp_conditioning_dim = temp_embedding_dim
        else:
            # Simple temperature feature
            temp_conditioning_dim = 1
        
        # --- Positional Encoding ---
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                d_model=embedding_dim,
                dropout=regression_dropout
            )
        
        # --- Feature Processing ---
        if use_enhanced_features:
            # Define feature dimensions
            # Default dimensions for common features (can be customized)
            self.feature_dims = {
                'normalized_resid': 1,
                'core_exterior_encoded': 1,
                'secondary_structure_encoded': 1,
                'relative_accessibility': 1,
                'phi_norm': 1,
                'psi_norm': 1,
                'voxel_rmsf': 1,
                'bfactor_norm': 1
            }
            
            feature_output_dim = min(64, embedding_dim // 2)  # Make it smaller than ESM embeddings
            
            self.feature_processor = FeatureProcessor(
                feature_dims=self.feature_dims,
                output_dim=feature_output_dim,
                dropout=regression_dropout
            )
            
            # Combine ESM embeddings with processed features
            combined_dim = embedding_dim + feature_output_dim
            
            self.feature_esm_fusion = nn.Sequential(
                nn.Linear(combined_dim, embedding_dim),
                nn.GELU(),
                nn.LayerNorm(embedding_dim),
                nn.Dropout(regression_dropout)
            )
        else:
            combined_dim = embedding_dim
        
        # --- Attention Mechanism ---
        if use_attention:
            self.attention = AttentionWithTemperature(
                embed_dim=embedding_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
                temp_dim=temp_conditioning_dim
            )
        
        # --- Create Temperature-Aware Regression Head ---
        regression_input_dim = embedding_dim
        
        logger.info(f"Creating regression head. Input dimension: {regression_input_dim}")

        if regression_hidden_dim > 0:
            self.regression_head = nn.Sequential(
                nn.LayerNorm(regression_input_dim),
                nn.Linear(regression_input_dim, regression_hidden_dim),
                nn.GELU(),
                nn.Dropout(regression_dropout),
                nn.Linear(regression_hidden_dim, 1)  # Output single RMSF value
            )
            logger.info(f"Using MLP regression head (LayerNorm -> Linear({regression_input_dim},{regression_hidden_dim}) -> GELU -> Dropout -> Linear({regression_hidden_dim},1))")
        else:  # Direct linear layer after LayerNorm
            self.regression_head = nn.Sequential(
                nn.LayerNorm(regression_input_dim),
                nn.Dropout(regression_dropout),
                nn.Linear(regression_input_dim, 1)
            )
            logger.info(f"Using Linear regression head (LayerNorm -> Dropout -> Linear({regression_input_dim},1))")

        self._log_parameter_counts()
        logger.info("EnhancedTemperatureAwareESMModel initialized successfully.")

    def _log_parameter_counts(self):
        total_params = sum(p.numel() for p in self.parameters())
        esm_params = sum(p.numel() for p in self.esm_model.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Parameter Counts:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  ESM-C parameters (frozen): {esm_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        if total_params > 0:
            logger.info(f"  Trainable percentage: {trainable_params/total_params:.4%}")

    def _process_protein_features(self, 
                                  batch_features: Dict[str, Any], 
                                  sequences: List[str], 
                                  device: torch.device) -> List[torch.Tensor]:
        """
        Process and align structural features to sequence lengths.
        
        Args:
            batch_features: Dictionary of feature tensors from dataloader
            sequences: List of sequence strings
            device: Device to place tensors on
            
        Returns:
            List of processed feature tensors aligned to sequence lengths
        """
        # Initialize list of per-sequence feature tensors
        processed_features = []
        batch_size = len(sequences)
        
        for i in range(batch_size):
            seq_length = len(sequences[i])
            seq_features = {}
            
            # Process each feature type
            for feature_name, feature_values in batch_features.items():
                if isinstance(feature_values, list):
                    # Per-residue feature
                    if i < len(feature_values):
                        feature_tensor = feature_values[i]
                        
                        # Align length with sequence if needed
                        if feature_tensor.size(0) > 0:  # Non-empty tensor
                            if feature_tensor.size(0) != seq_length:
                                # Trim or pad as needed
                                if feature_tensor.size(0) > seq_length:
                                    # Trim
                                    feature_tensor = feature_tensor[:seq_length]
                                else:
                                    # Pad with zeros
                                    padding = torch.zeros(seq_length - feature_tensor.size(0), 
                                                        dtype=feature_tensor.dtype, 
                                                        device=feature_tensor.device)
                                    feature_tensor = torch.cat([feature_tensor, padding], dim=0)
                            
                            # Ensure feature is on the correct device
                            feature_tensor = feature_tensor.to(device)
                            seq_features[feature_name] = feature_tensor
                        else:
                            # Create zero tensor for empty features
                            seq_features[feature_name] = torch.zeros(seq_length, dtype=torch.float32, device=device)
                    else:
                        # Create zero tensor if feature is missing for this sequence
                        seq_features[feature_name] = torch.zeros(seq_length, dtype=torch.float32, device=device)
                else:
                    # Global feature - replicate across sequence length
                    if feature_values.size(0) > i:
                        value = feature_values[i].to(device)
                        seq_features[feature_name] = value.expand(seq_length)
                    else:
                        # Create zero tensor if feature is missing for this sequence
                        seq_features[feature_name] = torch.zeros(seq_length, dtype=torch.float32, device=device)
            
            # Stack features for this sequence
            feature_list = []
            for feature_name in self.feature_dims.keys():
                if feature_name in seq_features:
                    # Add dimension if needed
                    if seq_features[feature_name].dim() == 1:
                        feature_tensor = seq_features[feature_name].unsqueeze(1)
                    else:
                        feature_tensor = seq_features[feature_name]
                    feature_list.append(feature_tensor)
                else:
                    # Create zero tensor if feature is completely missing
                    feature_list.append(torch.zeros(seq_length, 1, dtype=torch.float32, device=device))
            
            # Concatenate features for this sequence
            if feature_list:
                seq_feature_tensor = torch.cat(feature_list, dim=1)
                processed_features.append(seq_feature_tensor)
            else:
                # Empty feature tensor if no features available
                seq_feature_tensor = torch.zeros(seq_length, len(self.feature_dims), dtype=torch.float32, device=device)
                processed_features.append(seq_feature_tensor)
        
        return processed_features

    def forward(self,
                sequences: List[str],
                temperatures: torch.Tensor,  # Expecting a BATCHED tensor of SCALED temperatures
                target_rmsf_values: Optional[List[torch.Tensor]] = None,  # Optional targets
                features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Forward pass incorporating sequence embeddings, temperature, and structural features.

        Args:
            sequences: List of amino acid sequence strings (batch_size).
            temperatures: Tensor of SCALED temperature values for each sequence,
                          shape [batch_size]. MUST be pre-scaled.
            target_rmsf_values: Optional list of target RMSF tensors for loss calculation.
            features: Optional dictionary of structural features from dataloader.

        Returns:
            Dictionary containing 'predictions', 'loss', 'metrics'.
        """
        # --- Basic Input Checks ---
        if len(sequences) != len(temperatures):
            msg = f"Batch size mismatch: {len(sequences)} sequences vs {len(temperatures)} temperatures."
            logger.error(msg)
            raise ValueError(msg)
        if target_rmsf_values is not None and len(sequences) != len(target_rmsf_values):
            msg = f"Batch size mismatch: {len(sequences)} sequences vs {len(target_rmsf_values)} target RMSF values."
            logger.error(msg)
            raise ValueError(msg)

        # --- Setup Device ---
        # Infer device from parameters
        device = next(self.parameters()).device
        # Ensure ESM base model is on the correct device
        if next(self.esm_model.parameters()).device != device:
            self.esm_model.to(device)

        # --- Prepare ESMProtein objects ---
        proteins = []
        original_indices = []  # Store original batch index for each valid protein
        skipped_indices = []

        for i, seq_str in enumerate(sequences):
            if not seq_str or len(seq_str) == 0:
                logger.debug(f"Skipping empty sequence at original batch index {i}.")
                skipped_indices.append(i)
                continue
            try:
                proteins.append(ESMProtein(sequence=seq_str))
                original_indices.append(i)
            except Exception as e_prot:
                logger.warning(f"Could not create ESMProtein for sequence at index {i}. Error: {e_prot}. Skipping.")
                skipped_indices.append(i)

        if not proteins:
            logger.warning("No valid sequences found in the batch to process.")
            # Return structure consistent with successful run but empty preds/zero loss
            return {
                'predictions': [torch.tensor([], device=device) for _ in sequences],  # Match input batch size
                'loss': torch.tensor(0.0, device=device, requires_grad=True if self.training else False),
                'metrics': {'pearson_correlation': 0.0}
            }

        # --- Process Proteins ---
        all_predictions = []  # Store final per-residue predictions for each protein
        processed_indices_map = {}  # Map index in `all_predictions` back to original batch index

        try:
            # --- Advanced Temperature Processing ---
            if self.improved_temp_integration:
                # Create rich temperature embeddings (batch_size, temp_embedding_dim)
                batch_temp_embeddings = self.temp_encoder(temperatures.to(device))
            else:
                # Simple temperature feature (batch_size, 1)
                batch_temp_embeddings = temperatures.to(device).unsqueeze(1)
            
            # --- Process Structural Features ---
            batch_processed_features = None
            if self.use_enhanced_features and features is not None:
                # Process and align features with sequences
                batch_processed_features = self._process_protein_features(features, sequences, device)
            
            # --- Per-Protein Processing ---
            for protein_idx, protein in enumerate(proteins):
                original_batch_idx = original_indices[protein_idx]
                current_temp_embedding = batch_temp_embeddings[original_batch_idx]  # Get temperature embedding

                try:
                    # 1. Get ESM Embeddings (No Gradients for ESM part)
                    with torch.no_grad():
                        encoded_protein = self.esm_model.encode(protein)
                        logits_output = self.esm_model.logits(
                            encoded_protein,
                            LogitsConfig(sequence=True, return_embeddings=True)
                        )

                    if logits_output.embeddings is None:
                        logger.warning(f"No embeddings returned for protein {original_batch_idx}. Skipping.")
                        continue

                    # Embeddings shape: [1, seq_len_with_tokens, hidden_dim]
                    embeddings = logits_output.embeddings.to(device)
                    # Remove batch dimension: [seq_len_with_tokens, hidden_dim]
                    embeddings_tokens = embeddings.squeeze(0)
                    
                    # 2. Extract Residue Embeddings (Remove BOS/EOS tokens)
                    original_seq_len = len(protein.sequence)  # Length of the actual AA sequence
                    expected_tokens = original_seq_len + 2  # Assuming BOS and EOS tokens

                    if len(embeddings_tokens) >= expected_tokens:
                        # Slice: Start after BOS (index 1), end before EOS
                        residue_embeddings = embeddings_tokens[1:expected_tokens-1]
                    else:
                        logger.warning(f"Embedding tensor length ({len(embeddings_tokens)}) is shorter than "
                                      f"expected seq+BOS+EOS ({expected_tokens}) for original sequence {original_batch_idx}. "
                                      "Cannot reliably extract residue embeddings. Skipping.")
                        continue
                    
                    # Apply positional encoding if enabled
                    if self.use_positional_encoding:
                        residue_embeddings = self.positional_encoding(residue_embeddings)
                    
                    # 3. Process structural features if available
                    combined_features = residue_embeddings
                    if self.use_enhanced_features and batch_processed_features is not None and protein_idx < len(batch_processed_features):
                        # Get features for this protein
                        protein_features = batch_processed_features[protein_idx]
                        
                        # Ensure features align with residue embeddings
                        if protein_features.size(0) != residue_embeddings.size(0):
                            logger.warning(f"Feature length mismatch for protein {original_batch_idx}: "
                                         f"features={protein_features.size(0)}, embeddings={residue_embeddings.size(0)}. "
                                         "Adjusting feature length.")
                            
                            # Adjust feature length to match embeddings
                            if protein_features.size(0) > residue_embeddings.size(0):
                                protein_features = protein_features[:residue_embeddings.size(0)]
                            else:
                                padding = torch.zeros(
                                    residue_embeddings.size(0) - protein_features.size(0),
                                    protein_features.size(1),
                                    dtype=protein_features.dtype,
                                    device=protein_features.device
                                )
                                protein_features = torch.cat([protein_features, padding], dim=0)
                        
                        # Combine ESM embeddings with structural features
                        # Use feature processor to get aligned features
                        processed_features = self.feature_processor({
                            feature_name: protein_features[:, i].unsqueeze(1)
                            for i, feature_name in enumerate(self.feature_dims.keys())
                        })
                        
                        # Concatenate with ESM embeddings
                        combined_input = torch.cat([residue_embeddings, processed_features], dim=1)
                        
                        # Fuse ESM embeddings and features
                        combined_features = self.feature_esm_fusion(combined_input)
                    
                    # 4. Apply attention with temperature conditioning if enabled
                    if self.use_attention:
                        # Add batch dimension for attention
                        features_batch = combined_features.unsqueeze(0)  # [1, seq_len, hidden_dim]
                        temp_embed_batch = current_temp_embedding.unsqueeze(0)  # [1, temp_dim]
                        
                        # Apply attention with temperature conditioning
                        attended_features = self.attention(features_batch, temp_embed_batch)
                        
                        # Remove batch dimension
                        processed_embeddings = attended_features.squeeze(0)  # [seq_len, hidden_dim]
                    else:
                        # No attention - use combined features directly
                        processed_embeddings = combined_features
                    
                    # 5. Apply regression head to get predictions
                    # Ensure head is in correct mode (train/eval) based on model state
                    self.regression_head.train(self.training)
                    # Get per-residue predictions
                    token_predictions = self.regression_head(processed_embeddings).squeeze(-1)
                    
                    # Store predictions
                    all_predictions.append(token_predictions)
                    processed_indices_map[len(all_predictions)-1] = original_batch_idx

                except Exception as e_inner:
                    logger.error(f"Error processing protein at original batch index {original_batch_idx}: {e_inner}", exc_info=True)
                    # Continue to the next protein in the batch

        except Exception as e_outer:
            logger.error(f"Error during main forward loop: {e_outer}", exc_info=True)
            # Return empty/zero structure if outer loop fails catastrophically
            return {
                'predictions': [torch.tensor([], device=device) for _ in sequences],
                'loss': torch.tensor(0.0, device=device, requires_grad=True if self.training else False),
                'metrics': {'pearson_correlation': 0.0}
            }

        # --- Loss Calculation (Optional) ---
        loss = None
        metrics = {'pearson_correlation': 0.0}  # Default metrics

        if target_rmsf_values is not None:
            valid_losses = []
            valid_correlations = []
            num_valid_pairs = 0

            # Iterate through the predictions we successfully generated
            for pred_idx, prediction_tensor in enumerate(all_predictions):
                original_batch_idx = processed_indices_map[pred_idx]
                target_tensor = target_rmsf_values[original_batch_idx].to(device)

                # Align lengths (prediction might be slightly off if slicing warning occurred)
                min_len = min(len(prediction_tensor), len(target_tensor))
                if min_len <= 1:  # Need at least 2 points for correlation
                    continue

                pred_aligned = prediction_tensor[:min_len]
                target_aligned = target_tensor[:min_len]

                # Calculate standard MSE Loss
                mse = F.mse_loss(pred_aligned, target_aligned, reduction='mean')
                if not torch.isnan(mse) and not torch.isinf(mse):
                    valid_losses.append(mse)

                # Calculate Pearson Correlation for metrics reporting only
                # (not used in loss calculation)
                pearson_corr = self.safe_pearson_correlation(pred_aligned, target_aligned)
                if not torch.isnan(pearson_corr):
                    valid_correlations.append(pearson_corr)

                num_valid_pairs += 1

            # Average loss and correlation over valid pairs in the batch
            if valid_losses:
                # Average the loss across samples in the batch
                loss = torch.stack(valid_losses).mean()
                if torch.isnan(loss):  # Handle potential NaN if all losses were somehow NaN
                    loss = torch.tensor(0.0, device=device, requires_grad=True if self.training else False)
            else:  # No valid pairs, set loss to 0
                loss = torch.tensor(0.0, device=device, requires_grad=True if self.training else False)

            if valid_correlations:
                # Average correlation across samples for metrics reporting
                metrics['pearson_correlation'] = torch.stack(valid_correlations).mean().item()
            else:  # No valid correlations calculated
                metrics['pearson_correlation'] = 0.0  # Keep as float

        # Ensure loss is always a tensor, required by training loop
        if loss is None:
            loss = torch.tensor(0.0, device=device, requires_grad=True if self.training else False)

        # --- Reconstruct Output List ---
        # Create a list of tensors matching the original batch size, filling with
        # predictions where available and empty tensors otherwise.
        final_predictions_list = [torch.tensor([], device=device) for _ in sequences]
        for pred_idx, pred_tensor in enumerate(all_predictions):
            original_batch_idx = processed_indices_map[pred_idx]
            final_predictions_list[original_batch_idx] = pred_tensor

        return {'predictions': final_predictions_list, 'loss': loss, 'metrics': metrics}

    @staticmethod
    def safe_pearson_correlation(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """
        Calculate Pearson correlation safely, returning 0 for std dev near zero or len < 2.
        Used for metrics reporting only, not for optimization.
        """
        # Ensure float type
        x = x.float()
        y = y.float()

        # Check for conditions where correlation is undefined or unstable
        if len(x) < 2 or torch.std(x) < epsilon or torch.std(y) < epsilon:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        # Use matrix multiplication for covariance calculation for efficiency if needed,
        # but direct sum is fine for typical sequence lengths here.
        cov = torch.sum(vx * vy)
        sx = torch.sqrt(torch.sum(vx ** 2))
        sy = torch.sqrt(torch.sum(vy ** 2))
        denominator = sx * sy

        # Check for near-zero denominator
        if denominator < epsilon:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)

        corr = cov / denominator
        # Clamp to handle potential floating point inaccuracies near +/- 1
        corr = torch.clamp(corr, -1.0, 1.0)

        # Final NaN check (should be rare after previous checks, but just in case)
        if torch.isnan(corr):
            logger.warning("NaN detected during Pearson Correlation calculation despite checks. Returning 0.")
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)

        return corr

    @torch.no_grad()
    def predict(self,
                sequences: List[str],
                scaled_temperatures: torch.Tensor,  # Expecting tensor shape [batch_size]
                features: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """
        Predict RMSF values for sequences at given SCALED temperatures.

        Args:
            sequences: List of amino acid sequences.
            scaled_temperatures: Tensor of SCALED temperatures (one per sequence).
            features: Optional dictionary of structural features from dataloader.

        Returns:
            List of NumPy arrays containing predicted RMSF values for each sequence.
        """
        self.eval()  # Ensure evaluation mode

        if len(sequences) != len(scaled_temperatures):
            raise ValueError("Number of sequences must match number of temperatures for prediction.")

        # Pass sequences and scaled temperatures to the forward method
        outputs = self.forward(
            sequences=sequences,
            temperatures=scaled_temperatures.to(next(self.parameters()).device),
            target_rmsf_values=None,
            features=features
        )

        # Convert predictions tensor list to list of numpy arrays
        np_predictions = []
        for pred_tensor in outputs['predictions']:
            if pred_tensor is not None and pred_tensor.numel() > 0:
                np_predictions.append(pred_tensor.cpu().numpy())
            else:  # Handle cases where prediction failed for a sequence
                np_predictions.append(np.array([], dtype=np.float32))

        return np_predictions


# Factory function to create model based on config
def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Create a model instance based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    model_config = config.get('model', {})
    esm_version = model_config.get('esm_version', 'esmc_600m')
    
    regression_config = model_config.get('regression', {})
    hidden_dim = regression_config.get('hidden_dim', 64)
    dropout = regression_config.get('dropout', 0.1)
    
    architecture_config = model_config.get('architecture', {})
    use_enhanced_features = architecture_config.get('use_enhanced_features', True)
    use_attention = architecture_config.get('use_attention', True)
    attention_heads = architecture_config.get('attention_heads', 8)
    attention_dropout = architecture_config.get('attention_dropout', 0.1)
    improved_temp_integration = architecture_config.get('improved_temp_integration', True)
    
    logger.info(f"Creating model with config: esm_version={esm_version}, hidden_dim={hidden_dim}")
    logger.info(f"Enhanced features: {use_enhanced_features}, Attention: {use_attention}, "
               f"Improved temperature integration: {improved_temp_integration}")
    
    return EnhancedTemperatureAwareESMModel(
        esm_model_name=esm_version,
        regression_hidden_dim=hidden_dim,
        regression_dropout=dropout,
        use_attention=use_attention,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
        use_positional_encoding=True,
        use_enhanced_features=use_enhanced_features,
        improved_temp_integration=improved_temp_integration
    )