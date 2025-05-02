import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# Set up logging FIRST
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ONLY what's needed based on ESM-C Quickstart
try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import LogitsConfig, ESMProtein # Use ESMProtein object
except ImportError:
    logger.error("Failed to import 'ESMC', 'LogitsConfig', or 'ESMProtein' from the 'esm' library. "
                 "Please ensure 'esm' is installed (`pip install esm`).")
    raise

class ESMRegressionModel(nn.Module):
    """
    ESM-C based model for RMSF prediction using the native esm library API
    (encode -> logits) as demonstrated in the ESM-C Quickstart.
    """
    def __init__(self,
                 esm_model_name: str = "esmc_150m",
                 regression_hidden_dim: int = 32,
                 regression_dropout: float = 0.1):
        super().__init__()

        logger.info(f"Loading ESM-C Model using 'esm' library: {esm_model_name}")
        try:
            # Load the base ESMC model object
            self.esm_model = ESMC.from_pretrained(esm_model_name)
        except Exception as e:
            logger.error(f"Failed to load ESM-C model '{esm_model_name}'. Error: {e}")
            raise

        self.esm_model.eval() # Set base model to evaluation mode
        self.esm_model_name = esm_model_name

        # --- Freeze ESM-C parameters ---
        logger.info("Freezing ESM-C model parameters...")
        for param in self.esm_model.parameters():
            param.requires_grad = False

        # --- Detect embedding dimension and create regression head ---
        # Do a dummy forward pass to determine embedding dimension
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.esm_model.to(device)
        
        try:
            # Create a small test protein and get its embedding dimension
            with torch.no_grad():
                test_protein = ESMProtein(sequence="ACDEFGHIKLMNPQRSTVWY")  # 20 standard AAs
                encoded = self.esm_model.encode(test_protein)
                logits_output = self.esm_model.logits(
                    encoded, LogitsConfig(sequence=True, return_embeddings=True)
                )
                embedding_dim = logits_output.embeddings.size(-1)
                logger.info(f"Detected embedding dimension: {embedding_dim}")
                
                # Now create the regression head with the correct dimension
                self.hidden_dim = embedding_dim
                
                if regression_hidden_dim > 0:
                    self.regression_head = nn.Sequential(
                        nn.LayerNorm(self.hidden_dim),
                        nn.Linear(self.hidden_dim, regression_hidden_dim),
                        nn.GELU(),
                        nn.Dropout(regression_dropout),
                        nn.Linear(regression_hidden_dim, 1)
                    )
                    logger.info(f"Using MLP regression head with hidden dim {regression_hidden_dim}")
                else:
                    self.regression_head = nn.Sequential(
                        nn.LayerNorm(self.hidden_dim),
                        nn.Dropout(regression_dropout),
                        nn.Linear(self.hidden_dim, 1)
                    )
                    logger.info(f"Using linear regression head (Dropout={regression_dropout})")
                
        except Exception as e:
            logger.error(f"Error during dimension detection: {e}")
            raise ValueError(f"Could not determine embedding dimension for model {esm_model_name}.")

        self._log_parameter_counts()
        logger.info("ESM-C Regression model initialized successfully using 'esm' library API.")

    def _log_parameter_counts(self):
        total_params = sum(p.numel() for p in self.parameters())
        # Base model parameters (should all be frozen)
        esm_params = sum(p.numel() for p in self.esm_model.parameters())
        # Trainable parameters (only the regression head)
        trainable_params = sum(p.numel() for p in self.regression_head.parameters())

        logger.info(f"Parameter Counts:")
        logger.info(f"  Total parameters (incl. frozen ESM): {total_params:,}")
        logger.info(f"  ESM-C parameters (frozen): {esm_params:,}")
        logger.info(f"  Trainable parameters (regression head): {trainable_params:,}")
        if total_params > 0:
            logger.info(f"  Trainable percentage: {trainable_params/total_params:.4%}")

    def forward(self,
                sequences: List[str],
                rmsf_values: Optional[List[torch.Tensor]] = None
                ) -> Dict[str, Any]:
        """
        Forward pass using ESM-C's encode and logits methods for each protein individually.
        """
        # Ensure models are on the correct device
        device = next(self.regression_head.parameters()).device
        self.esm_model.to(device) # Move base model to device

        # --- Prepare ESMProtein objects ---
        proteins = []
        original_indices_map = {} # Map processed batch index back to original sequence list index
        valid_batch_indices = [] # Indices within the 'proteins' list being processed
        skipped_indices = []     # Original indices that were skipped

        for i, seq_str in enumerate(sequences):
            if not seq_str:
                 logger.warning(f"Skipping empty sequence at original index {i}.")
                 skipped_indices.append(i)
                 continue
            try:
                 # Create an ESMProtein object for each sequence
                 proteins.append(ESMProtein(sequence=seq_str))
                 current_processed_idx = len(proteins) - 1
                 original_indices_map[current_processed_idx] = i
                 valid_batch_indices.append(current_processed_idx)
            except Exception as e_prot:
                 logger.warning(f"Could not create ESMProtein for sequence at index {i}. Error: {e_prot}. Skipping.")
                 skipped_indices.append(i)

        if not proteins:
            logger.error("No valid sequences in the batch to process.")
            return {'predictions': [torch.tensor([], device=device)] * len(sequences),
                    'loss': torch.tensor(0.0, device=device, requires_grad=True),
                    'metrics': {'pearson_correlation': 0.0}}

        # --- ESM-C Inference (Process each protein individually) ---
        all_outputs = []  # Will hold all processed outputs
        processed_indices = []  # Will track which indices were successfully processed

        try:
            for i, protein in enumerate(proteins):
                try:
                    # Process one protein at a time
                    with torch.no_grad():  # No grad for ESM-C part
                        encoded_protein = self.esm_model.encode(protein)  # Encode a single protein
                        
                        # Get embeddings
                        logits_output = self.esm_model.logits(
                            encoded_protein,
                            LogitsConfig(sequence=True, return_embeddings=True)
                        )
                    
                    if logits_output.embeddings is not None:
                        # Move embeddings to device
                        embeddings = logits_output.embeddings.to(device)
                        
                        # embeddings shape is [batch_size=1, seq_len, hidden_dim]
                        # We need to process each position in the sequence
                        seq_len = embeddings.size(1)
                        
                        # Apply the regression head to each position in the sequence
                        # First reshape to [seq_len, hidden_dim]
                        embeddings_reshaped = embeddings.squeeze(0)
                        
                        # Now run through the regression head
                        predictions = self.regression_head(embeddings_reshaped).squeeze(-1)
                        
                        # Add to our processed outputs
                        all_outputs.append(predictions)
                        processed_indices.append(i)
                    else:
                        logger.warning(f"No embeddings returned for protein at index {i}. Skipping.")
                except Exception as e:
                    logger.error(f"Error processing protein at index {i}: {e}", exc_info=True)
                    continue

            if not all_outputs:
                raise ValueError("No proteins were successfully processed.")

        except Exception as e:
            logger.error(f"Error during ESM-C encode/logits: {e}", exc_info=True)
            return {'predictions': [torch.tensor([], device=device)] * len(sequences),
                    'loss': torch.tensor(0.0, device=device, requires_grad=True),
                    'metrics': {'pearson_correlation': 0.0}}

        # --- Extract Residue-Level Predictions (Remove BOS/EOS) ---
        # Assume embeddings include BOS/EOS and we need to remove them.
        # Use original sequence length to guide slicing.
        predictions_valid = [] # Holds predictions for successfully processed sequences
        for i, token_preds in enumerate(all_outputs):
            processed_idx = processed_indices[i]
            original_idx = original_indices_map[processed_idx]
            original_seq_len = len(sequences[original_idx])
            expected_len_with_special = original_seq_len + 2

            # Check if the prediction tensor is long enough
            if len(token_preds) >= expected_len_with_special:
                # Slice to remove BOS [0] and EOS [-1 relative to expected length]
                start_idx = 1
                end_idx = expected_len_with_special - 1
                sequence_predictions = token_preds[start_idx:end_idx]

                # Final length check
                if len(sequence_predictions) == original_seq_len:
                    predictions_valid.append((original_idx, sequence_predictions))
                else: # If length still doesn't match original seq after slicing
                    logger.warning(f"Length mismatch AFTER slicing for seq {original_idx}. "
                                 f"Expected {original_seq_len}, got {len(sequence_predictions)}. "
                                 f"Using this sliced prediction.")
                    predictions_valid.append((original_idx, sequence_predictions))
            else:
                # If the prediction tensor wasn't even long enough for seq+BOS+EOS
                logger.warning(f"Prediction tensor length ({len(token_preds)}) is shorter than "
                             f"expected seq+BOS+EOS ({expected_len_with_special}) for original sequence {original_idx}. "
                             "Cannot reliably slice BOS/EOS. Appending empty tensor.")
                predictions_valid.append((original_idx, torch.tensor([], device=device)))

        # --- Loss Calculation (Optional) ---
        loss = None
        metrics = {}
        if rmsf_values is not None:
            mse_losses = []; pearson_correlations = []
            
            for original_idx, pred in predictions_valid:
                if len(pred) == 0: continue  # Skip empty predictions
                
                target = rmsf_values[original_idx].to(device)
                
                # Align lengths of prediction and target for loss calculation
                min_len = min(len(pred), len(target))
                if min_len == 0: continue # Skip if either is empty after slicing/filtering

                pred_aligned = pred[:min_len]
                target_aligned = target[:min_len]

                mse_loss = F.mse_loss(pred_aligned, target_aligned)
                mse_losses.append(mse_loss)

                if min_len > 1:
                     pearson_corr = self.safe_pearson_correlation(pred_aligned, target_aligned)
                     pearson_correlations.append(pearson_corr)
                else: # Undefined correlation for < 2 points
                     pearson_correlations.append(torch.tensor(0.0, device=device))

            # Calculate average loss and correlation for the batch
            if mse_losses:
                loss = torch.stack(mse_losses).mean()
                if torch.isnan(loss): loss = torch.tensor(0.0, device=device, requires_grad=True) # Handle NaN
            # Ensure loss is always a valid tensor, even if no valid pairs were found
            if loss is None: loss = torch.tensor(0.0, device=device, requires_grad=True)

            if pearson_correlations:
                valid_corrs = [c for c in pearson_correlations if not torch.isnan(c)]
                metrics['pearson_correlation'] = torch.stack(valid_corrs).mean().item() if valid_corrs else 0.0
            else: metrics['pearson_correlation'] = 0.0

        # --- Reconstruct output list to match original input batch size ---
        final_predictions_list = [torch.tensor([], device=device)] * len(sequences)
        for original_idx, pred_tensor in predictions_valid:
            final_predictions_list[original_idx] = pred_tensor

        # Ensure loss is defined for return (needed by training loop)
        if loss is None: loss = torch.tensor(0.0, device=device, requires_grad=True)

        return {'predictions': final_predictions_list, 'loss': loss, 'metrics': metrics}

    @staticmethod
    def safe_pearson_correlation(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        x = x.float(); y = y.float()
        if len(x) < 2 or torch.std(x) < epsilon or torch.std(y) < epsilon: return torch.tensor(0.0, device=x.device)
        x_mean, y_mean = torch.mean(x), torch.mean(y); x_centered, y_centered = x - x_mean, y - y_mean
        covariance = torch.sum(x_centered * y_centered)
        x_std_dev = torch.sqrt(torch.sum(x_centered**2)); y_std_dev = torch.sqrt(torch.sum(y_centered**2))
        denominator = x_std_dev * y_std_dev
        correlation = covariance / (denominator + epsilon)
        correlation = torch.clamp(correlation, -1.0, 1.0)
        if torch.isnan(correlation): logger.warning("NaN detected during Pearson Correlation calculation. Returning 0."); return torch.tensor(0.0, device=x.device)
        return correlation

    @torch.no_grad()
    def predict(self, sequences: List[str]) -> List[np.ndarray]:
        self.eval()
        device = next(self.regression_head.parameters()).device
        self.esm_model.to(device)

        outputs = self.forward(sequences=sequences, rmsf_values=None)
        np_predictions = [pred.cpu().numpy() for pred in outputs['predictions']]
        return np_predictions