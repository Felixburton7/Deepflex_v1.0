# === FILE: model.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from torch.nn.utils.rnn import pad_sequence # Needed for padding

# PEFT for LoRA integration
try:
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
    peft_available = True
except ImportError:
    logging.warning("PEFT library not found. LoRA functionality will be disabled. Install with 'pip install peft'")
    peft_available = False
    class DummyPeftClass: pass
    LoraConfig, TaskType, PeftModel, PeftConfig = DummyPeftClass, DummyPeftClass, DummyPeftClass, DummyPeftClass
    def get_peft_model(model, config): return model

# ESM library for base model
try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig, ESMProteinTensor
    esm_available = True
except ImportError:
    logging.error("Failed to import ESM-C classes from 'esm' library. Ensure 'esm>=2.0.0' installed (`pip install esm`).")
    esm_available = False
    raise

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

# Known Embedding Dimensions
ESMC_DIMENSIONS = {"esmc_300m": 960, "esmc_600m": 1152, "esmc_6b": 2560}

class ESMRegressionModelWithLoRA(nn.Module):
    """ESM-C model with optional LoRA fine-tuning for RMSF prediction."""
    def __init__(self, config: Dict):
        super().__init__()
        if not esm_available: raise RuntimeError("ESM library not available.")
        self.config = config
        model_config = config['model']
        lora_config_dict = model_config.get('lora', {})
        regression_config = model_config['regression']
        esm_model_name = model_config['esm_version']
        self.lora_enabled = lora_config_dict.get('enabled', False) and peft_available

        logger.info(f"Initializing ESMRegressionModelWithLoRA: Base={esm_model_name}, LoRA={self.lora_enabled}")

        # --- Load Base Model ---
        try:
            self.base_model = ESMC.from_pretrained(esm_model_name)
            logger.info(f"Base model {esm_model_name} loaded.")
        except Exception as e:
            logger.error(f"Failed to load base ESM model '{esm_model_name}': {e}", exc_info=True); raise

        # --- Determine Embedding Dimension ---
        self.hidden_dim = self._get_hidden_dim(esm_model_name)
        logger.info(f"Using embedding dimension: {self.hidden_dim} for {esm_model_name}")

        # --- Configure PEFT ---
        if self.lora_enabled:
            logger.info("Configuring PEFT LoRA...")
            target_modules = lora_config_dict.get('target_modules', ["q_proj", "v_proj"])
            logger.info(f"Attempting to target LoRA modules: {target_modules}")
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS, inference_mode=False,
                r=lora_config_dict.get('r', 16), lora_alpha=lora_config_dict.get('lora_alpha', 32),
                lora_dropout=lora_config_dict.get('lora_dropout', 0.1),
                target_modules=target_modules, bias="none"
            )
            try:
                self.model = get_peft_model(self.base_model, peft_config)
                logger.info("PEFT LoRA adapters applied.")
                self.model.print_trainable_parameters()
            except ValueError as e: logger.error(f"Failed PEFT: {e}. Check 'target_modules'."); raise e
        else:
            logger.info("LoRA disabled. Freezing base model."); self.model = self.base_model
            for param in self.model.parameters(): param.requires_grad = False

        # --- Regression Head ---
        regression_hidden = regression_config.get('hidden_dim', 32)
        regression_dropout = regression_config.get('dropout', 0.1)
        head_layers = [nn.LayerNorm(self.hidden_dim)]
        if regression_hidden > 0:
            head_layers.extend([nn.Linear(self.hidden_dim, regression_hidden), nn.GELU(), nn.Dropout(regression_dropout), nn.Linear(regression_hidden, 1)])
            logger.info(f"Using MLP regression head (In: {self.hidden_dim}, Hidden: {regression_hidden}, Dropout: {regression_dropout})")
        else:
            head_layers.extend([nn.Dropout(regression_dropout), nn.Linear(self.hidden_dim, 1)])
            logger.info(f"Using Linear regression head (In: {self.hidden_dim}, Dropout: {regression_dropout})")
        self.regression_head = nn.Sequential(*head_layers)
        for param in self.regression_head.parameters(): param.requires_grad = True
        logger.info("Regression head configured and set to trainable.")
        self._log_parameter_counts()

    def _get_hidden_dim(self, model_name: str) -> int:
        if model_name in ESMC_DIMENSIONS: return ESMC_DIMENSIONS[model_name]
        else: raise ValueError(f"Embedding dimension for '{model_name}' unknown. Add to ESMC_DIMENSIONS.")

    def _log_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Params: Total={total:,}, Trainable={trainable:,} ({trainable/total:.4%})")

    def forward(self, sequences: List[str], rmsf_values: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        device = next(self.parameters()).device
        self.model.to(device)
        self.regression_head.to(device)

        batch_embeddings_list = []
        original_lengths = []
        processed_indices = [] # Keep track of indices successfully processed

        # --- Process sequences individually ---
        for i, seq_str in enumerate(sequences):
            if not (seq_str and isinstance(seq_str, str) and set(seq_str.upper()).issubset(set('ACDEFGHIKLMNPQRSTVWY'))):
                # logger.warning(f"Idx {i}: Invalid/empty sequence or chars. Skipping.")
                # Add placeholder for prediction list reconstruction later
                batch_embeddings_list.append(None)
                original_lengths.append(0)
                continue

            try:
                protein_obj = ESMProtein(sequence=seq_str)
                encoded_output = self.model.encode(protein_obj)
                logits_output = self.model.logits(encoded_output, LogitsConfig(sequence=True, return_embeddings=True))
                embeddings = logits_output.embeddings # Shape: [1, SeqLenPadded, Dim]

                if embeddings is None: raise ValueError("Logits call did not return embeddings.")

                # Extract the actual sequence embedding (remove potential batch dim)
                # and move to correct device
                single_embedding = embeddings.squeeze(0).to(device) # Shape: [SeqLenPadded, Dim]
                batch_embeddings_list.append(single_embedding)
                original_lengths.append(len(seq_str)) # Store original length
                processed_indices.append(i)

            except Exception as e:
                logger.error(f"Error processing sequence index {i}: {e}", exc_info=True)
                batch_embeddings_list.append(None) # Mark as failed
                original_lengths.append(0)
                # Continue processing other sequences in the batch

        # Filter out failed embeddings before padding
        valid_embeddings = [emb for emb in batch_embeddings_list if emb is not None]
        valid_lengths = [l for i, l in enumerate(original_lengths) if batch_embeddings_list[i] is not None]

        if not valid_embeddings: # If all sequences in batch failed
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True) if self.training else torch.tensor(0.0, device=device)
            return {'predictions': [torch.tensor([], device=device)] * len(sequences), 'loss': dummy_loss, 'metrics': {'pearson_correlation': 0.0}}

        # --- Pad and Stack Embeddings ---
        # pad_sequence expects a list of tensors [SeqLen, Dim]
        # batch_first=True makes output [Batch, MaxSeqLen, Dim]
        padded_embeddings = pad_sequence(valid_embeddings, batch_first=True, padding_value=0.0)

        # --- Regression Head ---
        # Pass padded batch through the head
        # Input: [ValidBatchSize, MaxSeqLen, Dim] -> Output: [ValidBatchSize, MaxSeqLen]
        token_predictions_padded = self.regression_head(padded_embeddings).squeeze(-1)

        # --- Extract valid per-residue predictions (Handle BOS/EOS and padding) ---
        predictions_valid = []
        targets_valid = []
        valid_original_indices_final = [] # Indices from original batch corresponding to predictions_valid

        for i in range(token_predictions_padded.shape[0]): # Iterate through the valid batch dimension
            original_len = valid_lengths[i]
            original_batch_idx = processed_indices[i] # Get original index

            start_idx, end_idx = 1, original_len + 1 # Slice BOS/EOS

            if end_idx <= token_predictions_padded.shape[1]: # Check against padded length
                sequence_predictions = token_predictions_padded[i, start_idx:end_idx]
                if len(sequence_predictions) == original_len:
                    predictions_valid.append(sequence_predictions)
                    valid_original_indices_final.append(original_batch_idx)
                    if rmsf_values is not None and original_batch_idx < len(rmsf_values):
                        targets_valid.append(rmsf_values[original_batch_idx].to(device))
                    elif rmsf_values is not None:
                         logger.error(f"Original index {original_batch_idx} out of bounds for rmsf_values list (len={len(rmsf_values)})")
                else: logger.warning(f"Idx {original_batch_idx}: Length mismatch post-slice. Expected {original_len}, got {len(sequence_predictions)}. Skipping.")
            else: logger.warning(f"Idx {original_batch_idx}: Padded prediction too short ({token_predictions_padded.shape[1]}) for slice end ({end_idx}). Skipping.")


        # --- Loss & Metrics ---
        loss = torch.tensor(0.0, device=device, requires_grad=True) if self.training else torch.tensor(0.0, device=device)
        metrics = {'pearson_correlation': 0.0}
        if rmsf_values is not None and predictions_valid and len(predictions_valid) == len(targets_valid):
            mse_losses = [F.mse_loss(pred, target) for pred, target in zip(predictions_valid, targets_valid) if len(pred) > 0]
            pearson_correlations = [self.safe_pearson_correlation(pred, target) for pred, target in zip(predictions_valid, targets_valid) if len(pred) > 1]
            if mse_losses:
                try: loss = torch.stack(mse_losses).mean(); loss = loss if not torch.isnan(loss) else torch.tensor(0.0, device=device, requires_grad=True)
                except RuntimeError: loss = torch.tensor(0.0, device=device, requires_grad=True)
            if pearson_correlations:
                valid_corrs = [c for c in pearson_correlations if not torch.isnan(c)]
                if valid_corrs:
                    try: metrics['pearson_correlation'] = torch.stack(valid_corrs).mean().item()
                    except RuntimeError: pass

        # --- Reconstruct output list to match original batch size ---
        final_predictions_list = [torch.tensor([], device=device)] * len(sequences)
        if len(predictions_valid) == len(valid_original_indices_final):
             for pred_tensor, original_idx in zip(predictions_valid, valid_original_indices_final):
                 final_predictions_list[original_idx] = pred_tensor
        else: logger.error("Mismatch count when reconstructing final prediction list.")

        return {'predictions': final_predictions_list, 'loss': loss, 'metrics': metrics}

    @staticmethod
    def safe_pearson_correlation(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Calculate Pearson correlation safely."""
        x, y = x.float(), y.float(); len_x = x.numel()
        if len_x < 2: return torch.tensor(0.0, device=x.device)
        if torch.allclose(x, x[0], atol=epsilon) or torch.allclose(y, y[0], atol=epsilon): return torch.tensor(0.0, device=x.device) # Check for near constant
        vx, vy = x - torch.mean(x), y - torch.mean(y)
        denom_sqrt_x = torch.sqrt(torch.sum(vx ** 2)); denom_sqrt_y = torch.sqrt(torch.sum(vy ** 2))
        if denom_sqrt_x < epsilon or denom_sqrt_y < epsilon: return torch.tensor(0.0, device=x.device)
        cost = torch.sum(vx * vy) / (denom_sqrt_x * denom_sqrt_y + epsilon)
        cost = torch.clamp(cost, -1.0, 1.0)
        return cost if not torch.isnan(cost) else torch.tensor(0.0, device=x.device)

    @torch.no_grad()
    def predict(self, sequences: List[str]) -> List[np.ndarray]:
        """Generates predictions for a list of sequences."""
        self.eval()
        outputs = self.forward(sequences=sequences, rmsf_values=None)
        return [pred.detach().cpu().numpy() for pred in outputs['predictions']]