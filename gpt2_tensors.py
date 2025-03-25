"""
GPT-2 weight loading and tensor transformation utilities.

This module handles loading and organizing the GPT-2 weights from safetensors format
into the structure expected by the model implementation.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from gpt2_loader import GPT2WeightLoader, download_vocab_encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPT2TensorManager:
    """Manages loading and organizing GPT-2 model weights and parameters."""

    def __init__(self, model_name: str = "openai-community/gpt2"):
        """
        Initialize the tensor manager.

        Args:
            model_name: Name of the model to load weights from
        """
        self.model_name = model_name
        self.loader: Optional[GPT2WeightLoader] = None
        self.hparams: Optional[Dict[str, Any]] = None

    def initialize(self) -> None:
        """Download necessary files and initialize the weight loader."""
        # Download vocabulary and encoder files
        download_vocab_encoder(".")

        # Initialize and download weights
        self.loader = GPT2WeightLoader(self.model_name)
        self.loader.download_weights()

        # Load configuration
        self.hparams = self.loader.config
        logger.info(f"Model configuration loaded: {self.hparams}")

    def get_tensor(self, name: str) -> np.ndarray:
        """
        Load a tensor by name with logging.

        Args:
            name: Name of the tensor to load

        Returns:
            The loaded numpy array
        """
        if self.loader is None:
            raise RuntimeError(
                "TensorManager not initialized. Call initialize() first."
            )

        tensor = self.loader.get_tensor(name)
        logger.info(f"Loaded tensor {name} with shape {tensor.shape}")
        return tensor

    def load_transformer_block(self, block_idx: int) -> Dict[str, Any]:
        """
        Load weights for a single transformer block.

        Args:
            block_idx: Index of the transformer block to load

        Returns:
            Dictionary containing organized weights for the block
        """
        prefix = f"h.{block_idx}"
        logger.info(f"\nLoading transformer block {block_idx}")

        # Load attention weights
        c_attn_w = self.get_tensor(f"{prefix}.attn.c_attn.weight")
        c_attn_b = self.get_tensor(f"{prefix}.attn.c_attn.bias")
        c_proj_w = self.get_tensor(f"{prefix}.attn.c_proj.weight")
        c_proj_b = self.get_tensor(f"{prefix}.attn.c_proj.bias")

        # Load MLP weights
        c_fc_w = self.get_tensor(f"{prefix}.mlp.c_fc.weight")
        c_fc_b = self.get_tensor(f"{prefix}.mlp.c_fc.bias")
        c_proj2_w = self.get_tensor(f"{prefix}.mlp.c_proj.weight")
        c_proj2_b = self.get_tensor(f"{prefix}.mlp.c_proj.bias")

        return {
            "ln_1": {
                "g": self.get_tensor(f"{prefix}.ln_1.weight"),
                "b": self.get_tensor(f"{prefix}.ln_1.bias"),
            },
            "ln_2": {
                "g": self.get_tensor(f"{prefix}.ln_2.weight"),
                "b": self.get_tensor(f"{prefix}.ln_2.bias"),
            },
            "mlp": {
                "c_fc": {"w": c_fc_w, "b": c_fc_b},
                "c_proj": {"w": c_proj2_w, "b": c_proj2_b},
            },
            "attn": {
                "c_attn": {"w": c_attn_w, "b": c_attn_b},
                "c_proj": {"w": c_proj_w, "b": c_proj_b},
            },
        }

    def load_model_weights(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load and organize all model weights.

        Returns:
            Tuple of (model parameters, hyperparameters)
        """
        if self.loader is None:
            self.initialize()

        if self.hparams is None:
            raise RuntimeError("Model hyperparameters not loaded")

        # Load embeddings
        wte = self.get_tensor("wte.weight")  # Token embeddings
        wpe = self.get_tensor("wpe.weight")  # Position embeddings

        # Load final layer norm
        ln_f = {"g": self.get_tensor("ln_f.weight"), "b": self.get_tensor("ln_f.bias")}

        # Load all transformer blocks
        blocks = []
        for i in range(self.hparams["n_layer"]):
            blocks.append(self.load_transformer_block(i))

        params = {"wte": wte, "wpe": wpe, "blocks": blocks, "ln_f": ln_f}

        return params, self.hparams
