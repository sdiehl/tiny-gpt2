"""
GPT-2 weight loading and tensor transformation utilities.

This module handles loading and organizing the GPT-2 weights from safetensors format
into the structure expected by the model implementation.

Abbreviation Dictionary:
    g    - Gamma (scale parameter for layer normalization)
    b    - Beta (bias parameter)
    w    - Weight matrix/array
    wte  - Word/Token Embeddings
    wpe  - Word Position Embeddings
    ln   - Layer Normalization
    mlp  - Multi-Layer Perceptron
    fc   - Fully Connected layer
    qkv  - Query, Key, Value (attention components)
    attn - Attention
    proj - Projection (linear transformation)
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
from gpt2_loader import GPT2WeightLoader, download_vocab_encoder
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LayerNormParams:
    """Layer normalization parameters."""

    g: np.ndarray  # Gamma (scale)
    b: np.ndarray  # Beta (bias)


@dataclass
class LinearParams:
    """Linear layer parameters."""

    w: np.ndarray  # Weight matrix
    b: np.ndarray  # Bias vector


@dataclass
class MLPParams:
    """MLP block parameters."""

    c_fc: LinearParams  # First linear layer
    c_proj: LinearParams  # Second linear layer


@dataclass
class AttentionParams:
    """Attention block parameters."""

    c_attn: LinearParams  # QKV projection
    c_proj: LinearParams  # Output projection


@dataclass
class TransformerBlockParams:
    """Parameters for a single transformer block."""

    ln_1: LayerNormParams  # First layer norm
    ln_2: LayerNormParams  # Second layer norm
    mlp: MLPParams  # MLP block
    attn: AttentionParams  # Attention block


@dataclass
class ModelParams:
    """Complete model parameters."""

    wte: np.ndarray  # Token embeddings
    wpe: np.ndarray  # Position embeddings
    blocks: List[TransformerBlockParams]  # Transformer blocks
    ln_f: LayerNormParams  # Final layer norm


@dataclass
class HParams:
    """Hyperparameters for the GPT-2 model."""

    n_layer: int  # Number of transformer blocks
    n_head: int  # Number of attention heads
    n_ctx: int  # Context length


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
        if self.loader is None:
            raise RuntimeError(
                "TensorManager not initialized. Call initialize() first."
            )

        tensor = self.loader.get_tensor(name)
        logger.info(f"Loaded tensor {name} with shape {tensor.shape}")
        return tensor

    def load_transformer_block(self, block_idx: int) -> TransformerBlockParams:
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

        return TransformerBlockParams(
            ln_1=LayerNormParams(
                g=self.get_tensor(f"{prefix}.ln_1.weight"),
                b=self.get_tensor(f"{prefix}.ln_1.bias"),
            ),
            ln_2=LayerNormParams(
                g=self.get_tensor(f"{prefix}.ln_2.weight"),
                b=self.get_tensor(f"{prefix}.ln_2.bias"),
            ),
            mlp=MLPParams(
                c_fc=LinearParams(w=c_fc_w, b=c_fc_b),
                c_proj=LinearParams(w=c_proj2_w, b=c_proj2_b),
            ),
            attn=AttentionParams(
                c_attn=LinearParams(w=c_attn_w, b=c_attn_b),
                c_proj=LinearParams(w=c_proj_w, b=c_proj_b),
            ),
        )

    def load_model_weights(self) -> Tuple[ModelParams, HParams]:
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
        ln_f = LayerNormParams(
            g=self.get_tensor("ln_f.weight"),
            b=self.get_tensor("ln_f.bias"),
        )

        # Load all transformer blocks
        blocks = []
        for i in range(self.hparams["n_layer"]):
            blocks.append(self.load_transformer_block(i))

        params = ModelParams(wte=wte, wpe=wpe, blocks=blocks, ln_f=ln_f)

        hparams = HParams(
            n_layer=self.hparams["n_layer"],
            n_head=self.hparams["n_head"],
            n_ctx=self.hparams["n_ctx"],
        )

        return params, hparams
