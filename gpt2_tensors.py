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

Safetensors is a fast and safe format for storing tensors. The format uses a simple key/value structure where:

- Keys are UTF-8 encoded strings representing tensor names (e.g. 'model.layers.0.attention.weight')
- Values are binary tensor data with a fixed header containing shape and dtype information
- A metadata section at the start of the file contains an index of all tensors and their offsets

This structure allows for direct memory mapping and random access to individual tensors
without loading the entire file into memory.

{
    "wpe.weight": np.array([1024, 768]),
    "wte.weight": np.array([50257, 768]),
    ...
    "h.0.attn.bias": np.array([1, 1, 1024, 1024]),
    "h.0.attn.c_attn.bias": np.array([2304]),
    "h.0.attn.c_attn.weight": np.array([768, 2304]),
    "h.0.attn.c_proj.bias": np.array([768]),
    "h.0.attn.c_proj.weight": np.array([768, 768]),
    "h.0.ln_1.bias": np.array([768]),
    "h.0.ln_1.weight": np.array([768]),
    "h.0.ln_2.bias": np.array([768]),
    "h.0.ln_2.weight": np.array([768]),
    "h.0.mlp.c_fc.bias": np.array([3072]),
    "h.0.mlp.c_fc.weight": np.array([768, 3072]),
    "h.0.mlp.c_proj.bias": np.array([768]),
    "h.0.mlp.c_proj.weight": np.array([3072, 768]),
    ...
    "ln_f.bias": np.array([768]),
    "ln_f.weight": np.array([768])
}
"""

import numpy as np
from pathlib import Path
import requests
from safetensors import safe_open
from dataclasses import dataclass
from typing import List, Tuple
import json

# URLs for vocabulary and encoder files
VOCAB_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe"
ENCODE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json"
HF_API_URL = "https://huggingface.co/api/models/"
HF_REPO_URL = "https://huggingface.co/"


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

    n_layer: int  # Number of transformer layers
    n_head: int  # Number of attention heads
    n_ctx: int  # Context length


def load_gpt2_weights(
    model_name: str = "openai-community/gpt2", cache_dir: str = "model"
) -> Tuple[ModelParams, HParams]:
    """Load GPT-2 weights from HuggingFace into structured dataclasses."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Download model info and config
    response = requests.get(f"{HF_API_URL}{model_name}")
    response.raise_for_status()

    # Get config
    config_path = cache_path / "config.json"
    if not config_path.exists():
        config_url = f"{HF_REPO_URL}{model_name}/resolve/main/config.json"
        response = requests.get(config_url)
        response.raise_for_status()
        config_path.write_text(response.text)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Download safetensors file - simplified since we know there's only one
    weights_path = cache_path / "model.safetensors"
    if not weights_path.exists():
        print("Downloading weights from HuggingFace...")
        weights_url = f"{HF_REPO_URL}{model_name}/resolve/main/model.safetensors"
        response = requests.get(weights_url)
        response.raise_for_status()
        weights_path.write_bytes(response.content)
        print(f"Weights downloaded to {weights_path}")

    # Load tensors
    tensors = {}
    with safe_open(weights_path, framework="numpy") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Build transformer blocks
    blocks = []
    for i in range(config["n_layer"]):
        prefix = f"h.{i}"
        block = TransformerBlockParams(
            ln_1=LayerNormParams(
                g=tensors[f"{prefix}.ln_1.weight"], b=tensors[f"{prefix}.ln_1.bias"]
            ),
            ln_2=LayerNormParams(
                g=tensors[f"{prefix}.ln_2.weight"], b=tensors[f"{prefix}.ln_2.bias"]
            ),
            mlp=MLPParams(
                c_fc=LinearParams(
                    w=tensors[f"{prefix}.mlp.c_fc.weight"],
                    b=tensors[f"{prefix}.mlp.c_fc.bias"],
                ),
                c_proj=LinearParams(
                    w=tensors[f"{prefix}.mlp.c_proj.weight"],
                    b=tensors[f"{prefix}.mlp.c_proj.bias"],
                ),
            ),
            attn=AttentionParams(
                c_attn=LinearParams(
                    w=tensors[f"{prefix}.attn.c_attn.weight"],
                    b=tensors[f"{prefix}.attn.c_attn.bias"],
                ),
                c_proj=LinearParams(
                    w=tensors[f"{prefix}.attn.c_proj.weight"],
                    b=tensors[f"{prefix}.attn.c_proj.bias"],
                ),
            ),
        )
        blocks.append(block)

    # Build final model params
    params = ModelParams(
        wte=tensors["wte.weight"],
        wpe=tensors["wpe.weight"],
        blocks=blocks,
        ln_f=LayerNormParams(g=tensors["ln_f.weight"], b=tensors["ln_f.bias"]),
    )

    # Extract hyperparameters
    hparams = HParams(
        n_layer=config["n_layer"], n_head=config["n_head"], n_ctx=config["n_ctx"]
    )

    return params, hparams
