"""
Module for downloading and loading GPT2 weights from HuggingFace safetensors.

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

# XXX: This module is not very efficient yet, it should load all the weights into memory at once.
"""

from pathlib import Path
from typing import Dict, List, Optional
import os
import json
import logging
import requests
from safetensors import safe_open
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL constants
HF_API_URL = "https://huggingface.co/api/models/"
HF_REPO_URL = "https://huggingface.co/"

# Vocabulary file URL
VOCAB_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe"
ENCODE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json"


class GPT2WeightLoader:
    """
    Utility class for downloading and loading GPT2 weights from HuggingFace.

    This class handles downloading safetensors files from HuggingFace and
    provides methods to access the individual layer weights.
    """

    def __init__(
        self, model_name: str = "openai-community/gpt2", cache_dir: Optional[str] = None
    ):
        """
        Initialize the GPT2WeightLoader.

        Args:
            model_name: The model identifier on HuggingFace (default: "openai-community/gpt2")
            cache_dir: Directory to cache downloaded files (default: ~/.cache/cudl/models)
        """
        self.model_name = model_name

        # if cache_dir is None:
        #     self.cache_dir = Path.home() / ".cache" / "cudl" / "models" / model_name.replace("/", "_")
        # else:
        #     self.cache_dir = Path(cache_dir) / model_name.replace("/", "_")

        self.cache_dir = Path(".")

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated after downloading
        self.safetensor_files: List[Path] = []
        self.config: Dict = {}

    def download_weights(self, force: bool = False) -> None:
        """
        Download model weights and configuration.

        Args:
            force: If True, re-download files even if they already exist in cache
        """
        # First get model info from HF API
        logger.info(f"Getting model info for {self.model_name}")
        response = requests.get(f"{HF_API_URL}{self.model_name}")
        response.raise_for_status()
        model_info = response.json()

        # Download config.json
        config_path = self.cache_dir / "config.json"
        if not config_path.exists() or force:
            logger.info("Downloading config.json")
            config_url = f"{HF_REPO_URL}{self.model_name}/resolve/main/config.json"
            response = requests.get(config_url)
            response.raise_for_status()
            with open(config_path, "w") as f:
                f.write(response.text)

        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Find and download safetensor files
        sibling_files = model_info.get("siblings", [])
        safetensor_files = [
            f for f in sibling_files if f["rfilename"].endswith(".safetensors")
        ]

        if not safetensor_files:
            raise ValueError(f"No safetensors files found for {self.model_name}")

        for file_info in safetensor_files:
            filename = file_info["rfilename"]
            file_path = self.cache_dir / filename

            if not file_path.exists() or force:
                logger.info(f"Downloading {filename}")
                download_url = f"{HF_REPO_URL}{self.model_name}/resolve/main/{filename}"
                response = requests.get(download_url, stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                with open(file_path, "wb") as binary_file:
                    if total_size == 0:
                        binary_file.write(response.content)
                    else:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                binary_file.write(chunk)
                                downloaded += len(chunk)
                                percent = 100 * downloaded / total_size
                                print(
                                    f"\rDownloading {filename}: {percent:.1f}% ({downloaded}/{total_size} bytes)",
                                    end="",
                                    flush=True,
                                )
                        print()

            self.safetensor_files.append(file_path)

        logger.info(
            f"Downloaded {len(self.safetensor_files)} safetensor files to {self.cache_dir}"
        )

    def get_tensor_names(self) -> List[str]:
        """
        Get a list of all tensor names in the model.

        Returns:
            List of tensor names
        """
        if not self.safetensor_files:
            raise ValueError(
                "No safetensor files loaded. Call download_weights() first."
            )

        tensor_names = []
        for file_path in self.safetensor_files:
            with safe_open(file_path, framework="numpy") as f:
                tensor_names.extend(f.keys())

        return sorted(tensor_names)

    def get_tensor(self, name: str):
        """
        Get a specific tensor by name.

        Args:
            name: The name of the tensor to retrieve

        Returns:
            The tensor as a numpy array

        Raises:
            ValueError: If the tensor is not found
        """
        if not self.safetensor_files:
            raise ValueError(
                "No safetensor files loaded. Call download_weights() first."
            )

        for file_path in self.safetensor_files:
            with safe_open(file_path, framework="numpy") as f:
                if name in f.keys():
                    return f.get_tensor(name)

        raise ValueError(f"Tensor {name} not found in any safetensor file")

    def dump_tensors(self, output_dir: Optional[str] = None) -> Dict[str, tuple]:
        """
        Dump all tensors with their shapes and data types.

        Args:
            output_dir: Optional directory to save tensors as numpy arrays

        Returns:
            Dictionary mapping tensor names to tuples of (shape, dtype)
        """
        if not self.safetensor_files:
            raise ValueError(
                "No safetensor files loaded. Call download_weights() first."
            )

        tensor_info = {}

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        tensor_names = self.get_tensor_names()
        for name in tensor_names:
            tensor = self.get_tensor(name)
            tensor_info[name] = (tensor.shape, tensor.dtype)

            logger.info(f"Tensor: {name}, Shape: {tensor.shape}, Dtype: {tensor.dtype}")

            if output_dir:
                output_path = os.path.join(output_dir, f"{name.replace('/', '_')}.npy")
                np.save(output_path, tensor)
                logger.info(f"Saved tensor to {output_path}")

        return tensor_info


# Download the vocab and encoder files


def download_vocab_encoder(output_dir: str) -> None:
    """
    Download the GPT2 vocabulary and encoder files.

    Args:
        output_dir: Directory to save the files
    """
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, "vocab.bpe")
    if not os.path.exists(vocab_path):
        logger.info(f"Downloading vocab.bpe to {vocab_path}")
        response = requests.get(VOCAB_URL)
        response.raise_for_status()
        with open(vocab_path, "wb") as f:
            f.write(response.content)

    encoder_path = os.path.join(output_dir, "encoder.json")
    if not os.path.exists(encoder_path):
        logger.info(f"Downloading encoder.json to {encoder_path}")
        response = requests.get(ENCODE_URL)
        response.raise_for_status()
        with open(encoder_path, "wb") as f:
            f.write(response.content)

    logger.info("Downloaded vocab.bpe and encoder.json")


def main() -> None:
    """
    Main function to demonstrate usage.
    """
    download_vocab_encoder(".")

    loader = GPT2WeightLoader("openai-community/gpt2")
    loader.download_weights()

    # Get all tensor names
    tensor_names = loader.get_tensor_names()
    print(f"\nTotal tensors: {len(tensor_names)}")
    print("\nAll tensor names:")
    for name in tensor_names:
        print(f"  {name}")

    # Dump tensor info
    tensor_info = loader.dump_tensors()

    # Group and print tensors by layer
    print("\nLayer groups:")
    layer_groups: Dict[str, List[str]] = {}
    for name in tensor_names:
        # More robust layer detection
        parts = name.split(".")
        if "h" in parts:
            # Find the index after "h"
            h_index = parts.index("h")
            if h_index + 1 < len(parts):
                layer_num = parts[h_index + 1]
                if layer_num not in layer_groups:
                    layer_groups[layer_num] = []
                layer_groups[layer_num].append(name)

    for layer, tensors in sorted(layer_groups.items(), key=lambda x: int(x[0])):
        print(f"\nLayer {layer} tensors:")
        for tensor in sorted(tensors):
            shape, dtype = tensor_info[tensor]
            print(f"  {tensor}: {shape}, {dtype}")


if __name__ == "__main__":
    main()
