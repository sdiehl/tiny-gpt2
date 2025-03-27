# gpt2-weights

Script to download GPT-2 weights from HuggingFace and do inference. A reference
implementation of GPT-2 in pure Python.

* [GPT2 Safetensors](https://huggingface.co/openai-community/gpt2/blob/main/model.safetensors)
* [Download Link](https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors) ( 548 MB )

Also need to download the encoder and vocab files `vocab.bpe` and `encoder.json`


## Running

```bash
poetry install

# Download the model weights
poetry run python gpt2_loader.py

# Run the inference
poetry run python gpt2_run.py
```

## Structure

The core logic is split into several files:

* `gpt2_loader.py` - Downloads the weights and saves them to `model.safetensors`
* `gpt2_tensors.py` - Loads the tensors into layer forms that can be used for inference
* `gpt2_ops.py` - Implements the GPT-2 primitives (gelu, layernorm, softmax, etc.)
* `gpt2_run.py` - Loads the weights and runs inference

In addition, the following files are included:

* `encoder.py` - The BPE encoder from the original GPT-2 repository
* `tokenize.py` - Example of how to use the encoder

In the model directory, you will find the following files:

* `model.safetensors` - The model weights
* `vocab.bpe` - The BPE vocab from the original GPT-2 repository
* `encoder.json` - The encoder json from the original GPT-2 repository
* `config.json` - The GPT-2 model configuration

## Loading Weights into Data Structures

The process of loading the weights into data structures is handled by the `GPT2WeightLoader` class and the `GPT2TensorManager` class.

### GPT2WeightLoader

The `GPT2WeightLoader` class is responsible for downloading and loading the GPT-2 weights from HuggingFace. It provides methods to download the weights, get tensor names, get specific tensors, and dump tensors with their shapes and data types.

### GPT2TensorManager

The `GPT2TensorManager` class organizes the weights into the structure expected by the model implementation. It handles the loading and organizing of the GPT-2 weights from safetensors format into the structure expected by the model implementation. It also provides methods to load transformer blocks and model weights.

### Unpacking Weights for Layers

The weights for the layers are unpacked and organized into the following data structures:

* `LayerNormParams` - Contains the gamma (scale) and beta (bias) parameters for layer normalization.
* `LinearParams` - Contains the weight matrix and bias vector for linear layers.
* `MLPParams` - Contains the parameters for the multi-layer perceptron (MLP) block.
* `AttentionParams` - Contains the parameters for the attention block.
* `TransformerBlockParams` - Contains the parameters for a single transformer block.
* `ModelParams` - Contains the complete model parameters, including token embeddings, position embeddings, transformer blocks, and final layer norm.
* `HParams` - Contains the hyperparameters for the GPT-2 model, including the number of transformer blocks, number of attention heads, and context length.

### Fields Inside Layers

Each layer in the GPT-2 model contains several fields that are essential for its operation. Here is a description of each field inside the layers:

* `ln_1` - The first layer normalization parameters, which include gamma (scale) and beta (bias).
* `ln_2` - The second layer normalization parameters, which include gamma (scale) and beta (bias).
* `mlp` - The multi-layer perceptron block parameters, which include the first linear layer (`c_fc`) and the second linear layer (`c_proj`).
* `attn` - The attention block parameters, which include the QKV projection (`c_attn`) and the output projection (`c_proj`).
