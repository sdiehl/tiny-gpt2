# gpt2-weights

Script to download GPT-2 weights from HuggingFace and do inference. A reference
implementation of GPT-2 in pure Python.

* [GPT2 Safetensors](https://huggingface.co/openai-community/gpt2/blob/main/model.safetensors)
* [Download Link](https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors) ( 548 MB )

Also need to download the encoder and vocab files `vocab.bpe` and `encoder.json`


## Running

```bash
poetry install
poetry run python gpt2_loader.py
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
* `vocab.bpe` - The BPE vocab from the original GPT-2 repository
* `encoder.json` - The encoder json from the original GPT-2 repository
* `config.json` - The GPT-2 model configuration
