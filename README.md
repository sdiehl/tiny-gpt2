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
