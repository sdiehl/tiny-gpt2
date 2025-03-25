# gpt2-weights

Script to download GPT2 weights from HuggingFace and dump out the individual weights as numpy arrays.

* [GPT2 Safetensors](https://huggingface.co/openai-community/gpt2/blob/main/model.safetensors)
* [Download Link](https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors) ( 548 MB )

Also downloads the encoder and vocab files.

```python
from encoder import get_encoder

encoder = get_encoder("", ".")
encoded = encoder.encode("Hello, world!")

print(encoded) # [15496, 11, 995, 0]
```
