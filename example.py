from gpt2_tensors import load_gpt2_weights
from gpt2_run import generate

# Load model parameters once
print("Loading model parameters...")
params, hparams = load_gpt2_weights()

example_prompts = [
    ("The rain in Spain falls mainly in the", 40),
    ("You're a wizard Harry,", 10),
    ("What is the capital of France?", 10),
    ("Stephen Hawking is a", 40),
    ("The quick brown fox jumped over", 10),
    ("Star Wars is a movie about", 40),
    ("Alan Turing theorized that computers would one day become", 10),
]

# Run each prompt and print the results
for prompt, max_tokens in example_prompts:
    print(f"\nPrompt: {prompt}")
    generate(params, hparams, prompt.strip(), max_tokens)
