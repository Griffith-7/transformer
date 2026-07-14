"""Text generation from a trained checkpoint."""
import os

import torch


def load_model(checkpoint_path, variant="standard", device=None):
    """Load a model from a checkpoint file.

    Returns:
        Tuple of (model, tokenizer_stoi, device).
    """
    from .model import TransformerLanguageModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    config["variant"] = variant

    model = TransformerLanguageModel(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint["tokenizer_stoi"], device


def generate(model, tokenizer_stoi, prompt, *, max_new_tokens=100, temperature=0.8, device=None):
    """Generate text from a prompt.

    Returns:
        Generated token indices as a 1-D tensor.
    """
    itos = {i: s for s, i in tokenizer_stoi.items()}
    encode = lambda text: [tokenizer_stoi.get(w, tokenizer_stoi.get("<unk>", 1)) for w in text.split()]
    decode = lambda indices: " ".join(itos.get(idx, "<unk>") for idx in indices)

    if device is None:
        device = next(model.parameters()).device

    encoded = encode(prompt)
    idx = torch.tensor([encoded], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    return generated[0].tolist(), decode


def interactive(model_path, variant="standard"):
    """Start an interactive generation session."""
    model, stoi, device = load_model(model_path, variant=variant)
    itos = {i: s for s, i in stoi.items()}
    decode = lambda indices: " ".join(itos.get(idx, "<unk>") for idx in indices)

    print("\nGeneration Ready! Type a prompt below (or 'quit' to exit):")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() in ("quit", "exit"):
            break
        indices, _ = generate(model, stoi, prompt, device=device)
        print(f"\nResponse: {decode(indices)}")
