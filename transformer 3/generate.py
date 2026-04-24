import os
import torch
from src.model import TransformerLanguageModel
from src.dataset import Tokenizer

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    vocab_path = os.path.join('checkpoints', 'vocab.pkl')
    model_path = os.path.join('checkpoints', 'best_model.pt')
    
    if not os.path.exists(model_path):
        print("No trained model found. Run train.py first!")
        return
    
    tokenizer = Tokenizer(max_vocab_size=20000)
    tokenizer.load(vocab_path)
    
    # HARDENED: dimension 256
    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.stoi),
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        seq_len=128
    )
    
    print(f"Loading checkpoint from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("\nGeneration Ready! Type a prompt Below:")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() == 'quit': break
        
        encoded = tokenizer.encode(prompt)
        idx = torch.tensor([encoded], dtype=torch.long, device=device)
        
        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens=100, temperature=0.8)
        
        print(f"\nResponse: {tokenizer.decode(generated[0].tolist())}")

if __name__ == "__main__":
    main()