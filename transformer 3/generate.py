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
    
    print("Loading vocabulary...")
    tokenizer = Tokenizer(max_vocab_size=20000)
    tokenizer.load(vocab_path)
    
    print("Loading model...")
    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.stoi),
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        seq_len=128
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({num_params / 1e6:.2f} M parameters)")
    
    print("\nGenerating text... (type 'quit' to exit)")
    while True:
        prompt = input("\nEnter prompt: ")
        if prompt.lower() == 'quit':
            break
        
        encoded = tokenizer.encode(prompt)
        if not encoded:
            print("Invalid input!")
            continue
        
        idx = torch.tensor([encoded], dtype=torch.long, device=device)
        
        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens=100, temperature=0.8)
        
        output = tokenizer.decode(generated[0].tolist())
        print(f"\nOutput: {output}")

if __name__ == "__main__":
    main()