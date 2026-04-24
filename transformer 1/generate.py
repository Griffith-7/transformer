import os
import torch
from src.model import TransformerLanguageModel
from src.dataset import Tokenizer

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join('checkpoints', 'best_model.pt')
    
    if not os.path.exists(model_path):
        print("No trained model found. Run train.py first!")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    tokenizer = Tokenizer()
    tokenizer.stoi = checkpoint['tokenizer_stoi']
    tokenizer.itos = {i: s for s, i in tokenizer.stoi.items()}
    
    model = TransformerLanguageModel(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("\nGeneration Ready! Type a prompt Below:")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() in ['quit', 'exit']: break
        encoded = tokenizer.encode(prompt)
        idx = torch.tensor([encoded], dtype=torch.long, device=device)
        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens=100, temperature=0.8)
        print(f"\nResponse: {tokenizer.decode(generated[0].tolist())}")

if __name__ == "__main__":
    main()
