import os
import torch
from src.model import TransformerLanguageModel
from src.dataset import Tokenizer

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading on device: {device}")

    # Load tokenizer
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join('checkpoints', 'vocab.pkl'))
    
    # Model parameters must match train.py exactly
    embed_dim = 256
    num_heads = 4
    num_layers = 4
    seq_len = 128
    
    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.stoi),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_len=seq_len
    )
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(os.path.join('checkpoints', 'best_model.pt'), map_location=device))
        print("Successfully loaded exactly trained 'best_model.pt'!")
    except Exception as e:
        print(f"Could not load best_model.pt: {e}")
        return
        
    model.to(device)
    model.eval()

    print("\n" + "="*50)
    print("Transformer Text Generator is Ready!")
    print("Type a prompt and press Enter. Type 'quit' to exit.")
    print("="*50 + "\n")

    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() == 'quit':
            break
            
        # Encode prompt
        idx = tokenizer.encode(prompt)
        idx = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0) # (1, T)
        
        # Generate raw response
        try:
            generated_idx = model.generate(idx, max_new_tokens=50, temperature=0.8)
            
            # Decode generated output, dropping the original prompt length
            # If you want to see the prompt too, use generated_idx[0].tolist()
            generated_text = tokenizer.decode(generated_idx[0].tolist())
            print(f"\nResponse: {generated_text}")
            
        except Exception as e:
            print(f"Oops, something went wrong generating text: {e}")

if __name__ == "__main__":
    main()
