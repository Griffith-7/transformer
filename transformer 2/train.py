import os
import torch
import geoopt
from torch.utils.data import DataLoader
from src.model import TransformerLanguageModel
from src.dataset import Tokenizer, WikiTextDataset

def main():
    embed_dim = 256
    num_heads = 4
    num_layers = 4
    seq_len = 128
    batch_size = 32
    learning_rate = 3e-4
    epochs = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    data_dir = os.path.join('data', 'wikitext-103')
    train_path = os.path.join(data_dir, 'wiki.train.tokens')
    valid_path = os.path.join(data_dir, 'wiki.valid.tokens')
    vocab_path = os.path.join('checkpoints', 'vocab.pkl')
    checkpoint_path = os.path.join('checkpoints', 'best_model.pt')

    tokenizer = Tokenizer()
    if os.path.exists(vocab_path): tokenizer.load(vocab_path)
    else: tokenizer.build_vocab(train_path); tokenizer.save(vocab_path)

    train_loader = DataLoader(WikiTextDataset(train_path, tokenizer, seq_len=seq_len), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(WikiTextDataset(valid_path, tokenizer, seq_len=seq_len), batch_size=batch_size, shuffle=False)

    model = TransformerLanguageModel(vocab_size=len(tokenizer.stoi), embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, seq_len=seq_len)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    print(f"Training Model: {sum(p.numel() for p in model.parameters())/1e6:.2f} M params")
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, loss = model(x, targets=y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx} | Loss {loss.item():.4f}")
                model.eval()
                v_losses = []
                with torch.no_grad():
                    for i, (vx, vy) in enumerate(valid_loader):
                        if i >= 20: break
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            _, v_loss = model(vx.to(device), vy.to(device))
                        v_losses.append(v_loss.item())
                avg_v = sum(v_losses)/len(v_losses)
                if avg_v < best_val_loss:
                    best_val_loss = avg_v
                    torch.save({'model_state_dict': model.state_dict(), 'config': model.config, 'tokenizer_stoi': tokenizer.stoi}, checkpoint_path)
                model.train()

if __name__ == "__main__":
    main()