import os
import torch
import math
import geoopt
from torch.utils.data import DataLoader
from src.model import TransformerLanguageModel
from src.dataset import Tokenizer, WikiTextDataset

# ==========================================
# HARDENED TRAINING: Adaptive Hybrid (ASLT)
# ==========================================

def main():
    embed_dim = 256
    num_heads = 4
    num_layers = 4
    seq_len = 128
    batch_size = 32
    max_vocab_size = 20000

    learning_rate = 3e-4
    epochs = 1
    eval_iters = 100
    log_interval = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Enable AMP for speed
    use_amp = device == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    data_dir = os.path.join('data', 'wikitext-103')
    train_path = os.path.join(data_dir, 'wiki.train.tokens')
    valid_path = os.path.join(data_dir, 'wiki.valid.tokens')
    vocab_path = os.path.join('checkpoints', 'vocab.pkl')
    checkpoint_path = os.path.join('checkpoints', 'best_model.pt')

    print(f"Using device: {device} | AMP: {use_amp}")

    if os.path.exists(vocab_path):
        tokenizer = Tokenizer(max_vocab_size=max_vocab_size)
        tokenizer.load(vocab_path)
    else:
        tokenizer = Tokenizer(max_vocab_size=max_vocab_size)
        tokenizer.build_vocab(train_path)
        tokenizer.save(vocab_path)

    train_dataset = WikiTextDataset(train_path, tokenizer, seq_len=seq_len)
    valid_dataset = WikiTextDataset(valid_path, tokenizer, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.stoi),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_len=seq_len
    )
    
    # Resume Logic
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created. Total trainable parameters: {num_params / 1e6:.2f} M")

    # RiemannianAdam is REQUIRED for ASLT manifold params
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=learning_rate, stabilize=10)

    best_val_loss = float('inf')

    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, loss = model(x, targets=y)

            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:5d} | Train Loss: {loss.item():.4f}")

            if batch_idx > 0 and batch_idx % eval_iters == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for v_idx, (vx, vy) in enumerate(valid_loader):
                        if v_idx >= 50: break
                        vx, vy = vx.to(device), vy.to(device)
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            _, v_loss = model(vx, targets=vy)
                        val_losses.append(v_loss.item())

                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"---> Eval at {batch_idx}: Val Loss {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), checkpoint_path)
                model.train()

    print("Training complete!")

if __name__ == "__main__":
    main()