import os
import torch
import math
import geoopt
from torch.utils.data import DataLoader
from src.model import TransformerLanguageModel
from src.dataset import Tokenizer, WikiTextDataset

def main():
    # --- Configurations ---
    embed_dim = 64
    num_heads = 4
    num_layers = 4
    seq_len = 128
    batch_size = 32    # RTX 3050 4GB should handle this
    max_vocab_size = 20000

    # Training parameters
    learning_rate = 3e-4
    epochs = 1
    eval_iters = 100
    log_interval = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths
    data_dir = os.path.join('data', 'wikitext-103')
    train_path = os.path.join(data_dir, 'wiki.train.tokens')
    valid_path = os.path.join(data_dir, 'wiki.valid.tokens')

    print(f"Using device: {device}")

    # --- Data Prep ---
    tokenizer = Tokenizer(max_vocab_size=max_vocab_size)
    vocab_path = os.path.join('checkpoints', 'vocab.pkl')
    if os.path.exists(vocab_path):
        print("Loading existing vocabulary...")
        tokenizer.load(vocab_path)
    else:
        tokenizer.build_vocab(train_path)
        tokenizer.save(vocab_path)

    print("Preparing training dataset...")
    train_dataset = WikiTextDataset(train_path, tokenizer, seq_len=seq_len)

    print("Preparing validation dataset...")
    valid_dataset = WikiTextDataset(valid_path, tokenizer, seq_len=seq_len)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Model Setup ---
    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.stoi),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_len=seq_len
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created. Total trainable parameters: {num_params / 1e6:.2f} M")

    # --- Optimizer Setup ---
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=learning_rate, stabilize=10)

    # --- Training Loop ---
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, loss = model(x, targets=y)

            loss.backward()

            optimizer.step()

            # Logging
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:5d} | Train Loss: {loss.item():.4f}")

            # Validation Step
            if batch_idx > 0 and batch_idx % eval_iters == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for v_idx, (vx, vy) in enumerate(valid_loader):
                        if v_idx >= 20:
                            break
                        vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                        _, v_loss = model(vx, targets=vy)
                        val_losses.append(v_loss.item())

                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"\n---> Evaluation at batch {batch_idx}: Validation Loss {avg_val_loss:.4f} <---")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"Saving checkpoint to 'checkpoints/best_model.pt'...")
                    torch.save(model.state_dict(), os.path.join('checkpoints', 'best_model.pt'))
                print()

                model.train()

    print("Training complete!")

if __name__ == "__main__":
    main()