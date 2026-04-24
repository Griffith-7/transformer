import os
import torch
from torch.utils.data import DataLoader
from src.model import TransformerLanguageModel
from src.dataset import Tokenizer, WikiTextDataset

def main():
    # STANDARD ATTENTION VERSION - same as T1 but compact
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

    use_amp = device == 'cuda'

    data_dir = os.path.join('data', 'wikitext-103')
    train_path = os.path.join(data_dir, 'wiki.train.tokens')
    valid_path = os.path.join(data_dir, 'wiki.valid.tokens')
    vocab_path = os.path.join('checkpoints', 'vocab.pkl')

    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")

    if os.path.exists(vocab_path):
        print("Loading existing vocabulary...")
        tokenizer = Tokenizer(max_vocab_size=max_vocab_size)
        tokenizer.load(vocab_path)
    else:
        print("Building new vocabulary...")
        tokenizer = Tokenizer(max_vocab_size=max_vocab_size)
        tokenizer.build_vocab(train_path)
        tokenizer.save(vocab_path)

    print("Preparing training dataset...")
    train_dataset = WikiTextDataset(train_path, tokenizer, seq_len=seq_len)

    print("Preparing validation dataset...")
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
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created. Total trainable parameters: {num_params / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    checkpoint_path = os.path.join('checkpoints', 'best_model.pt')
    if os.path.exists(checkpoint_path):
        print(f"Loading existing checkpoint from {checkpoint_path}...")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Successfully resumed from checkpoint.")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, loss = model(x, targets=y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:5d} | Train Loss: {loss.item():.4f}")

            if batch_idx > 0 and batch_idx % eval_iters == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for v_idx, (vx, vy) in enumerate(valid_loader):
                        if v_idx >= 20:
                            break
                        vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                        with torch.cuda.amp.autocast(enabled=use_amp):
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