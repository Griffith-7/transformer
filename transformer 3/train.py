import os
import torch
import math
import geoopt
from torch.utils.data import DataLoader
from src.model import TransformerLanguageModel
from src.dataset import Tokenizer, WikiTextDataset

# ==========================================
# PHASE 5: PROFESSIONAL RESEARCH TRAINING
# ==========================================

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    embed_dim = 256
    num_heads = 4
    num_layers = 4
    seq_len = 128
    batch_size = 32
    max_vocab_size = 20000

    learning_rate = 3e-4
    epochs = 1
    total_steps = 10000
    warmup_steps = 1000
    
    eval_iters = 200
    log_interval = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # MODERN API: torch.amp
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    data_dir = os.path.join('data', 'wikitext-103')
    train_path = os.path.join(data_dir, 'wiki.train.tokens')
    valid_path = os.path.join(data_dir, 'wiki.valid.tokens')
    vocab_path = os.path.join('checkpoints', 'vocab.pkl')
    checkpoint_path = os.path.join('checkpoints', 'best_model.pt')

    print(f"Starting ASLT-Professional Run | Device: {device} | AMP: {use_amp}")

    if os.path.exists(vocab_path):
        tokenizer = Tokenizer(max_vocab_size=max_vocab_size)
        tokenizer.load(vocab_path)
    else:
        tokenizer = Tokenizer(max_vocab_size=max_vocab_size)
        tokenizer.build_vocab(train_path)
        tokenizer.save(vocab_path)

    train_dataset = WikiTextDataset(train_path, tokenizer, seq_len=seq_len)
    valid_dataset = WikiTextDataset(valid_path, tokenizer, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.stoi),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_len=seq_len
    )
    
    model.to(device)
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=learning_rate, stabilize=10)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    best_val_loss = float('inf')

    print(f"Training Model: {sum(p.numel() for p in model.parameters())/1e6:.2f} M params")
    model.train()
    step = 0
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= total_steps: break
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, loss = model(x, targets=y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % log_interval == 0:
                print(f"Step {step:5d} | Loss {loss.item():.4f} | LR {scheduler.get_last_lr()[0]:.2e}")

            if step > 0 and step % eval_iters == 0:
                model.eval()
                v_losses = []
                with torch.no_grad():
                    for i, (vx, vy) in enumerate(valid_loader):
                        if i >= 50: break
                        vx, vy = vx.to(device), vy.to(device)
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            _, v_loss = model(vx, targets=vy)
                        v_losses.append(v_loss.item())
                
                avg_v = sum(v_losses) / len(v_losses)
                print(f"---> Eval at step {step}: Val Loss {avg_v:.4f}")
                
                if avg_v < best_val_loss:
                    best_val_loss = avg_v
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'config': model.config, # SMART: Save the architecture config
                        'tokenizer_stoi': tokenizer.stoi
                    }
                    torch.save(checkpoint, checkpoint_path)
                model.train()
            step += 1

    print("Success. Professional Grade Model Training Complete.")

if __name__ == "__main__":
    main()