"""Training loop with cosine LR schedule, AMP, gradient clipping, and checkpointing."""
import math
import os
import time

import torch
from torch.utils.data import DataLoader


def cosine_lr(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    model,
    tokenizer,
    train_path,
    valid_path,
    *,
    embed_dim=256,
    num_heads=4,
    num_layers=4,
    seq_len=128,
    batch_size=32,
    learning_rate=3e-4,
    total_steps=5000,
    warmup_steps=200,
    eval_every=500,
    checkpoint_dir="checkpoints",
    device=None,
    dataset_cls=None,
):
    """Train a :class:`TransformerLanguageModel` and return training stats.

    Args:
        model: An uninitialised model class (not an instance).
        tokenizer: Fitted :class:`Tokenizer`.
        train_path: Path to training corpus.
        valid_path: Path to validation corpus.
        embed_dim: Embedding dimension (used when constructing the model).
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        seq_len: Sequence length.
        batch_size: Batch size.
        learning_rate: Peak learning rate.
        total_steps: Total training steps.
        warmup_steps: Linear warmup steps.
        eval_every: Evaluate every N steps.
        checkpoint_dir: Directory to save best checkpoint.
        device: ``"cuda"`` or ``"cpu"``.
        dataset_cls: Dataset class to use (defaults to
            :class:`~hypertransformer.data.WikiTextDataset`).

    Returns:
        dict with keys ``final_loss``, ``best_val_loss``, ``elapsed``,
        ``train_losses``, ``val_losses``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset_cls is None:
        from .data import WikiTextDataset

        dataset_cls = WikiTextDataset

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    m = model(
        vocab_size=len(tokenizer.stoi),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_len=seq_len,
    ).to(device)

    params = sum(p.numel() for p in m.parameters())
    print(f"  Params: {params / 1e6:.2f}M | Device: {device}")

    train_ds = dataset_cls(train_path, tokenizer, seq_len=seq_len)
    valid_ds = dataset_cls(valid_path, tokenizer, seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    step = 0
    start_time = time.time()

    m.train()
    while step < total_steps:
        for x, y in train_loader:
            if step >= total_steps:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = m(x, targets=y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            lr_scale = cosine_lr(step, warmup_steps, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = learning_rate * lr_scale
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

            if step % 50 == 0:
                print(
                    f"  Step {step:4d}/{total_steps} | "
                    f"Loss: {loss.item():.4f} | LR: {learning_rate * lr_scale:.2e}"
                )

            if step > 0 and step % eval_every == 0:
                m.eval()
                v_losses = []
                with torch.no_grad():
                    for i, (vx, vy) in enumerate(valid_loader):
                        if i >= 30:
                            break
                        vx, vy = vx.to(device), vy.to(device)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            _, v_loss = m(vx, targets=vy)
                        v_losses.append(v_loss.item())
                avg_v = sum(v_losses) / len(v_losses)
                val_losses.append((step, avg_v))
                if avg_v < best_val_loss:
                    best_val_loss = avg_v
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": m.state_dict(),
                            "config": m.config,
                            "tokenizer_stoi": tokenizer.stoi,
                        },
                        os.path.join(checkpoint_dir, "best_model.pt"),
                    )
                print(f"  >>> Val Loss: {avg_v:.4f} (best: {best_val_loss:.4f})")
                m.train()

            step += 1

    elapsed = time.time() - start_time
    recent = train_losses[-50:] if train_losses else [float("inf")]
    ppl = math.exp(sum(recent) / len(recent))
    print(
        f"\n  Done in {elapsed:.1f}s | "
        f"Final loss: {sum(recent) / len(recent):.4f} | PPL: {ppl:.2f}"
    )
    return {
        "final_loss": sum(recent) / len(recent),
        "best_val_loss": best_val_loss,
        "elapsed": elapsed,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
