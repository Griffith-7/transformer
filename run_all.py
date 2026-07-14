"""Unified training and evaluation script for all 3 transformer versions.
Downloads WikiText-2, trains T1/T2/T3, compares results, generates text.
"""
import os
import sys
import time
import math
import torch
import pickle
import requests
import zipfile
import io
from torch.utils.data import DataLoader

# ─── Config ───────────────────────────────────────────────────────────
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
SEQ_LEN = 64
BATCH_SIZE = 16
LR = 3e-4
TOTAL_STEPS = 500
WARMUP_STEPS = 50
EVAL_EVERY = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_VOCAB = 10000
DATA_DIR = os.path.join('data', 'wikitext-2')
TRAIN_FILE = os.path.join(DATA_DIR, 'wiki.train.tokens')
VALID_FILE = os.path.join(DATA_DIR, 'wiki.valid.tokens')
TEST_FILE = os.path.join(DATA_DIR, 'wiki.test.tokens')
VOCAB_FILE = os.path.join('checkpoints', 'vocab.pkl')

# ─── Download WikiText-2 ──────────────────────────────────────────────
def download_wikitext2():
    if os.path.exists(TRAIN_FILE):
        print("WikiText-2 already downloaded.")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    print(f"Downloading WikiText-2 from {url}...")
    r = requests.get(url, allow_redirects=True, timeout=60)
    if r.status_code == 200 and len(r.content) > 10000:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall('data')
        print("Extracted to data/wikitext-2")
    else:
        print(f"  S3 URL returned {r.status_code}, generating synthetic corpus instead...")
        generate_synthetic_corpus()

def generate_synthetic_corpus():
    """Generate a synthetic corpus when download fails."""
    import random
    random.seed(42)
    vocab = [
        "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "shall", "can", "to", "of", "in", "for", "on", "with",
        "at", "by", "from", "as", "into", "through", "during", "before", "after",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either", "neither",
        "cat", "dog", "bird", "fish", "horse", "tree", "house", "river", "mountain", "sky",
        "sun", "moon", "star", "wind", "rain", "snow", "fire", "water", "earth", "air",
        "man", "woman", "child", "king", "queen", "world", "city", "land", "sea", "forest",
        "big", "small", "long", "short", "old", "new", "good", "bad", "great", "little",
        "ran", "sat", "walked", "flew", "sang", "played", "worked", "lived", "thought", "knew",
        "said", "told", "asked", "found", "gave", "took", "made", "came", "went", "saw",
    ]
    sentences = []
    templates = [
        "the {n} {v} in the {adj} {noun}",
        "a {n} {v} and the {noun} {v2}",
        "{n} and {n2} {v} in the {noun}",
        "the {adj} {noun} {v} the {adj} {n}",
        "in the {noun} , the {n} {v} {prep} {noun2}",
        "{n} was {adj} and the {noun} {v}",
        "the {v} {noun} and {n} {v2} the {adj} {noun2}",
        "when the {noun} {v} , the {n} {v2}",
    ]
    nouns = [w for w in vocab if w in ("cat","dog","bird","fish","horse","tree","house","river","mountain","sky","sun","moon","star","man","woman","child","king","queen","world","city","land","sea","forest")]
    adjs = [w for w in vocab if w in ("big","small","long","short","old","new","good","bad","great","little")]
    verbs = [w for w in vocab if w in ("ran","sat","walked","flew","sang","played","worked","lived","thought","knew","said","told","asked","found","gave","took","made","came","went","saw")]
    preps = ["in", "on", "with", "at", "by", "from", "to", "for"]
    
    for _ in range(2000):
        t = random.choice(templates)
        s = t.format(
            n=random.choice(nouns), n2=random.choice(nouns), noun=random.choice(nouns), noun2=random.choice(nouns),
            v=random.choice(verbs), v2=random.choice(verbs), adj=random.choice(adjs), prep=random.choice(preps)
        )
        sentences.append(s)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    split_sizes = [1600, 200, 200]
    texts = ["\n".join(sentences), "\n".join(sentences[1600:1800]), "\n".join(sentences[1800:])]
    for fname, text in zip(['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens'], texts):
        with open(os.path.join(DATA_DIR, fname), 'w') as f:
            f.write(text + "\n")
    print(f"  Generated {len(sentences)} synthetic sentences in {DATA_DIR}")

# ─── Shared imports (after download so paths exist) ──────────────────
def load_dataset_module(dir_name):
    for k in list(sys.modules.keys()):
        if k.startswith('src'):
            del sys.modules[k]
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), dir_name))
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    from src.dataset import Tokenizer, WikiTextDataset
    return Tokenizer, WikiTextDataset

def load_model_module(dir_name):
    for k in list(sys.modules.keys()):
        if k.startswith('src'):
            del sys.modules[k]
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), dir_name))
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    from src.model import TransformerLanguageModel
    return TransformerLanguageModel

# ─── LR Schedule ──────────────────────────────────────────────────────
def get_lr(step):
    if step < WARMUP_STEPS:
        return float(step) / float(max(1, WARMUP_STEPS))
    progress = float(step - WARMUP_STEPS) / float(max(1, TOTAL_STEPS - WARMUP_STEPS))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

# ─── Training loop ────────────────────────────────────────────────────
def train_model(name, model_cls, tokenizer):
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    model = model_cls(
        vocab_size=len(tokenizer.stoi), embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS, seq_len=SEQ_LEN
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params/1e6:.2f}M | Device: {DEVICE}")

    train_ds_cls = load_dataset_module(os.path.join(name, ''))[1]
    valid_ds_cls = load_dataset_module(os.path.join(name, ''))[1]

    train_ds = train_ds_cls(TRAIN_FILE, tokenizer, seq_len=SEQ_LEN)
    valid_ds = valid_ds_cls(VALID_FILE, tokenizer, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    use_amp = DEVICE == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    step = 0
    start_time = time.time()

    model.train()
    while step < TOTAL_STEPS:
        for x, y in train_loader:
            if step >= TOTAL_STEPS:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                _, loss = model(x, targets=y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            lr_scale = get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = LR * lr_scale
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

            if step % 50 == 0:
                print(f"  Step {step:4d}/{TOTAL_STEPS} | Loss: {loss.item():.4f} | LR: {LR * lr_scale:.2e}")

            if step > 0 and step % EVAL_EVERY == 0:
                model.eval()
                v_losses = []
                with torch.no_grad():
                    for i, (vx, vy) in enumerate(valid_loader):
                        if i >= 30:
                            break
                        vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            _, v_loss = model(vx, targets=vy)
                        v_losses.append(v_loss.item())
                avg_v = sum(v_losses) / len(v_losses)
                val_losses.append((step, avg_v))
                if avg_v < best_val_loss:
                    best_val_loss = avg_v
                    ckpt_dir = os.path.join('checkpoints', name)
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': model.config,
                        'tokenizer_stoi': tokenizer.stoi
                    }, os.path.join(ckpt_dir, 'best_model.pt'))
                print(f"  >>> Val Loss: {avg_v:.4f} (best: {best_val_loss:.4f})")
                model.train()

            step += 1

    elapsed = time.time() - start_time
    perplexity = math.exp(min(train_losses[-50:])) if train_losses[-1] < 10 else float('inf')
    print(f"\n  Done in {elapsed:.1f}s | Final train loss: {sum(train_losses[-50:])/50:.4f} | Approx PPL: {perplexity:.2f}")
    return {
        'name': name, 'params': params, 'final_loss': sum(train_losses[-50:])/50,
        'best_val_loss': best_val_loss, 'train_losses': train_losses,
        'val_losses': val_losses, 'elapsed': elapsed
    }

# ─── Generate text ────────────────────────────────────────────────────
def generate_text(name, tokenizer, prompt="the"):
    ckpt_dir = os.path.join('checkpoints', name)
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        print(f"  No checkpoint for {name}, skipping generation.")
        return

    model_cls = load_model_module(os.path.join(name, ''))
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = model_cls(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    enc = tokenizer.encode(prompt)
    idx = torch.tensor([enc], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=80, temperature=0.8)
    text = tokenizer.decode(generated[0].tolist())
    return text

# ─── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  EVOLUTION OF SPIKING HYPERBOLIC TRANSFORMERS")
    print("  Full Training Run — WikiText-2")
    print(f"  Device: {DEVICE} | AMP: {DEVICE == 'cuda'}")
    print("=" * 60)

    download_wikitext2()
    os.makedirs('checkpoints', exist_ok=True)

    Tokenizer, _ = load_dataset_module('transformer 1')
    tokenizer = Tokenizer(max_vocab_size=MAX_VOCAB)
    if os.path.exists(VOCAB_FILE):
        tokenizer.load(VOCAB_FILE)
    else:
        tokenizer.build_vocab(TRAIN_FILE)
        tokenizer.save(VOCAB_FILE)
    print(f"Vocab: {len(tokenizer.stoi)} tokens")

    results = []

    # T1
    T1_cls = load_model_module('transformer 1')
    r1 = train_model('transformer 1', T1_cls, tokenizer)
    results.append(r1)

    # T2
    T2_cls = load_model_module('transformer 2')
    r2 = train_model('transformer 2', T2_cls, tokenizer)
    results.append(r2)

    # T3
    T3_cls = load_model_module('transformer 3')
    r3 = train_model('transformer 3', T3_cls, tokenizer)
    results.append(r3)

    # ─── Comparison Table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)
    print(f"  {'Model':<25} {'Params':>8} {'Train Loss':>12} {'Val Loss':>12} {'Time':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*12} {'-'*8}")
    for r in results:
        name_short = r['name'].replace('transformer ', 'T')
        ppl = math.exp(r['final_loss']) if r['final_loss'] < 10 else 999
        print(f"  {name_short + ' (' + name_short + ')':<25} {r['params']/1e6:>7.2f}M {r['final_loss']:>12.4f} {r['best_val_loss']:>12.4f} {r['elapsed']:>7.1f}s")
    print()

    # ─── Generation ───────────────────────────────────────────────────
    print("=" * 70)
    print("  TEXT GENERATION")
    print("=" * 70)
    prompt = "the"
    for r in results:
        text = generate_text(r['name'], tokenizer, prompt)
        label = r['name'].replace('transformer ', 'T')
        print(f"\n  [{label}] Prompt: '{prompt}'")
        print(f"  {text}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
