"""Benchmark all three transformer variants and generate comparison charts.

Trains T1/T2/T3, records loss curves and throughput, saves plots to benchmarks/.
"""
import os
import sys
import math
import time
import json

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hypertransformer.data import Tokenizer, WikiTextDataset
from hypertransformer.model import TransformerLanguageModel
from hypertransformer.train import train, cosine_lr

# ── Config ─────────────────────────────────────────────────────────────
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
SEQ_LEN = 64
BATCH_SIZE = 16
LR = 3e-4
TOTAL_STEPS = 500
WARMUP_STEPS = 50
EVAL_EVERY = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join("data", "wikitext-2")
TRAIN_FILE = os.path.join(DATA_DIR, "wiki.train.tokens")
VALID_FILE = os.path.join(DATA_DIR, "wiki.valid.tokens")
BENCH_DIR = "benchmarks"

VARIANTS = ["standard", "spiking_lorentz", "adaptive"]
LABELS = {"standard": "T1 Standard", "spiking_lorentz": "T2 Spiking Lorentz", "adaptive": "T3 Adaptive Hybrid"}
COLORS = {"standard": "#2196F3", "spiking_lorentz": "#FF9800", "adaptive": "#4CAF50"}


def run_benchmarks():
    os.makedirs(BENCH_DIR, exist_ok=True)

    tokenizer = Tokenizer(max_vocab_size=10000)
    tokenizer.build_vocab(TRAIN_FILE)

    results = {}
    for v in VARIANTS:
        print(f"\n{'='*60}")
        print(f"  Benchmarking: {LABELS[v]}")
        print(f"{'='*60}")

        model_cls = lambda _v=v, **kw: TransformerLanguageModel(variant=_v, **kw)

        r = train(
            model_cls,
            tokenizer,
            TRAIN_FILE,
            VALID_FILE,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            seq_len=SEQ_LEN,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            total_steps=TOTAL_STEPS,
            warmup_steps=WARMUP_STEPS,
            eval_every=EVAL_EVERY,
            checkpoint_dir=os.path.join("checkpoints", v),
            device=DEVICE,
        )
        results[v] = r

    # Save raw results
    serializable = {}
    for v, r in results.items():
        serializable[v] = {
            "final_loss": r["final_loss"],
            "best_val_loss": r["best_val_loss"],
            "elapsed": r["elapsed"],
            "train_losses": r["train_losses"],
            "val_losses": r["val_losses"],
        }
    with open(os.path.join(BENCH_DIR, "results.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    generate_charts(results)
    print(f"\n  Charts saved to {BENCH_DIR}/")


def generate_charts(results):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 1. Training Loss Curves ──────────────────────────────────────
    ax = axes[0]
    for v in VARIANTS:
        losses = results[v]["train_losses"]
        # Smooth with moving average
        window = 10
        smoothed = []
        for i in range(len(losses)):
            start = max(0, i - window)
            smoothed.append(sum(losses[start:i + 1]) / (i - start + 1))
        ax.plot(smoothed, label=LABELS[v], color=COLORS[v], linewidth=2)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    # ── 2. Validation Loss Comparison ─────────────────────────────────
    ax = axes[1]
    names = [LABELS[v] for v in VARIANTS]
    val_losses = [results[v]["best_val_loss"] for v in VARIANTS]
    bar_colors = [COLORS[v] for v in VARIANTS]
    bars = ax.bar(names, val_losses, color=bar_colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, val_losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Best Validation Loss", fontsize=12)
    ax.set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(val_losses) * 1.2)

    # ── 3. Throughput Comparison ──────────────────────────────────────
    ax = axes[2]
    times = [results[v]["elapsed"] for v in VARIANTS]
    # Calculate approximate throughput (tokens/sec estimate)
    throughputs = []
    for v in VARIANTS:
        t = results[v]["elapsed"]
        tokens_per_step = BATCH_SIZE * SEQ_LEN
        total_tokens = TOTAL_STEPS * tokens_per_step
        throughputs.append(total_tokens / t / 1000)  # k tokens/sec

    bars = ax.bar(names, throughputs, color=bar_colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}k", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Throughput (k tokens/sec)", fontsize=12)
    ax.set_title("Training Throughput", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(throughputs) * 1.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BENCH_DIR, "benchmark_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4. Perplexity comparison ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ppls = []
    for v in VARIANTS:
        ppl = math.exp(results[v]["final_loss"]) if results[v]["final_loss"] < 10 else 999
        ppls.append(ppl)

    bars = ax.bar(names, ppls, color=bar_colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    ax.set_title("Final Perplexity Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(ppls) * 1.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BENCH_DIR, "perplexity_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_benchmarks()
