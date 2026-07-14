"""CLI entry point for hypertransformer."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="hypertransformer",
        description="Evolution of Spiking Hyperbolic Transformers",
    )
    sub = parser.add_subparsers(dest="command")

    # ── train ──
    train_p = sub.add_parser("train", help="Train a transformer variant")
    train_p.add_argument(
        "--variant",
        choices=["standard", "spiking_lorentz", "adaptive"],
        default="standard",
        help="Model variant (default: standard)",
    )
    train_p.add_argument("--embed-dim", type=int, default=256)
    train_p.add_argument("--num-heads", type=int, default=4)
    train_p.add_argument("--num-layers", type=int, default=4)
    train_p.add_argument("--seq-len", type=int, default=128)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--steps", type=int, default=5000)
    train_p.add_argument("--warmup", type=int, default=200)
    train_p.add_argument("--eval-every", type=int, default=500)
    train_p.add_argument("--checkpoint-dir", default="checkpoints")
    train_p.add_argument("--train-file", required=True)
    train_p.add_argument("--valid-file", required=True)
    train_p.add_argument("--vocab-file", default=None)

    # ── generate ──
    gen_p = sub.add_parser("generate", help="Generate text from a trained model")
    gen_p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    gen_p.add_argument("--variant", default="standard", help="Model variant")
    gen_p.add_argument("--prompt", default="the", help="Text prompt")
    gen_p.add_argument("--max-tokens", type=int, default=100)
    gen_p.add_argument("--temperature", type=float, default=0.8)

    # ── compare ──
    cmp_p = sub.add_parser("compare", help="Train all three variants and compare")
    cmp_p.add_argument("--embed-dim", type=int, default=256)
    cmp_p.add_argument("--num-heads", type=int, default=4)
    cmp_p.add_argument("--num-layers", type=int, default=4)
    cmp_p.add_argument("--seq-len", type=int, default=128)
    cmp_p.add_argument("--batch-size", type=int, default=32)
    cmp_p.add_argument("--lr", type=float, default=3e-4)
    cmp_p.add_argument("--steps", type=int, default=2000)
    cmp_p.add_argument("--train-file", required=True)
    cmp_p.add_argument("--valid-file", required=True)

    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "generate":
        _cmd_generate(args)
    elif args.command == "compare":
        _cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_train(args):
    import os
    import pickle

    import torch

    from .data import Tokenizer
    from .model import TransformerLanguageModel
    from .train import train

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer()
    if args.vocab_file and os.path.exists(args.vocab_file):
        tokenizer.load(args.vocab_file)
    else:
        tokenizer.build_vocab(args.train_file)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        vocab_path = args.vocab_file or os.path.join(args.checkpoint_dir, "vocab.pkl")
        tokenizer.save(vocab_path)

    variant_map = {
        "standard": lambda **kw: TransformerLanguageModel(variant="standard", **kw),
        "spiking_lorentz": lambda **kw: TransformerLanguageModel(variant="spiking_lorentz", **kw),
        "adaptive": lambda **kw: TransformerLanguageModel(variant="adaptive", **kw),
    }
    model_cls = variant_map[args.variant]

    train(
        model_cls,
        tokenizer,
        args.train_file,
        args.valid_file,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        total_steps=args.steps,
        warmup_steps=args.warmup,
        eval_every=args.eval_every,
        checkpoint_dir=os.path.join(args.checkpoint_dir, args.variant),
        device=device,
    )


def _cmd_generate(args):
    from .generate import interactive
    interactive(args.checkpoint, variant=args.variant)


def _cmd_compare(args):
    import math
    import os
    import pickle

    import torch

    from .data import Tokenizer
    from .model import TransformerLanguageModel
    from .generate import generate as gen_text
    from .train import train

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer()
    tokenizer.build_vocab(args.train_file)

    variants = ["standard", "spiking_lorentz", "adaptive"]
    results = {}

    for v in variants:
        model_cls = lambda _v=v, **kw: TransformerLanguageModel(variant=_v, **kw)
        ckpt_dir = os.path.join("checkpoints", v)
        r = train(
            model_cls,
            tokenizer,
            args.train_file,
            args.valid_file,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            total_steps=args.steps,
            checkpoint_dir=ckpt_dir,
            device=device,
        )
        results[v] = r

    print("\n" + "=" * 60)
    print("  RESULTS COMPARISON")
    print("=" * 60)
    print(f"  {'Variant':<20} {'Loss':>10} {'PPL':>10} {'Time':>8}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 8}")
    for v, r in results.items():
        ppl = math.exp(r["final_loss"]) if r["final_loss"] < 10 else 999
        print(f"  {v:<20} {r['final_loss']:>10.4f} {ppl:>10.2f} {r['elapsed']:>7.1f}s")


if __name__ == "__main__":
    main()
