# Benchmark Results

## Training Comparison (Synthetic WikiText-2, 70-token vocab)

All models: ~0.41M params, embed_dim=128, num_heads=4, num_layers=2, seq_len=64.

| Model | Variant | Train Loss | Perplexity | Time (s) | Throughput |
|-------|---------|-----------|-----------|---------|-----------|
| T1 | Standard | 2.168 | 8.74 | 3.6 | ~533k tok/s |
| T2 | Spiking Lorentz | 2.259 | 9.57 | 9.4 | ~204k tok/s |
| T3 | Adaptive Hybrid | 2.479 | 11.93 | 9.3 | ~207k tok/s |

![Benchmark Comparison](benchmark_comparison.png)
![Perplexity](perplexity_comparison.png)

## Reproducing

```bash
# Install
pip install -e ".[dev]"

# Train all three and compare
hypertransformer compare \
  --train-file data/wikitext-2/wiki.train.tokens \
  --valid-file data/wikitext-2/wiki.valid.tokens \
  --steps 2000

# Or use the unified script
python run_all.py
```
