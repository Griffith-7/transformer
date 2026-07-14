"""Tests for model.py across all three transformer versions."""

import os
import sys

import pytest
import torch


def _load_model_module(transformer_dir):
    sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", transformer_dir))
    for key in list(sys.modules.keys()):
        if key.startswith("src"):
            del sys.modules[key]
    if sys_path in sys.path:
        sys.path.remove(sys_path)
    sys.path.insert(0, sys_path)
    from src.model import TransformerLanguageModel

    return TransformerLanguageModel


def _make_small_model(cls, vocab_size=50, embed_dim=64, num_heads=4, num_layers=2, seq_len=32):
    return cls(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_len=seq_len,
    )


def _dummy_batch(vocab_size=50, batch_size=4, seq_len=32):
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))
    return x, y


class TestT1Standard:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.Model = _load_model_module("transformer 1")

    def test_instantiate(self):
        model = _make_small_model(self.Model)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0

    def test_forward_output_shape(self):
        model = _make_small_model(self.Model)
        x, y = _dummy_batch()
        logits, loss = model(x, targets=y)
        assert logits.shape == (4, 32, 50)
        assert loss.shape == ()

    def test_loss_is_finite(self):
        model = _make_small_model(self.Model)
        x, y = _dummy_batch()
        _, loss = model(x, targets=y)
        assert torch.isfinite(loss)

    def test_backward_pass(self):
        model = _make_small_model(self.Model)
        x, y = _dummy_batch()
        _, loss = model(x, targets=y)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_generate_shape(self):
        model = _make_small_model(self.Model)
        x = torch.randint(0, 50, (1, 5))
        gen = model.generate(x, max_new_tokens=10, temperature=1.0)
        assert gen.shape == (1, 15)

    def test_generate_in_range(self):
        model = _make_small_model(self.Model)
        x = torch.randint(0, 50, (1, 5))
        gen = model.generate(x, max_new_tokens=10, temperature=1.0)
        assert gen.min() >= 0 and gen.max() < 50

    def test_no_nans(self):
        model = _make_small_model(self.Model)
        x, y = _dummy_batch()
        logits, loss = model(x, targets=y)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(loss)

    def test_config_saved(self):
        model = _make_small_model(self.Model)
        assert hasattr(model, "config")
        assert model.config["vocab_size"] == 50
        assert model.config["embed_dim"] == 64

    def test_attention_is_causal(self):
        """First token's logits should not depend on later tokens."""
        model = _make_small_model(self.Model)
        model.eval()
        x1 = torch.randint(0, 50, (1, 10))
        x2 = torch.cat([x1, torch.randint(0, 50, (1, 5))], dim=1)
        with torch.no_grad():
            logits1, _ = model(x1)
            logits2, _ = model(x2)
        assert torch.allclose(logits1, logits2[:, :10, :], atol=1e-5)


class TestT2SpikingLorentz:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.Model = _load_model_module("transformer 2")

    def test_instantiate(self):
        model = _make_small_model(self.Model)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0

    def test_forward_output_shape(self):
        model = _make_small_model(self.Model)
        x, y = _dummy_batch()
        logits, loss = model(x, targets=y)
        assert logits.shape == (4, 32, 50)
        assert loss.shape == ()

    def test_loss_is_finite(self):
        model = _make_small_model(self.Model)
        x, y = _dummy_batch()
        _, loss = model(x, targets=y)
        assert torch.isfinite(loss)

    def test_backward_pass(self):
        model = _make_small_model(self.Model)
        x, y = _dummy_batch()
        _, loss = model(x, targets=y)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {p}"

    def test_no_nans(self):
        model = _make_small_model(self.Model)
        x, y = _dummy_batch()
        logits, loss = model(x, targets=y)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(loss)

    def test_generate(self):
        model = _make_small_model(self.Model)
        x = torch.randint(0, 50, (1, 5))
        gen = model.generate(x, max_new_tokens=10, temperature=1.0)
        assert gen.shape == (1, 15)
        assert gen.min() >= 0 and gen.max() < 50

    def test_spike_threshold_exists(self):
        model = _make_small_model(self.Model)
        thresholds = [p for n, p in model.named_parameters() if "spike_threshold" in n]
        assert len(thresholds) > 0

    def test_manifold_exists(self):
        model = _make_small_model(self.Model)
        attentions = [m for m in model.modules() if hasattr(m, "manifold")]
        assert len(attentions) > 0

    def test_attention_is_causal(self):
        model = _make_small_model(self.Model)
        model.eval()
        x1 = torch.randint(0, 50, (1, 10))
        x2 = torch.cat([x1, torch.randint(0, 50, (1, 5))], dim=1)
        with torch.no_grad():
            logits1, _ = model(x1)
            logits2, _ = model(x2)
        assert torch.allclose(logits1, logits2[:, :10, :], atol=1e-5)


class TestT3AdaptiveHyperbolic:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.Model = _load_model_module("transformer 3")

    def test_instantiate(self):
        model = _make_small_model(self.Model)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0

    def test_forward_output_shape(self):
        model = _make_small_model(self.Model)
        x, y = _dummy_batch()
        logits, loss = model(x, targets=y)
        assert logits.shape == (4, 32, 50)
        assert loss.shape == ()

    def test_loss_is_finite(self):
        model = _make_small_model(self.Model)
        x, y = _dummy_batch()
        _, loss = model(x, targets=y)
        assert torch.isfinite(loss)

    def test_backward_pass(self):
        model = _make_small_model(self.Model)
        x, y = _dummy_batch()
        _, loss = model(x, targets=y)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {p}"

    def test_no_nans(self):
        model = _make_small_model(self.Model)
        x, y = _dummy_batch()
        logits, loss = model(x, targets=y)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(loss)

    def test_generate(self):
        model = _make_small_model(self.Model)
        x = torch.randint(0, 50, (1, 5))
        gen = model.generate(x, max_new_tokens=10, temperature=1.0)
        assert gen.shape == (1, 15)
        assert gen.min() >= 0 and gen.max() < 50

    def test_log_k_exists(self):
        model = _make_small_model(self.Model)
        log_ks = [p for n, p in model.named_parameters() if "log_k" in n]
        assert len(log_ks) > 0

    def test_alpha_net_exists(self):
        model = _make_small_model(self.Model)
        alpha_nets = [p for n, p in model.named_parameters() if "alpha_net" in n]
        assert len(alpha_nets) > 0

    def test_curvature_learnable(self):
        model = _make_small_model(self.Model)
        log_k = [p for n, p in model.named_parameters() if "log_k" in n][0]
        assert log_k.requires_grad

    def test_qk_scale_exists(self):
        model = _make_small_model(self.Model)
        scales = [p for n, p in model.named_parameters() if "qk_scale" in n]
        assert len(scales) > 0

    def test_attention_is_causal(self):
        model = _make_small_model(self.Model)
        model.eval()
        x1 = torch.randint(0, 50, (1, 10))
        x2 = torch.cat([x1, torch.randint(0, 50, (1, 5))], dim=1)
        with torch.no_grad():
            logits1, _ = model(x1)
            logits2, _ = model(x2)
        assert torch.allclose(logits1, logits2[:, :10, :], atol=1e-5)


class TestSurrogateSpike:
    def _get_spike(self):
        for key in list(sys.modules.keys()):
            if key.startswith("src"):
                del sys.modules[key]
        t2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "transformer 2"))
        if t2_path in sys.path:
            sys.path.remove(t2_path)
        sys.path.insert(0, t2_path)
        from src.model import SurrogateSpike

        return SurrogateSpike

    def test_forward_threshold(self):
        SurrogateSpike = self._get_spike()
        scores = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        threshold = torch.tensor(0.5)
        result = SurrogateSpike.apply(scores, threshold)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        assert torch.equal(result, expected)

    def test_backward_uses_surrogate(self):
        SurrogateSpike = self._get_spike()
        scores = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], requires_grad=True)
        threshold = torch.tensor(0.5)
        result = SurrogateSpike.apply(scores, threshold)
        loss = result.sum()
        loss.backward()
        assert scores.grad is not None
        assert torch.isfinite(scores.grad).all()
