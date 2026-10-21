import pytest

torch = pytest.importorskip("torch")

from transformers import Olmo3Config, Olmo3ForCausalLM  # noqa: E402

from kv_cache_lab.eviction import select_retained  # noqa: E402
from kv_cache_lab.research import attach, forward  # noqa: E402
from kv_cache_lab.research_cache import ResearchCache, ResearchLayer  # noqa: E402

pytestmark = pytest.mark.research


def tiny_model():
    torch.manual_seed(42)
    config = Olmo3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=2,
    )
    config._attn_implementation = "sdpa"
    return Olmo3ForCausalLM(config).eval()


@torch.inference_mode()
def test_cache_baseline_matches_native_olmo_across_sliding_window_boundary():
    model = tiny_model()
    ids = torch.tensor([[3, 5, 7, 9, 11, 13, 15, 17]])
    # Native full forward is a reference independent of our cache and masks.
    expected = model(ids, use_cache=False).logits
    cache = ResearchCache(model.config, backend="torch")
    attach(model, cache)
    actual = []
    for index in range(ids.shape[1]):
        actual.append(forward(model, cache, ids[:, index : index + 1], index))
    torch.testing.assert_close(torch.stack(actual, dim=1), expected, atol=1e-5, rtol=1e-4)
    assert cache.get_seq_length() == 8
    assert cache.layers[0].positions.tolist() == [5, 6, 7]


@torch.inference_mode()
def test_eviction_preserves_absolute_positions_and_enforces_budget():
    model = tiny_model()
    cache = ResearchCache(model.config, quantized=True, budget=4, sink=1, recent=2, backend="torch")
    attach(model, cache)
    for index in range(12):
        logits = forward(model, cache, torch.tensor([[3 + index]]), index)
        assert torch.isfinite(logits).all()
    full = cache.layers[1]
    assert len(full.positions) == 4
    assert full.positions[0].item() == 0
    assert full.positions[-2:].tolist() == [10, 11]
    assert cache.get_seq_length() == 12
    assert full.keys.dtype == torch.int8


def test_policy_retains_sinks_heavy_hitters_recent_and_chronological_order():
    scores = torch.tensor([0.0, 10.0, 1.0, 8.0, 2.0, 3.0, 0.0, 0.0])
    assert select_retained(scores, 5, 1, 2).tolist() == [0, 1, 3, 6, 7]
    with pytest.raises(ValueError):
        select_retained(scores, 1, 1, 1)


def test_quantized_append_does_not_requantize_existing_tokens():
    layer = ResearchLayer(quantized=True, backend="torch")
    first = torch.randn(1, 2, 3, 16)
    layer.update(first, first, {"cache_position": torch.arange(3)})
    original = layer.keys.clone()
    layer.update(first[:, :, :1], first[:, :, :1], {"cache_position": torch.tensor([3])})
    assert torch.equal(layer.keys[:, :, :3], original)


@torch.inference_mode()
def test_chunked_prefill_matches_native_logits_with_sliding_attention():
    model = tiny_model()
    ids = torch.tensor([[3, 5, 7, 9, 11, 13, 15, 17, 19]])
    expected = model(ids, use_cache=False).logits
    cache = ResearchCache(model.config, backend="torch")
    attach(model, cache)
    for start, end in ((0, 3), (3, 8), (8, 9)):
        actual = forward(model, cache, ids[:, start:end], start)
        torch.testing.assert_close(actual, expected[:, end - 1], atol=1e-5, rtol=1e-4)
