import sys

import pytest

if sys.platform != "linux":
    pytest.skip("Triton kernels require Linux", allow_module_level=True)
torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from kv_cache_lab.kernels import compact, dequantize, quantize  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dim", [80, 128])
@pytest.mark.parametrize("tokens", [1, 127, 2048])
def test_quantization_error_is_bounded_and_compaction_is_exact(dim, tokens):
    torch.manual_seed(42)
    x = torch.randn((2, 3, tokens, dim), dtype=torch.float16, device="cuda")
    q, scales = quantize(x)
    restored = dequantize(q, scales, torch.float32)
    # One half-step plus input/FP32 rounding, including non-power-of-two head dims.
    assert torch.all((restored - x.float()).abs() <= scales * 0.501 + 1e-6)
    indices = torch.arange(0, tokens, 2, device="cuda")
    torch.testing.assert_close(compact(q, indices), q.index_select(2, indices), rtol=0, atol=0)
    torch.testing.assert_close(
        compact(scales, indices), scales.index_select(2, indices), rtol=0, atol=0
    )


def test_zero_vector_quantizes_without_nan():
    x = torch.zeros((1, 1, 8, 128), device="cuda")
    q, scales = quantize(x)
    assert torch.equal(dequantize(q, scales, torch.float32), x)
    assert torch.isfinite(scales).all()


@torch.inference_mode()
def test_triton_cache_integration_matches_torch_reference():
    from transformers import Olmo3Config, Olmo3ForCausalLM

    from kv_cache_lab.research import attach, forward
    from kv_cache_lab.research_cache import ResearchCache

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
    model = Olmo3ForCausalLM(config).cuda().eval()
    outputs = []
    for backend in ("torch", "triton"):
        cache = ResearchCache(config, quantized=True, budget=4, sink=1, recent=2, backend=backend)
        attach(model, cache)
        logits = []
        for position in range(12):
            logits.append(forward(model, cache, torch.tensor([[3 + position]]), position))
        outputs.append(torch.stack(logits))
        assert cache.layers[1].positions[-2:].tolist() == [10, 11]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)
