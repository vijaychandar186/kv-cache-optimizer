"""OpenAI Triton kernels. Layout: contiguous [batch, KV heads, tokens, head_dim].

INT8 symmetric quantization uses a separate FP32 scale per token/head vector.
K and V have independent scales. Compaction physically gathers retained tokens.
"""

import torch
import triton
import triton.language as tl
import triton.testing


@triton.jit
def _quantize(X, Q, S, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK)
    values = tl.load(X + row * D + dims, dims < D, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), axis=0) / 127.0, 1.0e-12)
    scaled = values / scale
    # Round half away from zero; explicit saturation avoids INT8 overflow.
    rounded = tl.where(scaled >= 0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
    quantized = tl.minimum(tl.maximum(rounded, -127), 127).to(tl.int8)
    tl.store(Q + row * D + dims, quantized, dims < D)
    tl.store(S + row, scale)


@triton.jit
def _dequantize(Q, S, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK)
    values = tl.load(Q + row * D + dims, dims < D, other=0).to(tl.float32)
    scale = tl.load(S + row)
    tl.store(Y + row * D + dims, values * scale, dims < D)


@triton.jit
def _compact(X, INDICES, Y, T, KEEP, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    token = row % KEEP
    head = row // KEEP
    original = tl.load(INDICES + token)
    dims = tl.arange(0, BLOCK)
    values = tl.load(X + (head * T + original) * D + dims, dims < D, other=0)
    tl.store(Y + row * D + dims, values, dims < D)


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda or x.ndim != 4 or not x.is_floating_point():
        raise ValueError("quantize requires a floating CUDA tensor [B,H,T,D]")
    x = x.contiguous()
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
    if x.numel():
        _quantize[(x.numel() // x.shape[-1],)](
            x, q, scales, x.shape[-1], triton.next_power_of_2(x.shape[-1])
        )
    return q, scales


def dequantize(q: torch.Tensor, scales: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    if not q.is_cuda or q.dtype != torch.int8 or q.ndim != 4:
        raise ValueError("dequantize requires an INT8 CUDA tensor [B,H,T,D]")
    if scales.shape != (*q.shape[:-1], 1) or scales.device != q.device:
        raise ValueError("Scales must match [B,H,T,1] on the same device")
    q, scales = q.contiguous(), scales.contiguous()
    y = torch.empty_like(q, dtype=dtype)
    if q.numel():
        _dequantize[(q.numel() // q.shape[-1],)](
            q, scales, y, q.shape[-1], triton.next_power_of_2(q.shape[-1])
        )
    return y


def compact(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or x.ndim != 4 or indices.ndim != 1:
        raise ValueError("compact requires CUDA [B,H,T,D] and one-dimensional indices")
    if indices.dtype != torch.int64 or indices.device != x.device:
        raise ValueError("Indices must be int64 and on the tensor's device")
    # The policy produces sorted in-range indices; validation belongs in policy tests.
    x, indices = x.contiguous(), indices.contiguous()
    b, h, t, d = x.shape
    keep = indices.numel()
    result = torch.empty((b, h, keep, d), device=x.device, dtype=x.dtype)
    if keep:
        _compact[(b * h * keep,)](x, indices, result, t, keep, d, triton.next_power_of_2(d))
    return result


def warmup(device, dtype, dim):
    """Compile the storage/scale dtypes and divisible/nondivisible gather sizes."""
    for tokens, kept in ((32, 16), (33, 17)):
        x = torch.zeros((1, 1, tokens, dim), device=device, dtype=dtype)
        indices = torch.arange(kept, device=device)
        q, scales = quantize(x)
        dequantize(q, scales, dtype)
        for tensor in (x, q, scales):
            compact(tensor, indices)


def microbenchmark(output, tokens=8192, heads=8, dim=128):
    from .artifacts import provenance, write_json

    if not torch.cuda.is_available():
        raise RuntimeError("A Linux NVIDIA GPU is required")
    torch.manual_seed(42)
    x = torch.randn((1, heads, tokens, dim), device="cuda", dtype=torch.bfloat16)
    q, scales = quantize(x)
    restored = dequantize(q, scales, x.dtype)
    error = (restored.float() - x.float()).abs()
    keep = torch.arange(0, tokens, 2, device="cuda")
    torch.testing.assert_close(compact(q, keep), q.index_select(2, keep), rtol=0, atol=0)
    result = provenance() | {
        "shape": list(x.shape),
        "gpu": torch.cuda.get_device_name(),
        "quantize_ms": triton.testing.do_bench(lambda: quantize(x)),
        "dequantize_ms": triton.testing.do_bench(lambda: dequantize(q, scales, x.dtype)),
        "compact_ms": triton.testing.do_bench(lambda: compact(q, keep)),
        "max_abs_error": error.max().item(),
        "mean_abs_error": error.mean().item(),
        "source_bytes": x.numel() * x.element_size(),
        "quantized_bytes_including_scales": q.numel() + scales.numel() * scales.element_size(),
        "scope": "one tensor; standalone kernels, not model throughput",
    }
    write_json(output, result)
    return result
