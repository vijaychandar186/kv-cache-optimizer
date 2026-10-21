"""End-to-end cache ablations on real OLMo3 weights, using the same decoder for each arm."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .artifacts import provenance, write_json
from .config import model_lock
from .prompts import prompt_ids
from .research_cache import ResearchCache, research_attention


def synchronize():
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def attach(model, cache):
    ALL_ATTENTION_FUNCTIONS["kv_lab"] = research_attention
    model.config._attn_implementation = "kv_lab"
    for layer in model.model.layers:
        layer.self_attn._kv_lab_cache = cache


def forward(model, cache, ids, start):
    device = model.get_input_embeddings().weight.device
    positions = torch.arange(start, start + ids.shape[1], device=device)
    result = model(
        input_ids=ids.to(device),
        past_key_values=cache,
        use_cache=True,
        cache_position=positions,
        position_ids=positions[None, :],
        attention_mask=cache.masks(positions, model.dtype, device),
        logits_to_keep=1,
    )
    cache.finish_step()
    return result.logits[:, -1, :]


@torch.inference_mode()
def run_variant(model, config, tokens, continuation, args, variant):
    quantized = variant in ("int8", "int8-evict")
    budget = args.budget if variant in ("evict", "int8-evict") else None
    cache = ResearchCache(config, quantized, budget, args.sink, args.recent, args.backend)
    attach(model, cache)
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)
    synchronize()
    begin = time.perf_counter()
    for start in range(0, len(tokens), args.prefill_chunk):
        logits = forward(
            model, cache, torch.tensor([tokens[start : start + args.prefill_chunk]]), start
        )
    synchronize()
    prefill_seconds = time.perf_counter() - begin
    # Teacher-force one common continuation so NLL measures the same targets in every arm.
    nll = []
    teacher_start = time.perf_counter()
    for index, token in enumerate(continuation):
        nll.append(-logits.float().log_softmax(-1)[0, token].item())
        logits = forward(model, cache, torch.tensor([[token]]), len(tokens) + index)
    synchronize()
    teacher_seconds = time.perf_counter() - teacher_start
    generated = []
    intervals = []
    generated_start = time.perf_counter()
    for index in range(args.output_tokens):
        tick = time.perf_counter()
        token = logits.argmax(-1).item()
        generated.append(token)
        # EOS is intentionally ignored for fixed-work decode measurements.
        logits = forward(
            model, cache, torch.tensor([[token]]), len(tokens) + len(continuation) + index
        )
        synchronize()
        intervals.append(time.perf_counter() - tick)
    elapsed = time.perf_counter() - generated_start
    result = {
        "variant": variant,
        "backend": args.backend,
        "input_tokens": len(tokens),
        "teacher_forced_tokens": len(continuation),
        "generated_tokens": len(generated),
        "prefill_seconds": prefill_seconds,
        "teacher_forced_seconds": teacher_seconds,
        "teacher_forced_mean_nll": sum(nll) / len(nll) if nll else None,
        "teacher_forced_nll": nll,
        "decode_seconds": elapsed,
        "decode_tokens_per_second": len(generated) / elapsed,
        "decode_step_seconds": intervals,
        "generated_token_ids": generated,
        "persistent_cache_bytes_including_metadata": cache.storage_bytes(),
        "peak_torch_allocated_bytes_by_gpu": {
            str(i): torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        },
        "peak_torch_reserved_bytes_by_gpu": {
            str(i): torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())
        },
        "retained_tokens_by_layer": [len(layer.positions) for layer in cache.layers],
        "budget_full_attention": budget,
        "scope": "HF sequential layer placement; isolated research decoder, batch size 1",
    }
    # Break references before the next arm, including references held by attention modules.
    for layer in model.model.layers:
        del layer.self_attn._kv_lab_cache
    del logits, cache
    torch.cuda.empty_cache()
    return result


def run(args):
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}; choose a new experiment path")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Research runs need a CUDA GPU. CPU correctness tests use tiny random models."
        )
    if args.backend == "triton" and args.device != "auto" and not args.device.startswith("cuda"):
        raise ValueError("Triton requires CUDA placement")
    lock = model_lock(args.model_lock)
    tokenizer = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"])
    ids = prompt_ids(tokenizer, args.context + args.quality_tokens, args.seed, args.corpus)
    tokens, continuation = ids[: args.context], ids[args.context :]
    model = AutoModelForCausalLM.from_pretrained(
        lock["model"],
        revision=lock["revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=args.device,
        max_memory={i: f"{args.max_gpu_memory_gib}GiB" for i in range(torch.cuda.device_count())},
    )
    model.eval()
    if model.config.model_type != "olmo3":
        raise ValueError("This decoder supports OLMo3 only")
    placements = {str(v) for v in model.hf_device_map.values()}
    if "cpu" in placements or "disk" in placements:
        raise ValueError("Model was offloaded to CPU/disk; increase GPU memory or use more GPUs")
    if len(ids) + args.output_tokens > model.config.max_position_embeddings:
        raise ValueError("Requested context and output exceed model position limit")
    if args.backend == "triton":
        from .kernels import warmup

        dim = model.config.hidden_size // model.config.num_attention_heads
        devices = {layer.self_attn.k_proj.weight.device for layer in model.model.layers}
        for device in devices:
            warmup(device, model.dtype, dim)
        synchronize()
    # Compile and warm up every arm before measuring; no warmup caches enter a measured arm.
    results = []
    for variant in args.variants:
        run_variant(model, model.config, tokens[: min(len(tokens), 32)], [], args, variant)
        result = run_variant(model, model.config, tokens, continuation, args, variant)
        result["text"] = tokenizer.decode(result["generated_token_ids"])
        results.append(result)
        print(
            f"{variant}: {result['decode_tokens_per_second']:.2f} decode tok/s; "
            f"{result['persistent_cache_bytes_including_metadata'] / 2**20:.1f} MiB cache",
            flush=True,
        )
    write_json(
        args.output,
        provenance()
        | {
            "model_lock": lock,
            "model_config": model.config.to_dict(),
            "settings": {
                "backend": args.backend,
                "variants": args.variants,
                "budget": args.budget,
                "sink": args.sink,
                "recent": args.recent,
                "device_map": {key: str(value) for key, value in model.hf_device_map.items()},
                "max_gpu_memory_gib": args.max_gpu_memory_gib,
            },
            "results": results,
            "corpus": str(args.corpus) if args.corpus else None,
            "seed": args.seed,
            "prefill_chunk": args.prefill_chunk,
            "quality_note": "NLL uses an identical teacher-forced continuation. Random-token NLL "
            "checks numerical drift; use a natural-text corpus for meaningful quality.",
            "measurement_note": "Includes Python dispatch, scoring, compaction, dequantization, "
            "and synchronization. No speculative decode in the research decoder.",
        },
    )
