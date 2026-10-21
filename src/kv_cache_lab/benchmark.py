"""Bounded-concurrency OpenAI-compatible streaming benchmark with auditable raw results."""

import asyncio
import hashlib
import json
import os
import time

import httpx

from .artifacts import provenance, write_json
from .metrics import summarize
from .prompts import prompt_ids
from .telemetry import GPUSampler


async def stream_completion(
    client: httpx.AsyncClient,
    model: str,
    ids: list[int],
    output_tokens: int,
    seed: int,
    request_id: int,
) -> dict:
    started = time.perf_counter()
    first = last = None
    chunks = []
    text_parts = []
    usage = None
    done = False
    record = {
        "request_id": request_id,
        "seed": seed,
        "input_tokens": len(ids),
        "input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "requested_output_tokens": output_tokens,
        "output_tokens": 0,
        "error": None,
    }
    try:
        async with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": model,
                "prompt": ids,
                "max_tokens": output_tokens,
                "temperature": 0,
                "seed": seed,
                "stream": True,
                "stream_options": {"include_usage": True},
                "ignore_eos": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                if event.get("error"):
                    raise ValueError(f"Server stream error: {event['error']}")
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    piece = choice.get("text", "")
                    if piece:
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        chunks.append(now - started)
                        text_parts.append(piece)
        if not done or usage is None or first is None:
            raise ValueError("Incomplete stream: require content, usage, and [DONE]")
        actual = usage["completion_tokens"]
        if usage["prompt_tokens"] != len(ids) or actual != output_tokens:
            raise ValueError(f"Token count mismatch: usage={usage}")
        record.update(
            output_tokens=actual,
            ttft_seconds=first - started,
            tpot_seconds=(last - first) / (actual - 1) if actual > 1 else None,
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record.update(
        latency_seconds=time.perf_counter() - started,
        content_chunk_times_seconds=chunks,
        text="".join(text_parts),
    )
    return record


async def run_cell(client, model, prompts, output_tokens, concurrency, seed):
    queue = asyncio.Queue()
    for index, ids in enumerate(prompts):
        queue.put_nowait((index, ids))
    records = []

    async def worker():
        while True:
            try:
                index, ids = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            records.append(
                await stream_completion(client, model, ids, output_tokens, seed + index, index)
            )

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(prompts)))))
    return sorted(records, key=lambda r: r["request_id"]), time.perf_counter() - started


async def benchmark(args) -> bool:
    from transformers import AutoTokenizer

    from .config import model_lock

    lock = model_lock(args.model_lock)
    tokenizer = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"])
    # Set up all token IDs before timing. Each cell uses exactly the same requests across profiles.
    prompts = {
        length: [
            prompt_ids(tokenizer, length, args.seed + i, args.corpus) for i in range(args.requests)
        ]
        for length in args.contexts
    }
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = provenance() | {
        "model_lock": lock,
        "profile": args.profile,
        "base_url": args.base_url,
        "workload": "corpus" if args.corpus else "random-token-ids",
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest()
        if args.corpus
        else None,
        "contexts": args.contexts,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "output_tokens": args.output_tokens,
        "repeats": args.repeats,
        "seed": args.seed,
        "warmup_requests_per_cell_minimum": args.warmup,
        "warmup_policy": "max(configured warmup count, cell concurrency)",
        "gpu_sampling_host": "client (run benchmark on the server VM)",
        "load_model": "closed loop: immediately replace each completed request",
    }
    headers = {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}
    all_ok = True
    async with httpx.AsyncClient(
        base_url=args.base_url,
        headers=headers,
        timeout=httpx.Timeout(args.timeout),
        limits=httpx.Limits(max_connections=max(args.concurrency) + 4),
    ) as client:
        (await client.get("/health")).raise_for_status()
        response = await client.get("/v1/models")
        response.raise_for_status()
        metadata["server_models"] = response.json()
        # Verify the launcher's model/profile record before measuring.
        manifest = json.loads(args.server_manifest.read_text(encoding="utf-8"))
        if manifest["model_lock"] != lock or manifest["profile"] != args.profile:
            raise ValueError("Server manifest does not match model lock/profile")
        model_ids = {entry["id"] for entry in metadata["server_models"]["data"]}
        if lock["model"] not in model_ids:
            raise ValueError("The server is not exposing the requested model")
        limit = manifest["config"]["max_model_len"]
        if max(args.contexts) + args.output_tokens > limit:
            raise ValueError("Input plus output exceeds the server's configured model length")
        metadata["server_manifest"] = manifest
        write_json(args.output / "metadata.json", metadata)
        for length in args.contexts:
            for concurrency in args.concurrency:
                for repeat in range(args.repeats):
                    name = f"ctx{length}-c{concurrency}-r{repeat}"
                    warm, _ = await run_cell(
                        client,
                        lock["model"],
                        prompts[length][: max(args.warmup, concurrency)],
                        args.output_tokens,
                        concurrency,
                        args.seed,
                    )
                    if any(r["error"] for r in warm):
                        write_json(args.output / f"{name}-warmup-errors.json", warm)
                        raise RuntimeError(f"Warmup failed for {name}; see warmup-errors.json")
                    sampler = GPUSampler()
                    task = asyncio.create_task(sampler.run())
                    try:
                        records, elapsed = await run_cell(
                            client,
                            lock["model"],
                            prompts[length],
                            args.output_tokens,
                            concurrency,
                            args.seed,
                        )
                    finally:
                        sampler.stop.set()
                        await task
                    summary = summarize(records, elapsed) | {
                        "context_tokens": length,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "profile": args.profile,
                        "gpu_memory": sampler.summary(),
                    }
                    all_ok &= summary["failed_requests"] == 0
                    write_json(args.output / f"{name}-summary.json", summary)
                    with (args.output / f"{name}-requests.jsonl").open("w", encoding="utf-8") as f:
                        for record in records:
                            f.write(json.dumps(record, allow_nan=False) + "\n")
                    write_json(args.output / f"{name}-gpu-samples.json", sampler.samples)
                    print(
                        f"{name}: {summary['output_tokens_per_second']:.2f} output tok/s; "
                        f"{summary['failed_requests']} failures",
                        flush=True,
                    )
    return all_ok
