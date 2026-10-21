import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from kv_cache_lab.benchmark import run_cell, stream_completion
from kv_cache_lab.metrics import summarize
from kv_cache_lab.prompts import prompt_ids


def response_events(prompt_tokens=3, completion_tokens=2, done=True):
    events = [
        {"choices": [{"text": "hello"}]},
        {"choices": [{"text": " world"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    ]
    content = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    return content + ("data: [DONE]\n\n" if done else "")


@pytest.mark.asyncio
async def test_usage_and_stream_termination_are_required():
    async def handler(request):
        return httpx.Response(200, text=response_events(done=False))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    ) as client:
        record = await stream_completion(client, "model", [1, 2, 3], 2, 42, 0)
    assert "Incomplete stream" in record["error"]


@pytest.mark.asyncio
async def test_exact_token_counts_and_content_are_saved():
    async def handler(request):
        body = json.loads(request.content)
        assert body["prompt"] == [1, 2, 3]
        assert body["ignore_eos"] is True
        return httpx.Response(200, text=response_events())

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    ) as client:
        record = await stream_completion(client, "model", [1, 2, 3], 2, 42, 0)
    assert record["error"] is None
    assert record["output_tokens"] == 2
    assert record["text"] == "hello world"
    assert 0 <= record["ttft_seconds"] <= record["latency_seconds"]


@pytest.mark.asyncio
async def test_mismatched_usage_is_failure():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, text=response_events(completion_tokens=1))
        ),
        base_url="http://test",
    ) as client:
        record = await stream_completion(client, "model", [1, 2, 3], 2, 42, 0)
    assert "Token count mismatch" in record["error"]
    assert record["output_tokens"] == 0


@pytest.mark.asyncio
async def test_http_failure_is_counted():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(503)), base_url="http://test"
    ) as client:
        record = await stream_completion(client, "model", [1, 2, 3], 2, 42, 0)
    assert "503" in record["error"]
    result = summarize([record], 2)
    assert result["failed_requests"] == 1
    assert result["output_tokens_per_second"] == 0
    assert result["ttft_seconds_p50"] is None


@pytest.mark.asyncio
async def test_concurrency_is_bounded_and_requests_not_dropped():
    active = peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, text=response_events())

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    ) as client:
        records, elapsed = await run_cell(client, "model", [[1, 2, 3]] * 9, 2, 3, 42)
    assert len(records) == 9
    assert peak == 3
    assert all(r["error"] is None for r in records)
    assert elapsed > 0


def test_failed_request_wall_time_is_in_throughput_denominator():
    records = [
        {
            "output_tokens": 20,
            "error": None,
            "latency_seconds": 1,
            "ttft_seconds": 0.1,
            "tpot_seconds": 0.05,
        },
        {"error": "timeout"},
    ]
    result = summarize(records, 10)
    assert result["output_tokens_per_second"] == 2
    assert result["successful_requests"] == 1
    assert result["failed_requests"] == 1


class Tokenizer:
    all_special_ids = [0, 4]

    def get_vocab(self):
        return {"a": 0, "b": 1, "c": 3, "d": 4}

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def test_prompts_have_exact_lengths_are_reproducible_and_exclude_special_tokens():
    first = prompt_ids(Tokenizer(), 8192, 42)
    assert len(first) == 8192
    assert first == prompt_ids(Tokenizer(), 8192, 42)
    assert set(first) == {1, 3}


def test_corpus_is_not_silently_repeated(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("too short")
    with pytest.raises(ValueError, match="larger corpus"):
        prompt_ids(Tokenizer(), 2048, 42, path)


@pytest.mark.asyncio
async def test_complete_benchmark_writes_auditable_artifacts(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    from kv_cache_lab.benchmark import benchmark
    from kv_cache_lab.report import report
    from kv_cache_lab.telemetry import GPUSampler

    model = {"model": "test-model", "revision": "a" * 40}
    model_path = tmp_path / "model-lock.json"
    model_path.write_text(json.dumps(model))
    manifest = tmp_path / "server.json"
    manifest.write_text(
        json.dumps({"profile": "baseline", "model_lock": model, "config": {"max_model_len": 12288}})
    )
    monkeypatch.setenv("VLLM_API_KEY", "test-only-secret")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        return httpx.Response(200, text=response_events())

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: client_class(transport=httpx.MockTransport(handler), **kw),
    )

    async def no_gpu(self):
        raise OSError("No GPU on test client")

    monkeypatch.setattr(GPUSampler, "sample_once", no_gpu)
    output = tmp_path / "run"
    args = SimpleNamespace(
        model_lock=model_path,
        server_manifest=manifest,
        profile="baseline",
        corpus=None,
        contexts=[3],
        concurrency=[1, 2],
        requests=4,
        repeats=2,
        warmup=2,
        seed=42,
        output=output,
        base_url="http://test",
        timeout=30,
        output_tokens=2,
    )
    assert await benchmark(args)
    summaries = list(output.glob("*-summary.json"))
    assert len(summaries) == 4
    for path in summaries:
        summary = json.loads(path.read_text())
        assert summary["output_tokens"] == 8
        assert summary["failed_requests"] == 0
        assert summary["gpu_memory"]["peak_simultaneous_total_used_mib"] is None
    records = list(output.glob("*-requests.jsonl"))
    assert all(len(path.read_text().splitlines()) == 4 for path in records)
    assert "test-only-secret" not in (output / "metadata.json").read_text()
    assert report(output).is_file()
