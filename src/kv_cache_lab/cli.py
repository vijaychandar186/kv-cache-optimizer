import argparse
import asyncio
import json
import os
import platform
import shlex
import subprocess
from pathlib import Path

from .artifacts import provenance, write_json
from .config import DEFAULT_MODEL, load_profile, model_lock, serve_command


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return result


def parser():
    root = argparse.ArgumentParser(description="OLMo Azure inference and KV-cache lab (use uv run)")
    sub = root.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Resolve and pin a model snapshot")
    download.add_argument("--model", default=DEFAULT_MODEL)
    download.add_argument("--revision", default="main")
    download.add_argument("--lock", type=Path, default=Path("models/model-lock.json"))
    download.add_argument("--tokenizer-only", action="store_true")
    serve = sub.add_parser("serve", help="Launch a pinned vLLM profile")
    serve.add_argument("--config", type=Path, default=Path("configs/serving.toml"))
    serve.add_argument(
        "--profile",
        choices=["baseline", "fp8", "speculative", "fp8-speculative"],
        default="baseline",
    )
    serve.add_argument("--tp", type=positive)
    serve.add_argument("--dry-run", action="store_true")
    serve.add_argument("--manifest", type=Path, default=Path("results/server.json"))
    check = sub.add_parser("doctor")
    check.add_argument("--tp", type=positive, default=8)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--model-lock", type=Path, default=Path("models/model-lock.json"))
    bench.add_argument("--server-manifest", type=Path, default=Path("results/server.json"))
    bench.add_argument("--base-url", default="http://127.0.0.1:8000")
    bench.add_argument("--profile", required=True)
    bench.add_argument("--contexts", type=positive, nargs="+", default=[2048, 4096, 8192])
    bench.add_argument("--concurrency", type=positive, nargs="+", default=[1, 4, 16])
    bench.add_argument("--requests", type=positive, default=64)
    bench.add_argument("--output-tokens", type=positive, default=256)
    bench.add_argument("--repeats", type=positive, default=3)
    bench.add_argument("--warmup", type=positive, default=2)
    bench.add_argument("--timeout", type=positive, default=600)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--corpus", type=Path)
    bench.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("path", type=Path)
    report.add_argument("--plot", action="store_true")
    kernel = sub.add_parser("kernels")
    kernel.add_argument("--tokens", type=positive, default=8192)
    kernel.add_argument("--output", type=Path, default=Path("results/kernels.json"))
    research = sub.add_parser("research")
    research.add_argument("--model-lock", type=Path, default=Path("models/model-lock.json"))
    research.add_argument("--context", type=positive, default=2048)
    research.add_argument("--output-tokens", type=positive, default=128)
    research.add_argument("--quality-tokens", type=positive, default=32)
    research.add_argument("--prefill-chunk", type=positive, default=128)
    research.add_argument("--budget", type=positive, default=1024)
    research.add_argument("--sink", type=int, default=4)
    research.add_argument("--recent", type=positive, default=128)
    research.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "int8", "evict", "int8-evict"],
        choices=["baseline", "int8", "evict", "int8-evict"],
    )
    research.add_argument("--backend", choices=["triton", "torch"], default="triton")
    research.add_argument("--device", default="auto")
    research.add_argument("--max-gpu-memory-gib", type=positive, default=70)
    research.add_argument("--corpus", type=Path)
    research.add_argument("--seed", type=int, default=42)
    research.add_argument("--output", type=Path, default=Path("results/research.json"))
    return root


def dispatch(args):
    if args.command == "download":
        from huggingface_hub import HfApi, snapshot_download

        # Reuse a previous lock unless the user explicitly selects a different revision/model.
        if args.lock.exists() and args.revision == "main":
            lock = model_lock(args.lock, args.model)
            sha = lock["revision"]
        else:
            sha = HfApi().model_info(args.model, revision=args.revision).sha
        patterns = ["*.json", "*.jinja", "*.model", "*.txt", "*.tiktoken"]
        if not args.tokenizer_only:
            patterns.append("*.safetensors")
        snapshot_download(args.model, revision=sha, allow_patterns=patterns)
        write_json(args.lock, {"model": args.model, "revision": sha})
        print(f"Pinned {args.model}@{sha} in {args.lock}")
    elif args.command == "serve":
        config = load_profile(args.config, args.profile, args.tp)
        lock = model_lock(Path(config["revision_file"]), config["model"])
        command = serve_command(config, lock["revision"])
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return 0
        if platform.system() != "Linux":
            raise ValueError(
                "Run vLLM on the Azure Linux VM; Windows supports client commands/tests"
            )
        if not os.environ.get("VLLM_API_KEY") or os.environ["VLLM_API_KEY"].startswith("replace-"):
            raise ValueError("Export a random VLLM_API_KEY first")
        from .doctor import doctor

        if not doctor(config["tensor_parallel_size"]):
            raise ValueError("GPU preflight failed")
        environment = os.environ.copy()
        environment.setdefault("VLLM_NO_USAGE_STATS", "1")
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        write_json(
            args.manifest,
            provenance()
            | {"profile": args.profile, "config": config, "model_lock": lock, "command": command},
        )
        return subprocess.call(command, env=environment)
    elif args.command == "doctor":
        from .doctor import doctor

        return 0 if doctor(args.tp) else 1
    elif args.command == "benchmark":
        from .benchmark import benchmark

        if not os.environ.get("VLLM_API_KEY"):
            raise ValueError("Export VLLM_API_KEY first")
        if args.warmup > args.requests:
            raise ValueError("Warmup count cannot exceed requests")
        if max(args.concurrency) > args.requests:
            raise ValueError("Request count must be at least the maximum concurrency")
        return 0 if asyncio.run(benchmark(args)) else 1
    elif args.command == "report":
        from .report import report

        print(report(args.path, args.plot))
    elif args.command == "kernels":
        from .kernels import microbenchmark

        print(json.dumps(microbenchmark(args.output, args.tokens), indent=2))
    elif args.command == "research":
        if args.sink < 0 or args.sink + args.recent > args.budget:
            raise ValueError("Require sink >= 0 and budget >= sink + recent")
        from .research import run

        run(args)
    return 0


def main():
    try:
        raise SystemExit(dispatch(parser().parse_args()))
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
