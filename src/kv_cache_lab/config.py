"""Validated serving profiles; no shell interpolation or secret-bearing command arguments."""

import json
import tomllib
from pathlib import Path

DEFAULT_MODEL = "allenai/Olmo-3-32B-Think"


def load_profile(path: Path, profile: str, tp: int | None = None) -> dict:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    profiles = data.pop("profiles")
    if profile not in profiles:
        raise ValueError(f"Unknown profile {profile!r}; choose from {', '.join(profiles)}")
    data.update(profiles[profile])
    if tp is not None:
        data["tensor_parallel_size"] = tp
    # OLMo has 40 query heads and 8 KV heads. Avoid KV replication in this lab.
    if data["tensor_parallel_size"] not in (1, 2, 4, 8):
        raise ValueError("OLMo TP must be 1, 2, 4, or 8")
    if not 0 < data["gpu_memory_utilization"] < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    if data["max_model_len"] < 8192 + 256:
        raise ValueError("max_model_len must leave room for 8K input plus output")
    return data


def model_lock(path: Path, model: str | None = None) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing {path}. First run: uv run kv-lab download")
    data = json.loads(path.read_text(encoding="utf-8"))
    if model is not None and data["model"] != model:
        raise ValueError("The model lock belongs to a different model")
    sha = data.get("revision", "")
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError("The model revision must be a full Hugging Face commit SHA")
    return data


def serve_command(config: dict, revision: str) -> list[str]:
    args = [
        "vllm",
        "serve",
        config["model"],
        "--revision",
        revision,
        "--tokenizer-revision",
        revision,
        "--dtype",
        "bfloat16",
        "--host",
        config["host"],
        "--port",
        str(config["port"]),
        "--distributed-executor-backend",
        "mp",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--generation-config",
        "vllm",
    ]
    for name in (
        "tensor_parallel_size",
        "max_model_len",
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_num_batched_tokens",
        "seed",
        "kv_cache_dtype",
    ):
        args.extend(["--" + name.replace("_", "-"), str(config[name])])
    if config.get("calculate_kv_scales"):
        args.append("--calculate-kv-scales")
    if "speculative_config" in config:
        args.extend(["--speculative-config", json.dumps(config["speculative_config"])])
    return args
