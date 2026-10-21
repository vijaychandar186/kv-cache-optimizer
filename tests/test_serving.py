from pathlib import Path

import pytest

from kv_cache_lab.config import load_profile, model_lock, serve_command


def test_profiles_use_independent_ablation_flags():
    baseline = serve_command(load_profile(Path("configs/serving.toml"), "baseline"), "a" * 40)
    fp8 = serve_command(load_profile(Path("configs/serving.toml"), "fp8"), "a" * 40)
    spec = serve_command(load_profile(Path("configs/serving.toml"), "speculative"), "a" * 40)
    assert "--calculate-kv-scales" not in baseline
    assert "--speculative-config" not in baseline
    assert "--calculate-kv-scales" in fp8
    assert "--speculative-config" in spec
    assert "--no-enable-prefix-caching" in baseline
    assert "--api-key" not in baseline


def test_invalid_tensor_parallelism_rejected():
    with pytest.raises(ValueError, match="TP"):
        load_profile(Path("configs/serving.toml"), "baseline", 5)


def test_model_lock_requires_commit_sha(tmp_path):
    path = tmp_path / "lock.json"
    path.write_text('{"model":"x","revision":"main"}')
    with pytest.raises(ValueError, match="SHA"):
        model_lock(path)
