import importlib.metadata
import platform
import shutil
import subprocess


def doctor(tp=8):
    checks = {"linux": platform.system() == "Linux", "uv": shutil.which("uv") is not None}
    print(f"Platform: {platform.platform()}")
    for name in ("vllm", "torch", "triton", "transformers"):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}: not installed")
            if name in ("vllm", "triton"):
                checks[name] = False
    try:
        import torch

        checks["cuda"] = torch.cuda.is_available()
        checks["gpu_count"] = torch.cuda.device_count() >= tp
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB, SM {p.major}.{p.minor}")
            checks[f"gpu_{i}_bf16_fp8"] = p.major >= 9
        if checks["cuda"]:
            # Check wheel/driver compatibility by actually executing a CUDA operation.
            value = torch.ones(16, device="cuda") * 2
            torch.cuda.synchronize()
            checks["cuda_operation"] = value.sum().item() == 32
    except (ImportError, RuntimeError, AssertionError) as exc:
        checks["cuda"] = False
        print(f"CUDA check: {exc}")
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "topo", "-m"], check=False)
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    return all(checks.values())
