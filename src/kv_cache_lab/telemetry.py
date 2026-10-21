"""Sample whole-device memory on the server host; never label it process memory."""

import asyncio
import csv
import io
import time


class GPUSampler:
    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.samples: list[dict] = []
        self.error: str | None = None
        self.stop = asyncio.Event()

    async def sample_once(self) -> None:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError("nvidia-smi timed out") from None
        if process.returncode:
            raise RuntimeError(stderr.decode(errors="replace").strip())
        devices = []
        for row in csv.reader(io.StringIO(stdout.decode())):
            index, uuid, used, total, utilization = (part.strip() for part in row)
            devices.append(
                {
                    "index": int(index),
                    "uuid": uuid,
                    "used_mib": float(used),
                    "total_mib": float(total),
                    "utilization_percent": float(utilization),
                }
            )
        if not devices:
            raise RuntimeError("nvidia-smi returned no devices")
        self.samples.append({"monotonic_seconds": time.perf_counter(), "gpus": devices})

    async def run(self) -> None:
        while not self.stop.is_set():
            try:
                await self.sample_once()
            except (OSError, RuntimeError, ValueError) as exc:
                self.error = str(exc)
                return
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=self.interval)
            except TimeoutError:
                pass

    def summary(self) -> dict:
        peaks = {}
        for sample in self.samples:
            for gpu in sample["gpus"]:
                peaks[gpu["uuid"]] = max(peaks.get(gpu["uuid"], 0), gpu["used_mib"])
        return {
            "source": "nvidia-smi: whole-device sampled used memory",
            "interval_seconds": self.interval,
            "samples": len(self.samples),
            "error": self.error,
            "peak_used_mib_by_gpu": peaks,
            "peak_simultaneous_total_used_mib": max(
                (sum(g["used_mib"] for g in s["gpus"]) for s in self.samples), default=None
            ),
        }
