"""Metrics computed from successful requests; failed requests still consume wall time."""

import math


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    offset = (len(ordered) - 1) * p / 100
    low, high = math.floor(offset), math.ceil(offset)
    return ordered[low] + (ordered[high] - ordered[low]) * (offset - low)


def summarize(records: list[dict], wall_seconds: float) -> dict:
    good = [r for r in records if r.get("error") is None]
    result = {
        "requests": len(records),
        "successful_requests": len(good),
        "failed_requests": len(records) - len(good),
        "wall_seconds": wall_seconds,
        "output_tokens": sum(r["output_tokens"] for r in good),
    }
    result["output_tokens_per_second"] = result["output_tokens"] / wall_seconds
    result["requests_per_second"] = len(good) / wall_seconds
    for field in ("latency_seconds", "ttft_seconds", "tpot_seconds"):
        values = [r[field] for r in good if r.get(field) is not None]
        for p in (50, 95, 99):
            result[f"{field}_p{p}"] = percentile(values, p)
    return result
