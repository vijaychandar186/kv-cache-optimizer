import csv
import json
from pathlib import Path


def report(root: Path, plot: bool = False) -> Path:
    rows = []
    for path in sorted(root.rglob("*-summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "context_tokens" not in data:
            continue
        memory = data.pop("gpu_memory")
        data["sampled_peak_gpu_used_mib"] = memory["peak_simultaneous_total_used_mib"]
        data["source"] = str(path)
        rows.append(data)
    if not rows:
        raise ValueError(f"No benchmark summaries under {root}")
    path = root / "comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        for profile in sorted({r["profile"] for r in rows}):
            for context in sorted({r["context_tokens"] for r in rows}):
                subset = [
                    r
                    for r in rows
                    if r["profile"] == profile
                    and r["context_tokens"] == context
                    and r["failed_requests"] == 0
                ]
                if not subset:
                    continue
                xs = sorted({r["concurrency"] for r in subset})
                groups = [
                    [r["output_tokens_per_second"] for r in subset if r["concurrency"] == x]
                    for x in xs
                ]
                ys = [sum(g) / len(g) for g in groups]
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[
                        [y - min(g) for y, g in zip(ys, groups, strict=True)],
                        [max(g) - y for y, g in zip(ys, groups, strict=True)],
                    ],
                    marker="o",
                    label=f"{profile}, {context} input tokens",
                )
        ax.set(xlabel="Concurrent requests", ylabel="Aggregate output tokens / second")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(root / "throughput.png", dpi=200)
        fig.savefig(root / "throughput.pdf")
        plt.close(fig)
    return path
