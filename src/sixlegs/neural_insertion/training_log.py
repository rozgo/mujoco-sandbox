"""Training records from PufferLib 5.0 dashboard logs and resource samples.

The trainer redraws a dashboard box every epoch. Each box reports steps,
throughput, losses, environment means and the dashboard's own GPU utilisation,
device memory and host memory readings. log_resources.sh adds per-minute GPU
power, temperature, trainer GPU memory and host CPU load for runs from align_v4.
"""

import argparse
import csv
import json
from pathlib import Path
import re

METRIC = re.compile(r"(SPS|Steps|Epoch|entropy|old_kl|kl|clipfrac|perf|collision|timeout|final_lateral_um|"
                    r"final_vertical_um|level|episode_return|episode_length|vessel_steps|placed|placement_um|thread_touch|"
                    r"dragged|missed)\s+(-?[0-9.]+[KMB]?)\b")
DASH = re.compile(r"GPU:\s*(\d+)%\s+VRAM:\s*([0-9.]+)/([0-9.]+)G\s+RAM:\s*([0-9.]+)G")
UPTIME = re.compile(r"Uptime\s+((?:\d+(?:ms|[dhms])\s*)+)")
UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1, "ms": 1e-3}


def number(text):
    scale = {"K": 1e3, "M": 1e6, "B": 1e9}.get(text[-1], 1)
    return float(text[:-1] if scale != 1 else text)*scale


def boxes(text):
    """One record per dashboard redraw, numbers in SI-like plain units."""
    out = []
    for box in text.replace("\r", "\n").split("╭")[1:]:
        row = {k: number(v) for k, v in METRIC.findall(box)}
        if "Steps" not in row:
            continue
        dash = DASH.search(box)
        if dash:
            row.update(gpu_pct=float(dash[1]), vram_gb=float(dash[2]), vram_total_gb=float(dash[3]), ram_gb=float(dash[4]))
        up = UPTIME.search(box)
        if up:
            row["uptime_s"] = sum(float(v)*UNITS[u] for v, u in re.findall(r"(\d+)(ms|[dhms])", up[1]))
        out.append(row)
    return out


def history(rows, every=4):
    """Learning curves for the journal: one record every `every` epochs, plus the last."""
    keep = ("Steps", "entropy", "kl", "clipfrac", "perf", "collision", "final_lateral_um", "final_vertical_um", "level",
            "placed", "placement_um", "thread_touch", "episode_length")
    picked = [r for i, r in enumerate(rows) if i % every == 0 or i == len(rows)-1]
    return [{k: r[k] for k in keep if k in r} for r in picked]


def resources(path):
    with open(path) as f:
        rows = [{k: float(v) for k, v in r.items() if k != "utc" and v not in ("", None)} for r in csv.DictReader(f)]
    if len(rows) > 2:
        rows = rows[1:]  # the first sample precedes the trainer's GPU allocation
    stat = lambda k: {"mean": sum(r[k] for r in rows)/len(rows), "max": max(r[k] for r in rows)} \
        if rows and all(k in r for r in rows) else None
    return {"samples": len(rows), "interval_s": 60, "gpu_util_pct": stat("gpu_util_pct"),
            "gpu_power_w": stat("gpu_power_w"), "gpu_temp_c": stat("gpu_temp_c"),
            "device_memory_mib": stat("gpu_mem_used_mib"), "trainer_gpu_memory_mib": stat("trainer_gpu_mem_mib"),
            "host_load_1min": stat("load1"), "trainer_cpu_pct": stat("trainer_cpu_pct"),
            "trainer_rss_mib": stat("trainer_rss_mib")}


def summary(rows, resource_csv=None):
    """Compute used by one run. Throughput is steps over uptime; the dashboard's per-epoch
    figure is reported as a median because the closing box prints a meaningless rate."""
    steady = rows[2:] or rows
    mean = lambda k: sum(r[k] for r in steady if k in r)/max(1, sum(k in r for r in steady))
    rates = sorted(r["SPS"] for r in steady if "SPS" in r)
    wall = rows[-1].get("uptime_s")
    out = {"steps": rows[-1]["Steps"], "epochs": int(rows[-1].get("Epoch", len(rows))),
           "wall_s": wall, "steps_per_s": rows[-1]["Steps"]/wall if wall else None,
           "dashboard_sps_median": rates[len(rates)//2] if rates else None,
           "dashboard": {"gpu_util_pct_mean": mean("gpu_pct"), "gpu_util_pct_max": max(r.get("gpu_pct", 0) for r in rows),
                         "device_memory_gb_max": max(r.get("vram_gb", 0) for r in rows),
                         "device_memory_total_gb": rows[-1].get("vram_total_gb"),
                         "host_memory_gb_max": max(r.get("ram_gb", 0) for r in rows)}}
    if resource_csv and Path(resource_csv).exists():
        out["sampled"] = resources(resource_csv)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--resources", type=Path)
    args = parser.parse_args()
    rows = boxes(args.log.read_text(errors="replace"))
    print(json.dumps(summary(rows, args.resources), indent=1))


if __name__ == "__main__":
    main()
