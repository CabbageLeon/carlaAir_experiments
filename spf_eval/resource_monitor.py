"""Record CARLA-Air host and GPU memory while an evaluation is running."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import time
from pathlib import Path


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0])
    return values


def _carla_pid() -> int | None:
    result = subprocess.run(
        ["pgrep", "-f", "CarlaUE4-Linux-Shipping"], capture_output=True, text=True, check=False
    )
    pids = [int(value) for value in result.stdout.split()]
    return pids[0] if pids else None


def _process_memory(pid: int | None) -> dict[str, int]:
    if pid is None:
        return {"VmRSS": 0, "RssAnon": 0, "RssFile": 0, "RssShmem": 0}
    try:
        values: dict[str, int] = {}
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key in {"VmRSS", "RssAnon", "RssFile", "RssShmem"}:
                values[key] = int(value.split()[0])
        return {key: values.get(key, 0) for key in ("VmRSS", "RssAnon", "RssFile", "RssShmem")}
    except FileNotFoundError:
        return {"VmRSS": 0, "RssAnon": 0, "RssFile": 0, "RssShmem": 0}


def _gpu() -> tuple[int, int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        used, total, utilization = (int(value.strip()) for value in result.stdout.splitlines()[0].split(","))
        return used, total, utilization
    except (IndexError, ValueError):
        return 0, 0, 0


def _stop(pid: int | None) -> None:
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-pid", type=int)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--critical-mem-mib", type=int, default=1024)
    parser.add_argument("--critical-swap-free-mib", type=int, default=256)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "state",
        "mem_available_mib",
        "swap_free_mib",
        "swap_total_mib",
        "carla_pid",
        "carla_rss_mib",
        "carla_anon_mib",
        "carla_file_mib",
        "carla_shmem_mib",
        "gpu_used_mib",
        "gpu_total_mib",
        "gpu_utilization_pct",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while True:
            mem = _meminfo()
            carla_pid = _carla_pid()
            process = _process_memory(carla_pid)
            gpu_used, gpu_total, gpu_utilization = _gpu()
            available_mib = mem.get("MemAvailable", 0) // 1024
            swap_free_mib = mem.get("SwapFree", 0) // 1024
            critical = (
                available_mib < args.critical_mem_mib
                or swap_free_mib < args.critical_swap_free_mib
            )
            writer.writerow(
                {
                    "timestamp": time.time(),
                    "state": "critical" if critical else "ok",
                    "mem_available_mib": available_mib,
                    "swap_free_mib": swap_free_mib,
                    "swap_total_mib": mem.get("SwapTotal", 0) // 1024,
                    "carla_pid": carla_pid or 0,
                    "carla_rss_mib": process["VmRSS"] // 1024,
                    "carla_anon_mib": process["RssAnon"] // 1024,
                    "carla_file_mib": process["RssFile"] // 1024,
                    "carla_shmem_mib": process["RssShmem"] // 1024,
                    "gpu_used_mib": gpu_used,
                    "gpu_total_mib": gpu_total,
                    "gpu_utilization_pct": gpu_utilization,
                }
            )
            handle.flush()
            if critical:
                _stop(args.runner_pid)
                _stop(carla_pid)
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
