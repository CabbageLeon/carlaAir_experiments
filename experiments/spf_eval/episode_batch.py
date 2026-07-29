"""Run independent SPF episodes with a fresh CARLA-Air process each time."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .metrics import EscortEpisode, LandingEpisode, escort_summary, landing_summary, timing_summary
from .runner import CarlaAirProcess, _write_json


def _worker_command(args: argparse.Namespace, mode: str, seed: int, worker_output: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.spf_eval.runner",
        args.task,
        mode,
        "--model",
        args.model,
        "--seeds",
        str(seed),
        "--episodes-per-seed",
        "1",
        "--seconds",
        str(args.seconds),
        "--output",
        str(worker_output),
        "--carla-port",
        str(args.carla_port),
        "--airsim-port",
        str(args.airsim_port),
        "--map",
        args.map,
        "--no-restart-carla-per-episode",
        "--allow-unpaired-landing",
    ]


def _run_mode(args: argparse.Namespace, mode: str, process: CarlaAirProcess) -> tuple[list[LandingEpisode], list[EscortEpisode], list[dict]]:
    output = Path(args.output) / args.task / "SPF" / mode
    workers = Path(args.output) / "worker_runs" / args.task / "SPF" / mode
    logs = Path(args.output) / "worker_logs" / args.task / "SPF" / mode
    landing: list[LandingEpisode] = []
    escort: list[EscortEpisode] = []
    all_events: list[dict] = []
    for seed in args.seeds:
        for episode_index in range(args.episodes_per_seed):
            worker_output = workers / f"seed-{seed}-episode-{episode_index:02d}"
            log_path = logs / f"seed-{seed}-episode-{episode_index:02d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            process.start(mode, seed, episode_index)
            try:
                with log_path.open("w", encoding="utf-8") as handle:
                    result = subprocess.run(
                        _worker_command(args, mode, seed, worker_output),
                        cwd=process.root,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if result.returncode != 0:
                    raise RuntimeError(f"episode worker failed: see {log_path}")
            finally:
                process.stop()
            source = worker_output / args.task / "SPF" / mode / f"seed-{seed}.json"
            if not source.exists():
                raise RuntimeError(f"episode worker did not write {source}")
            data = json.loads(source.read_text(encoding="utf-8"))
            destination = output / f"seed-{seed}-episode-{episode_index:02d}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            all_events.extend(data["events"])
            if args.task == "landing":
                landing.append(LandingEpisode(**data["episode"]))
            else:
                escort.append(EscortEpisode(**data["episode"]))
    _write_json(output / "timing.json", timing_summary(all_events))
    return landing, escort, all_events


def run(args: argparse.Namespace) -> dict[str, object]:
    modes = ("C0", "C1", "C2") if args.task == "landing" else ("C0", "C1")
    modes = modes if args.mode == "all" else (args.mode,)
    process = CarlaAirProcess(args)
    results: dict[str, object] = {}
    c0_landing: list[LandingEpisode] | None = None
    try:
        for mode in modes:
            landing, escort, _ = _run_mode(args, mode, process)
            if args.task == "landing":
                if mode == "C0":
                    c0_landing = landing
                if c0_landing is None:
                    raise ValueError("run C0 before C1/C2 so paired CG can be calculated")
                summary: dict[str, float] = landing_summary(landing, c0_landing)
            else:
                summary = escort_summary(escort)
            summary.update(
                json.loads((Path(args.output) / args.task / "SPF" / mode / "timing.json").read_text())
            )
            _write_json(Path(args.output) / args.task / "SPF" / mode / "summary.json", summary)
            results[mode] = summary
    finally:
        process.stop()
    return results if args.mode == "all" else results[args.mode]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("landing", "escort"))
    parser.add_argument("mode", choices=("C0", "C1", "C2", "all"))
    parser.add_argument("--model", default="qwen3-vl-flash")
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 22, 33))
    parser.add_argument("--episodes-per-seed", type=int, default=50)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--output", default="runs")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--airsim-port", type=int, default=41451)
    parser.add_argument("--map", default="Town10HD")
    parser.add_argument("--carla-start-timeout", type=float, default=90.0)
    parser.add_argument("--carla-warmup-seconds", type=float, default=15.0)
    parser.add_argument("--carla-cooldown-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.task == "escort" and args.mode == "C2":
        parser.error("C2 is defined only for moving-platform landing")
    if args.episodes_per_seed < 1:
        parser.error("--episodes-per-seed must be positive")
    args.seconds = args.seconds or (60.0 if args.task == "landing" else 90.0)
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
