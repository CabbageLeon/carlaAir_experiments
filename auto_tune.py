#!/usr/bin/env python3
"""自动调参 — 以降落成功率 (LSR) 为目标优化参数。

Usage:
    python auto_tune.py                    # 全参数搜索，60s episode
    python auto_tune.py --quick            # 30s 粗筛
    python auto_tune.py --cores 4          # 并行跑 4 组 (需要多个 CarlaAir 实例)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.spf_eval.environment import CarlaAirEnvironment
from experiments.spf_eval.metrics import LandingEpisode
from experiments.spf_eval.prompts import landing_prompt
from experiments.spf_eval.spf_policy import SPFPolicy

import numpy as np

PRESETS_FILE = Path.home() / ".carlaair" / "presets.json"


def run_one_episode(env, policy, seed, seconds):
    from concurrent.futures import ThreadPoolExecutor
    env.reset(seed, spawn_index=0)
    policy.reset()
    visible_seconds = 0.0
    stable_since = None
    stable = False
    started = time.monotonic()
    previous = started
    executor = ThreadPoolExecutor(max_workers=1)
    pending = None
    pending_state = None
    active_waypoint = None
    next_tracking = 0.0
    next_decision = 0.0
    decision_count = 0
    min_cargo_dist = float("inf")

    try:
        while time.monotonic() - started < seconds:
            now = env.tick()
            dt = now - previous
            previous = now
            visible = env.truck_in_camera_view()
            if visible:
                visible_seconds += dt

            state = env.drone_state()
            try:
                cargo_dist = float(np.linalg.norm((env.cargo_bed_ned() - state.ned)[:2]))
                if cargo_dist < min_cargo_dist:
                    min_cargo_dist = cargo_dist
            except Exception:
                pass

            if pending is not None and pending.done():
                try:
                    command = pending.result()
                    active_waypoint = command.target_ned
                    next_tracking = now
                    next_decision = command.inference_finished_at
                    decision_count += 1
                except Exception:
                    next_decision = now + 1.0
                pending = None
                pending_state = None

            if active_waypoint is not None and now >= next_tracking:
                velocity = env.track_waypoint(active_waypoint, duration=0.15)
                next_tracking = now + 0.1
                if float((velocity**2).sum() ** 0.5) < 0.05:
                    active_waypoint = None

            if pending is None and now >= next_decision:
                direction, motion, _ = env.cargo_relation()
                prompt = landing_prompt("C0", direction, motion, env.landing_phase())
                try:
                    frame = env.capture_rgb()
                    pending_state = state
                    pending = executor.submit(policy.act, frame, prompt, state.ned, state.yaw_rad)
                except Exception:
                    next_decision = now + 1.0

            try:
                on_cargo_bed = env.on_cargo_bed()
            except Exception:
                on_cargo_bed = False

            if on_cargo_bed and state.speed <= 0.3:
                stable_since = now if stable_since is None else stable_since
                stable = now - stable_since >= 2.0
            else:
                stable_since = None
            if stable:
                break
    finally:
        try:
            landed_on_bed = env.on_cargo_bed()
        except Exception:
            landed_on_bed = False
        executor.shutdown(wait=False, cancel_futures=True)
        env.close_episode()

    return LandingEpisode(seed, visible_seconds, landed_on_bed, stable), decision_count, min_cargo_dist


def evaluate_params(env, policy, seeds, seconds):
    episodes, total_dec, min_dists = [], 0, []
    for seed in seeds:
        ep, nd, md = run_one_episode(env, policy, seed, seconds)
        episodes.append(ep)
        total_dec += nd
        min_dists.append(md)
    tsr = sum(1 for ep in episodes if ep.tracked) / len(episodes)
    lsr = sum(1 for ep in episodes if ep.landed) / len(episodes)
    avg_visible = sum(ep.visible_seconds for ep in episodes) / len(episodes)
    avg_df = total_dec / (len(episodes) * seconds) if seconds > 0 else 0
    avg_min_dist = sum(min_dists) / len(min_dists) if min_dists else float("inf")
    return {
        "TSR": tsr, "LSR": lsr, "visible_s": avg_visible, "DF": avg_df,
        "decisions": total_dec, "min_dist": avg_min_dist,
    }


def print_header():
    print(f"{'#':>3} {'proj':>5} {'h_gain':>6} {'v_gain':>6} {'temp':>5}  "
          f"{'LSR':>6} {'TSR':>6} {'min_d':>6} {'DF':>6} {'vis':>5}")


def print_row(i, params, result):
    print(f"{i:>3} {params['proj']:>5.0f} {params['h_gain']:>6.1f} "
          f"{params['v_gain']:>6.1f} {params['temp']:>5.2f}  "
          f"{result['LSR']:>6.2f} {result['TSR']:>6.2f} {result['min_dist']:>6.1f} "
          f"{result['DF']:>6.2f} {result['visible_s']:>5.1f}")


def load_api_key():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        state_file = Path.home() / ".carlaair" / "state.json"
        if state_file.is_file():
            api_key = json.loads(state_file.read_text()).get("_api_key", "")
    if not api_key:
        print("No API key found. Run the experiment once first.")
        sys.exit(1)
    return api_key


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="30s episodes (faster)")
    parser.add_argument("--model", default="qwen3-vl-flash")
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 22, 33))
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--airsim-port", type=int, default=41451)
    parser.add_argument("--map", default="Town10HD")
    args = parser.parse_args()

    episode_seconds = 30.0 if args.quick else 60.0

    # ── Parameter grid: ordered by expected impact on LANDING ──
    param_grid = [
        # (proj, h_gain, v_gain, temp)
        # Baseline — current best for tracking
        (28.0, 5.0, 0.8, 0.2),

        # Higher vertical gain — faster descent
        (28.0, 5.0, 1.2, 0.2),
        (28.0, 5.0, 1.6, 0.2),
        (28.0, 5.0, 2.0, 0.2),

        # Lower horizontal gain — smoother alignment
        (28.0, 3.0, 1.6, 0.2),
        (28.0, 4.0, 1.6, 0.2),

        # Different projection distances
        (24.0, 5.0, 1.6, 0.2),
        (32.0, 5.0, 1.6, 0.2),
        (24.0, 4.0, 2.0, 0.2),
        (32.0, 4.0, 2.0, 0.2),

        # Combo: aggressive v_gain + moderate h_gain
        (28.0, 4.0, 1.4, 0.2),
        (28.0, 3.0, 1.4, 0.2),

        # Lower temperature for more consistent depth estimates
        (28.0, 5.0, 1.6, 0.1),
        (28.0, 4.0, 1.6, 0.1),

        # Wider exploration
        (20.0, 3.0, 1.0, 0.2),
        (36.0, 5.0, 1.8, 0.2),
        (28.0, 2.0, 1.0, 0.2),
        (28.0, 6.0, 1.8, 0.2),
    ]

    print(f"Auto-tuning for LANDING (map={args.map})")
    print(f"  Model: {args.model}  |  {len(args.seeds)} seeds × {episode_seconds:.0f}s  |  {len(param_grid)} combos")
    est_min = len(param_grid) * len(args.seeds) * (episode_seconds + 8) / 60
    print(f"  Est. ~{est_min:.0f} min")
    print()

    print("Connecting to CarlaAir...")
    env = CarlaAirEnvironment(carla_port=args.carla_port, airsim_port=args.airsim_port)
    print("Connected.\n")

    api_key = load_api_key()
    from experiments.spf_eval.config import SPFConfig, PROVIDERS
    config = SPFConfig(
        model=args.model,
        base_url=PROVIDERS["qianwenai"]["base_url"],
        api_key=api_key,
    )

    print_header()
    print("-" * 65)

    best, best_result = None, None
    all_results = []

    for i, (proj, h_gain, v_gain, temp) in enumerate(param_grid):
        config.projection_distance_max_m = proj
        config.horizontal_gain = h_gain
        config.vertical_gain = v_gain
        config.temperature = temp
        env.truck_speed = 4.0
        env.horizontal_gain = h_gain
        env.vertical_gain = v_gain
        policy = SPFPolicy(config=config)

        try:
            result = evaluate_params(env, policy, args.seeds, episode_seconds)
        except Exception as e:
            print(f"{i+1:>3} ERROR: {e}")
            continue

        p = {"proj": proj, "h_gain": h_gain, "v_gain": v_gain, "temp": temp}
        print_row(i + 1, p, result)
        all_results.append((p, result))

        # Optimize for LSR first, then TSR, then min_dist
        if best_result is None or (
            result["LSR"] > best_result["LSR"]
            or (result["LSR"] == best_result["LSR"] and result["TSR"] > best_result["TSR"])
            or (result["LSR"] == best_result["LSR"] and result["TSR"] == best_result["TSR"]
                and result["min_dist"] < best_result["min_dist"])
        ):
            best, best_result = p, result

    # ── Summary ──
    print()
    print("=" * 65)
    print("Top results (by LSR → TSR → min_dist):")
    print_header()
    sorted_results = sorted(all_results, key=lambda x: (
        -x[1]["LSR"], -x[1]["TSR"], x[1]["min_dist"]
    ))
    for p, r in sorted_results[:5]:
        mark = " ★" if p == best else ""
        print(f"     {p['proj']:>5.0f} {p['h_gain']:>6.1f} {p['v_gain']:>6.1f} {p['temp']:>5.2f}  "
              f"{r['LSR']:>6.2f} {r['TSR']:>6.2f} {r['min_dist']:>6.1f} {r['DF']:>6.2f} {r['visible_s']:>5.1f}{mark}")

    print()
    if best and best_result:
        print(f"Best: proj={best['proj']:.0f}m  h_gain={best['h_gain']:.1f}  "
              f"v_gain={best['v_gain']:.1f}  temp={best['temp']:.2f}")
        print(f"  LSR={best_result['LSR']:.2f}  TSR={best_result['TSR']:.2f}  "
              f"min_cargo_dist={best_result['min_dist']:.1f}m  DF={best_result['DF']:.2f}Hz")

        # Save
        PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        presets = json.loads(PRESETS_FILE.read_text()) if PRESETS_FILE.is_file() else {}
        name = f"landing_{args.map}"
        presets[name] = {
            "truck_speed": 4.0,
            "projection_distance_max_m": best['proj'],
            "horizontal_gain": best['h_gain'],
            "vertical_gain": best['v_gain'],
            "temperature": best['temp'],
            "LSR": best_result['LSR'],
            "TSR": best_result['TSR'],
            "min_dist": best_result['min_dist'],
        }
        PRESETS_FILE.write_text(json.dumps(presets, indent=2, ensure_ascii=False))
        print(f"  Saved → {PRESETS_FILE} [{name}]")

        if best_result["LSR"] > 0:
            print(f"\n  🎯 LSR > 0 achieved! Applying to config.py...")
            config_path = Path(__file__).resolve().parent / "experiments" / "spf_eval" / "config.py"
            content = config_path.read_text()
            for name, val in [
                ("DEFAULT_PROJECTION_DISTANCE_MAX_M", best['proj']),
                ("DEFAULT_HORIZONTAL_GAIN", best['h_gain']),
                ("DEFAULT_VERTICAL_GAIN", best['v_gain']),
                ("DEFAULT_TEMPERATURE", best['temp']),
            ]:
                content = re.sub(rf"^{name}: float = [\d.]+", f"{name}: float = {val}", content, flags=re.MULTILINE)
            config_path.write_text(content)
            print("  config.py updated!")

    env.shutdown()


if __name__ == "__main__":
    main()
