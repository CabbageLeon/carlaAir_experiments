#!/usr/bin/env python3
"""全自动降落调参 — 循环迭代直到 LSR >= 30% 或达到最大轮次。

Stages:
  A: 粗调 vertical_gain (0.8 → 3.0)
  B: 联调 v_gain × h_gain
  C: 精调 proj_max
  D: 综合微调 (all params)
  E: 修改 Prompt → 重来 (if all above fail)

Usage:
    python auto_land.py                    # 全自动模式
    python auto_land.py --quick            # 快速 (30s)
    python auto_land.py --max-rounds 5     # 最大迭代轮次
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
TARGET_LSR = 0.30


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
    last_visible_time = 0.0  # wall-clock time truck was last seen

    try:
        while time.monotonic() - started < seconds:
            now = env.tick()
            dt = now - previous
            previous = now
            visible = env.truck_in_camera_view()
            if visible:
                visible_seconds += dt
                last_visible_time = now
            elif now - last_visible_time > 3.0:
                break  # 卡车丢失超过3秒，立即中止
            state = env.drone_state()
            try:
                d = float(np.linalg.norm((env.cargo_bed_ned() - state.ned)[:2]))
                if d < min_cargo_dist:
                    min_cargo_dist = d
            except Exception:
                pass

            if pending is not None and pending.done():
                try:
                    command = pending.result()
                    active_waypoint = command.target_ned
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


def evaluate(env, policy, seeds, seconds):
    eps, nds, mds = [], [], []
    for s in seeds:
        ep, nd, md = run_one_episode(env, policy, s, seconds)
        eps.append(ep)
        nds.append(nd)
        mds.append(md)
    lsr = sum(1 for e in eps if e.landed) / len(eps)
    tsr = sum(1 for e in eps if e.tracked) / len(eps)
    return {"LSR": lsr, "TSR": tsr, "min_dist": sum(mds)/len(mds),
            "DF": sum(nds)/(len(eps)*seconds), "episodes": eps}


def run_stage(env, config, seeds, seconds, param_grid, stage_name):
    """Run one tuning stage, return best params + results."""
    print(f"\n{'='*60}")
    print(f"  Stage: {stage_name}  |  {len(param_grid)} combos × {len(seeds)} seeds × {seconds:.0f}s")
    print(f"{'='*60}")
    print(f"{'#':>3} {'proj':>5} {'h_g':>5} {'v_g':>5} {'t':>4}  {'LSR':>6} {'TSR':>6} {'min_d':>6} {'DF':>5}")
    print("-" * 55)

    best, best_r, best_p = None, None, None
    for i, (proj, hg, vg, tmp) in enumerate(param_grid):
        config.projection_distance_max_m = proj
        config.horizontal_gain = hg
        config.vertical_gain = vg
        config.temperature = tmp
        env.horizontal_gain = hg
        env.vertical_gain = vg
        policy = SPFPolicy(config=config)
        try:
            r = evaluate(env, policy, seeds, seconds)
        except Exception as e:
            print(f"{i+1:>3} ERROR: {e}")
            continue
        print(f"{i+1:>3} {proj:>5.0f} {hg:>5.1f} {vg:>5.1f} {tmp:>4.2f}  "
              f"{r['LSR']:>6.2f} {r['TSR']:>6.2f} {r['min_dist']:>6.1f} {r['DF']:>5.2f}")
        if best_r is None or r['LSR'] > best_r['LSR'] or (
            r['LSR'] == best_r['LSR'] and r['TSR'] > best_r['TSR']
        ):
            best, best_r, best_p = r, r, (proj, hg, vg, tmp)
    return best_p, best_r


def apply_best(proj, hg, vg, tmp):
    config_path = Path(__file__).resolve().parent / "experiments" / "spf_eval" / "config.py"
    content = config_path.read_text()
    for name, val in [
        ("DEFAULT_PROJECTION_DISTANCE_MAX_M", proj),
        ("DEFAULT_HORIZONTAL_GAIN", hg),
        ("DEFAULT_VERTICAL_GAIN", vg),
        ("DEFAULT_TEMPERATURE", tmp),
    ]:
        content = re.sub(rf"^{name}: float = [\d.]+", f"{name}: float = {val}", content, flags=re.MULTILINE)
    config_path.write_text(content)


def modify_prompt_for_landing():
    """Monkey-patch SPFPolicy._prompt to add landing guidance."""
    from experiments.spf_eval import spf_policy as spf_mod
    if hasattr(spf_mod.SPFPolicy, '_landing_modified'):
        return False

    original = spf_mod.SPFPolicy._prompt

    @staticmethod
    def landing_prompt(instruction: str) -> str:
        base = original(instruction)
        landing_hint = (
            "\n\nLANDING GUIDANCE: When the truck's cargo bed fills more than half the image, "
            "you are directly above it and ready to land. In this situation, place the point "
            "at the CENTER of the image (x=500, y=500) with depth=1. This commands the drone "
            "to descend straight down onto the cargo bed."
        )
        return base + landing_hint

    spf_mod.SPFPolicy._prompt = landing_prompt
    spf_mod.SPFPolicy._landing_modified = True
    return True


def restore_prompt():
    """Restore original prompt."""
    from experiments.spf_eval import spf_policy as spf_mod
    if hasattr(spf_mod.SPFPolicy, '_original_prompt'):
        spf_mod.SPFPolicy._prompt = spf_mod.SPFPolicy._original_prompt
        del spf_mod.SPFPolicy._landing_modified


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--model", default="qwen3-vl-flash")
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 22, 33))
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--airsim-port", type=int, default=41451)
    args = parser.parse_args()

    seconds = 30.0 if args.quick else 60.0

    print("=" * 60)
    print("  AUTO-LAND: 全自动降落调参")
    print(f"  Target LSR: {TARGET_LSR*100:.0f}%  |  Max rounds: {args.max_rounds}")
    print(f"  {len(args.seeds)} seeds × {seconds:.0f}s  |  Model: {args.model}")
    print("=" * 60)

    # Connect
    print("\nConnecting to CarlaAir...")
    env = CarlaAirEnvironment(carla_port=args.carla_port, airsim_port=args.airsim_port)
    print("Connected.")

    # Load config
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        sf = Path.home() / ".carlaair" / "state.json"
        if sf.is_file():
            api_key = json.loads(sf.read_text()).get("_api_key", "")
    if not api_key:
        print("No API key."); sys.exit(1)
    from experiments.spf_eval.config import SPFConfig, PROVIDERS
    config = SPFConfig(model=args.model, base_url=PROVIDERS["qianwenai"]["base_url"], api_key=api_key)
    env.truck_speed = 4.0

    # ── Tuning stages ──
    stages = [
        ("A: v_gain sweep", [
            (28.0, 5.0, vg, 0.2) for vg in [1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
        ]),
        ("B: v_gain × h_gain", [
            (28.0, hg, vg, 0.2)
            for hg in [3.0, 4.0, 5.0, 6.0]
            for vg in [1.4, 1.8, 2.2, 2.6]
        ]),
        ("C: proj_max tune", [
            (pj, 5.0, 2.0, 0.2) for pj in [20.0, 24.0, 28.0, 32.0, 36.0]
        ] + [
            (pj, 4.0, 2.2, 0.2) for pj in [24.0, 28.0, 32.0]
        ]),
        ("D: fine combo", [
            (pj, hg, vg, tmp)
            for pj in [24.0, 28.0, 32.0]
            for hg in [3.0, 4.0, 5.0]
            for vg in [1.6, 2.0, 2.4]
            for tmp in [0.1, 0.2]
        ][:20]),  # Cap at 20 combos
    ]

    best_lsr = 0.0
    best_params = (28.0, 5.0, 0.8, 0.2)
    round_num = 0
    prompt_modified = False

    while best_lsr < TARGET_LSR and round_num < args.max_rounds:
        round_num += 1
        stage_idx = min(round_num - 1, len(stages) - 1)
        name, grid = stages[stage_idx]

        params, result = run_stage(env, config, args.seeds, seconds, grid, name)
        if params is None:
            print(f"Round {round_num}: all combos failed. Trying next stage...")
            if stage_idx == len(stages) - 1 and not prompt_modified:
                print("\n  ⚠️  All stages exhausted. Modifying VLM prompt for landing...")
                modify_prompt_for_landing()
                prompt_modified = True
                round_num = 0  # restart
            continue

        best_params = params
        best_lsr = result['LSR']
        print(f"\n  Round {round_num} best: LSR={best_lsr:.2f} TSR={result['TSR']:.2f} "
              f"min_dist={result['min_dist']:.1f}m  proj={params[0]:.0f} h={params[1]:.1f} "
              f"v={params[2]:.1f} t={params[3]:.2f}")

        if best_lsr >= TARGET_LSR:
            break

        if best_lsr == 0.0 and stage_idx == len(stages) - 1 and not prompt_modified:
            print("\n  ⚠️  LSR still 0% after all stages. Modifying VLM prompt...")
            modify_prompt_for_landing()
            prompt_modified = True
            round_num = 0
            continue

    # ── Final ──
    print("\n" + "=" * 60)
    if best_lsr >= TARGET_LSR:
        print(f"  ✅ TARGET ACHIEVED! LSR = {best_lsr*100:.0f}%")
    else:
        print(f"  ⚠️  Max rounds reached. Best LSR = {best_lsr*100:.0f}%")

    print(f"  Best params: proj={best_params[0]:.0f}m  h_gain={best_params[1]:.1f}  "
          f"v_gain={best_params[2]:.1f}  temp={best_params[3]:.2f}")
    apply_best(*best_params)
    print("  Saved to config.py")

    if prompt_modified:
        print("  Prompt was modified for landing assistance.")
        print("  Restore original with: python -c 'from auto_land import restore_prompt; restore_prompt()'")

    env.shutdown()


if __name__ == "__main__":
    main()
