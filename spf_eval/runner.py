"""Run and report the SPF conditions specified in experiment.md."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .environment import CarlaAirEnvironment, TruckProfile
from .metrics import EscortEpisode, LandingEpisode, escort_summary, landing_summary, recovery_time, timing_summary
import math
from .prompts import escort_prompt, landing_prompt
from . import load_config
from .spf_policy import SPFPolicy
# OpenFlyPolicy imported lazily in _make_policy


class CarlaAirProcess:
    """Start one clean CARLA-Air renderer for each independent episode."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(__file__).resolve().parents[2]
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle = None

    @staticmethod
    def _running_pids() -> list[int]:
        result = subprocess.run(
            ["pgrep", "-f", "CarlaUE4-Linux-Shipping"],
            capture_output=True,
            text=True,
            check=False,
        )
        return [int(value) for value in result.stdout.split()]

    @staticmethod
    def _stop_pid(pid: int) -> None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.25)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def stop(self) -> None:
        pids = set(self._running_pids())
        if self.process is not None and self.process.poll() is None:
            pids.add(self.process.pid)
        for pid in pids:
            self._stop_pid(pid)
        if pids:
            time.sleep(self.args.carla_cooldown_seconds)
        self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def start(self, mode: str, seed: int, episode_index: int) -> None:
        self.stop()
        log_path = Path(self.args.output) / "carla_logs" / f"{mode}-seed-{seed}-episode-{episode_index:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        environment["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
        self.process = subprocess.Popen(
            [
                str(self.root / "carlaAir.sh"),
                self.args.map,
                "--port",
                str(self.args.carla_port),
                "--fg",
            ],
            cwd=self.root,
            env=environment,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + self.args.carla_warmup_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"CARLA-Air exited during startup with code {self.process.returncode}")
            time.sleep(0.25)


def _open_environment(args: argparse.Namespace, profile: TruckProfile | None = None) -> CarlaAirEnvironment:
    deadline = time.monotonic() + args.carla_start_timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return CarlaAirEnvironment(carla_port=args.carla_port, airsim_port=args.airsim_port, profile=profile, go_straight=args.go_straight)
        except Exception as exc:
            last_error = exc
            time.sleep(2.0)
    raise RuntimeError(f"CARLA-Air did not become ready within {args.carla_start_timeout:.0f}s") from last_error


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _state_sample(env: CarlaAirEnvironment, elapsed: float, visible: bool) -> dict:
    drone = env.drone_state()
    truck = env.truck_state()
    cargo = env.cargo_bed_ned()
    relative = cargo - drone.ned
    return {
        "t": elapsed,
        "type": "state",
        "truck_visible": visible,
        "on_cargo_bed": env.on_cargo_bed(),
        "drone_ned": drone.ned.tolist(),
        "truck_ned": truck.ned.tolist(),
        "cargo_bed_ned": cargo.tolist(),
        "cargo_horizontal_m": float((relative[0] ** 2 + relative[1] ** 2) ** 0.5),
        "cargo_down_m": float(relative[2]),
        "drone_speed_mps": drone.speed,
        "truck_speed_mps": truck.speed,
    }


def _debug_frame(frame_bgr: np.ndarray, prompt: str, raw_response: str,
                 point_xy: tuple[float, float] | None, depth: float | None,
                 waypoint_ned, frame_idx: int, target_dir: Path) -> None:
    """Draw VLM prediction overlay and save to debug directory."""
    h, w = frame_bgr.shape[:2]
    vis = frame_bgr.copy()

    # Draw crosshair at center
    cv2.line(vis, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 0), 1)
    cv2.line(vis, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 0), 1)

    # Draw VLM-predicted point
    if point_xy is not None:
        px = int(point_xy[0] / 1000.0 * w)
        py = int(point_xy[1] / 1000.0 * h)
        cv2.circle(vis, (px, py), 12, (0, 0, 255), 3)
        cv2.circle(vis, (px, py), 4, (0, 0, 255), -1)
        label = f"VLM point ({point_xy[0]:.0f},{point_xy[1]:.0f}) depth={depth:.1f}" if depth else ""
        cv2.putText(vis, label, (px + 15, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Overlay waypoint info
    y0 = 30
    if waypoint_ned is not None:
        wp = waypoint_ned
        lines = [f"Waypoint NED: [{wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}]"]
    else:
        lines = ["Waypoint: (none)"]
    if raw_response:
        lines.append(f"VLM: {raw_response[:120]}")
    for line in lines:
        for sub in textwrap.wrap(line, width=80):
            cv2.putText(vis, sub, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y0 += 18

    target_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target_dir / f"frame_{frame_idx:04d}.png"), vis)


def _run_landing(env: CarlaAirEnvironment, policy, policy_name: str, mode: str, seed: int, seconds: float,
                 spawn_index: int = 0, debug_dir: Path | None = None) -> tuple[LandingEpisode, list[dict]]:
    env.reset(seed, spawn_index=spawn_index)
    policy.reset()
    events: list[dict] = []
    next_decision = 0.0
    visible_seconds = 0.0
    stable_since: float | None = None
    stable = False
    started = time.monotonic()
    previous = started
    next_sample = 0.0
    executor = ThreadPoolExecutor(max_workers=1)
    pending = None
    pending_state = None
    pending_frame: np.ndarray | None = None
    active_waypoint = None
    target_yaw = None
    next_tracking = 0.0
    decision_idx = 0
    try:
        while time.monotonic() - started < seconds:
            now = env.tick()
            elapsed, dt = now - started, now - previous
            previous = now
            visible = env.truck_in_camera_view()
            if visible:
                visible_seconds += dt
            if elapsed >= next_sample:
                events.append(_state_sample(env, elapsed, visible))
                next_sample = elapsed + 1.0
            state = env.drone_state()
            if pending is not None and pending.done():
                try:
                    command = pending.result()
                    if pending_state is None:
                        raise RuntimeError("SPF decision is missing its observation state")
                    if mode == "C2":
                        env.apply_c2_speed(
                            command.target_ned,
                            pending_state.ned,
                            command.inference_finished_at - command.inference_started_at,
                        )
                    active_waypoint = command.target_ned
                    next_tracking = now
                    decision_time = command.inference_finished_at - command.inference_started_at

                    if policy_name == "OPENFLY":
                        # OpenFly outputs fwd=/lat=/vert=/turn= tokens
                        fwd_m = lat_m = vert_m = turn_d = 0.0
                        raw = command.raw_response
                        try:
                            if 'fwd=' in raw:
                                fwd_m = abs(float(raw.split('fwd=')[-1].split()[0]))
                                lat_m = abs(float(raw.split('lat=')[-1].split()[0]))
                                vert_m = abs(float(raw.split('vert=')[-1].split()[0]))
                                turn_d = abs(float(raw.split('turn=')[-1].replace(chr(176), '').split()[0]))
                        except Exception:
                            pass
                        is_stop = 'STOP' in raw
                        print(f"[vlm] dt={decision_time*1000:.0f}ms fwd={fwd_m:.2f} turn={turn_d:.1f}" + (" STOP" if is_stop else ""))
                        if not is_stop and (fwd_m > 0.1 or lat_m > 0.1 or vert_m > 0.1 or turn_d > 0.1):
                            if turn_d > 0.1:
                                sign = -1 if '-' in raw.split('turn=')[-1] else 1
                                target_yaw = pending_state.yaw_rad + math.radians(sign * turn_d)
                            else:
                                target_yaw = None
                        else:
                            active_waypoint = None
                            target_yaw = None
                            if not is_stop:
                                print("[vlm] all dims < 0.1, skipping")
                    else:
                        # SPF: waypoint comes from image point → NED projection; derive yaw from direction
                        delta = command.target_ned - pending_state.ned
                        point_y, point_x, point_d = 0.0, 0.0, 0.0
                        try:
                            point_y, point_x, point_d = SPFPolicy.parse_point(command.raw_response)
                        except Exception:
                            pass
                        print(f"[spf] dt={decision_time*1000:.0f}ms point=({point_x:.0f},{point_y:.0f}) depth={point_d:.0f}")
                        if np.linalg.norm(delta[:2]) < 0.1:
                            active_waypoint = None
                            target_yaw = None
                        else:
                            target_yaw = math.atan2(delta[1], delta[0])

                    next_decision = command.inference_finished_at
                    events.append(
                        {
                            "t": elapsed,
                            "type": "decision",
                            "prompt": command.prompt,
                            "waypoint_ned": command.target_ned.tolist(),
                            "inference_seconds": decision_time,
                            "ecl_seconds": max(0.0, time.monotonic() - command.inference_finished_at),
                            "raw_response": command.raw_response,
                        }
                    )
                    # Debug: save annotated frame
                    if debug_dir is not None and pending_frame is not None:
                        point_xy, depth = None, None
                        try:
                            y_val, x_val, d_val = SPFPolicy.parse_point(command.raw_response)
                            point_xy, depth = (x_val, y_val), d_val
                        except Exception:
                            pass
                        _debug_frame(pending_frame, command.prompt, command.raw_response,
                                     point_xy, depth, command.target_ned,
                                     decision_idx, debug_dir)
                        pending_frame = None
                        decision_idx += 1
                except Exception as exc:
                    next_decision = now + 1.0
                    events.append({"t": elapsed, "type": "no_waypoint", "error": str(exc)})
                pending = None
                pending_state = None
            if active_waypoint is not None and now >= next_tracking:
                velocity = env.track_waypoint(active_waypoint, duration=0.15, target_yaw=target_yaw)
                next_tracking = now + 0.1
                if float((velocity**2).sum() ** 0.5) < 0.05:
                    active_waypoint = None
            if pending is None and now >= next_decision:
                direction, motion, _ = env.cargo_relation()
                prompt = landing_prompt(policy_name, mode, direction, motion, env.landing_phase())
                try:
                    frame = env.capture_rgb()
                    pending_state = state
                    if debug_dir is not None:
                        pending_frame = frame.copy()
                    pending = executor.submit(policy.act, frame, prompt, state.ned, state.yaw_rad)
                except Exception as exc:
                    next_decision = now + 1.0
                    events.append({"t": elapsed, "type": "no_waypoint", "error": str(exc)})
            try:
                on_cargo_bed = env.on_cargo_bed()
            except Exception as exc:
                on_cargo_bed = False
                events.append({"t": elapsed, "type": "state_error", "error": str(exc)})
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
        except Exception as exc:
            landed_on_bed = False
            events.append({"t": time.monotonic() - started, "type": "state_error", "error": str(exc)})
        events.append({"t": time.monotonic() - started, "type": "episode_end"})
        executor.shutdown(wait=False, cancel_futures=True)
        env.close_episode()
    return LandingEpisode(seed, visible_seconds, landed_on_bed, stable), events


def _run_escort(env: CarlaAirEnvironment, policy, policy_name: str, mode: str, seed: int, seconds: float, spawn_index: int = 0) -> tuple[EscortEpisode, list[dict]]:
    env.reset(seed, spawn_index=spawn_index)
    policy.reset()
    events: list[dict] = []
    samples: list[tuple[float, float]] = []
    next_decision = 0.0
    started = time.monotonic()
    previous = started
    next_sample = 0.0
    occlusion_onsets: list[float] = []
    previous_occluded = False
    executor = ThreadPoolExecutor(max_workers=1)
    pending = None
    try:
        while time.monotonic() - started < seconds:
            now = env.tick()
            elapsed = now - started
            iou = env.truck_camera_iou()
            samples.append((elapsed, iou))
            occluded = iou < 0.15
            if elapsed >= next_sample:
                events.append(_state_sample(env, elapsed, iou >= 0.15) | {"iou": iou})
                next_sample = elapsed + 1.0
            if occluded and not previous_occluded:
                occlusion_onsets.append(elapsed)
                events.append({"t": elapsed, "type": "occlusion_onset", "iou": iou})
            previous_occluded = occluded
            if pending is not None and pending.done():
                try:
                    command = pending.result()
                    env.track_waypoint(command.target_ned, target_yaw=None)
                    decision_time = command.inference_finished_at - command.inference_started_at
                    next_decision = command.inference_finished_at
                    events.append(
                        {
                            "t": elapsed,
                            "type": "decision",
                            "prompt": command.prompt,
                            "waypoint_ned": command.target_ned.tolist(),
                            "inference_seconds": decision_time,
                            "ecl_seconds": max(0.0, time.monotonic() - command.inference_finished_at),
                            "raw_response": command.raw_response,
                        }
                    )
                except Exception as exc:
                    next_decision = now + 1.0
                    events.append({"t": elapsed, "type": "no_waypoint", "error": str(exc)})
                pending = None
            if pending is None and now >= next_decision:
                state = env.drone_state()
                direction, motion, _ = env.cargo_relation()
                prompt = escort_prompt(policy_name, mode, occluded, direction, motion, "occlusion recovery" if occluded else "normal escort")
                try:
                    frame = env.capture_rgb()
                    pending = executor.submit(policy.act, frame, prompt, state.ned, state.yaw_rad)
                except Exception as exc:
                    next_decision = now + 1.0
                    events.append({"t": elapsed, "type": "no_waypoint", "error": str(exc)})
            previous = now
    finally:
        events.append({"t": time.monotonic() - started, "type": "episode_end"})
        executor.shutdown(wait=False, cancel_futures=True)
        env.close_episode()
    rats = tuple(recovery_time(samples, onset) for onset in occlusion_onsets)
    recovered = sum(value < 15.0 for value in rats)
    return EscortEpisode(seed, recovered, len(rats), rats), events


def _execute_condition(
    policy: SPFPolicy, args: argparse.Namespace, mode: str, process: CarlaAirProcess | None,
    truck_profile: TruckProfile | None = None,
) -> tuple[list[LandingEpisode], list[EscortEpisode]]:
    landing: list[LandingEpisode] = []
    escort: list[EscortEpisode] = []
    all_events: list[dict] = []
    pn = args.policy.upper()
    output = Path(args.output) / args.task / pn / mode
    debug_dir = output / "debug" if args.debug else None
    # Pre-load model before episodes so the drone doesn't wait mid-flight
    if hasattr(policy, "warmup"):
        policy.warmup()
    shared_env = None if process is not None else _open_environment(args, truck_profile)
    try:
        for seed in args.seeds:
            for episode_index in range(args.episodes_per_seed):
                env = shared_env
                try:
                    if process is not None:
                        process.start(mode, seed, episode_index)
                        env = _open_environment(args, truck_profile)
                    if env is None:
                        raise RuntimeError("CARLA-Air environment is unavailable")
                    if args.task == "landing":
                        episode, events = _run_landing(env, policy, args.policy, mode, seed, args.seconds, spawn_index=args.spawn_index, debug_dir=debug_dir)
                        landing.append(episode)
                    else:
                        episode, events = _run_escort(env, policy, args.policy, mode, seed, args.seconds, spawn_index=args.spawn_index)
                        escort.append(episode)
                    all_events.extend(events)
                    name = (
                        f"seed-{seed}.json"
                        if args.episodes_per_seed == 1
                        else f"seed-{seed}-episode-{episode_index:02d}.json"
                    )
                    _write_json(output / name, {"episode": asdict(episode), "events": events})
                finally:
                    if process is not None:
                        if env is not None:
                            env.shutdown()
                        env = None
                        gc.collect()
                        process.stop()
    finally:
        if shared_env is not None:
            shared_env.shutdown()
    _write_json(output / "timing.json", timing_summary(all_events))
    return landing, escort


def _make_policy(args: argparse.Namespace):
    cfg = load_config()
    model_cfg = cfg["models"][args.policy]
    if args.policy == "openfly":
        from .openfly_policy import OpenFlyPolicy
        return OpenFlyPolicy(model_cfg)
    return SPFPolicy(model_cfg)


def run(args: argparse.Namespace, truck_profile: TruckProfile | None = None) -> dict[str, object]:
    policy = _make_policy(args)
    policy_name = args.policy.upper()
    process = CarlaAirProcess(args) if args.restart_carla_per_episode else None
    modes = ("C0", "C1", "C2") if args.task == "landing" else ("C0", "C1")
    modes = modes if args.mode == "all" else (args.mode,)
    results: dict[str, object] = {}
    c0_landing: list[LandingEpisode] | None = None
    try:
        for mode in modes:
            landing, escort = _execute_condition(policy, args, mode, process, truck_profile)
            if args.task == "landing":
                if mode == "C0":
                    c0_landing = landing
                elif c0_landing is None:
                    if not args.allow_unpaired_landing:
                        raise ValueError("run landing with mode 'all' so C1/C2 CG is paired with C0")
                    c0_landing = landing
                summary: dict[str, float] = landing_summary(landing, c0_landing)
            else:
                summary = escort_summary(escort)
            summary.update(json.loads((Path(args.output) / args.task / policy_name / mode / "timing.json").read_text()))
            _write_json(Path(args.output) / args.task / policy_name / mode / "summary.json", summary)
            results[mode] = summary
    finally:
        if process is not None:
            process.stop()
    return results if args.mode == "all" else results[args.mode]


KNOWN_TRUCKS = [
    ("European HGV (默认)",  "vehicle.carlamotors.european_hgv", TruckProfile()),
    ("Carlacola",            "vehicle.carlamotors.carlacola",     TruckProfile(bed_center_x=-2.30, bed_half_length=1.45, bed_half_width=1.10, bed_height=1.15)),
    ("Tesla Model 3",        "vehicle.tesla.model3",              TruckProfile(bed_center_x=-1.50, bed_half_length=0.80, bed_half_width=0.70, bed_height=0.75)),
    ("Audi TT",              "vehicle.audi.tt",                   TruckProfile(bed_center_x=-1.20, bed_half_length=0.60, bed_half_width=0.65, bed_height=0.70)),
    ("Ford Mustang",         "vehicle.ford.mustang",              TruckProfile(bed_center_x=-1.60, bed_half_length=0.75, bed_half_width=0.70, bed_height=0.75)),
    ("Mini Cooper S",        "vehicle.mini.cooper_s",             TruckProfile(bed_center_x=-1.00, bed_half_length=0.50, bed_half_width=0.60, bed_height=0.70)),
    ("BMW Grandtourer",      "vehicle.bmw.grandtourer",           TruckProfile(bed_center_x=-1.70, bed_half_length=0.80, bed_half_width=0.70, bed_height=0.75)),
    ("Nissan Patrol",        "vehicle.nissan.patrol",             TruckProfile(bed_center_x=-1.50, bed_half_length=0.85, bed_half_width=0.75, bed_height=0.85)),
    ("Mercedes Coupe",       "vehicle.mercedes.coupe",            TruckProfile(bed_center_x=-1.50, bed_half_length=0.75, bed_half_width=0.70, bed_height=0.72)),
]

KNOWN_DRONES = [
    ("SimpleFlight", "SimpleFlight", "SimpleFlight"),
]


def _pick_list(title: str, items: list[tuple[str, str, object]], default_idx: int = 0) -> object:
    """Generic interactive picker. Returns the third element of the chosen item."""
    print(title)
    for i, (label, _id, _payload) in enumerate(items, 1):
        print(f"  {i}. {label} ({_id})")
    while True:
        try:
            choice = input(f"选择 [1-{len(items)}, 默认={default_idx+1}]: ").strip()
            if choice == "":
                return items[default_idx][2]
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx][2]
        except (ValueError, EOFError, KeyboardInterrupt):
            pass
        print(f"请输入 1-{len(items)}")


def _choose_map() -> str:
    """Scan the CarlaUE4 Maps directory and prompt the user to select a map."""
    maps_dir = Path(__file__).resolve().parent.parent / "CarlaUE4" / "Content" / "Carla" / "Maps"
    if not maps_dir.is_dir():
        return "Town10HD"
    maps = sorted(
        {p.stem for p in maps_dir.glob("Town*.umap") if "_Opt" not in p.stem and "BuiltData" not in p.stem}
    )
    if not maps:
        return "Town10HD"
    items = [(name, name, name) for name in maps]
    return _pick_list("可用地图:", items, default_idx=maps.index("Town10HD") if "Town10HD" in maps else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SPF in CARLA-Air")
    parser.add_argument("task", choices=("landing", "escort"))
    parser.add_argument("mode", choices=("C0", "C1", "C2", "all"))
    parser.add_argument("--model", default="qwen3-vl-flash")
    parser.add_argument("--policy", choices=("spf", "openfly"), default="spf")
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 22, 33))
    parser.add_argument("--episodes-per-seed", type=int, default=50)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--output", default="runs")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--airsim-port", type=int, default=41451)
    parser.add_argument("--map", default="Town10HD")
    parser.add_argument("--truck", default="vehicle.carlamotors.european_hgv")
    parser.add_argument("--drone", default="SimpleFlight")
    parser.add_argument("--spawn-index", type=int, default=-1, help="卡车出生点索引 (0-4, 距离无人机最近的5个点)")
    parser.add_argument("--carla-start-timeout", type=float, default=90.0)
    parser.add_argument("--carla-warmup-seconds", type=float, default=15.0)
    parser.add_argument("--carla-cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--restart-carla-per-episode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-unpaired-landing", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Save annotated camera frames for each VLM decision")
    parser.add_argument("--go-straight", action="store_true", help="卡车优先直行 (路口不随机转弯)")
    args = parser.parse_args()
    if args.task == "escort" and args.mode == "C2":
        parser.error("C2 is defined only for moving-platform landing")
    if args.episodes_per_seed < 1:
        parser.error("--episodes-per-seed must be positive")
    args.seconds = args.seconds or (60.0 if args.task == "landing" else 90.0)

    # Map uses default unless explicitly given (no interactive prompt)
    truck_profile: TruckProfile | None = None
    for label, bid, profile in KNOWN_TRUCKS:
        if args.truck.lower() in bid.lower() or args.truck.lower() in label.lower():
            truck_profile = profile
            break
    if truck_profile is None:
        truck_profile = TruckProfile(blueprint_id=args.truck)
    if not args.drone:
        print(f"无人机: SimpleFlight (CarlaAir 内置唯一型号)")
        args.drone = "SimpleFlight"

    if args.spawn_index < 0:
        while True:
            try:
                val = input("卡车出生点索引 (0, 1, 2, ... 或回车=0): ").strip()
                args.spawn_index = int(val) if val else 0
                break
            except (ValueError, EOFError):
                print("请输入整数")
        if args.spawn_index < 0:
            args.spawn_index = 0

    print(f"地图={args.map}  车辆={truck_profile.blueprint_id}  无人机={args.drone}  出生点=#{args.spawn_index}")
    print(json.dumps(run(args, truck_profile), indent=2))


if __name__ == "__main__":
    main()
