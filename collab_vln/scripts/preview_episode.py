#!/usr/bin/env python3
"""
Interactive episode preview — spawn colored trucks and manually fly to inspect.

Controls:
    Drone (terminal focus):
        W/S       Forward / Back
        A/D       Left / Right
        Q/E       Up / Down
        ←/→       Yaw left / right

    Truck:
        ↑/↓       Throttle / Brake
        Num4/Num6 Steer (keypad ←/→)

    Global:
        R         Reset positions
        N         Next episode
        P         Previous episode
        C         Capture drone bird's-eye view (saves PNG)
        ESC       Quit

Usage:
    conda activate carlaAir
    python collab_vln/scripts/preview_episode.py --input collab_vln/episodes/town10hd_templates.json
"""

import argparse
import json
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

import airsim
import carla
import cv2
import numpy as np

W, H = 1280, 720

START_TRUCK = "vehicle.mini.cooper_s"               # small car at UGV start
GOAL_TRUCK = "vehicle.carlamotors.european_hgv"     # big truck at goal


class KeyboardReader:
    """Raw-mode terminal keyboard reader with sticky-key support."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        self.running = False
        self._thread = None

        # ── Drone keys (sticky — held while key is down) ──
        self.drone_fwd = False
        self.drone_back = False
        self.drone_left = False
        self.drone_right = False
        self.drone_up = False
        self.drone_down = False
        self.drone_yaw_left = False
        self.drone_yaw_right = False
        self.truck_throttle = False
        self.truck_brake = False
        self.truck_steer_left = False
        self.truck_steer_right = False

        # ── One-shot triggers ──
        self.reset = False
        self.next_ep = False
        self.prev_ep = False
        self.capture = False
        self.quit = False

        # Track held keys
        self._held = set()

    def start(self):
        tty.setraw(self.fd)
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def _read_seq(self) -> bytes:
        ch = os.read(self.fd, 1)
        if ch == b'\x1b':
            ready, _, _ = select.select([self.fd], [], [], 0.02)
            if ready:
                return b'\x1b' + os.read(self.fd, 5)
            return b'\x1b'
        return ch

    def _loop(self):
        while self.running:
            try:
                ready, _, _ = select.select([self.fd], [], [], 0.05)
                if not ready:
                    continue
                seq = self._read_seq()
                self._handle(seq)
            except (OSError, IOError):
                break

    def _handle(self, seq: bytes):
        # ── Drone movement ──
        if seq in (b'w', b'W'):
            self.drone_fwd = True; self._held.add(b'w')
        if seq in (b's', b'S'):
            self.drone_back = True; self._held.add(b's')
        if seq in (b'a', b'A'):
            self.drone_left = True; self._held.add(b'a')
        if seq in (b'd', b'D'):
            self.drone_right = True; self._held.add(b'd')
        if seq in (b'q', b'Q'):
            self.drone_up = True; self._held.add(b'q')
        if seq in (b'e', b'E'):
            self.drone_down = True; self._held.add(b'e')

        # ── Drone yaw ──
        if seq in (b'\x1b[D', b'\x1bOD'):  # ← arrow
            self.drone_yaw_left = True; self._held.add(b'yawL')
        if seq in (b'\x1b[C', b'\x1bOC'):  # → arrow
            self.drone_yaw_right = True; self._held.add(b'yawR')

        # ── Truck control (arrows for throttle, keypad for steer) ──
        if seq in (b'\x1b[A', b'\x1bOA'):  # ↑
            self.truck_throttle = True; self._held.add(b'thr')
        if seq in (b'\x1b[B', b'\x1bOB'):  # ↓
            self.truck_brake = True; self._held.add(b'brk')

        # ── Steer: use < and > or comma/period ──
        if seq in (b',', b'<'):
            self.truck_steer_left = True; self._held.add(b'stL')
        if seq in (b'.', b'>'):
            self.truck_steer_right = True; self._held.add(b'stR')

        # ── One-shot keys ──
        if seq in (b'r', b'R'):
            self.reset = True
        if seq in (b'n', b'N'):
            self.next_ep = True
        if seq in (b'p', b'P'):
            self.prev_ep = True
        if seq in (b'c', b'C'):
            self.capture = True
        if seq in (b'\x1b', b'x', b'X'):
            self.quit = True

    def release_sticky(self):
        """Release all sticky keys so they must be re-pressed."""
        self.drone_fwd = False
        self.drone_back = False
        self.drone_left = False
        self.drone_right = False
        self.drone_up = False
        self.drone_down = False
        self.drone_yaw_left = False
        self.drone_yaw_right = False
        self.truck_throttle = False
        self.truck_brake = False
        self.truck_steer_left = False
        self.truck_steer_right = False
        self._held.clear()


class EpisodePreview:
    def __init__(self, templates_path: str):
        data = json.loads(Path(templates_path).read_text(encoding="utf-8"))
        self.episodes = data["episodes"]
        self.map_name = data["meta"]["map"]
        self.idx = 0
        self._pic_dir = Path(__file__).resolve().parent.parent / "captures"
        self._pic_dir.mkdir(exist_ok=True)

    @property
    def current(self) -> dict:
        return self.episodes[self.idx]

    @property
    def total(self) -> int:
        return len(self.episodes)

    def next(self):
        self.idx = (self.idx + 1) % self.total

    def prev(self):
        self.idx = (self.idx - 1) % self.total


def setup_carla_airsim() -> tuple[carla.Client, carla.World, airsim.MultirotorClient]:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    air = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=10)
    air.confirmConnection()
    air.enableApiControl(True)
    air.armDisarm(True)
    return client, world, air


def clear_vehicles(world):
    for v in world.get_actors().filter("vehicle.*"):
        try:
            v.destroy()
        except RuntimeError:
            pass


def spawn_truck(world, bp_name: str, location: carla.Location, yaw: float) -> carla.Vehicle:
    bp = world.get_blueprint_library().find(bp_name)
    tf = carla.Transform(location, carla.Rotation(yaw=yaw))
    actor = world.try_spawn_actor(bp, tf)
    if actor is None:
        raise RuntimeError(f"Cannot spawn {bp_name} at ({location.x:.1f}, {location.y:.1f})")
    return actor


def neds_to_carla(ned, airsim_offset):
    """Convert NED to CARLA world coordinates."""
    delta = np.asarray(ned) - airsim_offset
    return carla.Location(x=float(delta[0]), y=float(delta[1]), z=float(-delta[2]))


def carla_to_neds(loc, airsim_offset):
    return airsim_offset + np.array([loc.x, loc.y, -loc.z], dtype=float)


def get_airsim_offset(air, world):
    """Align AirSim NED and CARLA coordinates via the built-in drone actor."""
    drone = next((a for a in world.get_actors() if "drone" in a.type_id.lower()), None)
    if drone is None:
        raise RuntimeError("No CarlaAir drone actor found")
    state = air.getMultirotorState().kinematics_estimated.position
    dl = drone.get_location()
    return np.array([state.x_val - dl.x, state.y_val - dl.y, state.z_val + dl.z], dtype=float)


def move_drone_to(air, loc: carla.Location, offset: np.ndarray, yaw: float = 0.0):
    ned = carla_to_neds(loc, offset)
    pose = airsim.Pose(
        airsim.Vector3r(float(ned[0]), float(ned[1]), float(ned[2])),
        airsim.to_quaternion(0.0, 0.0, math.radians(yaw)),
    )
    # Ensure drone is flying before teleporting
    try:
        state = air.getMultirotorState()
        if state.landed_state == airsim.LandedState.Landed:
            air.takeoffAsync().join()
    except Exception:
        air.takeoffAsync()
        time.sleep(1.0)
    air.simSetVehiclePose(pose, ignore_collision=True)
    time.sleep(0.2)
    try:
        air.hoverAsync()
        time.sleep(0.1)
    except Exception:
        pass


def capture_drone_view(air) -> np.ndarray:
    """Capture RGB from the AirSim drone front camera '0'."""
    responses = air.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    if responses and responses[0].width > 0:
        return np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(
            responses[0].height, responses[0].width, 3
        )
    raise RuntimeError("Empty drone frame")


def draw_hud_info(frame, ep, key, offset, idx, total, drone_pos):
    """Overlay episode info on image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"Episode {idx+1}/{total}: {ep['id']}",
        f"Distance: {ep['distance_m']:.0f}m",
        f"Instruction: {ep['instruction']}",
        f"UGV spawn #{ep['ugv_spawn']['index']} -> Goal ({ep['goal']['x']:.0f}, {ep['goal']['y']:.0f})",
        "",
        f"Drone NED: ({drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f})",
        "",
        "Keys: WASD/QE=Drone  ↑↓=Throttle/Brk  ,.=Steer",
        "  R=Reset  C=Capture  ESC=Quit",
    ]
    y0 = 30
    for line in lines:
        cv2.putText(frame, line, (10, y0), font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        y0 += 18
    return frame


def _pick_file(episodes_dir: Path) -> Path:
    """Interactive file picker for episode JSON files."""
    files = sorted(episodes_dir.glob("*_templates.json")) + sorted(episodes_dir.glob("*_episodes.json"))
    files = list(dict.fromkeys(files))  # dedup
    if not files:
        print("No episode files found in", episodes_dir)
        sys.exit(1)

    print("\nAvailable episode files:")
    for i, f in enumerate(files):
        print(f"  {i+1:>2}. {f.name}")
    while True:
        try:
            choice = input(f"\n选择 [1-{len(files)}, 回车=1]: ").strip()
            idx = int(choice) - 1 if choice else 0
            if 0 <= idx < len(files):
                return files[idx]
        except (ValueError, EOFError):
            pass
        print(f"输入 1-{len(files)}")


def _pick_episode(preview: EpisodePreview) -> None:
    """Interactive episode picker within a loaded file."""
    print(f"\n{preview.total} episodes in {preview.map_name}:")
    for i, ep in enumerate(preview.episodes):
        g = ep["goal"]
        us = ep["ugv_spawn"]
        print(f"  {i+1:>2}. {ep['id']}  |  #{us['index']} -> ({g['x']:.0f},{g['y']:.0f})  |  {ep['distance_m']:.0f}m")
    while True:
        try:
            choice = input(f"\n选择 episode [1-{preview.total}, 回车=1]: ").strip()
            idx = int(choice) - 1 if choice else 0
            if 0 <= idx < preview.total:
                preview.idx = idx
                return
        except (ValueError, EOFError):
            pass
        print(f"输入 1-{preview.total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to episode templates JSON (omit for interactive picker)")
    args = parser.parse_args()

    episodes_dir = Path(__file__).resolve().parent.parent / "episodes"
    path = Path(args.input) if args.input else _pick_file(episodes_dir)

    preview = EpisodePreview(str(path))
    _pick_episode(preview)
    ep = preview.current

    key = KeyboardReader()

    print(f"\n  Episode: {ep['id']}  |  Map: {preview.map_name}  |  Distance: {ep['distance_m']:.0f}m")
    print(f"  START = {START_TRUCK} (Mini Cooper)  |  GOAL = {GOAL_TRUCK} (HGV)")
    print(f"  Start #{ep['ugv_spawn']['index']}: ({ep['ugv_spawn']['x']:.0f}, {ep['ugv_spawn']['y']:.0f})")
    print(f"  Goal:           ({ep['goal']['x']:.0f}, {ep['goal']['y']:.0f})")
    print("\n  Drone:  W/S=Forward/Back  A/D=Left/Right  Q/E=Up/Down  ←/→=Yaw")
    print("  Truck:  ↑/↓=Throttle/Brk  ,.=Steer")
    print("  R=Reset  C=Capture  ESC=Quit\n")

    client, world, air = setup_carla_airsim()
    offset = get_airsim_offset(air, world)
    key.start()

    blue_truck = None
    red_truck = None
    truck_camera = None
    overhead_camera = None
    latest_truck_img = [None]
    latest_overhead_img = [None]

    def truck_img_cb(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        latest_truck_img[0] = arr.reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]

    def overhead_img_cb(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        latest_overhead_img[0] = arr.reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]

    def spawn_overhead_camera(ep):
        """Spawn a static top-down camera high above, covering start and goal."""
        g = ep["goal"]
        us = ep["ugv_spawn"]
        mid_x = (us["x"] + g["x"]) / 2
        mid_y = (us["y"] + g["y"]) / 2
        alt = 150.0

        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "640")
        cam_bp.set_attribute("image_size_y", "640")
        cam_bp.set_attribute("fov", "110")
        cam_tf = carla.Transform(
            carla.Location(x=mid_x, y=mid_y, z=alt),
            carla.Rotation(pitch=-90, yaw=0, roll=0),  # straight down
        )
        cam = world.spawn_actor(cam_bp, cam_tf)
        cam.listen(overhead_img_cb)
        return cam

    def draw_overhead_markers(frame, ep):
        """Draw start/goal markers on overhead view."""
        h, w = frame.shape[:2]
        g = ep["goal"]
        us = ep["ugv_spawn"]
        # Camera is at midpoint, altitude 150m, FOV 110deg
        mid_x = (us["x"] + g["x"]) / 2
        mid_y = (us["y"] + g["y"]) / 2
        alt = 150.0
        fov = 110.0
        half_w = alt * math.tan(math.radians(fov / 2))
        scale = (w / 2) / half_w

        def proj(wx, wy):
            px = int(w / 2 + (wx - mid_x) * scale)
            py = int(h / 2 - (wy - mid_y) * scale)
            return max(0, min(w-1, px)), max(0, min(h-1, py))

        gx, gy = proj(g["x"], g["y"])
        sx, sy = proj(us["x"], us["y"])

        # Goal (red cross)
        cv2.circle(frame, (gx, gy), 15, (0, 0, 255), 2)
        cv2.circle(frame, (gx, gy), 4, (0, 0, 255), -1)
        cv2.putText(frame, "GOAL", (gx+10, gy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        # Start (blue)
        cv2.circle(frame, (sx, sy), 12, (255, 0, 0), 2)
        cv2.circle(frame, (sx, sy), 3, (255, 0, 0), -1)
        cv2.putText(frame, "START", (sx+10, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        # Line
        cv2.line(frame, (sx, sy), (gx, gy), (0, 255, 255), 1)
        # Info
        cv2.putText(frame, f"{ep['id']} | {ep['distance_m']:.0f}m", (5, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        return frame

    def spawn_episode(ep):
        nonlocal blue_truck, red_truck, truck_camera, overhead_camera
        if truck_camera:
            try: truck_camera.stop(); truck_camera.destroy()
            except: pass
        if overhead_camera:
            try: overhead_camera.stop(); overhead_camera.destroy()
            except: pass

        clear_vehicles(world)
        time.sleep(0.3)

        us = ep["ugv_spawn"]
        ugv_loc = carla.Location(x=us["x"], y=us["y"], z=us["z"] + 0.5)
        blue_truck = spawn_truck(world, START_TRUCK, ugv_loc, us["yaw"])

        # Red at goal
        g = ep["goal"]
        goal_loc = carla.Location(x=g["x"], y=g["y"], z=g["z"] + 0.5)
        red_truck = spawn_truck(world, GOAL_TRUCK, goal_loc, 0.0)

        # Drone 30m above blue truck
        drone_loc = carla.Location(x=us["x"], y=us["y"], z=us["z"] + 30.0)
        move_drone_to(air, drone_loc, offset, yaw=0.0)

        # Truck chase camera
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(W))
        cam_bp.set_attribute("image_size_y", str(H))
        cam_bp.set_attribute("fov", "100")
        cam_tf = carla.Transform(carla.Location(x=-8.0, z=4.0), carla.Rotation(pitch=-15))
        truck_camera = world.spawn_actor(cam_bp, cam_tf, attach_to=blue_truck)
        truck_camera.listen(truck_img_cb)

        # Overhead camera
        overhead_camera = spawn_overhead_camera(ep)

    spawn_episode(preview.current)

    try:
        running = True
        while running:
            if key.quit:
                running = False
                break

            ep = preview.current

            # ── Reset ──
            if key.reset:
                key.reset = False
                if truck_camera:
                    truck_camera.stop()
                    truck_camera.destroy()
                spawn_episode(preview.current)
                print(f"  [Reset] Episode {preview.current['id']}")

            # ── Capture drone view ──
            if key.capture:
                key.capture = False
                try:
                    frame = capture_drone_view(air)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = preview._pic_dir / f"{preview.current['id']}_{ts}.png"
                    cv2.imwrite(str(path), frame)
                    print(f"  [Capture] {path}")
                except Exception as e:
                    print(f"  [Capture] ERROR: {e}")

            # ── Drone control (position-based, avoids msgpack thread issues) ──
            step = 2.0   # meters per keypress
            yaw_step = 10.0  # degrees per keypress
            yaw_deg = 0.0

            try:
                # Track last known yaw
                pass
            except Exception:
                pass

            dx, dy, dz = 0.0, 0.0, 0.0
            if key.drone_fwd:    dx += step
            if key.drone_back:   dx -= step
            if key.drone_left:   dy -= step
            if key.drone_right:  dy += step
            if key.drone_up:     dz -= step
            if key.drone_down:   dz += step
            if key.drone_yaw_left:   yaw_deg -= yaw_step
            if key.drone_yaw_right:  yaw_deg += yaw_step

            if dx or dy or dz or yaw_deg:
                try:
                    state = air.getMultirotorState()
                    pos = state.kinematics_estimated.position
                    _, _, cur_yaw = airsim.to_eularian_angles(state.orientation)
                    new_yaw = math.degrees(cur_yaw) + yaw_deg
                    new_pose = airsim.Pose(
                        airsim.Vector3r(pos.x_val + dx, pos.y_val + dy, pos.z_val + dz),
                        airsim.to_quaternion(0.0, 0.0, math.radians(new_yaw)),
                    )
                    air.simSetVehiclePose(new_pose, ignore_collision=True)
                except Exception:
                    pass

            # ── Truck control ──
            if blue_truck:
                ctrl = carla.VehicleControl()
                if key.truck_throttle:
                    ctrl.throttle = 0.6
                if key.truck_brake:
                    ctrl.brake = 1.0
                steer = 0.0
                if key.truck_steer_left:
                    steer = -0.5
                if key.truck_steer_right:
                    steer = 0.5
                ctrl.steer = steer
                blue_truck.apply_control(ctrl)

            key.release_sticky()

            # ── Display: Truck chase cam ──
            drone_pos = np.array([0.0, 0.0, 0.0])
            try:
                s = air.getMultirotorState().kinematics_estimated.position
                drone_pos = np.array([s.x_val, s.y_val, s.z_val])
            except Exception:
                pass

            if latest_truck_img[0] is not None:
                frame = latest_truck_img[0].copy()
                frame = draw_hud_info(frame, ep, key, offset, preview.idx, preview.total, drone_pos)
                cv2.imshow("Truck Chase (arrow=drive  ,.=steer)", frame)

            # ── Display: Overhead view (separate window) ──
            if latest_overhead_img[0] is not None:
                oh = latest_overhead_img[0].copy()
                oh = draw_overhead_markers(oh, ep)
                cv2.imshow(f"Overhead [{ep['id']}]", oh)
            else:
                # Keep the window name alive
                cv2.imshow(f"Overhead [{ep['id']}]", np.zeros((300, 300, 3), dtype=np.uint8))

            key_code = cv2.waitKey(10) & 0xFF
            if key_code == 27:
                running = False

            time.sleep(0.02)

    finally:
        key.stop()
        if truck_camera:
            try: truck_camera.stop(); truck_camera.destroy()
            except: pass
        if overhead_camera:
            try: overhead_camera.stop(); overhead_camera.destroy()
            except: pass
        cv2.destroyAllWindows()
        print("  Done.\n")


if __name__ == "__main__":
    main()
