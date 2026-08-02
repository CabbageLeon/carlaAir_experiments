#!/usr/bin/env python3
"""
control_car.py — Keyboard-controlled European HGV truck (same as SPF runner)
============================================================================
Spawns the same truck model used by the SPF evaluation runner and drives it
with the keyboard.

Reads arrow keys directly from the terminal (not pygame window), so keyboard
focus on the terminal is sufficient — no need to click the camera window.

Controls:
    ↑ / ↓       Throttle / Brake
    ← / →       Steer left / right
    Space        Handbrake
    R            Toggle reverse
    N            Next weather
    ESC / Q      Quit

Usage:
    conda activate carlaAir
    python3 carlaAir_experiments/control_car.py
"""

import carla
import airsim
import cv2
import pygame
import numpy as np
import math
import sys
import os
import termios
import tty
import select
import threading
from datetime import datetime

W, H = 1280, 720

TRUCK_BLUEPRINT = "vehicle.carlamotors.european_hgv"  # same as SPF TruckProfile

WEATHERS = [
    ("Clear Day", carla.WeatherParameters.ClearNoon),
    ("Sunset", carla.WeatherParameters(
        cloudiness=30, precipitation=0, precipitation_deposits=0,
        wind_intensity=30, sun_azimuth_angle=180, sun_altitude_angle=5,
        fog_density=10, fog_distance=50, fog_falloff=2, wetness=0)),
    ("Rain", carla.WeatherParameters.SoftRainNoon),
    ("Night", carla.WeatherParameters(
        cloudiness=10, precipitation=0, precipitation_deposits=0,
        wind_intensity=5, sun_azimuth_angle=0, sun_altitude_angle=-90,
        fog_density=2, fog_distance=0, fog_falloff=0, wetness=0)),
]


class TerminalKeyReader:
    """Read arrow keys and other keys from stdin in raw mode, non-blocking."""

    # ANSI escape sequences for arrow keys
    ARROW_UP = b'\x1b[A'
    ARROW_DOWN = b'\x1b[B'
    ARROW_RIGHT = b'\x1b[C'
    ARROW_LEFT = b'\x1b[D'
    # Alternate arrow key codes (some terminals)
    ARROW_UP_ALT = b'\x1bOA'
    ARROW_DOWN_ALT = b'\x1bOB'
    ARROW_RIGHT_ALT = b'\x1bOC'
    ARROW_LEFT_ALT = b'\x1bOD'

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        self._running = False
        self._thread = None

        # Shared state, protected by GIL (single-byte writes are atomic in CPython)
        self.throttle = False
        self.brake = False
        self.steer_left = False
        self.steer_right = False
        self.space = False
        self.toggle_reverse = False
        self.toggle_weather = False
        self.capture_drone = False
        self.quit = False

    def start(self):
        """Set terminal to raw mode and start the reader thread."""
        tty.setraw(self.fd)
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Restore terminal settings and stop the reader thread."""
        self._running = False
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def _read_loop(self):
        """Continuously read keystrokes from stdin."""
        buf = b''
        while self._running:
            try:
                ready, _, _ = select.select([self.fd], [], [], 0.02)
                if not ready:
                    continue
                ch = os.read(self.fd, 1)
                if not ch:
                    continue

                # ESC starts an escape sequence (or is a lone ESC)
                if ch == b'\x1b':
                    # Check if more bytes are available (arrow key sequence)
                    ready2, _, _ = select.select([self.fd], [], [], 0.01)
                    if ready2:
                        buf = b'\x1b' + os.read(self.fd, 2)
                    else:
                        buf = b'\x1b'  # Lone ESC
                else:
                    buf = ch

                self._handle(buf)
                buf = b''

            except (OSError, IOError):
                break

    def _handle(self, seq: bytes):
        if seq == self.ARROW_UP or seq == self.ARROW_UP_ALT:
            self.throttle = True
        elif seq == self.ARROW_DOWN or seq == self.ARROW_DOWN_ALT:
            self.brake = True
        elif seq == self.ARROW_LEFT or seq == self.ARROW_LEFT_ALT:
            self.steer_left = True
        elif seq == self.ARROW_RIGHT or seq == self.ARROW_RIGHT_ALT:
            self.steer_right = True
        elif seq == b' ':
            self.space = True
        elif seq in (b'r', b'R'):
            self.toggle_reverse = True
        elif seq in (b'n', b'N'):
            self.toggle_weather = True
        elif seq in (b'p', b'P'):
            self.capture_drone = True
        elif seq in (b'q', b'Q', b'\x1b'):  # q or ESC
            self.quit = True
        # Key-up: terminal sends the same sequence for press; we need to use
        # a timeout-based approach instead — see _update_state below.

    def _start_key_timeout(self):
        """Called after processing a key; keys auto-release after a short timeout."""
        pass

    def update(self):
        """
        Must be called from the main thread every frame.
        Keys auto-release after ~100ms of not being pressed, giving smooth control.
        This is a simple sticky-keys model: each press toggles the key on,
        and it stays on for a short window. For throttle/brake we want them
        to release quickly when the finger lifts.
        """
        pass  # Keys are set by the reader thread; no extra processing needed


def main():
    actors = []
    key_reader = None
    try:
        print("\n  Connecting to CarlaAir...")
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()

        # ── Connect to AirSim for drone camera capture ──
        air_client = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=10)
        air_client.confirmConnection()
        print("  AirSim drone connected.")

        # ── Clear all existing vehicles ──
        print("  Removing existing vehicles...")
        for v in world.get_actors().filter("vehicle.*"):
            try:
                v.destroy()
            except RuntimeError:
                pass
        print("  All existing vehicles destroyed.")

        # ── Find the CarlaAir built-in drone for coordinate reference ──
        drone_actor = None
        for actor in world.get_actors():
            if "drone" in actor.type_id.lower():
                drone_actor = actor
                break
        if drone_actor is None:
            raise RuntimeError(
                "CarlaAir drone actor not found. Is CarlaAir running?"
            )
        drone_location = drone_actor.get_location()
        print(f"  Drone at: ({drone_location.x:.1f}, {drone_location.y:.1f}, {drone_location.z:.1f})")

        # ── Spawn the European HGV truck near the drone ──
        truck_bp = bp_lib.find(TRUCK_BLUEPRINT)
        if truck_bp is None:
            raise RuntimeError(f"Blueprint '{TRUCK_BLUEPRINT}' not found.")
        all_spawn_points = world.get_map().get_spawn_points()
        if not all_spawn_points:
            raise RuntimeError("No spawn points found on current map.")

        spawn_points = sorted(
            all_spawn_points,
            key=lambda pt: pt.location.distance(drone_location),
        )

        truck = None
        for candidate in spawn_points:
            truck = world.try_spawn_actor(truck_bp, candidate)
            if truck is not None:
                break
        if truck is None:
            raise RuntimeError("Cannot spawn European HGV.")
        actors.append(truck)
        truck_loc = truck.get_location()
        print(f"  Spawned: European HGV at ({truck_loc.x:.1f}, {truck_loc.y:.1f}, {truck_loc.z:.1f})")

        # ── Chase camera ──
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(W))
        cam_bp.set_attribute("image_size_y", str(H))
        cam_bp.set_attribute("fov", "100")
        cam_tf = carla.Transform(
            carla.Location(x=-10.0, z=5.0),
            carla.Rotation(pitch=-15),
        )
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=truck)
        actors.append(camera)

        latest_image = [None]

        def on_image(img):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8)
            latest_image[0] = arr.reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]

        camera.listen(on_image)

        # ── Create output directory for drone captures ──
        pic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pic")
        os.makedirs(pic_dir, exist_ok=True)

        # ── Start terminal key reader ──
        key_reader = TerminalKeyReader()
        key_reader.start()
        print("\n  *** Keyboard control ACTIVE (terminal-based, no window focus needed) ***")
        print("  ↑↓=Throttle/Brake  ←→=Steer  Space=Handbrake  R=Reverse  N=Weather  P=Capture  Q/ESC=Quit\n")

        # ── Pygame window (display only) ──
        pygame.init()
        display = pygame.display.set_mode((W, H))
        pygame.display.set_caption("CarlaAir — Truck Camera (control from terminal)")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("monospace", 18, bold=True)

        weather_idx = 0
        world.set_weather(WEATHERS[0][1])
        reverse = False
        running = True

        while running:
            clock.tick(60)

            # Process pygame events (window close, resize etc)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False

            # ── Read keyboard state from terminal reader ──
            if key_reader.quit:
                running = False
            if key_reader.toggle_reverse:
                reverse = not reverse
                key_reader.toggle_reverse = False
            if key_reader.toggle_weather:
                weather_idx = (weather_idx + 1) % len(WEATHERS)
                world.set_weather(WEATHERS[weather_idx][1])
                key_reader.toggle_weather = False
            if key_reader.capture_drone:
                key_reader.capture_drone = False
                try:
                    responses = air_client.simGetImages([
                        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
                    ])
                    if responses and responses[0].width > 0:
                        img_bgr = np.frombuffer(
                            responses[0].image_data_uint8, dtype=np.uint8
                        ).reshape(responses[0].height, responses[0].width, 3)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        path = os.path.join(pic_dir, f"drone_{ts}.png")
                        cv2.imwrite(path, img_bgr)
                        print(f"  [Capture] saved: {path}")
                    else:
                        print("  [Capture] WARNING: empty frame from drone camera")
                except Exception as e:
                    print(f"  [Capture] ERROR: {e}")

            throttle_on = key_reader.throttle
            brake_on = key_reader.brake
            steer_left = key_reader.steer_left
            steer_right = key_reader.steer_right
            handbrake_on = key_reader.space

            # ── Build and apply vehicle control ──
            ctrl = carla.VehicleControl()
            ctrl.throttle = 0.8 if throttle_on else 0.0
            ctrl.brake = 1.0 if brake_on else 0.0

            steer = 0.0
            if steer_left:
                steer = -0.5
            elif steer_right:
                steer = 0.5
            ctrl.steer = steer

            ctrl.hand_brake = handbrake_on
            ctrl.reverse = reverse
            truck.apply_control(ctrl)

            # Clear key states for next frame (sticky until next terminal event)
            key_reader.throttle = False
            key_reader.brake = False
            key_reader.steer_left = False
            key_reader.steer_right = False
            key_reader.space = False

            # ── Render camera feed ──
            if latest_image[0] is not None:
                surf = pygame.surfarray.make_surface(latest_image[0].swapaxes(0, 1))
                display.blit(surf, (0, 0))
            else:
                display.fill((30, 30, 30))

            # ── HUD ──
            vel = truck.get_velocity()
            spd = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            rev_str = " [R]" if reverse else ""

            key_parts = []
            if throttle_on: key_parts.append("↑THR")
            if brake_on: key_parts.append("↓BRK")
            if steer_left: key_parts.append("←L")
            if steer_right: key_parts.append("→R")
            key_state = "|".join(key_parts) if key_parts else "idle"

            hud = (f"{WEATHERS[weather_idx][0]}  |  {spd:.0f} km/h{rev_str}  "
                   f"|  keys: [{key_state}]  "
                   f"|  ↑↓←→=Drive  Space=Brake  R=Reverse  N=Weather  P=Capture  Q=Quit")
            hs = font.render(hud, True, (0, 230, 180))
            hbg = pygame.Surface((W, 26))
            hbg.set_alpha(180)
            hbg.fill((0, 0, 0))
            display.blit(hbg, (0, H - 26))
            display.blit(hs, (8, H - 24))

            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        if key_reader:
            key_reader.stop()
        for a in actors:
            try:
                if hasattr(a, 'stop'):
                    a.stop()
            except Exception:
                pass
            try:
                a.destroy()
            except Exception:
                pass
        try:
            pygame.quit()
        except Exception:
            pass
        print("  Done.\n")


if __name__ == "__main__":
    main()
