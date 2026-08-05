#!/usr/bin/env python3
"""无人机双视角实时查看 — 前视 + 下视左右拼接.

Usage:
    conda activate carlaAir
    python show_drone_cam.py                 # 前视┃下视拼接 (默认)
    python show_drone_cam.py --view front    # 只看前视
    python show_drone_cam.py --view down     # 只看下视

键盘: Q/ESC=退出  V=切换视角  M=开关小地图  S=截图
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime

import airsim
import carla
import cv2
import numpy as np
import pygame
import time

MINIMAP_SIZE = 240
MINIMAP_RANGE = 40.0
MARGIN = 10


# ═══════════════════════════════════════════════════════════════════
#  AirSim capture
# ═══════════════════════════════════════════════════════════════════

def _capture(air: airsim.MultirotorClient, camera: str) -> np.ndarray | None:
    responses = air.simGetImages([airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)])
    if not responses or responses[0].width == 0:
        return None
    return np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(
        responses[0].height, responses[0].width, 3
    )


# ═══════════════════════════════════════════════════════════════════
#  Minimap
# ═══════════════════════════════════════════════════════════════════

def _mm_xy(ned: np.ndarray, center: np.ndarray, s: float, cx: float, cy: float) -> tuple[int, int]:
    return (
        int(np.clip((ned[1] - center[1]) * s + cx, -MINIMAP_SIZE, MINIMAP_SIZE * 2)),
        int(np.clip((center[0] - ned[0]) * s + cy, -MINIMAP_SIZE, MINIMAP_SIZE * 2)),
    )


def _draw_triangle(surf: pygame.Surface, x: float, y: float, yaw: float, sz: float, color: tuple):
    pts = [(x + math.cos(yaw + math.pi * 0.8) * sz, y - math.sin(yaw + math.pi * 0.8) * sz),
           (x + math.cos(yaw) * sz * 1.6, y - math.sin(yaw) * sz * 1.6),
           (x + math.cos(yaw - math.pi * 0.8) * sz, y - math.sin(yaw - math.pi * 0.8) * sz)]
    pygame.draw.polygon(surf, color, pts)


def _minimap(drone_ned: np.ndarray, drone_yaw: float,
             truck_ned: np.ndarray | None, truck_yaw: float,
             cargo_ned: np.ndarray | None, font_sm) -> pygame.Surface:
    mm = pygame.Surface((MINIMAP_SIZE, MINIMAP_SIZE), pygame.SRCALPHA)
    mm.fill((10, 10, 16, 210))
    cx = cy = MINIMAP_SIZE / 2.0
    s = MINIMAP_SIZE / (2.0 * MINIMAP_RANGE)

    for m in range(-int(MINIMAP_RANGE), int(MINIMAP_RANGE) + 10, 10):
        px, _ = _mm_xy(np.array([0., float(m)]), drone_ned, s, cx, cy)
        pygame.draw.line(mm, (44, 44, 50, 130), (px, 0), (px, MINIMAP_SIZE), 1)
        _, py = _mm_xy(np.array([float(m), 0.]), drone_ned, s, cx, cy)
        pygame.draw.line(mm, (44, 44, 50, 130), (0, py), (MINIMAP_SIZE, py), 1)

    if truck_ned is not None:
        tx, ty = _mm_xy(truck_ned, drone_ned, s, cx, cy)
        ts = max(3, 6.0 * s)
        rect = pygame.Surface((ts, ts // 2), pygame.SRCALPHA)
        rect.fill((80, 160, 255, 200))
        rotated = pygame.transform.rotate(rect, -math.degrees(truck_yaw))
        mm.blit(rotated, (tx - rotated.get_width() // 2, ty - rotated.get_height() // 2))
        if cargo_ned is not None:
            cpx, cpy = _mm_xy(cargo_ned, drone_ned, s, cx, cy)
            pygame.draw.circle(mm, (255, 200, 0, 200), (cpx, cpy), 3)

    dx, dy = _mm_xy(drone_ned, drone_ned, s, cx, cy)
    _draw_triangle(mm, dx, dy, drone_yaw, 8.0, (0, 255, 100, 240))

    fov_len = 7.0 * s
    fov_half = math.radians(45)
    fov_pts = [(dx, dy)]
    for a in [drone_yaw - fov_half, drone_yaw - fov_half * 0.5, drone_yaw,
              drone_yaw + fov_half * 0.5, drone_yaw + fov_half]:
        fov_pts.append((dx + math.cos(a) * fov_len, dy - math.sin(a) * fov_len))
    if len(fov_pts) >= 3:
        pygame.draw.polygon(mm, (0, 255, 100, 50), fov_pts)

    # scale bar
    bar_m, bar_px = 10, int(10 * s)
    bar_x, bar_y = MINIMAP_SIZE - bar_px - 12, MINIMAP_SIZE - 15
    pygame.draw.line(mm, (180, 180, 180), (bar_x, bar_y), (bar_x + bar_px, bar_y), 3)
    mm.blit(font_sm.render(f"{bar_m}m", True, (200, 200, 200)), (bar_x + bar_px // 2 - 8, bar_y - 14))
    mm.blit(font_sm.render("N", True, (255, 255, 255)), (6, 4))

    if truck_ned is not None:
        d = float(np.linalg.norm(drone_ned[:2] - truck_ned[:2]))
        h = drone_ned[2] - truck_ned[2]
        mm.blit(font_sm.render(f"D→T:{d:.1f}m  H:{h:.1f}m", True, (170, 170, 170)),
                (6, MINIMAP_SIZE - 30))

    pygame.draw.rect(mm, (100, 100, 100), mm.get_rect(), 1)
    return mm


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Drone dual-camera viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--view", choices=("stitched", "front", "down"), default="stitched")
    args = parser.parse_args()

    air = airsim.MultirotorClient(ip=args.host, port=args.port, timeout_value=20)
    air.confirmConnection()
    print(f"AirSim: {args.host}:{args.port}")

    carla_world = None
    try:
        cc = carla.Client(args.host, args.carla_port)
        cc.set_timeout(5.0)
        carla_world = cc.get_world()
        print(f"CARLA : {args.host}:{args.carla_port}")
    except Exception:
        print("CARLA unavailable")

    pygame.init()
    W, H = args.width, args.height
    display = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 13, bold=True)
    font_sm = pygame.font.SysFont("monospace", 11, bold=False)

    running = True
    show_mm = False
    view = args.view

    fps_hist: list[float] = []
    last_t = time.monotonic()

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if ev.key == pygame.K_v:
                    view = {"stitched": "front", "front": "down", "down": "stitched"}[view]
                if ev.key == pygame.K_m:
                    show_mm = not show_mm
                if ev.key == pygame.K_s:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs("screenshots", exist_ok=True)
                    pygame.image.save(display, f"screenshots/drone_{ts}.png")
                    print(f"  [screenshot] screenshots/drone_{ts}.png")

        # ── Capture both cameras ──
        front = _capture(air, "0")
        down = None
        if view in ("stitched", "down"):
            try:
                down = _capture(air, "1")
            except Exception:
                pass

        # ── FPS ──
        now = time.monotonic()
        dt = now - last_t
        last_t = now
        fps_hist.append(1.0 / dt if dt > 0 else 0)
        if len(fps_hist) > 30:
            fps_hist.pop(0)
        fps = sum(fps_hist) / len(fps_hist)

        # ── Drone state ──
        drone_ned = np.zeros(3)
        drone_yaw = 0.0
        try:
            s = air.getMultirotorState()
            p = s.kinematics_estimated.position
            drone_ned = np.array([p.x_val, p.y_val, p.z_val])
            _, _, drone_yaw = airsim.to_eularian_angles(s.kinematics_estimated.orientation)
        except Exception:
            pass

        # ── Truck state ──
        truck_ned, truck_yaw, cargo_ned = None, 0.0, None
        if carla_world is not None:
            try:
                for actor in carla_world.get_actors().filter("vehicle.*"):
                    loc = actor.get_location()
                    rot = actor.get_transform().rotation
                    truck_ned = np.array([loc.x, loc.y, -loc.z])
                    truck_yaw = math.radians(rot.yaw)
                    fwd = np.array([math.cos(truck_yaw), math.sin(truck_yaw)])
                    cargo_ned = truck_ned + np.array([fwd[0] * (-2.3), fwd[1] * (-2.3), -1.15])
                    break
            except Exception:
                pass

        # ── Layout ──
        display.fill((16, 16, 16))

        pane_w = W // 2 if view == "stitched" else W
        pane_h = H - 26  # leave room for status bar

        def blit_img(img: np.ndarray | None, x: int, y: int, w: int, h: int, label: str, label_color: tuple):
            """Resize and blit an image into a region, with a label at top-left."""
            if img is not None:
                resized = cv2.resize(img, (w, h))
                display.blit(pygame.surfarray.make_surface(resized.swapaxes(0, 1)), (x, y))
            else:
                pygame.draw.rect(display, (30, 30, 30), (x, y, w, h))
            lb = font_sm.render(label, True, label_color)
            display.blit(lb, (x + 6, y + 4))

        if view == "stitched":
            blit_img(front, 0, 0, pane_w, pane_h, "前视", (100, 255, 100))
            sep_x = W // 2
            pygame.draw.line(display, (80, 80, 80), (sep_x, 0), (sep_x, H - 26), 2)
            blit_img(down, sep_x + 1, 0, pane_w, pane_h, "下视", (100, 200, 255))
        elif view == "front":
            blit_img(front, 0, 0, pane_w, pane_h, "前视", (100, 255, 100))
        else:  # down
            blit_img(down, 0, 0, pane_w, pane_h, "下视", (100, 200, 255))

        # ── Minimap ──
        if show_mm:
            mm = _minimap(drone_ned, drone_yaw, truck_ned, truck_yaw, cargo_ned, font_sm)
            display.blit(mm, (W - MINIMAP_SIZE - MARGIN, MARGIN))

        # ── Status bar ──
        bar = pygame.Surface((W, 26))
        bar.set_alpha(200)
        bar.fill((18, 18, 22))
        display.blit(bar, (0, H - 26))
        ti = f"Truck:[{truck_ned[0]:.1f},{truck_ned[1]:.1f}]" if truck_ned is not None else "Truck:--"
        status_text = (
            f"FPS:{fps:5.1f}  NED:[{drone_ned[0]:.1f},{drone_ned[1]:.1f},{drone_ned[2]:.1f}]"
            f"  Yaw:{math.degrees(drone_yaw):.0f}°  {ti}"
            f"  |  V:视角({view})  M:小地图  S:截图  Q:退出"
        )
        display.blit(font.render(status_text, True, (210, 210, 210)), (6, H - 22))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()
