#!/usr/bin/env python3
"""Spawn colored trucks at start/goal, capture top-down views from above both positions.

Output per episode:
  - {id}_above_goal.png    — 100m above goal, red truck visible
  - {id}_above_start.png   — 50m above start, blue truck visible

Usage:
  conda activate carlaAir
  python collab_vln/scripts/birdview_goals.py --input collab_vln/episodes/town10hd_templates.json
"""

import argparse, json, math, time
from pathlib import Path

import airsim, carla, cv2
import numpy as np

BLUE_TRUCK_BP = "vehicle.carlamotors.european_hgv"
RED_TRUCK_BP = "vehicle.tesla.model3"


# ── connect & calibrate ──────────────────────────────────────

def connect():
    cl = carla.Client("127.0.0.1", 2000)
    cl.set_timeout(10.0)
    world = cl.get_world()

    drone_actor = next((a for a in world.get_actors() if "drone" in a.type_id.lower()), None)
    if drone_actor is None:
        raise RuntimeError("CarlaAir drone not found")

    air = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=15)
    air.confirmConnection()
    air.enableApiControl(True)
    air.armDisarm(True)
    try:
        air.takeoffAsync().join()
    except Exception:
        pass

    ap = air.getMultirotorState().kinematics_estimated.position
    dl = drone_actor.get_location()
    offset = np.array([ap.x_val - dl.x, ap.y_val - dl.y, ap.z_val + dl.z])
    return cl, world, air, offset


# ── coordinate helpers ──────────────────────────────────────

def carla_to_ned(loc, offset):
    ned = offset + np.array([loc.x, loc.y, -loc.z])
    return float(ned[0]), float(ned[1]), float(ned[2])


def world_to_pixel(wx, wy, cam_x, cam_y, img_w, img_h, altitude, fov_h=90.0):
    half_w = altitude * math.tan(math.radians(fov_h / 2))
    scale = (img_w / 2) / half_w
    px = img_w / 2 + (wx - cam_x) * scale
    py = img_h / 2 - (wy - cam_y) * scale
    return max(0, min(img_w - 1, int(px))), max(0, min(img_h - 1, int(py)))


# ── spawn ───────────────────────────────────────────────────

def spawn_truck(world, bp_name, x, y, z, yaw=0.0, color="0,0,0"):
    bp = world.get_blueprint_library().find(bp_name)
    bp.set_attribute("color", color)
    tf = carla.Transform(carla.Location(x=float(x), y=float(y), z=float(z) + 0.3),
                         carla.Rotation(yaw=float(yaw)))
    return world.try_spawn_actor(bp, tf)


def clear_vehicles(world):
    for v in world.get_actors().filter("vehicle.*"):
        try:
            v.destroy()
        except RuntimeError:
            pass
    time.sleep(0.3)


# ── capture ─────────────────────────────────────────────────

def capture_at(air, offset, x, y, z, altitude):
    nx, ny, nz = carla_to_ned(carla.Location(x=x, y=y, z=altitude), offset)
    orientation = airsim.to_quaternion(math.radians(-90.0), 0.0, 0.0)
    air.simSetVehiclePose(airsim.Pose(airsim.Vector3r(nx, ny, nz), orientation),
                          ignore_collision=True)
    time.sleep(0.5)
    responses = air.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    return np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(
        responses[0].height, responses[0].width, 3).copy()


# ── draw markers ────────────────────────────────────────────

def draw_markers(frame, ep, altitude, cam_above_goal):
    h, w = frame.shape[:2]
    g = ep["goal"]
    us = ep["ugv_spawn"]

    if cam_above_goal:
        cam_x, cam_y = g["x"], g["y"]
        goal_px, goal_py = w // 2, h // 2
        ugv_px, ugv_py = world_to_pixel(us["x"], us["y"], cam_x, cam_y, w, h, altitude)
        cam_label = f"Above GOAL  {altitude:.0f}m"
    else:
        cam_x, cam_y = us["x"], us["y"]
        ugv_px, ugv_py = w // 2, h // 2
        goal_px, goal_py = world_to_pixel(g["x"], g["y"], cam_x, cam_y, w, h, altitude)
        cam_label = f"Above START  {altitude:.0f}m"

    # Goal marker (red)
    cv2.circle(frame, (goal_px, goal_py), 60, (0, 0, 255), 3)
    cv2.circle(frame, (goal_px, goal_py), 8, (0, 0, 255), -1)
    cv2.line(frame, (goal_px - 40, goal_py), (goal_px + 40, goal_py), (0, 0, 255), 3)
    cv2.line(frame, (goal_px, goal_py - 40), (goal_px, goal_py + 40), (0, 0, 255), 3)
    cv2.putText(frame, "GOAL", (goal_px + 20, goal_py - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Start marker (blue)
    cv2.circle(frame, (ugv_px, ugv_py), 40, (255, 0, 0), 3)
    cv2.circle(frame, (ugv_px, ugv_py), 6, (255, 0, 0), -1)
    cv2.putText(frame, "START", (ugv_px + 15, ugv_py - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2, cv2.LINE_AA)

    # Line
    cv2.line(frame, (ugv_px, ugv_py), (goal_px, goal_py), (0, 255, 255), 2, cv2.LINE_AA)

    # Info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    info = [
        f"{ep['id']}  |  {cam_label}  |  {ep['distance_m']:.0f}m  |  {ep['map']}",
        f"Goal: ({g['x']:.0f}, {g['y']:.0f})   |   Start #{us['index']}: ({us['x']:.0f}, {us['y']:.0f})",
    ]
    for i, line in enumerate(info):
        cv2.putText(frame, line, (10, 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

    return frame


# ── main ────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--alt-goal", type=float, default=100.0, help="Altitude above goal")
    p.add_argument("--alt-start", type=float, default=50.0, help="Altitude above start")
    args = p.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    episodes = data["episodes"]
    out_dir = Path(__file__).resolve().parent.parent / "birdviews"
    out_dir.mkdir(exist_ok=True)

    print(f"{len(episodes)} episodes | goal@{args.alt_goal:.0f}m | start@{args.alt_start:.0f}m")
    cl, world, air, offset = connect()

    for ep in episodes:
        g = ep["goal"]
        us = ep["ugv_spawn"]
        eid = ep["id"]
        print(f"  {eid}: start #{us['index']} -> goal ({g['x']:.0f},{g['y']:.0f})", end="", flush=True)

        try:
            clear_vehicles(world)
            red = spawn_truck(world, RED_TRUCK_BP, g["x"], g["y"], g["z"], 0.0, "255,0,0")
            blue = spawn_truck(world, BLUE_TRUCK_BP, us["x"], us["y"], us["z"], us["yaw"], "0,0,255")
            time.sleep(0.3)

            if red and blue:
                # View from above goal
                f = capture_at(air, offset, g["x"], g["y"], g["z"], args.alt_goal)
                f = draw_markers(f, ep, args.alt_goal, cam_above_goal=True)
                cv2.imwrite(str(out_dir / f"{eid}_above_goal.png"), f)

                # View from above start
                f = capture_at(air, offset, us["x"], us["y"], us["z"], args.alt_start)
                f = draw_markers(f, ep, args.alt_start, cam_above_goal=False)
                cv2.imwrite(str(out_dir / f"{eid}_above_start.png"), f)

                print(" -> OK")
            else:
                print(" -> spawn FAILED")

            if red: red.destroy()
            if blue: blue.destroy()
            time.sleep(0.1)

        except Exception as e:
            print(f" -> ERROR: {e}")

    clear_vehicles(world)
    print(f"\nDone -> {out_dir}/")


if __name__ == "__main__":
    main()
