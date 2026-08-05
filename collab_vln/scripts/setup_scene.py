#!/usr/bin/env python3
"""Terminal 3 — Spawn vehicles, position drone, keep scene alive.

Usage:
  python collab_vln/scripts/setup_scene.py [--episode-file episodes/xxx.json] [--episode-id town10hd_001]
"""

import argparse, json, math, sys, time, threading
from pathlib import Path

import airsim, carla, cv2
import numpy as np

START_TRUCK = "vehicle.mini.cooper_s"
GOAL_TRUCK = "vehicle.carlamotors.european_hgv"
DRONE_ALTITUDE = 60.0


def connect():
    cl = carla.Client("127.0.0.1", 2000)
    cl.set_timeout(10.0)
    world = cl.get_world()

    drone = next((a for a in world.get_actors() if "drone" in a.type_id.lower()), None)
    if drone is None:
        raise RuntimeError("CarlaAir drone not found")

    air = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=15)
    air.confirmConnection()
    air.enableApiControl(True)
    air.armDisarm(True)
    # Only takeoff if landed
    try:
        if air.getMultirotorState().landed_state == airsim.LandedState.Landed:
            air.takeoffAsync().join()
    except Exception:
        pass
    # Hold still
    try:
        air.moveByVelocityAsync(0, 0, 0, 0.5).join()
    except Exception:
        pass

    ap = air.getMultirotorState().kinematics_estimated.position
    dl = drone.get_location()
    offset = np.array([ap.x_val - dl.x, ap.y_val - dl.y, ap.z_val + dl.z])
    return cl, world, air, offset


def carla_to_ned(loc, offset):
    ned = offset + np.array([loc.x, loc.y, -loc.z])
    return float(ned[0]), float(ned[1]), float(ned[2])


def clear_vehicles(world):
    for v in world.get_actors().filter("vehicle.*"):
        try: v.destroy()
        except RuntimeError: pass
    time.sleep(0.3)


def spawn_vehicle(world, bp_name, x, y, z, yaw=0.0):
    bp = world.get_blueprint_library().find(bp_name)
    tf = carla.Transform(carla.Location(x=float(x), y=float(y), z=float(z)+0.3),
                         carla.Rotation(yaw=float(yaw)))
    return world.try_spawn_actor(bp, tf)


def position_drone(air, offset, x, y, z, yaw=0.0):
    nx, ny, nz = carla_to_ned(carla.Location(x=x, y=y, z=z), offset)
    orientation = airsim.to_quaternion(0.0, 0.0, math.radians(yaw))
    air.simSetVehiclePose(
        airsim.Pose(airsim.Vector3r(nx, ny, nz), orientation),
        ignore_collision=True,
    )
    time.sleep(0.3)
    # Hold still — zero velocity prevents drift better than hoverAsync
    try:
        air.moveByVelocityAsync(0, 0, 0, 0.5).join()
    except Exception:
        pass
    time.sleep(0.1)


def pick_episode(episodes_dir: Path):
    files = sorted(episodes_dir.glob("*_templates.json")) + sorted(episodes_dir.glob("*_episodes.json"))
    files = list(dict.fromkeys(files))
    if not files:
        print("No episode files found in", episodes_dir)
        sys.exit(1)
    print("\nAvailable episode files:")
    for i, f in enumerate(files):
        print(f"  {i+1:>2}. {f.name}")
    choice = input(f"\n选择 [1-{len(files)}, 回车=1]: ").strip()
    idx = int(choice)-1 if choice else 0
    f = files[max(0, min(len(files)-1, idx))]

    data = json.loads(f.read_text(encoding="utf-8"))
    episodes = data["episodes"]
    print(f"\n{len(episodes)} episodes:")
    for i, ep in enumerate(episodes):
        g = ep["goal"]; us = ep["ugv_spawn"]
        print(f"  {i+1:>2}. {ep['id']}  #{us['index']} -> ({g['x']:.0f},{g['y']:.0f})  {ep['distance_m']:.0f}m")
    choice = input(f"\n选择 [1-{len(episodes)}, 回车=1]: ").strip()
    idx = int(choice)-1 if choice else 0
    return episodes[max(0, min(len(episodes)-1, idx))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode-file", default=None, help="Path to episode JSON")
    p.add_argument("--episode-id", default=None, help="Episode ID to load")
    args = p.parse_args()

    episodes_dir = Path(__file__).resolve().parent.parent / "episodes"
    if args.episode_file and args.episode_id:
        data = json.loads(Path(args.episode_file).read_text(encoding="utf-8"))
        ep = next(e for e in data["episodes"] if e["id"] == args.episode_id)
    else:
        ep = pick_episode(episodes_dir)

    cl, world, air, offset = connect()
    clear_vehicles(world)
    time.sleep(0.3)

    us = ep["ugv_spawn"]
    g = ep["goal"]

    # Spawn vehicles
    start_car = spawn_vehicle(world, START_TRUCK, us["x"], us["y"], us["z"], us["yaw"])
    if start_car is None:
        print("ERROR: failed to spawn start car")
        sys.exit(1)
    goal_car = spawn_vehicle(world, GOAL_TRUCK, g["x"], g["y"], g["z"], 0.0)
    if goal_car is None:
        print("ERROR: failed to spawn goal car")
        sys.exit(1)

    # Drone 60m above start
    position_drone(air, offset, us["x"], us["y"], us["z"] + DRONE_ALTITUDE)
    time.sleep(0.3)

    drone_state = air.getMultirotorState().kinematics_estimated.position
    print(f"\n{'='*55}")
    print(f"  Scene Ready — {ep['id']}  |  {ep['distance_m']:.0f}m")
    print(f"  Start: #{us['index']} Mini Cooper  ({us['x']:.0f}, {us['y']:.0f})")
    print(f"  Goal:  HGV             ({g['x']:.0f}, {g['y']:.0f})")
    print(f"  Drone: {DRONE_ALTITUDE:.0f}m above start")
    print(f"{'='*55}")
    print(f"\n  Terminal 4: python collab_vln/scripts/control.py")
    print(f"  Overhead window open. Press Ctrl+C to stop.\n")

    # ── Overhead camera (static, high above) ──
    mid_x = (us["x"] + g["x"]) / 2
    mid_y = (us["y"] + g["y"]) / 2
    alt = 150.0

    oh_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    oh_bp.set_attribute("image_size_x", "640")
    oh_bp.set_attribute("image_size_y", "640")
    oh_bp.set_attribute("fov", "110")
    oh_tf = carla.Transform(
        carla.Location(x=mid_x, y=mid_y, z=alt),
        carla.Rotation(pitch=-90),
    )
    overhead_cam = world.spawn_actor(oh_bp, oh_tf)
    latest_oh = [None]

    def oh_cb(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        latest_oh[0] = arr.reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]

    overhead_cam.listen(oh_cb)

    # ── Chase camera (attached to goal HGV, behind + above) ──
    chase_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    chase_bp.set_attribute("image_size_x", "960")
    chase_bp.set_attribute("image_size_y", "540")
    chase_bp.set_attribute("fov", "100")
    chase_tf = carla.Transform(
        carla.Location(x=-12.0, z=6.0),   # 12m behind, 6m above
        carla.Rotation(pitch=-15),
    )
    chase_cam = world.spawn_actor(chase_bp, chase_tf, attach_to=goal_car)
    latest_chase = [None]

    def chase_cb(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        latest_chase[0] = arr.reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]

    chase_cam.listen(chase_cb)

    # Marker helpers
    fov, hw = 110.0, alt * math.tan(math.radians(110.0 / 2))
    scale = 320.0 / hw

    def proj(wx, wy):
        px = int(320 + (wx - mid_x) * scale)
        py = int(320 - (wy - mid_y) * scale)
        return max(0, min(639, px)), max(0, min(639, py))

    try:
        while True:
            # ── Overhead view ──
            if latest_oh[0] is not None:
                oh = latest_oh[0].copy()
                gx, gy = proj(g["x"], g["y"])
                sx, sy = proj(us["x"], us["y"])
                cv2.circle(oh, (gx, gy), 15, (0, 0, 255), 2)
                cv2.circle(oh, (gx, gy), 4, (0, 0, 255), -1)
                cv2.putText(oh, "GOAL", (gx+10, gy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.circle(oh, (sx, sy), 12, (255, 0, 0), 2)
                cv2.circle(oh, (sx, sy), 3, (255, 0, 0), -1)
                cv2.putText(oh, "START", (sx+10, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.line(oh, (sx, sy), (gx, gy), (0, 255, 255), 1)
                cv2.putText(oh, f"{ep['id']} | {ep['distance_m']:.0f}m", (5, 630),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.imshow(f"Overhead - {ep['id']}", oh)

            # ── Chase view (attached to goal HGV) ──
            if latest_chase[0] is not None:
                chase = latest_chase[0].copy()
                cv2.putText(chase, f"Chase [HGV GOAL] - {ep['id']}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow(f"Chase [HGV@GOAL] - {ep['id']}", chase)

            if cv2.waitKey(50) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if overhead_cam:
            try: overhead_cam.stop(); overhead_cam.destroy()
            except: pass
        if chase_cam:
            try: chase_cam.stop(); chase_cam.destroy()
            except: pass
        print("\n  Cleaning up...")
        clear_vehicles(world)
        print("  Done.")


if __name__ == "__main__":
    main()
