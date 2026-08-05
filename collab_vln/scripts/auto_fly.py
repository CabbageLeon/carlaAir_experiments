#!/usr/bin/env python3
"""Terminal 5 — Auto-fly drone to goal position.

Two-phase flight:
  1. Yaw-rotate to face the goal
  2. Fly forward to above the goal

Body-frame velocity commands, same format as control.py.

Usage:
  python collab_vln/scripts/auto_fly.py
"""

import math, time

import airsim, carla
import numpy as np


def connect():
    air = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=10)
    air.confirmConnection()
    cl = carla.Client("127.0.0.1", 2000)
    cl.set_timeout(5.0)
    world = cl.get_world()

    # Calibrate: CARLA <-> AirSim NED offset
    drone = next((a for a in world.get_actors() if "drone" in a.type_id.lower()), None)
    if drone is None:
        raise RuntimeError("CarlaAir drone not found")
    ap = air.getMultirotorState().kinematics_estimated.position
    dl = drone.get_location()
    offset = np.array([ap.x_val - dl.x, ap.y_val - dl.y, ap.z_val + dl.z])

    return air, world, offset


def carla_to_ned(loc, offset):
    ned = offset + np.array([loc.x, loc.y, -loc.z])
    return float(ned[0]), float(ned[1]), float(ned[2])


def get_drone_state(air):
    s = air.getMultirotorState().kinematics_estimated
    pos = np.array([s.position.x_val, s.position.y_val, s.position.z_val])
    _, _, yaw = airsim.to_eularian_angles(s.orientation)
    return pos, yaw


GOAL_BP = "vehicle.carlamotors.european_hgv"
START_BP = "vehicle.mini.cooper_s"

def find_goal_truck(world):
    """Find the HGV goal truck by blueprint type."""
    for v in world.get_actors().filter("vehicle.*"):
        if GOAL_BP in v.type_id:
            return v
    return None


def yaw_to_target(air, current_yaw, target_yaw, yaw_speed=60.0):
    """Rotate in place to face target_yaw. Shortest path."""
    diff = (target_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
    if abs(diff) < 0.02:
        print(f"  already facing target (yaw={math.degrees(current_yaw):.0f}°)")
        return

    yaw_rate = yaw_speed if diff > 0 else -yaw_speed
    dur = abs(diff) / math.radians(yaw_speed)
    print(f"  [1/2] rotating {abs(math.degrees(diff)):.0f}° {'right' if diff>0 else 'left'} "
          f"({dur:.1f}s @ {abs(yaw_rate):.0f}°/s)")
    air.moveByVelocityAsync(0, 0, 0, dur,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(True, yaw_rate),
    ).join()
    time.sleep(0.2)


def fly_forward_body(air, target_yaw, distance, height_diff=0.0, speed=8.0):
    """Fly forward in body frame. Uses target_yaw for body→global conversion.

    After Phase 1 yaw rotation, the drone faces target_yaw.
    Body forward (speed, 0, 0) → global (speed*cos(yaw), speed*sin(yaw), 0).
    """
    dur = distance / speed
    # body forward (speed, 0, 0) rotated to global NED
    cy, sy = math.cos(target_yaw), math.sin(target_yaw)
    gvx = speed * cy       # body fwd cos(yaw) + body right * 0
    gvy = speed * sy
    gvz = height_diff / dur if dur > 0 else 0.0

    print(f"  [2/2] body fwd {distance:.0f}m @ {speed:.0f}m/s ({dur:.1f}s)"
          + (f"  alt delta {height_diff:.0f}m" if abs(height_diff) > 0.5 else ""))
    print(f"         body->global vel ({gvx:.1f}, {gvy:.1f}, {gvz:.1f})")
    air.moveByVelocityAsync(gvx, gvy, gvz, dur,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(False, math.degrees(target_yaw)),
    ).join()
    time.sleep(0.2)


def auto_fly_to_goal(air, world, offset, target_altitude=None, yaw_only=False):
    """Two-phase flight to goal truck position."""
    drone_pos, drone_yaw = get_drone_state(air)

    goal_truck = find_goal_truck(world)
    if goal_truck is None:
        print("  ERROR: goal truck not found. Run setup_scene.py first.")
        return False

    goal_loc = goal_truck.get_location()
    # Convert CARLA -> NED
    gx, gy, gz = carla_to_ned(goal_loc, offset)
    goal_pos = np.array([gx, gy, gz])

    target_alt = target_altitude if target_altitude is not None else drone_pos[2]
    target = np.array([goal_pos[0], goal_pos[1], target_alt])

    # Horizontal vector
    dx = target[0] - drone_pos[0]
    dy = target[1] - drone_pos[1]
    horiz_dist = math.hypot(dx, dy)
    target_yaw = math.atan2(dy, dx)  # NED: 0=north, pi/2=east

    height_diff = target[2] - drone_pos[2]

    print(f"\n  Drone:   ({drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f}) yaw={math.degrees(drone_yaw):.0f}°")
    print(f"  Goal:    ({goal_pos[0]:.1f}, {goal_pos[1]:.1f}, {goal_pos[2]:.1f})")
    print(f"  Target:  ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")
    print(f"  Distance: {horiz_dist:.1f}m horizontal  |  height diff: {height_diff:.1f}m")
    print(f"  Target yaw: {math.degrees(target_yaw):.0f}°  |  Current yaw: {math.degrees(drone_yaw):.0f}°")

    if horiz_dist < 1.0:
        print("  Already at goal!")
        return True

    # ── Closed-loop: iterate until within tolerance ──
    TOLERANCE = 3.0  # meters
    MAX_ITER = 5
    speed = 8.0

    for iteration in range(MAX_ITER):
        drone_pos, drone_yaw = get_drone_state(air)
        dx = target[0] - drone_pos[0]
        dy = target[1] - drone_pos[1]
        dz = target[2] - drone_pos[2]
        horiz_dist = math.hypot(dx, dy)
        err = horiz_dist

        if err < TOLERANCE and abs(dz) < TOLERANCE:
            print(f"  [iter {iteration+1}] error={err:.1f}m < {TOLERANCE}m — arrived!")
            return True

        target_yaw = math.atan2(dy, dx)
        print(f"\n  [iter {iteration+1}] error={err:.1f}m  dist={horiz_dist:.1f}m  dz={dz:.1f}m")

        # Phase 1: yaw to face goal
        diff = (target_yaw - drone_yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) > 0.03:
            yaw_to_target(air, drone_yaw, target_yaw)

        if yaw_only and iteration == 0:
            print("  [yaw-only mode] stopped. Check orientation in terminal 2.")
            return True

        # Phase 2: fly with a cap so large errors get multiple corrections
        distance = min(horiz_dist, 80.0)  # max 80m per iteration
        fly_forward_body(air, target_yaw, distance, dz, speed)

        # Hover between iterations
        air.hoverAsync()
        time.sleep(0.2)

    # Final check
    drone_pos2, _ = get_drone_state(air)
    err = math.hypot(drone_pos2[0] - target[0], drone_pos2[1] - target[1])
    print(f"  Final error: {err:.1f}m (max iterations reached)")
    return err < TOLERANCE


def main():
    import argparse
    p = argparse.ArgumentParser(description="Auto-fly drone to goal")
    p.add_argument("--altitude", type=float, default=None,
                   help="Target altitude (default: keep current)")
    p.add_argument("--loop", action="store_true",
                   help="Continuously fly back and forth (test mode)")
    p.add_argument("--yaw-only", action="store_true",
                   help="Only rotate to face goal, don't fly forward")
    args = p.parse_args()

    air, world, offset = connect()
    print("Connected. Auto-fly ready.\n")

    if args.loop:
        # Test mode: fly to goal, back to start, repeat
        start_pos = None
        while True:
            try:
                if start_pos is None:
                    start_pos, _ = get_drone_state(air)

                print("\n=== Flying to goal ===")
                auto_fly_to_goal(air, world, offset, args.altitude, yaw_only=args.yaw_only)

                print("\n=== Flying back to start ===")
                drone_pos, drone_yaw = get_drone_state(air)
                dx = start_pos[0] - drone_pos[0]
                dy = start_pos[1] - drone_pos[1]
                horiz_dist = math.hypot(dx, dy)
                target_yaw = math.atan2(dy, dx)
                yaw_to_target(air, drone_yaw, target_yaw)
                height_diff = start_pos[2] - drone_pos[2]
                fly_forward_body(air, target_yaw, horiz_dist, height_diff, speed=8.0)
                air.hoverAsync()
                print("\n--- Cycle complete. Press Ctrl+C to stop. ---\n")
                time.sleep(2)
            except KeyboardInterrupt:
                break
    else:
        auto_fly_to_goal(air, world, offset, args.altitude, yaw_only=args.yaw_only)
        print("\nDone.")


if __name__ == "__main__":
    main()
