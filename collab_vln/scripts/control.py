#!/usr/bin/env python3
"""Terminal 4 — Velocity-based drone control + throttle/steer car control.

Commands:
  d  vx vy vz [yaw_rate] [dur]   Drone velocity (m/s, deg/s, default 1s)
  h                             Drone hover (zero velocity)
  c  throttle steer             Car drive (0-1, -1..1)
  C  spawn_idx                  Car teleport to spawn point
  s                             Stop car (brake)
  w                             Where — show positions
  q                             Quit

Usage:
  python collab_vln/scripts/control.py
"""

import cmd, math, time, threading

import airsim, carla
import numpy as np


class Control(cmd.Cmd):
    intro = "Commands: d vx vy vz [yaw_rate] [dur] | h | c thr steer | C idx | s | w | q"
    prompt = "> "

    def __init__(self):
        super().__init__()
        self.cl = carla.Client("127.0.0.1", 2000); self.cl.set_timeout(5.0)
        self.world = self.cl.get_world()
        self.air = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=10)
        self.air.confirmConnection()
        self._dpos = np.zeros(3); self._dyaw = 0.0
        self._hover_timer = None
        self._sync()

    def _sync(self):
        try:
            s = self.air.getMultirotorState().kinematics_estimated
            self._dpos = np.array([s.position.x_val, s.position.y_val, s.position.z_val])
            _, _, self._dyaw = airsim.to_eularian_angles(s.orientation)
        except: pass

    def _vehicles(self):
        return list(self.world.get_actors().filter("vehicle.*"))

    # ── Drone: velocity-based ──

    def do_d(self, arg):
        """d vx vy vz [yaw_rate_deg/s] [duration_sec]

        Body-frame velocities: vx=forward, vy=right, vz=down (m/s).
        Automatically rotated to global NED using current drone yaw.
        """
        p = arg.strip().split()
        if len(p) < 3: return print("Usage: d fwd right down [yaw_rate_deg/s] [duration_sec]")
        try:
            bx, by, bz = float(p[0]), float(p[1]), float(p[2])
            yr = float(p[3]) if len(p) > 3 else 0.0
            dur = float(p[4]) if len(p) > 4 else 1.0
        except ValueError: return print("Numbers please")

        self._sync()
        cy, sy = math.cos(self._dyaw), math.sin(self._dyaw)
        # body → global NED: fwd=(cy,sy), right=(-sy,cy)
        gx = bx * cy - by * sy
        gy = bx * sy + by * cy
        gz = bz

        print(f"  drone body({bx:.1f},{by:.1f},{bz:.1f}) -> global({gx:.1f},{gy:.1f},{gz:.1f})m/s  "
              f"yaw={math.degrees(self._dyaw):.0f}°  yaw_rate={yr:.0f}°/s  dur={dur:.1f}s")
        try:
            self.air.moveByVelocityAsync(
                gx, gy, gz, dur,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(True, yr),
            )
            if self._hover_timer:
                self._hover_timer.cancel()
            self._hover_timer = threading.Timer(dur + 0.1, self._hover_after_move)
            self._hover_timer.start()
        except Exception as e:
            print(f"  ERROR: {e}")

    def _hover_after_move(self):
        try:
            self.air.hoverAsync()
        except Exception:
            pass

    def do_h(self, arg):
        """h — hover in place"""
        try:
            self.air.hoverAsync()
            print("  hover")
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── Car ──

    def do_c(self, arg):
        """c throttle steer"""
        p = arg.strip().split()
        if len(p) < 2: return print("Usage: c throttle steer")
        try: thr, st = float(p[0]), float(p[1])
        except ValueError: return print("Numbers please")

        cars = self._vehicles()
        if not cars: return print("No car")
        car = cars[0]
        ctrl = carla.VehicleControl()
        ctrl.throttle = max(0, min(1, thr))
        ctrl.steer = max(-1, min(1, st))
        car.apply_control(ctrl)

        v = car.get_velocity(); spd = 3.6 * math.hypot(v.x, v.y)
        l = car.get_location()
        print(f"  car thr={thr:.1f} steer={st:.1f}  {spd:.0f}km/h  ({l.x:.0f},{l.y:.0f})")

    def do_C(self, arg):
        """C spawn_idx"""
        p = arg.strip().split()
        if not p: return print("Usage: C spawn_idx")
        try: idx = int(p[0])
        except ValueError: return print("Integer please")
        cars = self._vehicles()
        if not cars: return print("No car")
        pts = self.world.get_map().get_spawn_points()
        if 0 <= idx < len(pts):
            sp = pts[idx]
            cars[0].set_transform(carla.Transform(sp.location, sp.rotation))
            print(f"  car -> spawn #{idx} ({sp.location.x:.0f},{sp.location.y:.0f})")
        else:
            print(f"  bad index (0-{len(pts)-1})")

    # ── stop ──

    def do_s(self, arg):
        cars = self._vehicles()
        if not cars: return print("No car")
        ctrl = carla.VehicleControl(); ctrl.brake = 1.0
        cars[0].apply_control(ctrl)
        print("  brake")

    # ── where ──

    def do_w(self, arg):
        self._sync()
        print(f"  drone: ({self._dpos[0]:.1f},{self._dpos[1]:.1f},{self._dpos[2]:.1f}) yaw={math.degrees(self._dyaw):.0f}°")
        cars = self._vehicles()
        if cars:
            l = cars[0].get_location()
            v = cars[0].get_velocity(); spd = 3.6 * math.hypot(v.x, v.y)
            print(f"  car:   ({l.x:.1f},{l.y:.1f},{l.z:.1f}) yaw={cars[0].get_transform().rotation.yaw:.0f}° {spd:.0f}km/h")
        if len(cars) > 1:
            gl = cars[1].get_location()
            d = math.hypot(gl.x - l.x, gl.y - l.y)
            print(f"  goal:  ({gl.x:.1f},{gl.y:.1f}) dist={d:.0f}m")

    def do_q(self, arg):
        if self._hover_timer:
            self._hover_timer.cancel()
        return True
    def do_quit(self, arg):
        return self.do_q(arg)
    def emptyline(self): pass


if __name__ == "__main__":
    Control().cmdloop()
