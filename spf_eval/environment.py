"""CARLA-Air scene control for the two cooperative SPF evaluations."""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass

import airsim
import carla
import numpy as np


@dataclass(frozen=True)
class TruckProfile:
    blueprint_id: str = "vehicle.carlamotors.european_hgv"
    # Validated against the visible rear platform of the CARLA European HGV.
    bed_center_x: float = -2.30
    bed_center_y: float = 0.00
    bed_height: float = 1.15
    bed_half_length: float = 1.45
    bed_half_width: float = 1.10


@dataclass(frozen=True)
class GroundRobotProfile:
    blueprint_id: str = "vehicle.mini.cooper_s"
    max_speed_mps: float = 5.0
    follow_offset_m: float = 10.0
    pursuit_radius_m: float = 4.0


@dataclass(frozen=True)
class VehicleState:
    ned: np.ndarray
    yaw_rad: float
    speed: float


class CarlaAirEnvironment:
    """Owns the evaluation truck and clears pre-existing traffic before each episode."""

    def __init__(self, carla_port: int = 2000, airsim_port: int = 41451, profile: TruckProfile | None = None, go_straight: bool = False, robot_profile: GroundRobotProfile | None = None):
        self.client = carla.Client("127.0.0.1", carla_port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.air = airsim.MultirotorClient(ip="127.0.0.1", port=airsim_port, timeout_value=20)
        self.air.confirmConnection()
        # AirSim's msgpack RPC session is not thread-safe. Image requests are issued
        # from a worker so the shared world can advance one render tick; keep them on
        # their own session and leave flight/state operations on ``self.air``.
        self.camera_air = airsim.MultirotorClient(
            ip="127.0.0.1", port=airsim_port, timeout_value=20
        )
        self.camera_air.confirmConnection()
        self.profile = profile or TruckProfile()
        self.truck: carla.Vehicle | None = None
        self.robot_dog: carla.Vehicle | None = None
        self.robot_profile = robot_profile or GroundRobotProfile()
        self._airsim_offset: np.ndarray | None = None
        self._previous_settings: carla.WorldSettings | None = None
        self._last_target_speed = 4.0
        self._fixed_delta_seconds = 0.05
        self._next_tick_wall: float | None = None
        self.go_straight = go_straight

    def reset(self, seed: int, spawn_index: int = 0) -> None:
        self.close_episode()
        random.seed(seed)
        self._previous_settings = self.world.get_settings()
        self._clear_vehicles()
        time.sleep(0.5)
        blueprint = self.world.get_blueprint_library().find(self.profile.blueprint_id)
        drone_location = self._drone_actor_location()
        all_points = self.world.get_map().get_spawn_points()
        total = len(all_points)

        # Try spawn_index first, then wrap around to nearby points
        spawn_index = spawn_index % total
        order = [(spawn_index + offset) % total for offset in range(total)]

        self.truck = None
        for idx in order:
            actor = self.world.try_spawn_actor(blueprint, all_points[idx])
            if actor is not None:
                self.truck = actor
                spawn_index = idx
                break
        if self.truck is None:
            raise RuntimeError("no free spawn point for the experiment truck")
        spawn_loc = self.truck.get_location()
        dist = spawn_loc.distance(drone_location)
        print(f"[env] 卡车: {self.profile.blueprint_id}  出生点 #{spawn_index}/{total}  距无人机={dist:.1f}m  位置=({spawn_loc.x:.1f}, {spawn_loc.y:.1f}, {spawn_loc.z:.1f})")
        self.air.enableApiControl(True)
        self.air.armDisarm(True)
        if self.air.getMultirotorState().landed_state == airsim.LandedState.Landed:
            self.air.takeoffAsync().join()
        self.air.hoverAsync().join()
        self._calibrate_airsim_offset()
        # Start behind and above the truck, looking along its route.  This is an
        # initial condition, not a policy-side geometric cue.
        truck_yaw = math.radians(self.truck.get_transform().rotation.yaw)
        start = self._to_ned(self.truck.get_location()) + np.array(
            [-12.0 * math.cos(truck_yaw), -12.0 * math.sin(truck_yaw), -6.0], dtype=float
        )
        # Episode initialization is fixed by the scenario, not selected by SPF.
        # Teleporting here makes every seed start from the same settled observation
        # pose; policy actions after this point use only the P controller below.
        pose = airsim.Pose(
            airsim.Vector3r(float(start[0]), float(start[1]), float(start[2])),
            airsim.to_quaternion(0.0, 0.0, truck_yaw),
        )
        self.air.simSetVehiclePose(pose, ignore_collision=True)
        time.sleep(0.1)
        self.air.hoverAsync().join()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._fixed_delta_seconds
        self.world.apply_settings(settings)
        self.world.tick()
        if self.go_straight:
            self.truck.set_autopilot(False)
            self.truck.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
        else:
            self.truck.apply_control(carla.VehicleControl(brake=1.0))
            self.truck.set_autopilot(True, 8000)
            self.traffic_manager.ignore_lights_percentage(self.truck, 100.0)
            self.set_truck_speed(4.0)
        self.world.tick()
        self._next_tick_wall = time.monotonic()

    def close_episode(self) -> None:
        if self.truck is not None:
            try:
                self.truck.set_autopilot(False)
                self.truck.destroy()
            except RuntimeError:
                pass
            self.truck = None
        if self._previous_settings is not None:
            self.world.apply_settings(self._previous_settings)
            self._previous_settings = None
        self._next_tick_wall = None

    def _clear_vehicles(self) -> None:
        """Match the quickstart cleanup so the HGV is the only vehicle in an episode."""
        for actor in self.world.get_actors().filter("vehicle.*"):
            try:
                actor.destroy()
            except RuntimeError:
                pass

    def shutdown(self) -> None:
        self.close_episode()
        try:
            self.air.hoverAsync()
            self.air.armDisarm(False)
            self.air.enableApiControl(False)
        except Exception:
            pass

    def tick(self) -> float:
        now = self._tick_world()
        if self.go_straight and self.truck is not None:
            self._drive_straight()
        return now

    def _drive_straight(self) -> None:
        """Lane-keeping with straight preference: follow current lane, go straight at junctions."""
        truck_loc = self.truck.get_location()
        truck_yaw = math.radians(self.truck.get_transform().rotation.yaw)
        wp = self.world.get_map().get_waypoint(truck_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return

        # Get next waypoints 5m ahead
        next_wps = wp.next(5.0)
        if not next_wps:
            return

        # Prefer same road & lane (straight), else same road, else first
        target = next_wps[0]
        for nw in next_wps:
            if nw.road_id == wp.road_id and nw.lane_id == wp.lane_id:
                target = nw
                break
        for nw in next_wps:
            if nw.road_id == wp.road_id:
                target = nw
                break

        # Steer toward target waypoint
        target_loc = target.transform.location
        dx = target_loc.x - truck_loc.x
        dy = target_loc.y - truck_loc.y
        target_angle = math.atan2(dy, dx)
        yaw_error = target_angle - truck_yaw
        # Normalize to [-pi, pi]
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
        steer = max(-1.0, min(1.0, yaw_error * 2.0))

        self.truck.apply_control(carla.VehicleControl(
            throttle=0.5,
            steer=steer,
        ))

    def _tick_world(self) -> float:
        """Advance the fixed-step world at its configured real-time cadence."""
        if self.world.get_settings().synchronous_mode:
            target = self._next_tick_wall
            if target is not None:
                remaining = target - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
            tick_started = time.monotonic()
            self.world.tick()
            self._next_tick_wall = tick_started + self._fixed_delta_seconds
        else:
            self.world.tick()
        return time.monotonic()

    def _calibrate_airsim_offset(self) -> None:
        """Use CARLA-Air's own drone actor to align CARLA and AirSim coordinates."""
        drone_actor = self._drone_actor()
        position = self.air.getMultirotorState().kinematics_estimated.position
        location = drone_actor.get_location()
        self._airsim_offset = np.array(
            [position.x_val - location.x, position.y_val - location.y, position.z_val + location.z],
            dtype=float,
        )

    def _drone_actor(self) -> carla.Actor:
        drone_actor = next(
            (actor for actor in self.world.get_actors() if "drone" in actor.type_id.lower()), None
        )
        if drone_actor is None:
            raise RuntimeError("CARLA-Air drone actor is unavailable for coordinate calibration")
        return drone_actor

    def _drone_actor_location(self) -> carla.Location:
        return self._drone_actor().get_location()

    def _to_ned(self, location: carla.Location) -> np.ndarray:
        if self._airsim_offset is None:
            raise RuntimeError("environment has not been reset")
        return self._airsim_offset + np.array([location.x, location.y, -location.z], dtype=float)

    def _to_carla(self, ned: np.ndarray) -> carla.Location:
        if self._airsim_offset is None:
            raise RuntimeError("environment has not been reset")
        delta = np.asarray(ned) - self._airsim_offset
        return carla.Location(
            x=float(delta[0]),
            y=float(delta[1]),
            z=-float(delta[2]),
        )

    def drone_state(self) -> VehicleState:
        state = self.air.getMultirotorState().kinematics_estimated
        position = state.position
        velocity = state.linear_velocity
        _, _, yaw = airsim.to_eularian_angles(state.orientation)
        return VehicleState(
            ned=np.array([position.x_val, position.y_val, position.z_val], dtype=float),
            yaw_rad=float(yaw),
            speed=float(math.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2)),
        )

    def truck_state(self) -> VehicleState:
        if self.truck is None:
            raise RuntimeError("truck is not spawned")
        transform = self.truck.get_transform()
        velocity = self.truck.get_velocity()
        return VehicleState(
            ned=self._to_ned(transform.location),
            yaw_rad=math.radians(transform.rotation.yaw),
            speed=float(math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)),
        )

    def cargo_bed_ned(self) -> np.ndarray:
        if self.truck is None:
            raise RuntimeError("truck is not spawned")
        location = self.truck.get_transform().transform(
            carla.Location(
                x=self.profile.bed_center_x,
                y=self.profile.bed_center_y,
                z=self.profile.bed_height,
            )
        )
        return self._to_ned(location)

    def cargo_relation(self) -> tuple[str, float, float]:
        """Coarse C1 direction, truck-motion phrase, and phase-supporting range."""
        drone = self.drone_state()
        bed = self.cargo_bed_ned()
        relative = bed - drone.ned
        forward = math.cos(drone.yaw_rad) * relative[0] + math.sin(drone.yaw_rad) * relative[1]
        right = -math.sin(drone.yaw_rad) * relative[0] + math.cos(drone.yaw_rad) * relative[1]
        horizontal = math.hypot(forward, right)
        if abs(right) < horizontal * 0.25:
            direction = "forward" if forward >= 0 else "rear"
        elif abs(forward) < horizontal * 0.25:
            direction = "right" if right >= 0 else "left"
        else:
            direction = ("forward" if forward >= 0 else "rear") + ("-right" if right >= 0 else "-left")
        motion = "moving slowly" if self.truck_state().speed < 2.5 else "moving"
        return direction, motion, horizontal

    def landing_phase(self) -> str:
        drone = self.drone_state()
        bed = self.cargo_bed_ned()
        horizontal = float(np.linalg.norm((bed - drone.ned)[:2]))
        height = bed[2] - drone.ned[2]
        if horizontal > 4.0:
            return "approach"
        if horizontal > 1.0:
            return "align"
        return "descend" if height > 0.5 else "touchdown"

    def set_truck_speed(self, speed: float) -> None:
        if self.truck is None:
            raise RuntimeError("truck is not spawned")
        speed = max(0.1, min(6.0, speed))
        speed_limit_mps = max(self.truck.get_speed_limit() / 3.6, 0.1)
        percentage_difference = 100.0 * (1.0 - speed / speed_limit_mps)
        self.traffic_manager.vehicle_percentage_speed_difference(self.truck, percentage_difference)
        self._last_target_speed = speed

    def apply_c2_speed(self, waypoint: np.ndarray, current: np.ndarray, inference_seconds: float) -> None:
        """The exact SPF waypoint adaptation from B.2: displacement / inference period."""
        commanded_forward_speed = float(np.linalg.norm((waypoint - current)[:2])) / max(inference_seconds, 1e-3)
        self.set_truck_speed(4.0 * min(1.5, max(0.5, commanded_forward_speed / 2.0)))

    def capture_rgb(self) -> np.ndarray:
        response = self._request_images([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        if response.width == 0 or response.height == 0:
            raise RuntimeError("AirSim returned an empty RGB frame")
        return np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)

    def _request_images(self, requests: list[airsim.ImageRequest]) -> list[airsim.ImageResponse]:
        """Pair an AirSim image RPC with the next CARLA synchronous render tick.

        In CARLA-Air, an AirSim image request issued after ``world.tick()`` waits
        for another renderer frame.  Dispatching the request first and then
        advancing one shared tick avoids that deadlock and keeps the image tied to
        a single physics state.
        """
        result: dict[str, object] = {}
        issued = threading.Event()

        def request() -> None:
            issued.set()
            try:
                result["responses"] = self.camera_air.simGetImages(requests)
            except Exception as exc:
                result["error"] = exc

        worker = threading.Thread(target=request, daemon=True)
        worker.start()
        issued.wait(timeout=1.0)
        if self.world.get_settings().synchronous_mode:
            time.sleep(0.01)
            self._tick_world()
        worker.join(timeout=20.0)
        if worker.is_alive():
            raise TimeoutError("AirSim image request did not complete after a CARLA tick")
        if "error" in result:
            raise result["error"]  # type: ignore[misc]
        return result["responses"]  # type: ignore[return-value]

    def track_waypoint(
        self,
        waypoint: np.ndarray,
        horizontal_gain: float = 3,
        vertical_gain: float = 0.8,
        duration: float = 0.05,
        target_yaw: float | None = None,
    ) -> np.ndarray:
        """Track the SPF waypoint with independent horizontal and vertical P gains."""
        state = self.drone_state()
        error = np.asarray(waypoint) - state.ned
        velocity = np.array(
            [
                horizontal_gain * error[0],
                horizontal_gain * error[1],
                vertical_gain * error[2],
            ]
        )
        speed = float(np.linalg.norm(velocity))
        if speed < 0.05:
            self.air.hoverAsync()
            return velocity
        if speed > 8.0:
            velocity *= 8.0 / speed
        self.air.moveByVelocityAsync(
            float(velocity[0]),
            float(velocity[1]),
            float(velocity[2]),
            duration=max(duration, 0.1),
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, math.degrees(target_yaw if target_yaw is not None else state.yaw_rad)),
        )
        return velocity

    def truck_in_camera_view(self) -> bool:
        """Geometric camera-frustum test used by TSR, independent of the policy prompt."""
        drone, target = self.drone_state(), self.truck_state()
        relative = target.ned - drone.ned
        forward = math.cos(drone.yaw_rad) * relative[0] + math.sin(drone.yaw_rad) * relative[1]
        right = -math.sin(drone.yaw_rad) * relative[0] + math.cos(drone.yaw_rad) * relative[1]
        if forward <= 0.1:
            return False
        horizontal_angle = abs(math.degrees(math.atan2(right, forward)))
        vertical_angle = abs(math.degrees(math.atan2(-relative[2], forward)))
        return horizontal_angle <= 54.0 and vertical_angle <= 55.0

    def truck_camera_iou(self) -> float:
        """Depth-verified target visibility for escort recovery.

        The protocol's 0.15 IoU threshold is evaluated as the visible fraction of
        the projected UGV box.  Depth samples inside the projected box prevent a
        truck that is geometrically in-frame but hidden by an object from counting
        as recovered.
        """
        if self.truck is None or not self.truck_in_camera_view():
            return 0.0
        drone, target = self.drone_state(), self.truck_state()
        response = self._request_images(
            [airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)]
        )[0]
        depth = np.asarray(response.image_data_float, dtype=np.float32).reshape(response.height, response.width)
        half_horizontal = math.radians(90.0 / 2.0)
        half_vertical = math.atan(math.tan(half_horizontal) * response.height / response.width)

        projected: list[tuple[float, float, float]] = []
        transform = self.truck.get_transform()
        for vertex in self.truck.bounding_box.get_world_vertices(transform):
            relative = self._to_ned(vertex) - drone.ned
            forward = math.cos(drone.yaw_rad) * relative[0] + math.sin(drone.yaw_rad) * relative[1]
            if forward <= 0.1:
                continue
            right = -math.sin(drone.yaw_rad) * relative[0] + math.cos(drone.yaw_rad) * relative[1]
            horizontal = math.atan2(right, forward)
            vertical = math.atan2(-relative[2], forward)
            pixel_x = (horizontal / half_horizontal + 1.0) * response.width / 2.0
            pixel_y = (1.0 - vertical / half_vertical) * response.height / 2.0
            if 0 <= pixel_x < response.width and 0 <= pixel_y < response.height:
                projected.append((pixel_x, pixel_y, forward))
        if not projected:
            return 0.0

        xs = [point[0] for point in projected]
        ys = [point[1] for point in projected]
        min_x, max_x = max(0, int(min(xs))), min(response.width - 1, int(max(xs)))
        min_y, max_y = max(0, int(min(ys))), min(response.height - 1, int(max(ys)))
        if min_x >= max_x or min_y >= max_y:
            return 0.0

        target_forward = target.ned - drone.ned
        target_depth = math.cos(drone.yaw_rad) * target_forward[0] + math.sin(drone.yaw_rad) * target_forward[1]
        depth_threshold = max(0.0, target_depth - max(self.truck.bounding_box.extent.x, 1.0) - 1.0)
        visible = 0
        total = 0
        for pixel_y in np.linspace(min_y, max_y, num=6, dtype=int):
            for pixel_x in np.linspace(min_x, max_x, num=6, dtype=int):
                total += 1
                if float(depth[pixel_y, pixel_x]) >= depth_threshold:
                    visible += 1
        return visible / total if total else 0.0

    def on_cargo_bed(self) -> bool:
        if self.truck is None:
            return False
        drone_location = self._to_carla(self.drone_state().ned)
        transform = self.truck.get_transform()
        delta_x, delta_y = drone_location.x - transform.location.x, drone_location.y - transform.location.y
        yaw = math.radians(transform.rotation.yaw)
        local_x = math.cos(yaw) * delta_x + math.sin(yaw) * delta_y
        local_y = -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y
        local_z = drone_location.z - transform.location.z
        return (
            abs(local_x - self.profile.bed_center_x) <= self.profile.bed_half_length
            and abs(local_y - self.profile.bed_center_y) <= self.profile.bed_half_width
            and abs(local_z - self.profile.bed_height) <= 0.35
        )
