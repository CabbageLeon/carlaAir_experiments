#!/usr/bin/env python3
"""Terminal 6 — Auto-drive with VLM image-output route planning.

1. Generate clean street map → show to user
2. Send to qwen-vl-max via DashScope multimodal API (image+text → image)
3. Extract green route pixels from VLM output → waypoints
4. Drive car following waypoints

Usage:
  python collab_vln/scripts/auto_drive.py
"""

import base64, json, math, re, time
from pathlib import Path

import carla, cv2
import numpy as np
import dashscope
from dashscope import MultiModalConversation

GOAL_BP = "vehicle.carlamotors.european_hgv"
LOOKAHEAD = 6.0
MAX_SPEED = 10.0
STEER_GAIN = 0.4
TOLERANCE = 10.0
MAX_TIME = 180.0
MAP_SIZE = 640
MAP_ALT = 150.0


# ═══════════════════ connect & utils ═══════════════════

def connect():
    cl = carla.Client("127.0.0.1", 2000); cl.set_timeout(5.0)
    return cl, cl.get_world()

def find_vehicles(world):
    start_car = goal_car = None
    for v in world.get_actors().filter("vehicle.*"):
        if GOAL_BP in v.type_id: goal_car = v
        else: start_car = v
    return start_car, goal_car

def load_api_config():
    cfg = json.loads((Path(__file__).resolve().parent.parent.parent / "config.json").read_text())
    return cfg["models"]["spf"]["api_key"]


# ═══════════════════ street map ═══════════════════

def generate_street_map(world, min_x, max_x, min_y, max_y, img_size=1280):
    margin, road_width_m = 50, 16.0
    sx = (img_size - 2*margin) / (max_x - min_x) if max_x > min_x else 1
    sy = (img_size - 2*margin) / (max_y - min_y) if max_y > min_y else 1
    scale = min(sx, sy)
    thickness = max(3, int(scale * road_width_m))
    mid_x = (min_x + max_x) / 2; mid_y = (min_y + max_y) / 2

    def proj(wx, wy):
        px = int(img_size/2 + (wy - mid_y) * scale)
        py = int(img_size/2 - (wx - mid_x) * scale)
        return max(0, min(img_size-1, px)), max(0, min(img_size-1, py))

    img = np.zeros((img_size, img_size), dtype=np.uint8)
    spawn_pts = world.get_map().get_spawn_points()
    bmin_x, bmax_x = min_x - 80, max_x + 80
    bmin_y, bmax_y = min_y - 80, max_y + 80

    for sp in spawn_pts:
        spx, spy = sp.location.x, sp.location.y
        if not (bmin_x < spx < bmax_x and bmin_y < spy < bmax_y): continue
        wp = world.get_map().get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None: continue
        queue, seen = [(wp, 0)], set()
        while queue:
            cur, d = queue.pop(0)
            if d > 300: continue
            k = (round(cur.transform.location.x/2), round(cur.transform.location.y/2))
            if k in seen: continue
            seen.add(k)
            for nxt in cur.next(4.0):
                nx, ny = nxt.transform.location.x, nxt.transform.location.y
                if not (bmin_x < nx < bmax_x and bmin_y < ny < bmax_y): continue
                cv2.line(img, proj(cur.transform.location.x, cur.transform.location.y),
                         proj(nx, ny), 255, thickness)
                queue.append((nxt, d + 1))

    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return bgr, scale, mid_x, mid_y


# ═══════════════════ VLM: image+text → image ═══════════════════

def vlm_plan_route(street_map_img, start_xy, goal_xy, scale, mid_x, mid_y):
    api_key = load_api_config()
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    _, buf = cv2.imencode(".jpg", street_map_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    prompt = f"生成起点到终点的最短路径的像素坐标序列（网格按像素尺寸划分）"

    print("  Calling qwen3.8-max...")
    resp = MultiModalConversation.call(
        model="qwen3.8-max",
        api_key=api_key,
        messages=[{"role": "user", "content": [
            {"image": "data:image/jpeg;base64," + img_b64},
            {"text": prompt},
        ]}],
    )

    # Parse: output.choices[0].message.content[0]["text"]
    text = ""
    try:
        choices = resp.output.choices if resp.output else []
        for item in choices[0].message.content:
            if isinstance(item, dict) and "text" in item:
                text += item["text"]
    except Exception as e:
        print(f"  Parse error: {e}"); return None

    print(f"  VLM raw ({len(text)} chars):\n{text[:500]}\n---end---")

    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*"points".*\[.*\].*\}', text, re.DOTALL)
        data = json.loads(m.group()) if m else {}

    points = data.get("points") or data.get("waypoints") or data.get("coordinates") or []
    if not points:
        m = re.search(r'\[\[[\d,\s]+\]\]', text)
        if m: points = json.loads(m.group())
    if not points:
        print("  VLM: no points found"); return None

    waypoints = []
    for w in points:
        wy = (float(w[0]) - 640) / scale + mid_y
        wx = -(float(w[1]) - 640) / scale + mid_x
        waypoints.append((wx, wy))

    preview = street_map_img.copy()
    for i in range(1, len(points)):
        p1 = (int(float(points[i-1][0])), int(float(points[i-1][1])))
        p2 = (int(float(points[i][0])), int(float(points[i][1])))
        cv2.line(preview, p1, p2, (0, 255, 0), 3)
        cv2.circle(preview, p1, 3, (0, 255, 0), -1)
    if points:
        cv2.circle(preview, (int(float(points[-1][0])), int(float(points[-1][1]))), 5, (0, 255, 0), -1)
    cv2.putText(preview, f"{len(waypoints)} pts — any key", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("VLM Route Preview", preview)
    cv2.waitKey(0); cv2.destroyAllWindows()

    print(f"  VLM: {len(waypoints)} waypoints")
    return waypoints


# ═══════════════════ fallback route ═══════════════════

def plan_route(world, start_loc, goal_loc):
    pts = [(start_loc.x, start_loc.y)]
    wp = world.get_map().get_waypoint(start_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None: return pts
    gx, gy = goal_loc.x, goal_loc.y
    for _ in range(500):
        choices = wp.next(4.0)
        if not choices: break
        best = min(choices, key=lambda w: math.hypot(w.transform.location.x - gx, w.transform.location.y - gy)
                   - (2 if w.lane_id == wp.lane_id else 0))
        wp = best
        x, y = wp.transform.location.x, wp.transform.location.y
        pts.append((x, y))
        if math.hypot(x - gx, y - gy) < TOLERANCE: break
    return pts


# ═══════════════════ route map display ═══════════════════

def draw_route_map(route, cx, cy, gx, gy, mid_x, mid_y, scale):
    img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    def pj(wx, wy):
        return (max(0, min(MAP_SIZE-1, int(MAP_SIZE/2 + (wy-mid_y)*scale))),
                max(0, min(MAP_SIZE-1, int(MAP_SIZE/2 - (wx-mid_x)*scale))))
    for i in range(1, len(route)):
        cv2.line(img, pj(*route[i-1]), pj(*route[i]), (0, 200, 0), 1)
    gx_p, gy_p = pj(gx, gy)
    cv2.circle(img, (gx_p, gy_p), 10, (0, 0, 255), -1)
    cx_p, cy_p = pj(cx, cy)
    cv2.circle(img, (cx_p, cy_p), 6, (255, 0, 0), -1)
    return img


# ═══════════════════ main ═══════════════════

def main():
    cl, world = connect()
    start_car, goal_car = find_vehicles(world)
    if not start_car or not goal_car:
        print("ERROR: vehicles not found"); return

    sl = start_car.get_location(); gl = goal_car.get_location()
    gx, gy = gl.x, gl.y
    print(f"Start: ({sl.x:.0f},{sl.y:.0f})  Goal: ({gx:.0f},{gy:.0f})  Dist: {math.hypot(gx-sl.x,gy-sl.y):.0f}m")

    # Street map → annotate → VLM
    margin = 60
    min_x, max_x = min(sl.x, gx)-margin, max(sl.x, gx)+margin
    min_y, max_y = min(sl.y, gy)-margin, max(sl.y, gy)+margin
    smap, scl, mid_x, mid_y = generate_street_map(world, min_x, max_x, min_y, max_y)

    def proj(wx, wy):
        return (max(0,min(1279,int(640+(wy-mid_y)*scl))), max(0,min(1279,int(640-(wx-mid_x)*scl))))

    annotated = smap.copy()
    cv2.circle(annotated, proj(sl.x, sl.y), 40, (255,0,0), 3)
    cv2.putText(annotated, "START", (proj(sl.x,sl.y)[0]-30, proj(sl.x,sl.y)[1]-50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
    cv2.circle(annotated, proj(gx, gy), 40, (0,0,255), 3)
    cv2.line(annotated, (proj(gx,gy)[0]-30, proj(gx,gy)[1]),(proj(gx,gy)[0]+30, proj(gx,gy)[1]),(0,0,255),2)
    cv2.line(annotated, (proj(gx,gy)[0], proj(gx,gy)[1]-30),(proj(gx,gy)[0], proj(gx,gy)[1]+30),(0,0,255),2)
    cv2.putText(annotated, "GOAL", (proj(gx,gy)[0]+30, proj(gx,gy)[1]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # Show map, call VLM
    cv2.imshow("Street Map", annotated)
    print("  Press any key to call VLM, ESC to skip...")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    route = plan_route(world, sl, gl)  # fallback
    if key != 27:
        vlm_route = vlm_plan_route(annotated, (sl.x, sl.y), (gx, gy), scl, mid_x, mid_y)
        if vlm_route: route = vlm_route

    print(f"Route: {len(route)} points")
    wp_idx = 0

    # Chase camera
    chase_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    chase_bp.set_attribute("image_size_x", "960"); chase_bp.set_attribute("image_size_y", "540")
    chase_bp.set_attribute("fov", "100")
    chase_cam = world.spawn_actor(chase_bp,
        carla.Transform(carla.Location(x=-8.0, z=4.0), carla.Rotation(pitch=-15)),
        attach_to=start_car)
    latest_chase = [None]
    chase_cam.listen(lambda img: latest_chase.__setitem__(0,
        np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))[:,:,:3][:,:,::-1]))

    # Drive
    started = time.monotonic(); it = 0; ok = False
    try:
        while time.monotonic() - started < MAX_TIME:
            it += 1
            cloc = start_car.get_location(); cyaw = math.radians(start_car.get_transform().rotation.yaw)
            cx, cy = cloc.x, cloc.y
            dist = math.hypot(gx-cx, gy-cy)

            if dist < TOLERANCE:
                start_car.apply_control(carla.VehicleControl(brake=1.0))
                print(f"\n  ARRIVED! {dist:.1f}m {time.monotonic()-started:.1f}s")
                ok = True; time.sleep(1); break

            while wp_idx < len(route)-1:
                wx, wy = route[wp_idx+1]
                if math.hypot(cx-wx, cy-wy) < LOOKAHEAD+3: wp_idx += 1
                else: break
            tx, ty = route[wp_idx+1] if wp_idx < len(route)-1 else (gx, gy)

            fwd_x, fwd_y = math.cos(cyaw), math.sin(cyaw)
            right_x, right_y = -math.sin(cyaw), math.cos(cyaw)
            lateral = (tx-cx)*right_x + (ty-cy)*right_y
            fwd_dist = max(LOOKAHEAD, (tx-cx)*fwd_x + (ty-cy)*fwd_y)
            steer = max(-1.0, min(1.0, STEER_GAIN * lateral / fwd_dist))

            cur_spd = math.hypot(*[getattr(start_car.get_velocity(), a) for a in ('x','y')])
            if dist < 20: ts = max(1.5, dist*0.4)
            elif abs(steer) > 0.5: ts = MAX_SPEED*0.3
            elif abs(steer) > 0.3: ts = MAX_SPEED*0.5
            else: ts = MAX_SPEED
            err = ts - cur_spd
            throttle = max(0.0, min(1.0, 0.6*err + 0.15))
            brake = min(1.0, -2.0*err) if err < -1.0 else 0.0

            start_car.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

            if it % 20 == 0:
                print(f"  [{it}] dist={dist:.0f}m wp={wp_idx}/{len(route)} speed={cur_spd:.1f} steer={steer:+.2f}")

            if latest_chase[0] is not None:
                ch = latest_chase[0].copy()
                cv2.putText(ch, f"dist={dist:.0f}m {cur_spd:.1f}m/s", (10,500),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
                cv2.imshow("Chase", ch)
            cv2.imshow("Route", draw_route_map(route, cx, cy, gx, gy, mid_x, mid_y, scl))
            if cv2.waitKey(10) & 0xFF == 27: break
            time.sleep(0.05)
    finally:
        start_car.apply_control(carla.VehicleControl(brake=1.0))
        try: chase_cam.stop(); chase_cam.destroy()
        except: pass
        cv2.destroyAllWindows()

    print(f"\n{'SUCCESS' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
