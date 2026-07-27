#!/usr/bin/env python3
"""实时显示 SPF 使用的无人机相机画面 (camera "0")。

Usage:
    python show_drone_cam.py              # 默认端口
    python show_drone_cam.py --port 41451 # 指定 AirSim 端口
    python show_drone_cam.py --save /tmp/frames/  # 保存截图
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import airsim
import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Show SPF drone camera feed")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--save", type=str, default="", help="Save frames to directory")
    args = parser.parse_args()

    print(f"Connecting to AirSim on port {args.port}...")
    client = airsim.MultirotorClient(ip="127.0.0.1", port=args.port, timeout_value=10)
    client.confirmConnection()
    print("Connected. Press 'q' to quit, 's' to save a screenshot.")

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    cv2.namedWindow("Drone Camera (SPF View)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Camera (SPF View)", 960, 720)

    try:
        while True:
            try:
                responses = client.simGetImages(
                    [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
                )
            except Exception as e:
                print(f"  Image error: {e}", end="\r")
                time.sleep(0.5)
                continue

            response = responses[0]
            if response.width == 0 or response.height == 0:
                time.sleep(0.1)
                continue

            img = np.frombuffer(
                response.image_data_uint8, dtype=np.uint8
            ).reshape(response.height, response.width, 3)

            # SPF 内部用的是 BGR，直接显示
            cv2.imshow("Drone Camera (SPF View)", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"drone_frame_{frame_idx:04d}.png"
                cv2.imwrite(str(save_dir / fname) if save_dir else fname, img)
                print(f"  Saved: {fname}")
                frame_idx += 1

            if save_dir:
                cv2.imwrite(str(save_dir / f"frame_{frame_idx:06d}.png"), img)
                frame_idx += 1
                if frame_idx % 30 == 0:
                    print(f"  Saved {frame_idx} frames...", end="\r")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("\nDone.")


if __name__ == "__main__":
    main()
