"""SPF's image-point waypoint planner over an OpenAI-compatible VLM endpoint."""

from __future__ import annotations

import base64
import json
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
from openai import OpenAI

from .config import SPFConfig, get_config
from .policy import Waypoint


class SPFPolicy:
    """One VLM call produces one spatial point and one AirSim NED waypoint."""

    def __init__(self, config: SPFConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    @classmethod
    def from_environment(cls, model: str | None = None) -> "SPFPolicy":
        """Create a policy from environment variables and provider presets.

        Set SPF_PROVIDER to switch providers, or override with
        OPENAI_BASE_URL / OPENAI_MODEL.  See config.py for presets.
        """
        config = get_config(model_override=model)
        return cls(config)

    @classmethod
    def from_interactive(cls, model: str | None = None) -> "SPFPolicy":
        """Interactive provider/model/key selection then create the policy."""
        from .interactive import interactive_config

        config = interactive_config(model_arg=model)
        return cls(config)

    def reset(self) -> None:
        """SPF has no episode state to reset."""

    @staticmethod
    def _prompt(instruction: str) -> str:
        return f"""You are a drone navigation expert analyzing a drone camera view.

Task: {instruction}

Target object: the moving truck. For the landing task, target its flat rear cargo bed.
Locate that truck in the image and place a single point DIRECTLY ON the target. When the
truck is visible, do not return an empty list; choose its center even when it is small or distant.

Return in this exact JSON format:
[{{\"point\": [x, y], \"depth\": depth_value, \"label\": \"action description\"}}]

Coordinate system:
- x: 0-1000 scale (500=center, >500=right, <500=left)
- y: 0-1000 scale (lower values=higher in image/sky)
- depth: 1-10 scale where 1 is very close and 10 is far away.

IMPORTANT: Place the point precisely on the center of the target object. Assess depth from
its image size. Do not use a road, building, sky, or unrelated vehicle as the target."""

    @staticmethod
    def _clean_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(line for line in lines if not line.strip().startswith("```"))
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end < start:
            raise ValueError("SPF response does not contain a JSON array")
        return text[start : end + 1]

    @classmethod
    def parse_point(cls, text: str) -> tuple[float, float, float]:
        """Parse Qwen's conventional [x, y] point and its 1--10 depth."""
        data, _ = json.JSONDecoder().raw_decode(cls._clean_json(text))
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ValueError("SPF response has no point object")
        point = data[0].get("point")
        depth = data[0].get("depth")
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("SPF point must be [x, y]")
        x, y = (float(point[0]), float(point[1]))
        depth = float(depth)
        if not 0 <= x <= 1000 or not 0 <= y <= 1000 or not 1 <= depth <= 10:
            raise ValueError("SPF point/depth is outside its documented range")
        return y, x, depth

    def act(
        self, image_bgr: np.ndarray, prompt: str, current_ned: np.ndarray, yaw_rad: float
    ) -> Waypoint:
        """Map SPF's 2-D point to a single waypoint, matching its non-adaptive AirSim mode."""
        height, width = image_bgr.shape[:2]
        ok, encoded = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise RuntimeError("could not JPEG-encode UAV frame")
        started = time.monotonic()
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt(prompt)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,"
                                + base64.b64encode(encoded.tobytes()).decode("ascii")
                            },
                        },
                    ],
                }
            ],
        )
        finished = time.monotonic()
        raw = response.choices[0].message.content or ""
        y, x, depth = self.parse_point(raw)

        distance = depth / 10.0 * self.config.projection_distance_max_m
        fov_factor = math.tan(math.radians(self.config.horizontal_fov_deg / 2.0))
        lateral = ((x / 1000.0 * width - width / 2.0) / (width / 2.0)) * distance * fov_factor
        up = ((height / 2.0 - y / 1000.0 * height) / (height / 2.0)) * distance * fov_factor
        forward = distance
        forward_axis = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
        right_axis = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0])
        target = np.asarray(current_ned, dtype=float) + forward * forward_axis + lateral * right_axis
        target[2] -= up  # NED down is positive.
        return Waypoint(
            target_ned=target,
            issued_at=finished,
            inference_started_at=started,
            inference_finished_at=finished,
            prompt=prompt,
            raw_response=raw,
        )
