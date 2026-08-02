"""OpenFly-Agent VLA policy — local 7B model produces 8-DoF drone actions from camera images.

Follows the ``PolicyAdapter`` protocol defined in ``.policy`` so it can be swapped in
place of ``SPFPolicy``.

Frame semantics (from OpenFly paper, arXiv 2502.18041):
  - The model was trained with 3 *distinct* frames: current observation + 2 historical keyframes.
  - Historical keyframes are compressed to 1 visual token each (current frame: 256 tokens).
  - We maintain a rolling buffer of past frames; when history is unavailable (first steps),
    the current frame is repeated as a placeholder (matching training-time initial-step behaviour).

Optimisation levels:
  - flash_attn:   flash_attention_2 kernel (~2× vs SDPA)
  - torch.compile: JIT-compile vision backbone (~1.5× vision speedup)
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

from ..train.extern.hf.configuration_prismatic import OpenFlyConfig
from ..train.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from ..train.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from .policy import Waypoint

# ── register custom model classes with HuggingFace auto-classes ─────────────
AutoConfig.register("openvla", OpenFlyConfig)
AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)

# ── 10 discrete action templates ────────────────────────────────────────────
_TEMPLATES = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],  #  0: stop
        [0, 3, 0, 0, 0, 0, 0, 0],  #  1: forward 3 m
        [0, 0, 15, 0, 0, 0, 0, 0],  #  2: turn left 15°
        [0, 0, 0, 15, 0, 0, 0, 0],  #  3: turn right 15°
        [0, 0, 0, 0, 2, 0, 0, 0],  #  4: up 2 m
        [0, 0, 0, 0, 0, 2, 0, 0],  #  5: down 2 m
        [0, 0, 0, 0, 0, 0, 5, 0],  #  6: move left 5 m
        [0, 0, 0, 0, 0, 0, 0, 5],  #  7: move right 5 m
        [0, 6, 0, 0, 0, 0, 0, 0],  #  8: fast forward 6 m
        [0, 9, 0, 0, 0, 0, 0, 0],  #  9: fastest forward 9 m
    ],
    dtype=np.float32,
)


class OpenFlyPolicy:
    """Local VLA policy that replaces the SPF cloud-VLM call.

    Maintains a rolling buffer of the last 2 observed frames as historical
    keyframes (matching OpenFly's training: current + 2 historical keyframes).
    """

    _HISTORY_SIZE: int = 2   # number of historical keyframes the model expects
    _TOTAL_FRAMES: int = _HISTORY_SIZE + 1  # current + history

    control_method = "p_controller"

    def __init__(self, config: dict):
        self.config = config
        self._model: Optional[OpenVLAForActionPrediction] = None
        self._processor: Optional[PrismaticProcessor] = None
        self._frame_history: deque[np.ndarray] = deque(maxlen=self._TOTAL_FRAMES)

    @classmethod
    def from_environment(cls, model_path: str | None = None) -> "OpenFlyPolicy":
        from . import load_config
        cfg = dict(load_config()["models"]["openfly"])
        if model_path:
            cfg["model_path"] = model_path
        return cls(cfg)

    # ── lazy loading ────────────────────────────────────────────────────────

    @property
    def model(self) -> OpenVLAForActionPrediction:
        if self._model is None:
            self._load()
        return self._model  # type: ignore[return-value]

    @property
    def processor(self) -> PrismaticProcessor:
        if self._processor is None:
            self._load()
        return self._processor  # type: ignore[return-value]

    @staticmethod
    def _detect_attn() -> str:
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            return "sdpa"

    def _load(self) -> None:
        started = time.monotonic()
        attn = self._detect_attn()
        print(f"[OpenFly] attention: {attn}  |  "
              f'compile_vision: {self.config["compile_vision"]}  |  '
              f"history: {self._HISTORY_SIZE} keyframes")

        self._processor = AutoProcessor.from_pretrained(
            self.config["model_path"], trust_remote_code=True
        )

        load_kwargs: dict = dict(
            attn_implementation=attn,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        free_gb = 0.0
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info(self.config["device"])[0] / (1024**3)
        if free_gb < 14.0:
            print(f"[OpenFly] GPU free {free_gb:.1f} GB → 4-bit quantisation")
            load_kwargs["load_in_4bit"] = True
            load_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
            load_kwargs["bnb_4bit_use_double_quant"] = True

        self._model = AutoModelForVision2Seq.from_pretrained(
            self.config["model_path"], **load_kwargs
        ).to(self.config["device"]).eval()

        if self.config["compile_vision"]:
            torch._dynamo.config.suppress_errors = True
            self._model.vision_backbone = torch.compile(
                self._model.vision_backbone, mode="reduce-overhead"
            )
            print("[OpenFly] vision backbone compiled with torch.compile")

        elapsed = time.monotonic() - started
        mem_used = torch.cuda.memory_allocated(self.config["device"]) / (1024**3) if torch.cuda.is_available() else 0
        print(f"[OpenFly] loaded in {elapsed:.1f}s, GPU mem: {mem_used:.1f} GB")

    def warmup(self) -> None:
        """Pre-load the model and trigger torch.compile (if enabled).

        Call this BEFORE the episode loop so the drone doesn't sit idle
        waiting for model loading on the first ``act()`` call.
        """
        print("[OpenFly] loading model — this takes ~11s on first run...")
        _ = self.model       # trigger _load()
        _ = self.processor   # trigger _load()
        print("[OpenFly] model ready, starting episode")

    # ── PolicyAdapter interface ────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the rolling frame buffer at episode start."""
        self._frame_history.clear()

    def get_fifo_frames(self) -> list:
        """Return current FIFO frames [oldest, ..., newest] (up to 3 frames)."""
        return list(self._frame_history)

    def _preprocess_frame(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Convert a BGR image to a preprocessed 6-channel pixel tensor (1, 6, 224, 224)."""
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        return torch.as_tensor(
            self.processor.image_processor([image])["pixel_values"][0]
        )

    def act(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        current_ned: np.ndarray,
        yaw_rad: float,
    ) -> Waypoint:
        device = self.config["device"]
        started = time.monotonic()

        # FIFO history: current frame + last 2 observed frames (matches his_step=2)
        self._frame_history.append(image_bgr)
        hist = list(self._frame_history)
        frames = [
            hist[-1],
            hist[-2] if len(hist) > 1 else hist[-1],
            hist[-3] if len(hist) > 2 else hist[-1],
        ]
        images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

        # Combined processor call (official OpenFly style: 3-image list)
        inputs = self.processor(prompt, images).to(device, dtype=torch.bfloat16)

        # Predict action (official generate-based predict_action)
        with torch.no_grad():
            action = self.model.predict_action(**inputs, unnorm_key=self.config["unnorm_key"], do_sample=False)
        finished = time.monotonic()

        # 6. Decode action vector
        a0, a1, a2, a3, a4, a5, a6, a7 = action.ravel().tolist()

        # 7. Stop-flag check
        if float(a0) > 0.5:
            return Waypoint(
                target_ned=np.asarray(current_ned, dtype=float),
                issued_at=time.monotonic(),
                inference_started_at=started,
                inference_finished_at=finished,
                prompt=prompt,
                raw_response=f"raw_action={np.array2string(action.ravel(), precision=2, separator=',')}  STOP",
            )

        # 8. Optional discretisation
        raw_action = action.ravel().copy()
        if self.config["use_discrete"]:
            best = int(np.argmin(np.linalg.norm(_TEMPLATES - raw_action, axis=1)))
            _, a1, a2, a3, a4, a5, a6, a7 = _TEMPLATES[best].tolist()

        # 9. Net displacements (body frame)
        forward_m = min(float(a1), self.config["max_forward_m"])
        lateral_m = float(a7 - a6)
        vertical_m = float(a5 - a4)
        turn_deg = float(a2 - a3)

        # 10. Compute NED waypoint
        adjusted_yaw = yaw_rad + math.radians(turn_deg)
        forward_axis = np.array([math.cos(adjusted_yaw), math.sin(adjusted_yaw), 0.0])
        right_axis = np.array([-math.sin(adjusted_yaw), math.cos(adjusted_yaw), 0.0])

        target = (
            np.asarray(current_ned, dtype=float)
            + forward_m * forward_axis
            + lateral_m * right_axis
        )
        target[2] += vertical_m

        return Waypoint(
            target_ned=target,
            issued_at=finished,
            inference_started_at=started,
            inference_finished_at=finished,
            prompt=prompt,
            raw_response=(
                f"raw_action={np.array2string(raw_action, precision=2, separator=',')}  "
                f"fwd={forward_m:.2f} lat={lateral_m:.2f} vert={vertical_m:.2f} "
                f"turn={turn_deg:.1f}°"
            ),
        )
