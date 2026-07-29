"""Policy boundary shared by SPF and future aerial-navigation baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Waypoint:
    """A single waypoint in the AirSim NED world frame."""

    target_ned: np.ndarray
    issued_at: float
    inference_started_at: float
    inference_finished_at: float
    prompt: str
    raw_response: str


class PolicyAdapter(Protocol):
    """Common policy API; future methods only need to implement this boundary."""

    def reset(self) -> None:
        """Reset episode-local policy state."""

    def act(
        self, image_bgr: np.ndarray, prompt: str, current_ned: np.ndarray, yaw_rad: float
    ) -> Waypoint:
        """Return the next native waypoint from the current UAV image and prompt."""
