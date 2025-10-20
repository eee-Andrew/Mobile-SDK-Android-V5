"""Camera model utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import json
import numpy as np

from .models import Intrinsics, hash_intrinsics_profile


@dataclass(slots=True)
class CameraModel:
    """Interpolates intrinsics from a calibration lookup table."""

    lut_json_path: Path

    def __post_init__(self) -> None:
        with open(self.lut_json_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        self._profiles = {
            float(key): Intrinsics(
                zoom=float(key),
                K=np.asarray(value["K"], dtype=float),
                distortion=value.get("dist"),
                image_size=tuple(value.get("image_size", ())) or None,
            )
            for key, value in raw.items()
        }
        if not self._profiles:
            raise ValueError("The intrinsics LUT cannot be empty")
        self._sorted_zoom = sorted(self._profiles)
        self.profile_hash = hash_intrinsics_profile(sorted(raw.items()))

    def _interpolate(self, zoom: float) -> Intrinsics:
        if zoom in self._profiles:
            return self._profiles[zoom]
        # clamp to bounds
        if zoom <= self._sorted_zoom[0]:
            return self._profiles[self._sorted_zoom[0]]
        if zoom >= self._sorted_zoom[-1]:
            return self._profiles[self._sorted_zoom[-1]]

        for low, high in zip(self._sorted_zoom[:-1], self._sorted_zoom[1:]):
            if low <= zoom <= high:
                alpha = (zoom - low) / (high - low)
                K_low = self._profiles[low].K
                K_high = self._profiles[high].K
                K_interp = (1 - alpha) * K_low + alpha * K_high
                return Intrinsics(zoom=zoom, K=K_interp)
        raise RuntimeError("Interpolation failed for zoom level")

    def K_for_zoom(self, zoom: float) -> np.ndarray:
        """Return the 3x3 intrinsic matrix for a zoom level."""

        return self._interpolate(zoom).K

    def dir_cam(self, u: float, v: float, K: np.ndarray) -> np.ndarray:
        """Compute the unit ray in camera coordinates for an image point."""

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x = (u - cx) / fx
        y = (v - cy) / fy
        ray = np.array([x, y, 1.0], dtype=float)
        return ray / np.linalg.norm(ray)

    def intrinsics_hash(self) -> str:
        """Return the provenance hash of the LUT file."""

        return self.profile_hash
