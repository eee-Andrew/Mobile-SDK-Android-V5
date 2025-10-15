"""Geoid helpers for datum consistency."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
try:  # pragma: no cover - optional dependency runtime check
    from geographiclib.geoids import Geoid  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    Geoid = None  # type: ignore


@dataclass(slots=True)
class GeoidModel:
    """Wrapper around :mod:`geographiclib` that exposes convenience methods."""

    model: str = "EGM2008"
    grid_path: Optional[str] = None
    _geoid: Optional[object] = field(init=False, default=None)
    _fallback: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        kwargs = {"name": self.model}
        if self.grid_path:
            kwargs["path"] = self.grid_path
        try:
            if Geoid is None:
                raise RuntimeError("geographiclib geoid grids not available")
            self._geoid = Geoid(**kwargs)
            self._fallback = None
        except RuntimeError:
            # geographiclib raises RuntimeError when the grid file is not present.
            # For unit tests or offline usage we fall back to a zero separation
            # approximation which is still datum consistent (albeit less accurate).
            self._geoid = None
            self._fallback = 0.0

    def ellipsoidal_to_orthometric(self, lat: float, lon: float, h_ellipsoidal: float) -> float:
        """Convert ellipsoidal height to orthometric height."""

        separation = self._height(lat, lon)
        return h_ellipsoidal - separation

    def orthometric_to_ellipsoidal(self, lat: float, lon: float, H: float) -> float:
        separation = self._height(lat, lon)
        return H + separation

    @property
    def model_name(self) -> str:
        if self._geoid is not None:
            return self._geoid.description
        return f"{self.model} (fallback)"

    def _height(self, lat: float, lon: float) -> float:
        if self._geoid is None:
            return 0.0 if self._fallback is None else self._fallback
        return self._geoid.Height(lat, lon)
