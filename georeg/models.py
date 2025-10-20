"""Data models for the georegistration pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(slots=True)
class Pose:
    """Camera pose in the world frame.

    Attributes
    ----------
    R_wc:
        Rotation matrix that maps camera coordinates into the world frame.
    t_wc:
        Camera origin expressed in the world frame.  The world frame is
        expected to be Earth centred, Earth fixed (ECEF) or a local ENU frame
        that is consistent with the DEM georeferencing.
    timestamp:
        Optional timestamp in UTC for provenance logging.
    """

    R_wc: np.ndarray
    t_wc: np.ndarray
    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.R_wc.shape != (3, 3):
            raise ValueError("R_wc must be a 3x3 rotation matrix")
        if self.t_wc.shape != (3,):
            raise ValueError("t_wc must be a 3-vector")


@dataclass(slots=True)
class Intrinsics:
    """Camera intrinsics for a given zoom level."""

    zoom: float
    K: np.ndarray
    distortion: Optional[Sequence[float]] = None
    image_size: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.K.shape != (3, 3):
            raise ValueError("Intrinsic matrix K must be 3x3")


@dataclass(slots=True)
class Provenance:
    """Provenance metadata returned with every georegistered detection."""

    dem_product: str
    tile_ids: Sequence[str]
    geoid: str
    intrinsics_profile: str
    residual_m: float
    iters: int
    lrf_used: bool
    confidence: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "dem_product": self.dem_product,
            "tile_ids": list(self.tile_ids),
            "geoid": self.geoid,
            "intrinsics_profile": self.intrinsics_profile,
            "residual_m": float(self.residual_m),
            "iters": int(self.iters),
            "lrf_used": bool(self.lrf_used),
            "confidence": float(self.confidence),
        }
        base.update(self.extra)
        return base


@dataclass(slots=True)
class DetectionResult:
    """Result returned per detection."""

    lat: float
    lon: float
    alt_orthometric: float
    residual_m: float
    iters: int
    lrf_used: bool
    provenance: Provenance

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "lat": float(self.lat),
            "lon": float(self.lon),
            "alt_orthometric": float(self.alt_orthometric),
            "residual_m": float(self.residual_m),
            "iters": int(self.iters),
            "lrf_used": bool(self.lrf_used),
        }
        data.update({"provenance": self.provenance.to_dict()})
        return data


@dataclass(slots=True)
class LRFMeasurement:
    """Simple representation of a laser range finder measurement."""

    range_m: float
    variance: float
    valid: bool = True


@dataclass(slots=True)
class DEMTileRecord:
    """Metadata describing a cached DEM tile."""

    product: str
    identifier: str
    path: str
    footprint: Any
    resolution: float
    nodata: Optional[float]


@dataclass(slots=True)
class Corridor:
    """Container storing the flight corridor geometries."""

    polygon_wgs84: Any
    polygon_projected: Any
    projected_crs: Any


def hash_intrinsics_profile(entries: Iterable[Tuple[str, Any]]) -> str:
    """Create a deterministic hash string for provenance.

    The helper keeps provenance logic in a single location so that hashing can
    be improved without touching multiple files.
    """

    import hashlib
    import json

    serialised = json.dumps(list(entries), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return digest
