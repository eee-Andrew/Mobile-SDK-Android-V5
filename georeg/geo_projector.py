"""Geo projection and ray/DEM intersection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import logging
import math

import numpy as np
import pyproj

from .camera_model import CameraModel
from .dem_manager import DEMManager
from .geoid import GeoidModel
from .models import DetectionResult, Provenance

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GeoProjector:
    """Perform ray-terrain intersection using bounded Newton iterations."""

    dem: DEMManager
    geoid: GeoidModel
    camera: CameraModel
    eps_m: float = 0.05
    max_iters: int = 2
    dem_variance: float = 1.0
    _ecef_to_geo: pyproj.Transformer = field(init=False)
    _geo_to_ecef: pyproj.Transformer = field(init=False)

    def __post_init__(self) -> None:
        self._ecef_to_geo = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
        self._geo_to_ecef = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)

    def _pose_to_ecef(self, R_wc: np.ndarray, t_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if R_wc.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        if t_wc.shape != (3,):
            raise ValueError("Translation must be 3-vector")
        origin = t_wc.astype(float)
        return origin, R_wc.astype(float)

    def _ecef_to_llh(self, point: np.ndarray) -> tuple[float, float, float]:
        lon, lat, h = self._ecef_to_geo.transform(point[0], point[1], point[2])
        return lat, lon, h

    def _llh_to_ecef(self, lat: float, lon: float, h: float) -> np.ndarray:
        x, y, z = self._geo_to_ecef.transform(lon, lat, h)
        return np.array([x, y, z])

    def intersect_ray_dem(
        self,
        u: float,
        v: float,
        zoom: float,
        R_wc: np.ndarray,
        t_wc: np.ndarray,
        lrf: Optional[Dict[str, float]] = None,
        meta_in: Optional[Dict[str, object]] = None,
    ) -> DetectionResult:
        origin, R = self._pose_to_ecef(R_wc, t_wc)
        K = self.camera.K_for_zoom(zoom)
        dir_cam = self.camera.dir_cam(u, v, K)
        d_world = R @ dir_cam
        d_world = d_world / np.linalg.norm(d_world)

        h0 = self.dem.initial_height()
        # Convert initial height to ellipsoidal
        lat0, lon0, h_ellip0 = self._ecef_to_llh(origin)
        h0_ellip = self.geoid.orthometric_to_ellipsoidal(lat0, lon0, h0)
        w = (h0_ellip - origin[2]) / d_world[2]

        residual = math.inf
        used_iters = 0
        last_height = h0
        last_lat = lat0
        last_lon = lon0
        for it in range(self.max_iters):
            candidate = origin + w * d_world
            lat, lon, h_ellip = self._ecef_to_llh(candidate)
            H = self.geoid.ellipsoidal_to_orthometric(lat, lon, h_ellip)
            dem_height, dem_meta = self.dem.sample(lat, lon)
            residual = H - dem_height
            last_height = dem_height
            last_lat = lat
            last_lon = lon
            used_iters = it + 1
            if abs(residual) < self.eps_m:
                break
            # numeric derivative
            delta = 0.01
            candidate_delta = origin + (w + delta) * d_world
            lat_d, lon_d, h_ellip_d = self._ecef_to_llh(candidate_delta)
            H_d = self.geoid.ellipsoidal_to_orthometric(lat_d, lon_d, h_ellip_d)
            deriv = (H_d - H) / delta
            if abs(deriv) < 1e-6:
                LOGGER.debug("Derivative too small, breaking early")
                break
            w -= residual / deriv

        lrf_used = False
        fused_height = last_height
        variance = self.dem_variance
        if lrf and lrf.get("valid", True):
            range_m = float(lrf["range_m"])
            var_lrf = float(lrf.get("variance", 1.0))
            w_lrf = range_m
            # Convert range to orthometric height at intersection point
            candidate_lrf = origin + w_lrf * d_world
            lat_lrf, lon_lrf, h_lrf = self._ecef_to_llh(candidate_lrf)
            H_lrf = self.geoid.ellipsoidal_to_orthometric(lat_lrf, lon_lrf, h_lrf)
            fused_height = (last_height / variance + H_lrf / var_lrf) / (1.0 / variance + 1.0 / var_lrf)
            lrf_used = True

        provenance = Provenance(
            dem_product=dem_meta["dem_product"],
            tile_ids=dem_meta["tile_ids"],
            geoid=self.geoid.model_name,
            intrinsics_profile=self.camera.intrinsics_hash(),
            residual_m=float(abs(residual)),
            iters=used_iters,
            lrf_used=lrf_used,
            confidence=float(math.exp(-abs(residual))),
            extra=meta_in or {},
        )

        return DetectionResult(
            lat=last_lat,
            lon=last_lon,
            alt_orthometric=fused_height,
            residual_m=float(abs(residual)),
            iters=used_iters,
            lrf_used=lrf_used,
            provenance=provenance,
        )
