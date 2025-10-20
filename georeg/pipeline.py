"""High level orchestration utilities for the georegistration pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .camera_model import CameraModel
from .config import Config
from .dem_manager import DEMManager
from .geo_projector import GeoProjector
from .geoid import GeoidModel
from .models import DetectionResult


@dataclass(slots=True)
class GeoregistrationPipeline:
    """Convenience wrapper assembling the full pipeline end-to-end."""

    config: Config
    dem_manager: DEMManager
    camera: CameraModel
    geoid: GeoidModel
    projector: GeoProjector

    @classmethod
    def from_config(
        cls,
        config: Config,
        waypoints_wgs84: Sequence[Tuple[float, float]],
    ) -> "GeoregistrationPipeline":
        """Build a ready-to-use pipeline from a :class:`Config` instance."""

        dem_manager = DEMManager(config.dem)
        dem_manager.build_corridor(waypoints_wgs84, buffer_m=config.dem.buffer_m)
        dem_manager.ensure_tiles()

        camera = CameraModel(config.camera.lut_json)
        geoid = GeoidModel(model=config.geoid.model, grid_path=(str(config.geoid.grid_path) if config.geoid.grid_path else None))
        projector = GeoProjector(
            dem=dem_manager,
            geoid=geoid,
            camera=camera,
            eps_m=config.runtime.eps_m,
            max_iters=config.runtime.max_iters,
            dem_variance=config.runtime.dem_variance,
        )
        return cls(
            config=config,
            dem_manager=dem_manager,
            camera=camera,
            geoid=geoid,
            projector=projector,
        )

    def process_detection(
        self,
        u: float,
        v: float,
        zoom: float,
        R_wc: np.ndarray,
        t_wc: np.ndarray,
        lrf: Optional[Mapping[str, float]] = None,
        meta: Optional[MutableMapping[str, object]] = None,
    ) -> DetectionResult:
        """Run the complete georegistration flow for one detection."""

        lrf_payload = dict(lrf) if lrf is not None else None
        extra_meta = dict(meta) if meta is not None else None
        return self.projector.intersect_ray_dem(
            u=u,
            v=v,
            zoom=zoom,
            R_wc=R_wc,
            t_wc=t_wc,
            lrf=lrf_payload,
            meta_in=extra_meta,
        )

    def close(self) -> None:
        """Release open file handles and cached datasets."""

        self.dem_manager.close()
