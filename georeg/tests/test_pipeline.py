from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from georeg.config import CameraConfig, Config, DEMCacheConfig, DEMConfig, DEMProviderConfig, GeoidConfig, RuntimeConfig
from georeg.models import DetectionResult, Provenance
from georeg.pipeline import GeoregistrationPipeline


class StubDEM:
    def __init__(self, config):
        self.config = config
        self.corridor_args = None
        self.closed = False
        self.ensure_called = False

    def build_corridor(self, waypoints, buffer_m=None):
        self.corridor_args = (list(waypoints), buffer_m)

    def ensure_tiles(self):
        self.ensure_called = True

    def close(self):
        self.closed = True


class StubProjector:
    def __init__(self, dem, geoid, camera, eps_m, max_iters, dem_variance):
        self.dem = dem
        self.geoid = geoid
        self.camera = camera
        self.params = (eps_m, max_iters, dem_variance)

    def intersect_ray_dem(self, **_: object) -> DetectionResult:
        provenance = Provenance(
            dem_product="stub",
            tile_ids=["tile"],
            geoid="stub-geoid",
            intrinsics_profile="hash",
            residual_m=0.0,
            iters=1,
            lrf_used=False,
            confidence=1.0,
        )
        return DetectionResult(
            lat=1.0,
            lon=2.0,
            alt_orthometric=3.0,
            residual_m=0.0,
            iters=1,
            lrf_used=False,
            provenance=provenance,
        )


def test_pipeline_wires_dependencies(monkeypatch, tmp_path):
    lut_path = tmp_path / "calib.json"
    lut_path.write_text(
        """
        {
          "4.0": {
            "K": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
          }
        }
        """.strip()
    )

    monkeypatch.setattr("georeg.pipeline.DEMManager", StubDEM)
    monkeypatch.setattr("georeg.pipeline.GeoProjector", StubProjector)

    provider = DEMProviderConfig(name="stub", stac_url="https://example.com", collection="x")
    cache_cfg = DEMCacheConfig(directory=tmp_path / "cache", max_bytes=1024 * 1024)
    dem_cfg = DEMConfig(primary=provider, fallbacks=[], cache=cache_cfg, buffer_m=1500.0)
    camera_cfg = CameraConfig(lut_json=lut_path)
    geoid_cfg = GeoidConfig(model="EGM2008")
    runtime_cfg = RuntimeConfig(eps_m=0.05, max_iters=2, use_lrf=True, dem_variance=1.5)
    config = Config(dem=dem_cfg, camera=camera_cfg, geoid=geoid_cfg, runtime=runtime_cfg)

    waypoints = [(0.0, 0.0), (0.0, 0.01)]
    pipeline = GeoregistrationPipeline.from_config(config, waypoints)

    assert isinstance(pipeline.dem_manager, StubDEM)
    assert pipeline.dem_manager.corridor_args == (list(waypoints), 1500.0)
    assert pipeline.dem_manager.ensure_called

    R = np.eye(3)
    t = np.zeros(3)
    result = pipeline.process_detection(u=0.0, v=0.0, zoom=4.0, R_wc=R, t_wc=t)
    assert isinstance(result, DetectionResult)

    pipeline.close()
    assert pipeline.dem_manager.closed
