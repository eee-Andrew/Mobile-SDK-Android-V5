import pytest

np = pytest.importorskip("numpy")

from georeg.geo_projector import GeoProjector
from georeg.models import DetectionResult


class StubDEM:
    def initial_height(self) -> float:
        return 0.0

    def sample(self, lat: float, lon: float):
        return 0.0, {"dem_product": "stub", "tile_ids": ["stub-tile"]}


class StubGeoid:
    def ellipsoidal_to_orthometric(self, lat: float, lon: float, h: float) -> float:
        return h

    def orthometric_to_ellipsoidal(self, lat: float, lon: float, H: float) -> float:
        return H

    @property
    def model_name(self) -> str:
        return "stub"


class StubCamera:
    def K_for_zoom(self, zoom: float):
        return np.eye(3)

    def dir_cam(self, u: float, v: float, K: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    def intrinsics_hash(self) -> str:
        return "stub"


def test_newton_converges_flat_terrain(monkeypatch):
    projector = GeoProjector(dem=StubDEM(), geoid=StubGeoid(), camera=StubCamera(), eps_m=0.01, max_iters=2)

    monkeypatch.setattr(GeoProjector, "_ecef_to_llh", lambda self, point: (point[1], point[0], point[2]))
    monkeypatch.setattr(GeoProjector, "_pose_to_ecef", lambda self, R, t: (t, R))

    R = np.diag([1.0, 1.0, -1.0])
    t = np.array([0.0, 0.0, 100.0])

    result = projector.intersect_ray_dem(u=0.0, v=0.0, zoom=1.0, R_wc=R, t_wc=t)
    assert isinstance(result, DetectionResult)
    assert abs(result.alt_orthometric) < 1e-6
    assert result.iters <= 2
