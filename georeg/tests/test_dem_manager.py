from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")
shapely = pytest.importorskip("shapely")
from rasterio.transform import from_origin
box = shapely.geometry.box

from georeg.config import DEMCacheConfig, DEMConfig, DEMProviderConfig
from georeg.dem_manager import DEMManager


class FakeAsset:
    def __init__(self, href: str):
        self.href = href
        self.extra_fields = {"gsd": 30, "nodata": None}


class FakeItem:
    def __init__(self, identifier: str, href: str, geometry):
        self.id = identifier
        self.assets = {"data": FakeAsset(href)}
        self.geometry = geometry
        self.properties = {}


class FakeSearch:
    def __init__(self, items):
        self._items = items

    def get_items(self):
        for item in self._items:
            yield item


class FakeClient:
    def __init__(self, items):
        self._items = items

    @classmethod
    def open(cls, url: str):
        return cls(cls._items)

    def search(self, **_: object):
        return FakeSearch(self._items)


def test_dem_fetch_and_cache(monkeypatch, tmp_path):
    raster_path = tmp_path / "tile.tif"
    data = np.array([[100.0, 101.0], [102.0, 103.0]], dtype=np.float32)
    transform = from_origin(-0.01, 0.01, 0.01, 0.01)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    geom = box(-0.02, -0.02, 0.02, 0.02)
    FakeClient._items = [FakeItem("tile-001", str(raster_path), geom.__geo_interface__)]

    monkeypatch.setattr("georeg.dem_manager.Client", FakeClient)
    monkeypatch.setattr(DEMManager, "_download_cog", lambda self, href: Path(href))

    provider = DEMProviderConfig(name="test", stac_url="https://example", collection="x")
    cache_cfg = DEMCacheConfig(directory=tmp_path, max_bytes=10 * 1024**2, index_file="index.json", http_timeout_sec=5.0)
    config = DEMConfig(primary=provider, fallbacks=[], cache=cache_cfg, buffer_m=1000)
    dem = DEMManager(config)
    dem.build_corridor([(0.0, 0.0), (0.0, 0.01)])
    dem.ensure_tiles()

    height, meta = dem.sample(0.0, 0.0)
    assert 100.0 <= height <= 103.0
    assert meta["dem_product"] == "test"
    assert meta["tile_ids"] == ["tile-001"]
