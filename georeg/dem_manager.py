"""DEM management, discovery and sampling."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import logging
import math
import os
import threading

import cachetools
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point, shape
from pystac_client import Client
import requests

from .config import DEMConfig, DEMProviderConfig
from .corridor import CorridorBuilder
from .models import Corridor, DEMTileRecord

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DEMManager:
    """Discover, download and sample DEM tiles."""

    config: DEMConfig
    _corridor: Optional[Corridor] = field(init=False, default=None)
    _providers: Sequence[DEMProviderConfig] = field(init=False)
    _cache_dir: Path = field(init=False)
    _tile_records: List[DEMTileRecord] = field(init=False, default_factory=list)
    _dataset_cache: cachetools.LRUCache = field(init=False)
    _lock: threading.Lock = field(init=False)

    def __post_init__(self) -> None:
        self._providers = self.config.providers
        self._cache_dir = Path(self.config.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._tile_records = []
        self._dataset_cache = cachetools.LRUCache(maxsize=8)
        self._lock = threading.Lock()

    def build_corridor(self, waypoints_wgs84: Sequence[Tuple[float, float]], buffer_m: float | None = None) -> None:
        builder = CorridorBuilder(buffer_m=buffer_m or self.config.buffer_m)
        self._corridor = builder.build_corridor(waypoints_wgs84)

    @property
    def corridor(self) -> Corridor:
        if self._corridor is None:
            raise RuntimeError("Corridor has not been built yet")
        return self._corridor

    def ensure_tiles(self) -> None:
        if self._corridor is None:
            raise RuntimeError("Corridor must be built before fetching tiles")

        for provider in self._providers:
            try:
                self._fetch_tiles_from_provider(provider)
                if self._tile_records:
                    return
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.warning("DEM provider %s failed: %s", provider.name, exc)
        if not self._tile_records:
            raise RuntimeError("Failed to acquire DEM tiles from all providers")

    def _fetch_tiles_from_provider(self, provider: DEMProviderConfig) -> None:
        client = Client.open(provider.stac_url)
        corridor_geom = self._corridor.polygon_wgs84.__geo_interface__
        search = client.search(collections=[provider.collection], intersects=corridor_geom)
        items = list(search.get_items())
        if not items:
            raise RuntimeError(f"No DEM tiles found for provider {provider.name}")
        for item in items:
            asset = item.assets.get("data") or next(iter(item.assets.values()))
            href = asset.href
            path = self._download_cog(href)
            geom = shape(item.geometry)
            tile = DEMTileRecord(
                product=provider.name,
                identifier=item.id,
                path=str(path),
                footprint=geom,
                resolution=float(item.properties.get("gsd", asset.extra_fields.get("gsd", 30))),
                nodata=asset.extra_fields.get("nodata"),
            )
            self._tile_records.append(tile)

    def _download_cog(self, href: str) -> Path:
        filename = os.path.basename(href)
        target = self._cache_dir / filename
        if target.exists():
            return target

        LOGGER.info("Downloading DEM tile %s", href)
        with requests.get(href, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(target, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 512):
                    handle.write(chunk)
        return target

    def _dataset(self, tile: DEMTileRecord) -> rasterio.io.DatasetReader:
        with self._lock:
            ds = self._dataset_cache.get(tile.path)
            if ds is None:
                ds = rasterio.open(tile.path)
                self._dataset_cache[tile.path] = ds
        return ds

    def sample(self, lat: float, lon: float) -> Tuple[float, Dict[str, object]]:
        if not self._tile_records:
            raise RuntimeError("DEM tiles have not been fetched")

        point = Point(lon, lat)
        for tile in self._tile_records:
            if tile.footprint.contains(point):
                height = self._sample_tile(tile, lat, lon)
                meta = {
                    "dem_product": tile.product,
                    "tile_ids": [tile.identifier],
                }
                return height, meta
        raise ValueError("Location outside of DEM coverage")

    def _sample_tile(self, tile: DEMTileRecord, lat: float, lon: float) -> float:
        dataset = self._dataset(tile)
        row, col = dataset.index(lon, lat)
        window = Window(col_off=col - 1, row_off=row - 1, width=2, height=2)
        window = window.round_offsets().round_lengths()
        arr = dataset.read(1, window=window, boundless=True, fill_value=dataset.nodata)
        transform = dataset.window_transform(window)
        # convert to float, handle nodata
        nodata = dataset.nodata if dataset.nodata is not None else tile.nodata
        arr = arr.astype(float)
        if nodata is not None:
            arr[arr == nodata] = np.nan
        x, y = rasterio.transform.rowcol(transform, lon, lat)
        fx = x - math.floor(x)
        fy = y - math.floor(y)
        top = (1 - fx) * arr[0, 0] + fx * arr[0, 1]
        bottom = (1 - fx) * arr[1, 0] + fx * arr[1, 1]
        value = (1 - fy) * top + fy * bottom
        if np.isnan(value):
            raise ValueError("DEM sample contains NoData")
        return float(value)

    def initial_height(self) -> float:
        if not self._tile_records:
            return 0.0
        heights = []
        for tile in self._tile_records[:3]:
            ds = self._dataset(tile)
            sample = ds.read(1, window=Window(0, 0, 1, 1))
            heights.append(float(sample[0, 0]))
        return float(np.nanmedian(heights)) if heights else 0.0

    def close(self) -> None:
        for ds in list(self._dataset_cache.values()):
            ds.close()
        self._dataset_cache.clear()
