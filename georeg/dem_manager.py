"""DEM management, discovery and sampling."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import logging
import math
import os
import threading
import time
import json

import cachetools
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point, shape
from pystac_client import Client
import requests

from .config import DEMCacheConfig, DEMConfig, DEMProviderConfig
from .corridor import CorridorBuilder
from .models import Corridor, DEMTileRecord

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DEMManager:
    """Discover, download and sample DEM tiles."""

    config: DEMConfig
    _corridor: Optional[Corridor] = field(init=False, default=None)
    _providers: Sequence[DEMProviderConfig] = field(init=False)
    _cache_cfg: DEMCacheConfig = field(init=False)
    _cache_dir: Path = field(init=False)
    _cache_index_path: Path = field(init=False)
    _cache_index: Dict[str, Dict[str, float]] = field(init=False)
    _tile_records: List[DEMTileRecord] = field(init=False, default_factory=list)
    _dataset_cache: cachetools.LRUCache = field(init=False)
    _lock: threading.Lock = field(init=False)

    def __post_init__(self) -> None:
        self._providers = self.config.providers
        self._cache_cfg = self.config.cache
        self._cache_dir = Path(self._cache_cfg.directory)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_index_path = self._cache_dir / self._cache_cfg.index_file
        self._cache_index = self._load_cache_index()
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
        if provider.stac_url:
            self._fetch_via_stac(provider)
        elif provider.tile_template and provider.root:
            self._fetch_via_template(provider)
        else:
            raise RuntimeError(
                f"Provider {provider.name} does not expose a supported acquisition mechanism"
            )

    def _fetch_via_stac(self, provider: DEMProviderConfig) -> None:
        assert self._corridor is not None
        client = Client.open(provider.stac_url)
        corridor_geom = self._corridor.polygon_wgs84.__geo_interface__
        search_kwargs: Dict[str, object] = {"intersects": corridor_geom}
        if provider.collection:
            search_kwargs["collections"] = [provider.collection]
        search = client.search(**search_kwargs)
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
                resolution=float(item.properties.get("gsd", asset.extra_fields.get("gsd", provider.resolution_arcsec or 30))),
                nodata=asset.extra_fields.get("nodata", provider.nodata),
            )
            self._tile_records.append(tile)

    def _fetch_via_template(self, provider: DEMProviderConfig) -> None:
        from shapely.geometry import box

        assert self._corridor is not None
        minx, miny, maxx, maxy = self._corridor.polygon_wgs84.bounds
        lat_start = math.floor(miny)
        lat_end = math.ceil(maxy)
        lon_start = math.floor(minx)
        lon_end = math.ceil(maxx)
        for lat_deg in range(lat_start, lat_end):
            for lon_deg in range(lon_start, lon_end):
                href, identifier = self._template_href(provider, lat_deg, lon_deg)
                path = self._download_cog(href)
                geom = box(lon_deg, lat_deg, lon_deg + 1, lat_deg + 1)
                tile = DEMTileRecord(
                    product=provider.name,
                    identifier=identifier,
                    path=str(path),
                    footprint=geom,
                    resolution=float(provider.resolution_arcsec or 1.0) * 30.0,
                    nodata=provider.nodata,
                )
                self._tile_records.append(tile)

    def _template_href(self, provider: DEMProviderConfig, lat_deg: int, lon_deg: int) -> Tuple[str, str]:
        if not provider.tile_template:
            raise RuntimeError("Tile template missing for template-based provider")
        lat_band = "N" if lat_deg >= 0 else "S"
        lon_band = "E" if lon_deg >= 0 else "W"
        lat_fmt = f"{abs(int(lat_deg)):02d}"
        lon_fmt = f"{abs(int(lon_deg)):03d}"
        identifier = provider.tile_template.format(NS=lat_band, EW=lon_band, LAT=lat_fmt, LON=lon_fmt)
        root = provider.root or ""
        if root and not root.endswith("/"):
            root = root + "/"
        href = root + identifier
        return href, identifier

    def _download_cog(self, href: str) -> Path:
        filename = os.path.basename(href.rstrip("/")) or os.path.basename(href)
        target = self._cache_dir / filename
        if target.exists():
            self._touch_cache_entry(target)
            return target

        LOGGER.info("Downloading DEM tile %s", href)
        if href.startswith("http://") or href.startswith("https://"):
            with requests.get(href, stream=True, timeout=self._cache_cfg.http_timeout_sec) as response:
                response.raise_for_status()
                with open(target, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 512):
                        handle.write(chunk)
        else:
            source = Path(href)
            if not source.exists():
                raise FileNotFoundError(href)
            with open(source, "rb") as src, open(target, "wb") as dst:
                dst.write(src.read())
        self._touch_cache_entry(target)
        self._enforce_cache_limit()
        return target

    def _touch_cache_entry(self, path: Path) -> None:
        stat = path.stat()
        self._cache_index[str(path)] = {"size": float(stat.st_size), "last_used": time.time()}
        self._save_cache_index()

    def _enforce_cache_limit(self) -> None:
        max_bytes = self._cache_cfg.max_bytes
        entries = list(self._cache_index.items())
        total = sum(entry[1]["size"] for entry in entries)
        if total <= max_bytes:
            return
        entries.sort(key=lambda item: item[1].get("last_used", 0.0))
        for path_str, _ in entries:
            if total <= max_bytes:
                break
            path = Path(path_str)
            LOGGER.info("Evicting DEM tile %s to honor cache limits", path.name)
            with self._lock:
                ds = self._dataset_cache.pop(str(path), None)
                if ds is not None:
                    ds.close()
            if path.exists():
                path.unlink()
            entry = self._cache_index.pop(path_str, None)
            if entry:
                total -= entry["size"]
        self._save_cache_index()

    def _load_cache_index(self) -> Dict[str, Dict[str, float]]:
        if self._cache_index_path.exists():
            try:
                with open(self._cache_index_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except (json.JSONDecodeError, OSError):  # pragma: no cover - corruption guard
                LOGGER.warning("Failed to parse DEM cache index, rebuilding")
        return {}

    def _save_cache_index(self) -> None:
        with open(self._cache_index_path, "w", encoding="utf-8") as handle:
            json.dump(self._cache_index, handle)

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
            if tile.footprint.contains(point) or tile.footprint.touches(point):
                height = self._sample_tile(tile, lat, lon)
                self._touch_cache_entry(Path(tile.path))
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
