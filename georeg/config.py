"""Configuration helpers for the georegistration pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml


@dataclass(slots=True)
class DEMProviderConfig:
    """Configuration for a single DEM provider."""

    name: str
    data_type: str = "COG"
    root: Optional[str] = None
    stac_url: Optional[str] = None
    collection: Optional[str] = None
    tile_template: Optional[str] = None
    resolution_arcsec: Optional[float] = None
    nodata: Optional[float] = None
    note: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DEMProviderConfig":
        return cls(
            name=str(data["name"]),
            data_type=str(data.get("type", data.get("data_type", "COG"))),
            root=data.get("root"),
            stac_url=data.get("stac") or data.get("stac_url"),
            collection=data.get("collection"),
            tile_template=data.get("tile_template"),
            resolution_arcsec=(
                float(data["resolution_arcsec"]) if data.get("resolution_arcsec") is not None else None
            ),
            nodata=(float(data["nodata"]) if data.get("nodata") is not None else None),
            note=data.get("note"),
        )


@dataclass(slots=True)
class DEMCacheConfig:
    """Settings describing how DEM assets are cached locally."""

    directory: Path
    max_bytes: int
    index_file: str = "cache_index.json"
    prefer_full_tile_download: bool = True
    http_timeout_sec: float = 30.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DEMCacheConfig":
        directory = Path(data.get("dir") or data.get("directory"))
        max_bytes_raw = data.get("max_bytes")
        if max_bytes_raw is None and data.get("max_cache_gb") is not None:
            max_bytes_raw = float(data["max_cache_gb"]) * (1024**3)
        if max_bytes_raw is None:
            raise ValueError("DEM cache configuration requires 'max_bytes' or 'max_cache_gb'")
        return cls(
            directory=directory,
            max_bytes=int(max_bytes_raw),
            index_file=str(data.get("index_file", "cache_index.json")),
            prefer_full_tile_download=bool(data.get("prefer_full_tile_download", True)),
            http_timeout_sec=float(data.get("http_timeout_sec", 30.0)),
        )


@dataclass(slots=True)
class DEMConfig:
    """Full DEM configuration including providers and cache."""

    primary: DEMProviderConfig
    fallbacks: Sequence[DEMProviderConfig] = field(default_factory=tuple)
    cache: DEMCacheConfig = field(
        default_factory=lambda: DEMCacheConfig(directory=Path("./dem_cache"), max_bytes=int(5 * 1024**3))
    )
    buffer_m: float = 2000.0

    @property
    def providers(self) -> Sequence[DEMProviderConfig]:
        return (self.primary,) + tuple(self.fallbacks)


@dataclass(slots=True)
class CameraConfig:
    lut_json: Path


@dataclass(slots=True)
class GeoidConfig:
    model: str = "EGM2008"
    grid_path: Optional[Path] = None
    height_type_in_dem: str = "orthometric"
    height_type_from_gnss: str = "ellipsoidal"


@dataclass(slots=True)
class RuntimeConfig:
    eps_m: float = 0.05
    max_iters: int = 2
    use_lrf: bool = True
    dem_variance: float = 1.0


@dataclass(slots=True)
class Config:
    dem: DEMConfig
    camera: CameraConfig
    geoid: GeoidConfig = field(default_factory=GeoidConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _coerce_provider_list(raw: Iterable[Mapping[str, Any]]) -> List[DEMProviderConfig]:
    return [DEMProviderConfig.from_mapping(entry) for entry in raw]


def _load_dem_config(dem_section: Mapping[str, Any]) -> DEMConfig:
    # Backwards compatible path with legacy 'providers'
    if "providers" in dem_section:
        providers = _coerce_provider_list(dem_section.get("providers", []))
        if not providers:
            raise ValueError("At least one DEM provider must be configured")
        cache_dir = Path(dem_section["cache_dir"])
        max_gb = float(dem_section.get("max_cache_gb", 10.0))
        cache = DEMCacheConfig(
            directory=cache_dir,
            max_bytes=int(max_gb * (1024**3)),
            index_file=str(dem_section.get("index_file", "cache_index.json")),
            prefer_full_tile_download=True,
            http_timeout_sec=float(dem_section.get("http_timeout_sec", 30.0)),
        )
        return DEMConfig(
            primary=providers[0],
            fallbacks=providers[1:],
            cache=cache,
            buffer_m=float(dem_section.get("buffer_m", 2000.0)),
        )

    if "primary" not in dem_section:
        raise ValueError("DEM configuration must include a 'primary' provider")

    primary = DEMProviderConfig.from_mapping(dem_section["primary"])
    fallback_section = dem_section.get("fallback")
    if isinstance(fallback_section, Mapping):
        fallbacks = [DEMProviderConfig.from_mapping(fallback_section)]
    elif isinstance(fallback_section, Iterable) and not isinstance(fallback_section, (str, bytes)):
        fallbacks = _coerce_provider_list(fallback_section)
    else:
        fallbacks = []

    cache_cfg = dem_section.get("cache")
    if cache_cfg is None:
        raise ValueError("DEM configuration requires a 'cache' section")
    cache = DEMCacheConfig.from_mapping(cache_cfg)

    return DEMConfig(
        primary=primary,
        fallbacks=fallbacks,
        cache=cache,
        buffer_m=float(dem_section.get("buffer_m", dem_section.get("corridor_buffer_m", 2000.0))),
    )


def load_config(path: str | Path) -> Config:
    """Load the YAML configuration file."""

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    dem_cfg = _load_dem_config(data.get("dem", {}))

    cam = data.get("camera", {})
    if "lut_json" not in cam:
        raise ValueError("Camera configuration requires 'lut_json'")
    camera_cfg = CameraConfig(lut_json=Path(cam["lut_json"]))

    geoid_data = data.get("geoid", {})
    geoid_cfg = GeoidConfig(
        model=str(geoid_data.get("model", "EGM2008")),
        grid_path=(Path(geoid_data["grid_path"]) if geoid_data.get("grid_path") else None),
        height_type_in_dem=str(geoid_data.get("height_type_in_dem", "orthometric")),
        height_type_from_gnss=str(geoid_data.get("height_type_from_gnss", "ellipsoidal")),
    )

    runtime_data = data.get("runtime", {})
    runtime_cfg = RuntimeConfig(
        eps_m=float(runtime_data.get("eps_m", 0.05)),
        max_iters=int(runtime_data.get("max_iters", 2)),
        use_lrf=bool(runtime_data.get("use_lrf", True)),
        dem_variance=float(runtime_data.get("dem_variance", 1.0)),
    )

    return Config(
        dem=dem_cfg,
        camera=camera_cfg,
        geoid=geoid_cfg,
        runtime=runtime_cfg,
    )
