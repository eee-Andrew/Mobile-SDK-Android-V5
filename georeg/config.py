"""Configuration helpers for the georegistration pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml


@dataclass(slots=True)
class DEMProviderConfig:
    name: str
    stac_url: str
    collection: str


@dataclass(slots=True)
class DEMConfig:
    providers: Sequence[DEMProviderConfig]
    cache_dir: Path
    max_cache_gb: float = 10.0
    buffer_m: float = 2000.0


@dataclass(slots=True)
class CameraConfig:
    lut_json: Path


@dataclass(slots=True)
class GeoidConfig:
    model: str = "EGM2008"
    grid_path: Optional[Path] = None


@dataclass(slots=True)
class RuntimeConfig:
    eps_m: float = 0.05
    max_iters: int = 2
    use_lrf: bool = True


@dataclass(slots=True)
class Config:
    dem: DEMConfig
    camera: CameraConfig
    geoid: GeoidConfig = field(default_factory=GeoidConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _coerce_providers(raw: Iterable[Dict[str, Any]]) -> List[DEMProviderConfig]:
    providers: List[DEMProviderConfig] = []
    for entry in raw:
        providers.append(
            DEMProviderConfig(
                name=str(entry["name"]),
                stac_url=str(entry["stac_url"]),
                collection=str(entry["collection"]),
            )
        )
    return providers


def load_config(path: str | Path) -> Config:
    """Load the YAML configuration file."""

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    dem = data.get("dem", {})
    providers = _coerce_providers(dem.get("providers", []))
    if not providers:
        raise ValueError("At least one DEM provider must be configured")

    dem_cfg = DEMConfig(
        providers=providers,
        cache_dir=Path(dem["cache_dir"]),
        max_cache_gb=float(dem.get("max_cache_gb", 10.0)),
        buffer_m=float(dem.get("buffer_m", 2000.0)),
    )

    cam = data.get("camera", {})
    camera_cfg = CameraConfig(lut_json=Path(cam["lut_json"]))

    geoid_data = data.get("geoid", {})
    geoid_cfg = GeoidConfig(
        model=str(geoid_data.get("model", "EGM2008")),
        grid_path=(Path(geoid_data["grid_path"]) if geoid_data.get("grid_path") else None),
    )

    runtime_data = data.get("runtime", {})
    runtime_cfg = RuntimeConfig(
        eps_m=float(runtime_data.get("eps_m", 0.05)),
        max_iters=int(runtime_data.get("max_iters", 2)),
        use_lrf=bool(runtime_data.get("use_lrf", True)),
    )

    return Config(
        dem=dem_cfg,
        camera=camera_cfg,
        geoid=geoid_cfg,
        runtime=runtime_cfg,
    )
