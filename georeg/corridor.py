"""Mission corridor construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from shapely.geometry import LineString
from shapely.ops import transform
import pyproj

from .models import Corridor


@dataclass(slots=True)
class CorridorBuilder:
    """Build a mission corridor around waypoints."""

    buffer_m: float = 2000.0

    def build_corridor(self, waypoints_wgs84: Sequence[Tuple[float, float]]) -> Corridor:
        if len(waypoints_wgs84) < 2:
            raise ValueError("At least two waypoints are required to build a corridor")

        line = LineString([(lon, lat) for lat, lon in waypoints_wgs84])

        # Determine a UTM zone from the centroid
        centroid = line.centroid
        utm_crs = self._utm_crs(centroid.y, centroid.x)
        project_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True).transform
        project_to_wgs = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True).transform

        corridor_utm = transform(project_to_utm, line).buffer(self.buffer_m)
        corridor_wgs84 = transform(project_to_wgs, corridor_utm)

        return Corridor(
            polygon_wgs84=corridor_wgs84,
            polygon_projected=corridor_utm,
            projected_crs=utm_crs,
        )

    @staticmethod
    def _utm_crs(lat: float, lon: float) -> pyproj.CRS:
        zone = int((lon + 180) / 6) + 1
        is_northern = lat >= 0
        return pyproj.CRS.from_dict(
            {
                "proj": "utm",
                "zone": zone,
                "datum": "WGS84",
                "south": not is_northern,
            }
        )
