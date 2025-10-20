import pytest

shapely = pytest.importorskip("shapely")
Point = shapely.geometry.Point

from georeg.corridor import CorridorBuilder


def test_corridor_contains_waypoints():
    builder = CorridorBuilder(buffer_m=500)
    waypoints = [
        (37.0, -122.0),
        (37.005, -122.01),
        (37.01, -122.02),
    ]
    corridor = builder.build_corridor(waypoints)

    for lat, lon in waypoints:
        assert corridor.polygon_wgs84.contains(Point(lon, lat))
