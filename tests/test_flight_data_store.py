from pathlib import Path
import json
import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("PIL")
pytest.importorskip("transformers")
pytest.importorskip("ultralytics")

from scripts.autonomous_highway_monitor import ESP32FlightDataStore, HighwayMonitor, TruckDetection


def write_samples(tmp_path: Path) -> Path:
    samples = [
        {"utc_time": "2024-05-01T12:00:00Z", "latitude": 42.0, "longitude": -71.0},
        {"utc_time": "2024-05-01T12:00:01Z", "latitude": 42.1, "longitude": -71.1, "altitude": 10},
    ]
    log_path = tmp_path / "flight_log.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample) + "\n")
    return log_path


def test_store_reads_jsonl(tmp_path):
    log_path = write_samples(tmp_path)
    store = ESP32FlightDataStore(log_path)
    samples = store.refresh()
    assert len(samples) == 2
    assert store.latest()["latitude"] == 42.1


def test_photo_metadata_prefers_esp32(tmp_path):
    # Minimal gimbal + detector setup is not required for metadata builder
    log_path = write_samples(tmp_path)
    store = ESP32FlightDataStore(log_path)
    store.refresh()

    dummy_truck = TruckDetection(bbox=(0, 0, 10, 10), confidence=0.9, zoom_level=2.0, segment_index=0)
    monitor = HighwayMonitor(
        gimbal=None,  # type: ignore
        road_detector=None,  # type: ignore
        truck_detector=None,  # type: ignore
        stream=None,  # type: ignore
        log_file=tmp_path / "log.json",
        esp32_store=store,
        uploader=None,
    )
    monitor.last_truck = dummy_truck
    monitor.telemetry = {"lat": 0.0, "lon": 0.0}
    monitor.esp32_samples = store.samples
    meta = monitor._build_photo_metadata()
    assert meta["gps"]["source"] == "ESP32"
    assert meta["gps"]["lat"] == 42.1
