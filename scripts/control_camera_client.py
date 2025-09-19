import csv
import json
import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import cv2
from ultralytics import YOLO

# Connection and stream configuration
HOST = os.getenv("RC_HOST", "192.168.0.161")
PORT = int(os.getenv("RC_PORT", "8989"))
RTSP_URL = os.getenv(
    "RC_RTSP_URL",
    "rtsp://user:192.168.0.160@192.168.0.161:8554/streaming/live/1",
)

# Flight log used to replay gimbal poses
_DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parent.parent
    / "DJIFlightRecord_2025-08-25_[12-41-47].csv"
)
LOG_PATH = Path(os.getenv("GIMBAL_LOG_PATH", _DEFAULT_LOG_PATH))
ZOOM_COLUMN_NAME = os.getenv("ZOOM_COLUMN_NAME")
POSE_OVERRIDES_PATH = os.getenv("POSE_OVERRIDES_PATH")
MAX_POSES = int(os.getenv("MAX_POSES", "0"))  # 0 = use all poses

# Behaviour tuning
DEFAULT_ZOOM = float(os.getenv("DEFAULT_ZOOM", "1.0"))
POSE_SETTLE_SECONDS = float(os.getenv("POSE_SETTLE_SECONDS", "1.5"))
DETECTION_TIMEOUT_SECONDS = float(os.getenv("DETECTION_TIMEOUT_SECONDS", "4.0"))
BETWEEN_POSE_PAUSE_SECONDS = float(
    os.getenv("BETWEEN_POSE_PAUSE_SECONDS", "0.75")
)
DISPLAY_PREVIEW = os.getenv("DISPLAY_PREVIEW", "0") == "1"

# Queue-length API configuration
QUEUE_API_URL = os.getenv(
    "QUEUE_API_URL", "http://143.198.57.23:8100/api/v1/queue-length"
)
QUEUE_API_TOKEN = os.getenv("QUEUE_API_TOKEN", "DUTH_SPATRA")
QUEUE_API_ENABLED = os.getenv("QUEUE_API_ENABLED", "1") != "0"
QUEUE_API_TIMEOUT = float(os.getenv("QUEUE_API_TIMEOUT", "10"))


@dataclass
class CameraPose:
    """A single gimbal pose extracted from the flight record."""

    timestamp: Optional[datetime]
    pitch: float
    yaw: float
    zoom: Optional[float] = None

    def command(self, default_zoom: float) -> str:
        zoom_value = self.zoom if self.zoom is not None else default_zoom
        return f"SET {self.yaw:.2f} {self.pitch:.2f} {zoom_value:.2f}\n"


@dataclass
class DetectionResult:
    """Information about a truck detection event."""

    pose_index: int
    measurement_time: datetime
    latitude: float
    longitude: float
    yaw: float
    pitch: float
    zoom: float
    range_m: Optional[float] = None

    def as_payload(self) -> Dict[str, float]:
        return {
            "measurement_time": self.measurement_time.isoformat().replace(
                "+00:00", "Z"
            ),
            "last_truck_lat": self.latitude,
            "last_truck_lon": self.longitude,
        }


model = YOLO("best.pt")


def _normalise_time_string(time_str: str) -> str:
    """Pad fractional seconds so that datetime.strptime can parse them."""

    if "." not in time_str:
        return time_str
    body, suffix = time_str.split(".", 1)
    if " " in suffix:
        fraction, ampm = suffix.split(" ", 1)
    else:
        fraction, ampm = suffix, ""
    fraction = (fraction + "000000")[:6]
    return f"{body}.{fraction} {ampm}".strip()


def parse_log_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """Return a naive datetime parsed from the DJI CSV timestamps."""

    if not date_str or not time_str:
        return None
    normalised_time = _normalise_time_string(time_str)
    try:
        return datetime.strptime(
            f"{date_str} {normalised_time}", "%m/%d/%Y %I:%M:%S.%f %p"
        )
    except ValueError:
        return None


def _load_pose_overrides(path: Path) -> Dict[int, Dict[str, float]]:
    """Load per-index pose overrides from a JSON document."""

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    overrides: Dict[int, Dict[str, float]] = {}
    if isinstance(data, list):
        for index, value in enumerate(data):
            if isinstance(value, dict):
                overrides[index] = value
    elif isinstance(data, dict):
        for key, value in data.items():
            try:
                overrides[int(key)] = value
            except (TypeError, ValueError):
                continue
    return overrides


def _extract_zoom_value(
    header: List[str],
    row: List[str],
    overrides: Optional[Dict[int, Dict[str, float]]],
    pose_index: int,
) -> Optional[float]:
    """Return a zoom value for the pose, using overrides when available."""

    if overrides and pose_index in overrides:
        zoom_value = overrides[pose_index].get("zoom")
        if zoom_value is not None:
            return float(zoom_value)

    if ZOOM_COLUMN_NAME:
        try:
            index = header.index(ZOOM_COLUMN_NAME)
        except ValueError:
            pass
        else:
            value = row[index]
            if value:
                try:
                    return float(value)
                except ValueError:
                    return None

    for candidate, column_name in enumerate(header):
        lowered = column_name.lower()
        if "zoom" in lowered and "is" not in lowered:
            value = row[candidate]
            if value:
                try:
                    return float(value)
                except ValueError:
                    continue
    return None


def load_camera_poses(path: Path, limit: Optional[int] = None) -> List[CameraPose]:
    """Parse the flight record and build an ordered list of gimbal poses."""

    if not path.exists():
        raise FileNotFoundError(f"Flight log not found: {path}")

    overrides = None
    if POSE_OVERRIDES_PATH:
        override_path = Path(POSE_OVERRIDES_PATH)
        if override_path.exists():
            overrides = _load_pose_overrides(override_path)

    poses: List[CameraPose] = []
    with path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        try:
            next(reader)  # Skip sep= line
        except StopIteration:
            return poses
        try:
            header = next(reader)
        except StopIteration:
            return poses

        header = [item.strip() for item in header]
        try:
            idx_photo = header.index("CAMERA.isPhoto")
            idx_date = header.index("CUSTOM.date [local]")
            idx_time = header.index("CUSTOM.updateTime [local]")
            idx_pitch = header.index("GIMBAL.pitch")
            idx_yaw = header.index("GIMBAL.yaw")
        except ValueError as exc:
            raise RuntimeError("Expected columns were not found in the CSV") from exc

        prev_photo = False
        for pose_index, row in enumerate(reader):
            current_photo = row[idx_photo] == "True"
            if current_photo and not prev_photo:
                timestamp = parse_log_datetime(row[idx_date], row[idx_time])
                try:
                    pitch = float(row[idx_pitch])
                    yaw = float(row[idx_yaw])
                except ValueError:
                    prev_photo = current_photo
                    continue

                zoom = _extract_zoom_value(header, row, overrides, len(poses))
                poses.append(CameraPose(timestamp=timestamp, pitch=pitch, yaw=yaw, zoom=zoom))
                if limit and len(poses) >= limit:
                    break
            prev_photo = current_photo
    return poses


def parse_response(resp: str) -> Dict[str, float]:
    """Parse server response into a dictionary of floats."""

    tokens = resp.split()
    data: Dict[str, float] = {}
    for i in range(0, len(tokens) - 1, 2):
        key = tokens[i]
        try:
            value = float(tokens[i + 1])
        except ValueError:
            continue
        data[key] = value
    return data


def detect_truck(model: YOLO, frame) -> bool:
    """Return True if a truck is detected in the provided frame."""

    results = model(frame, verbose=False)[0]
    if results.boxes is None or results.boxes.cls is None:
        return False
    names = results.names or {}
    for cls_idx in results.boxes.cls.tolist():
        if names.get(int(cls_idx)) == "truck":
            return True
    return False


def publish_detection(result: DetectionResult) -> bool:
    """Send a detection result to the queue-length API."""

    if not QUEUE_API_ENABLED:
        print(
            "Queue-length API disabled; skipping publish for",
            f"pose {result.pose_index} at {result.latitude}, {result.longitude}",
        )
        return False

    payload = json.dumps(result.as_payload()).encode("utf-8")
    request = Request(
        QUEUE_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {QUEUE_API_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=QUEUE_API_TIMEOUT) as response:
            response_body = response.read().decode("utf-8")
            print(
                "Published detection for pose",
                result.pose_index,
                "response:",
                response_body,
            )
            return True
    except HTTPError as exc:
        print("Failed to publish detection (HTTP error)", exc)
    except URLError as exc:
        print("Failed to publish detection (connection error)", exc)
    except TimeoutError:
        print("Failed to publish detection (timeout)")
    return False


def process_camera_poses(poses: Iterable[CameraPose]) -> None:
    """Replay each gimbal pose, run detection, and publish results."""

    poses = list(poses)
    if not poses:
        print("No camera poses available to process.")
        return

    detections: List[DetectionResult] = []
    try:
        with socket.create_connection((HOST, PORT)) as sock:
            cap = cv2.VideoCapture(RTSP_URL)
            if not cap.isOpened():
                print("Failed to open RTSP stream")
                return
            try:
                for index, pose in enumerate(poses):
                    command = pose.command(DEFAULT_ZOOM)
                    sock.sendall(command.encode())
                    print(
                        f"Pose {index}: yaw={pose.yaw:.2f} pitch={pose.pitch:.2f} "
                        f"zoom={(pose.zoom if pose.zoom is not None else DEFAULT_ZOOM):.2f}"
                    )

                    time.sleep(POSE_SETTLE_SECONDS)

                    detected = False
                    start_time = time.time()
                    while time.time() - start_time <= DETECTION_TIMEOUT_SECONDS:
                        ret, frame = cap.read()
                        if not ret:
                            print("Failed to read frame from RTSP stream")
                            break

                        if detect_truck(model, frame):
                            measurement_time = datetime.now(timezone.utc)
                            sock.sendall(b"GET\n")
                            resp = sock.recv(1024).decode().strip()
                            data = parse_response(resp)
                            lat = data.get("LAT") or data.get("LATITUDE")
                            lon = data.get("LON") or data.get("LONGITUDE")
                            if lat is None or lon is None:
                                print("Truck detected but coordinates missing in response:", resp)
                                break

                            detection = DetectionResult(
                                pose_index=index,
                                measurement_time=measurement_time,
                                latitude=lat,
                                longitude=lon,
                                yaw=pose.yaw,
                                pitch=pose.pitch,
                                zoom=pose.zoom if pose.zoom is not None else DEFAULT_ZOOM,
                                range_m=data.get("RANGE"),
                            )
                            detections.append(detection)
                            print(
                                "Detected truck:",
                                f"range {detection.range_m} m lat {lat} lon {lon}",
                            )
                            detected = True
                            break

                        if DISPLAY_PREVIEW:
                            cv2.imshow("H20 Stream", frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                detected = True
                                break
                    if DISPLAY_PREVIEW:
                        cv2.waitKey(1)

                    if not detected:
                        print(f"No truck detected at pose {index}")

                    time.sleep(BETWEEN_POSE_PAUSE_SECONDS)
            finally:
                cap.release()
                if DISPLAY_PREVIEW:
                    cv2.destroyAllWindows()
    finally:
        if detections:
            for detection in detections:
                publish_detection(detection)
        else:
            print("No trucks detected during the mission.")


def main() -> None:
    limit = MAX_POSES if MAX_POSES > 0 else None
    poses = load_camera_poses(LOG_PATH, limit=limit)
    print(f"Loaded {len(poses)} camera poses from {LOG_PATH}")
    process_camera_poses(poses)


if __name__ == "__main__":
    main()
