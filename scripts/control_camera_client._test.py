import csv
import json
import os
import socket
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from fractions import Fraction
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

# Image capture configuration
CAPTURE_DIR = Path(
    os.getenv("CAPTURE_DIR")
    or (Path(__file__).resolve().parent / "captures")
).resolve()

# Pose sequence selection
POSE_SEQUENCE_SOURCE = os.getenv("POSE_SEQUENCE_SOURCE", "preset").strip().lower()

# Preset gimbal orientations extracted from the flight log. A per-entry ``zoom``
# field is left as ``None`` so that operators can fill in the appropriate zoom
# level when it becomes available.
GIMBAL_PHOTOS: List[Dict[str, Optional[float]]] = [
    {"pitch": -6.3, "yaw": -81.5, "yaw360": 278.5, "zoom": None},
    {"pitch": -6.3, "yaw": -81.5, "yaw360": 278.5, "zoom": None},
    {"pitch": -8.0, "yaw": -80.5, "yaw360": 279.5, "zoom": None},
    {"pitch": -8.0, "yaw": -80.5, "yaw360": 279.5, "zoom": None},
    {"pitch": -10.4, "yaw": -79.8, "yaw360": 280.2, "zoom": None},
    {"pitch": -10.4, "yaw": -79.8, "yaw360": 280.2, "zoom": None},
    {"pitch": -13.7, "yaw": -78.8, "yaw360": 281.2, "zoom": None},
    {"pitch": -13.7, "yaw": -78.8, "yaw360": 281.2, "zoom": None},
    {"pitch": -17.3, "yaw": -77.5, "yaw360": 282.5, "zoom": None},
    {"pitch": -17.3, "yaw": -77.5, "yaw360": 282.5, "zoom": None},
    {"pitch": -23.2, "yaw": -73.6, "yaw360": 286.4, "zoom": None},
    {"pitch": -23.2, "yaw": -73.6, "yaw360": 286.4, "zoom": None},
    {"pitch": -24.8, "yaw": -73.6, "yaw360": 286.4, "zoom": None},
    {"pitch": -31.4, "yaw": -68.6, "yaw360": 291.4, "zoom": None},
    {"pitch": -31.4, "yaw": -68.6, "yaw360": 291.4, "zoom": None},
    {"pitch": -44.3, "yaw": -57.3, "yaw360": 302.7, "zoom": None},
    {"pitch": -44.3, "yaw": -57.3, "yaw360": 302.7, "zoom": None},
    {"pitch": -54.9, "yaw": -37.1, "yaw360": 322.9, "zoom": None},
    {"pitch": -54.9, "yaw": -37.1, "yaw360": 322.9, "zoom": None},
    {"pitch": -54.9, "yaw": -34.0, "yaw360": 326.0, "zoom": None},
    {"pitch": -60.5, "yaw": -10.1, "yaw360": 349.9, "zoom": None},
    {"pitch": -60.5, "yaw": -10.1, "yaw360": 349.9, "zoom": None},
    {"pitch": -58.9, "yaw": 29.7, "yaw360": 29.7, "zoom": None},
    {"pitch": -58.9, "yaw": 29.7, "yaw360": 29.7, "zoom": None},
    {"pitch": -35.8, "yaw": 68.7, "yaw360": 68.7, "zoom": None},
    {"pitch": -35.8, "yaw": 68.7, "yaw360": 68.7, "zoom": None},
    {"pitch": -25.1, "yaw": 76.3, "yaw360": 76.3, "zoom": None},
    {"pitch": -25.1, "yaw": 76.3, "yaw360": 76.3, "zoom": None},
    {"pitch": -16.7, "yaw": 81.8, "yaw360": 81.8, "zoom": None},
    {"pitch": -16.7, "yaw": 81.8, "yaw360": 81.8, "zoom": None},
    {"pitch": -10.8, "yaw": 84.6, "yaw360": 84.6, "zoom": None},
    {"pitch": -10.5, "yaw": 84.6, "yaw360": 84.6, "zoom": None},
    {"pitch": -8.7, "yaw": 85.2, "yaw360": 85.2, "zoom": None},
    {"pitch": -8.7, "yaw": 85.2, "yaw360": 85.2, "zoom": None},
    {"pitch": -7.1, "yaw": 85.3, "yaw360": 85.3, "zoom": None},
    {"pitch": -7.1, "yaw": 85.3, "yaw360": 85.3, "zoom": None},
    {"pitch": -6.0, "yaw": 85.3, "yaw360": 85.3, "zoom": None},
    {"pitch": -6.0, "yaw": 85.3, "yaw360": 85.3, "zoom": None},
    {"pitch": -5.6, "yaw": 85.0, "yaw360": 85.0, "zoom": None},
    {"pitch": -5.0, "yaw": 84.8, "yaw360": 84.8, "zoom": None},
    {"pitch": -5.0, "yaw": 84.8, "yaw360": 84.8, "zoom": None},
    {"pitch": -4.4, "yaw": 84.4, "yaw360": 84.4, "zoom": None},
    {"pitch": -4.4, "yaw": 84.4, "yaw360": 84.4, "zoom": None},
    {"pitch": -4.0, "yaw": 83.9, "yaw360": 83.9, "zoom": None},
    {"pitch": -4.0, "yaw": 83.9, "yaw360": 83.9, "zoom": None},
    {"pitch": -3.7, "yaw": 83.4, "yaw360": 83.4, "zoom": None},
    {"pitch": -3.7, "yaw": 83.4, "yaw360": 83.4, "zoom": None},
    {"pitch": -3.2, "yaw": 82.3, "yaw360": 82.3, "zoom": None},
    {"pitch": -3.0, "yaw": 81.8, "yaw360": 81.8, "zoom": None},
    {"pitch": -3.0, "yaw": 81.8, "yaw360": 81.8, "zoom": None},
    {"pitch": -2.8, "yaw": 81.4, "yaw360": 81.4, "zoom": None},
    {"pitch": -2.8, "yaw": 81.4, "yaw360": 81.4, "zoom": None},
    {"pitch": -2.7, "yaw": 81.0, "yaw360": 81.0, "zoom": None},
    {"pitch": -2.7, "yaw": 81.0, "yaw360": 81.0, "zoom": None},
    {"pitch": -2.6, "yaw": 80.6, "yaw360": 80.6, "zoom": None},
    {"pitch": -2.6, "yaw": 80.6, "yaw360": 80.6, "zoom": None},
    {"pitch": -2.5, "yaw": 80.2, "yaw360": 80.2, "zoom": None},
    {"pitch": -2.5, "yaw": 80.2, "yaw360": 80.2, "zoom": None},
    {"pitch": -2.2, "yaw": 80.0, "yaw360": 80.0, "zoom": None},
    {"pitch": -2.2, "yaw": 80.0, "yaw360": 80.0, "zoom": None},
    {"pitch": -2.2, "yaw": 79.8, "yaw360": 79.8, "zoom": None},
    {"pitch": -2.2, "yaw": 79.8, "yaw360": 79.8, "zoom": None},
    {"pitch": -2.1, "yaw": 79.5, "yaw360": 79.5, "zoom": None},
    {"pitch": -2.1, "yaw": 79.5, "yaw360": 79.5, "zoom": None},
    {"pitch": -2.1, "yaw": 79.5, "yaw360": 79.5, "zoom": None},
    {"pitch": -2.0, "yaw": 79.2, "yaw360": 79.2, "zoom": None},
    {"pitch": -2.0, "yaw": 79.2, "yaw360": 79.2, "zoom": None},
    {"pitch": -1.9, "yaw": 79.1, "yaw360": 79.1, "zoom": None},
    {"pitch": -1.9, "yaw": 79.1, "yaw360": 79.1, "zoom": None},
    {"pitch": -1.8, "yaw": 79.2, "yaw360": 79.2, "zoom": None},
    {"pitch": -1.8, "yaw": 79.2, "yaw360": 79.2, "zoom": None},
]


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
    image_path: Optional[Path] = None

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


def load_camera_poses_from_log(path: Path, limit: Optional[int] = None) -> List[CameraPose]:
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


def build_preset_camera_poses(limit: Optional[int] = None) -> List[CameraPose]:
    """Convert the predefined gimbal photo list into CameraPose objects."""

    entries = GIMBAL_PHOTOS if limit is None else GIMBAL_PHOTOS[:limit]
    poses: List[CameraPose] = []
    for entry in entries:
        pitch = entry.get("pitch")
        yaw = entry.get("yaw")
        if pitch is None or yaw is None:
            continue

        zoom_value = entry.get("zoom")
        zoom: Optional[float]
        if zoom_value in {None, ""}:
            zoom = None
        else:
            try:
                zoom = float(zoom_value)
            except (TypeError, ValueError):
                print(f"Invalid zoom value in GIMBAL_PHOTOS entry: {zoom_value}")
                zoom = None

        poses.append(
            CameraPose(
                timestamp=None,
                pitch=float(pitch),
                yaw=float(yaw),
                zoom=zoom,
            )
        )
    return poses


def resolve_camera_pose_sequence(limit: Optional[int] = None) -> Tuple[List[CameraPose], str]:
    """Return the configured camera pose sequence and a description."""

    source = POSE_SEQUENCE_SOURCE
    if source == "log":
        poses = load_camera_poses_from_log(LOG_PATH, limit=limit)
        return poses, f"flight log {LOG_PATH}"

    if source != "preset":
        print(
            f"Unknown POSE_SEQUENCE_SOURCE '{source}', using preset gimbal photo list."
        )

    poses = build_preset_camera_poses(limit=limit)
    return poses, "preset gimbal photo list"


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


def fetch_position(sock: socket.socket) -> Optional[Dict[str, float]]:
    """Request the current drone position from the remote controller."""

    sock.sendall(b"GET\n")
    resp = sock.recv(1024).decode().strip()
    if not resp:
        print("Empty response received when requesting position data")
        return None
    data = parse_response(resp)
    lat = data.get("LAT") or data.get("LATITUDE")
    lon = data.get("LON") or data.get("LONGITUDE")
    if lat is None or lon is None:
        print("Coordinates missing in response:", resp)
        return None
    data["latitude"] = lat
    data["longitude"] = lon
    return data


def _decimal_to_dms_fractions(value: float) -> List[Fraction]:
    """Convert a decimal coordinate to EXIF-compatible DMS fractions."""

    abs_value = abs(value)
    degrees = int(abs_value)
    minutes_full = (abs_value - degrees) * 60
    minutes = int(minutes_full)
    seconds = round((minutes_full - minutes) * 60, 6)
    return [
        Fraction(degrees, 1),
        Fraction(minutes, 1),
        Fraction(seconds).limit_denominator(1_000_000),
    ]


def _fractions_to_bytes(values: Iterable[Fraction]) -> bytes:
    """Pack Fraction values into EXIF rational byte representation."""

    packed = []
    for value in values:
        numerator = value.numerator
        denominator = value.denominator if value.denominator else 1
        packed.append(struct.pack("<II", numerator, denominator))
    return b"".join(packed)


def _build_gps_exif_bytes(latitude: float, longitude: float) -> bytes:
    """Create an EXIF payload containing GPS metadata."""

    lat_ref = b"N\x00" if latitude >= 0 else b"S\x00"
    lon_ref = b"E\x00" if longitude >= 0 else b"W\x00"

    lat_rationals = _fractions_to_bytes(_decimal_to_dms_fractions(latitude))
    lon_rationals = _fractions_to_bytes(_decimal_to_dms_fractions(longitude))

    tiff_header = b"II*\x00\x08\x00\x00\x00"
    ifd0_count = struct.pack("<H", 1)
    ifd0_entry = struct.pack("<HHI", 0x8825, 4, 1)
    gps_ifd_offset = len(tiff_header) + len(ifd0_count) + 12 + 4
    ifd0_entry += struct.pack("<I", gps_ifd_offset)
    ifd0 = ifd0_count + ifd0_entry + struct.pack("<I", 0)

    gps_entries_metadata = [
        (1, 2, 2, lat_ref.ljust(4, b"\x00"), None),
        (2, 5, 3, lat_rationals, "lat"),
        (3, 2, 2, lon_ref.ljust(4, b"\x00"), None),
        (4, 5, 3, lon_rationals, "lon"),
    ]

    gps_entries: List[bytes] = []
    gps_data = b""
    gps_ifd_header_len = 2 + len(gps_entries_metadata) * 12 + 4
    gps_data_offset = gps_ifd_offset + gps_ifd_header_len
    for tag, type_id, count, value_bytes, data_label in gps_entries_metadata:
        if type_id == 5:
            entry = struct.pack("<HHI", tag, type_id, count)
            entry += struct.pack("<I", gps_data_offset)
            gps_entries.append(entry)
            gps_data += value_bytes
            gps_data_offset += len(value_bytes)
        else:
            entry = struct.pack("<HHI", tag, type_id, count)
            entry += value_bytes[:4]
            gps_entries.append(entry)
    gps_ifd = (
        struct.pack("<H", len(gps_entries_metadata))
        + b"".join(gps_entries)
        + struct.pack("<I", 0)
        + gps_data
    )

    exif_payload = b"Exif\x00\x00" + tiff_header + ifd0 + gps_ifd
    return exif_payload


def _strip_existing_exif(image_bytes: bytes) -> bytes:
    """Remove existing EXIF APP1 segments from a JPEG byte sequence."""

    idx = 2
    while idx + 4 <= len(image_bytes):
        if image_bytes[idx] != 0xFF:
            break
        marker = image_bytes[idx : idx + 2]
        if marker == b"\xff\xda":
            break
        length = struct.unpack(">H", image_bytes[idx + 2 : idx + 4])[0]
        if marker == b"\xff\xe1":
            segment_end = idx + 2 + length
            segment_data = image_bytes[idx + 4 : segment_end]
            if segment_data.startswith(b"Exif\x00\x00"):
                image_bytes = image_bytes[:idx] + image_bytes[segment_end:]
                continue
        idx += 2 + length
    return image_bytes


def _insert_exif_segment(image_bytes: bytes, exif_payload: bytes) -> bytes:
    """Insert an EXIF APP1 segment into a JPEG byte sequence."""

    image_bytes = _strip_existing_exif(image_bytes)
    if not image_bytes.startswith(b"\xff\xd8"):
        raise ValueError("Only JPEG images are supported for EXIF tagging")

    segment = b"\xff\xe1" + struct.pack(">H", len(exif_payload) + 2) + exif_payload
    insert_pos = 2
    idx = 2
    while idx + 4 <= len(image_bytes):
        if image_bytes[idx] != 0xFF:
            break
        marker = image_bytes[idx : idx + 2]
        if marker == b"\xff\xe0":
            length = struct.unpack(">H", image_bytes[idx + 2 : idx + 4])[0]
            idx += 2 + length
            insert_pos = idx
            continue
        if marker in {b"\xff\xe1", b"\xff\xda"}:
            break
        break
    return image_bytes[:insert_pos] + segment + image_bytes[insert_pos:]


def save_geotagged_image(
    frame,
    pose_index: int,
    measurement_time: datetime,
    latitude: float,
    longitude: float,
    detected: bool,
) -> Path:
    """Persist a captured frame to disk and embed GPS EXIF metadata."""

    try:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create capture directory {CAPTURE_DIR}: {exc}")

    timestamp_str = measurement_time.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "truck" if detected else "no-truck"
    filename = f"pose_{pose_index:04d}_{label}_{timestamp_str}.jpg"
    output_path = CAPTURE_DIR / filename

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write captured image to {output_path}")

    try:
        exif_payload = _build_gps_exif_bytes(latitude, longitude)
        image_bytes = output_path.read_bytes()
        tagged = _insert_exif_segment(image_bytes, exif_payload)
        output_path.write_bytes(tagged)
    except Exception as exc:  # noqa: BLE001 - best effort geotagging
        print(f"Failed to embed EXIF metadata for {output_path}: {exc}")

    return output_path


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
                    zoom_value = pose.zoom if pose.zoom is not None else DEFAULT_ZOOM
                    command = pose.command(DEFAULT_ZOOM)
                    sock.sendall(command.encode())
                    print(
                        f"Pose {index}: yaw={pose.yaw:.2f} pitch={pose.pitch:.2f} "
                        f"zoom={zoom_value:.2f}"
                    )

                    time.sleep(POSE_SETTLE_SECONDS)
                    # Resend only the zoom value after the gimbal settles to ensure
                    # that the H20 zoom command is applied reliably.
                    zoom_command = f"ZOOM {zoom_value:.2f}\n"
                    sock.sendall(zoom_command.encode())

                    last_frame = None
                    detected = False
                    start_time = time.time()
                    while time.time() - start_time <= DETECTION_TIMEOUT_SECONDS:
                        ret, frame = cap.read()
                        if not ret:
                            print("Failed to read frame from RTSP stream")
                            break

                        last_frame = frame

                        if detect_truck(model, frame):
                            measurement_time = datetime.now(timezone.utc)
                            position = fetch_position(sock)
                            if position is None:
                                print(
                                    "Truck detected but coordinates unavailable; "
                                    "retrying position fetch."
                                )
                                continue

                            lat = position["latitude"]
                            lon = position["longitude"]
                            image_path = save_geotagged_image(
                                frame,
                                pose_index=index,
                                measurement_time=measurement_time,
                                latitude=lat,
                                longitude=lon,
                                detected=True,
                            )

                            detection = DetectionResult(
                                pose_index=index,
                                measurement_time=measurement_time,
                                latitude=lat,
                                longitude=lon,
                                yaw=pose.yaw,
                                pitch=pose.pitch,
                                zoom=zoom_value,
                                range_m=position.get("RANGE"),
                                image_path=image_path,
                            )
                            detections.append(detection)
                            print(
                                "Detected truck:",
                                f"range {detection.range_m} m lat {lat} lon {lon}",
                                f"image {image_path}",
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
                        if last_frame is not None:
                            position = fetch_position(sock)
                            if position is None:
                                print(
                                    f"No truck detected at pose {index} and "
                                    "coordinates were unavailable"
                                )
                            else:
                                measurement_time = datetime.now(timezone.utc)
                                image_path = save_geotagged_image(
                                    last_frame,
                                    pose_index=index,
                                    measurement_time=measurement_time,
                                    latitude=position["latitude"],
                                    longitude=position["longitude"],
                                    detected=False,
                                )
                                print(
                                    f"No truck detected at pose {index}; saved image {image_path}"
                                )
                        else:
                            print(f"No truck detected at pose {index} (no frame captured)")

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
    poses, description = resolve_camera_pose_sequence(limit)
    print(f"Loaded {len(poses)} camera poses from {description}")
    process_camera_poses(poses)


if __name__ == "__main__":
    main()
