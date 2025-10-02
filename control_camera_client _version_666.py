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

# ===============================
# STEP 2: Force RTSP over TCP and tune FFmpeg buffering
# (can still be overridden by your environment)
# ===============================
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    # tcp, longer stimeout, bigger probe/analyze, reasonable max_delay & buffer
    "rtsp_transport;tcp|stimeout;20000000|probesize;5000000|analyzeduration;5000000|max_delay;5000000|buffer_size;262144"
)

# Optional: use GStreamer instead of FFmpeg if your OpenCV was built with it.
# Set USE_GSTREAMER=1 to enable. If your stream is HEVC/H.265, also set FORCE_HEVC=1.
USE_GSTREAMER = os.getenv("USE_GSTREAMER", "0") == "1"
FORCE_HEVC = os.getenv("FORCE_HEVC", "0") == "1"

# Connection and stream configuration
HOST = os.getenv("RC_HOST","192.168.1.10")
PORT = int(os.getenv("RC_PORT", "8989"))

# ===============================
# STEP 1: Use a clean RTSP URL by default (no user:pass@ in the URL)
# You can still override with RC_RTSP_URL env var.
# ===============================
RTSP_URL = os.getenv(
    "RC_RTSP_URL",
    "rtsp://user:192.168.0.160@192.168.1.10:8554/streaming/live/1",
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

try:
    _zoom_repeat = int(os.getenv("ZOOM_COMMAND_REPEAT", "2"))
except (TypeError, ValueError):
    _zoom_repeat = 2
ZOOM_COMMAND_REPEAT = max(1, _zoom_repeat)

try:
    ZOOM_COMMAND_INTERVAL = float(os.getenv("ZOOM_COMMAND_INTERVAL", "0.4"))
except (TypeError, ValueError):
    ZOOM_COMMAND_INTERVAL = 0.4

try:
    ZOOM_SETTLE_SECONDS = float(os.getenv("ZOOM_SETTLE_SECONDS", "0.5"))
except (TypeError, ValueError):
    ZOOM_SETTLE_SECONDS = 0.5

# ===== NEW: detection thresholds =====
TRUCK_MIN_COUNT = int(os.getenv("TRUCK_MIN_COUNT", "2"))          # need ≥2 trucks
TRUCK_CONF_THRESHOLD = float(os.getenv("TRUCK_CONF_THRESHOLD", "0.25"))

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

# Example manual positions as (zoom, pitch, yaw). These are used when
# POSE_SEQUENCE_SOURCE=manual.
MANUAL_POSITIONS: List[Tuple[float, float, float]] = [
    (20.0, -8.0, -80.5),
    (10.0, -25.1, 76.3),
    (5.0, -35.8, 68.7),
]

# ---------------------------
# GIMBAL_PHOTOS: keep ONLY the dict entries (removed raw lat,lon lines that broke Python)
# ---------------------------
GIMBAL_PHOTOS = [
    {"pitch": -6.3,  "yaw": -90.5, "yaw360": 283, "zoom": 35},   # pose 0
    {"pitch": -8.0,  "yaw": -80.5, "yaw360": 285.5, "zoom": 30},   # pose 1

    {"pitch": -9.4, "yaw": -80.8, "yaw360": 285, "zoom": 20},   # pose 2
    {"pitch": -13.7, "yaw": -78.8, "yaw360": 285.5, "zoom": 10},   # pose 3

    {"pitch": -17.3, "yaw": -77.5, "yaw360": 285.5, "zoom": 10},   # pose 4
    {"pitch": -23.2, "yaw": -75.6, "yaw360": 286.4, "zoom": 5},    # pose 5

    {"pitch": -24.8, "yaw": -70.6, "yaw360": 290.4, "zoom": 5},    # pose 6
    {"pitch": -31.4, "yaw": -68.6, "yaw360": 291.4, "zoom": 2},    # pose 7

    {"pitch": -44.3, "yaw": -57.3, "yaw360": 302.7, "zoom": 2},    # pose 8
    {"pitch": -54.9, "yaw": -37.1, "yaw360": 322.9, "zoom": 2},    # pose 9

    {"pitch": -57.9, "yaw": -34.0, "yaw360": 326.0, "zoom": 2},    # pose 10
    {"pitch": -60.5, "yaw": -10.1, "yaw360": 349.9, "zoom": 2},    # pose 11

    {"pitch": -60.9, "yaw": 40.7,  "yaw360": 380.7,  "zoom": 2},    # pose 12
    {"pitch": -62.9, "yaw": 65.1,  "yaw360": 390.7,  "zoom": 5},    # pose 13

    {"pitch": -59, "yaw": 76.3,  "yaw360": 405.7,  "zoom": 10},   # pose 14
    {"pitch": -52, "yaw": 81.8,  "yaw360": 420,  "zoom": 10},   # pose 15

    {"pitch": -30, "yaw": 84.6,  "yaw360": 440,  "zoom": 20},   # pose 16
    {"pitch": -25.5, "yaw": 84.6,  "yaw360": 84.6,  "zoom": 20},   # pose 17

    {"pitch": -20,  "yaw": 85.2,  "yaw360": 85.2,  "zoom": 40},   # pose 18
    {"pitch": -18,  "yaw": 85.3,  "yaw360": 87.3,  "zoom": 40},   # pose 19

    {"pitch": -15,  "yaw": 85.3,  "yaw360": 91.3,  "zoom": 40},   # pose 20
    {"pitch": -13,  "yaw": 85.0,  "yaw360": 92.0,  "zoom": 40},   # pose 21

    {"pitch": -12.0,  "yaw": 84.8,  "yaw360": 92.0,  "zoom": 40},   # pose 22
    {"pitch": -11,  "yaw": 84.4,  "yaw360": 92.0,  "zoom": 80},   # pose 23

    {"pitch": -10,  "yaw": 83.9,  "yaw360": 92,  "zoom": 80},   # pose 24
    {"pitch": -9,  "yaw": 83.4,  "yaw360": 92,  "zoom": 80},   # pose 25

    {"pitch": -8,  "yaw": 82.3,  "yaw360": 94,  "zoom": 80},   # pose 26
    {"pitch": -7,  "yaw": 81.8,  "yaw360": 94,  "zoom": 80},   # pose 27

    {"pitch": -6.5,  "yaw": 81.4,  "yaw360": 96,  "zoom": 23},   # pose 28
    {"pitch": -6,  "yaw": 81.0,  "yaw360": 96,  "zoom": 200},   # pose 29

    {"pitch": -2.6,  "yaw": 80.6,  "yaw360": 80.6,  "zoom": 160},  # pose 30
    {"pitch": -2.5,  "yaw": 80.2,  "yaw360": 80.2,  "zoom": 160},  # pose 31

    {"pitch": -2.2,  "yaw": 80.0,  "yaw360": 80.0,  "zoom": 160},  # pose 32
    {"pitch": -2.2,  "yaw": 79.8,  "yaw360": 79.8,  "zoom": 160},  # pose 33

    {"pitch": -2.1,  "yaw": 79.5,  "yaw360": 79.5,  "zoom": 160},  # pose 34
    {"pitch": -2.0,  "yaw": 79.2,  "yaw360": 79.2,  "zoom": 160},  # pose 35

    {"pitch": -1.9,  "yaw": 79.1,  "yaw360": 79.1,  "zoom": 160},  # pose 36
    {"pitch": -1.8,  "yaw": 79.2,  "yaw360": 79.2,  "zoom": 160},  # pose 37
]

# ---------------------------
# NEW: Coordinates linked to each POSE (now 1:1)
# These were previously pair-aligned; we now map direct by pose index.
# ---------------------------
PHOTO_COORDS: List[Tuple[float, float]] = [
    (45.046841, 19.115122),
    (45.046402, 19.118282),
    (45.045933, 19.120916),
    (45.045532, 19.123236),
    (45.045292, 19.124672),
    (45.045094, 19.126293),
    (45.044933, 19.127426),
    (45.044796, 19.128511),
    (45.044735, 19.129240),
    (45.044689, 19.129770),
    (45.044582, 19.130400),
    (45.044445, 19.132080),
    (45.044343, 19.133410),
    (45.044136, 19.135601),
    (45.044456, 19.146797),
    (45.044136, 19.135601),
    (45.04394158772911, 19.14139711900882),
    (45.04398493123204, 19.144309328894472),
    (45.044039114499114, 19.1470746075316),
    (45.04432723520292, 19.153625332808836),
    (45.04439416706496, 19.154301970819287),
    (45.04446428607475, 19.15530790594515),
    (45.04457265164956, 19.157031077339152),
    (45.04488181112077, 19.159841380437804),
    (45.045028422245174, 19.161081883410436),
    (45.04518459455412, 19.16256597604556),
    (45.04547144053743, 19.164830457825634),
    (45.0456849805027, 19.166273952197095),
    (45.04590808110798, 19.167749023002767),
    (45.046666598484684, 19.17245311481021),
    (45.04702120148974, 19.174550141395216),
    (45.04717444664346, 19.17532842868806),
    (45.04729163383292, 19.176078008683177),
    (45.04825424029357, 19.183190936575908),
    (45.048349850651064, 19.184661496441176),
]

def wrap_to_180(angle: float) -> float:
    """Normalize an angle to the [-180, 180] range."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    if wrapped <= -180.0:
        wrapped += 360.0
    return wrapped

def normalise_heading(angle: float) -> float:
    """Normalize an angle to the [0, 360) range."""
    return angle % 360.0

@dataclass
class CameraPose:
    """A single gimbal pose extracted from the flight record."""
    timestamp: Optional[datetime]
    pitch: float
    yaw: float
    zoom: Optional[float] = None
    absolute_yaw: Optional[float] = None

    def effective_zoom(self, default_zoom: float) -> float:
        return self.zoom if self.zoom is not None else default_zoom

    def resolve_relative_yaw(self, aircraft_yaw: Optional[float]) -> float:
        if self.absolute_yaw is not None and aircraft_yaw is not None:
            return wrap_to_180(self.absolute_yaw - aircraft_yaw)
        return self.yaw

    def command(
        self, default_zoom: float, aircraft_yaw: Optional[float] = None
    ) -> Tuple[str, float, float]:
        zoom_value = self.effective_zoom(default_zoom)
        yaw_value = self.resolve_relative_yaw(aircraft_yaw)
        command = f"SET {yaw_value:.2f} {self.pitch:.2f} {zoom_value:.2f}\n"
        return command, yaw_value, zoom_value

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
    absolute_yaw: Optional[float] = None
    range_m: Optional[float] = None
    image_path: Optional[Path] = None
    # ===== NEW: gate publishing so we don't double-post =====
    published: bool = False

    def as_payload(self) -> Dict[str, float]:
        return {
            "measurement_time": self.measurement_time.isoformat().replace(
                "+00:00", "Z"
            ),
            "last_truck_lat": self.latitude,
            "last_truck_lon": self.longitude,
        }

# Load your model
model = YOLO(r"C:\Users\User\Downloads\Mobile-SDK-Android-V5-codex-update-python-script-for-gimbal-and-truck-detection (1)\Mobile-SDK-Android-V5-codex-update-python-script-for-gimbal-and-truck-detection\best.pt")

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

        try:
            idx_yaw360 = header.index("GIMBAL.yaw [360]")
        except ValueError:
            idx_yaw360 = None

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

                absolute_yaw = None
                if idx_yaw360 is not None:
                    yaw360_value = row[idx_yaw360]
                    if yaw360_value:
                        try:
                            absolute_yaw = float(yaw360_value)
                        except ValueError:
                            absolute_yaw = None

                zoom = _extract_zoom_value(header, row, overrides, len(poses))
                poses.append(
                    CameraPose(
                        timestamp=timestamp,
                        pitch=pitch,
                        yaw=yaw,
                        zoom=zoom,
                        absolute_yaw=absolute_yaw,
                    )
                )
                if limit and len(poses) >= limit:
                    break
            prev_photo = current_photo
    return poses

def build_manual_camera_poses(limit: Optional[int] = None) -> List[CameraPose]:
    """Create camera poses from the hard-coded MANUAL_POSITIONS list."""
    entries = MANUAL_POSITIONS if limit is None else MANUAL_POSITIONS[:limit]
    poses: List[CameraPose] = []
    for zoom, pitch, yaw in entries:
        poses.append(
            CameraPose(
                timestamp=None,
                pitch=float(pitch),
                yaw=float(yaw),
                zoom=float(zoom),
            )
        )
    return poses

def build_preset_camera_poses(limit: Optional[int] = None) -> List[CameraPose]:
    """
    Convert the predefined gimbal photo list into CameraPose objects.
    Use EVERY row 1:1 (no dedup or slicing).
    """
    raw = GIMBAL_PHOTOS if limit is None else GIMBAL_PHOTOS[:limit]
    entries = raw  # ← use all rows exactly as you wrote them

    poses: List[CameraPose] = []
    for entry in entries:
        pitch = entry.get("pitch")
        yaw = entry.get("yaw")
        if pitch is None or yaw is None:
            continue

        # zoom
        zoom_value = entry.get("zoom")
        if zoom_value in {None, ""}:
            zoom: Optional[float] = None
        else:
            try:
                zoom = float(zoom_value)
            except (TypeError, ValueError):
                print(f"Invalid zoom value in GIMBAL_PHOTOS entry: {zoom_value}")
                zoom = None

        # absolute yaw (yaw360)
        absolute_value = entry.get("yaw360")
        if absolute_value in {None, ""}:
            absolute_yaw: Optional[float] = None
        else:
            try:
                absolute_yaw = float(absolute_value)
            except (TypeError, ValueError):
                print(f"Invalid yaw360 value in GIMBAL_PHOTOS entry: {absolute_value}")
                absolute_yaw = None

        poses.append(
            CameraPose(
                timestamp=None,
                pitch=float(pitch),
                yaw=float(yaw),
                zoom=zoom,
                absolute_yaw=absolute_yaw,
            )
        )
    return poses


def resolve_camera_pose_sequence(limit: Optional[int] = None) -> Tuple[List[CameraPose], str]:
    """Return the configured camera pose sequence and a description."""
    source = POSE_SEQUENCE_SOURCE
    if source == "log":
        poses = load_camera_poses_from_log(LOG_PATH, limit=limit)
        return poses, f"flight log {LOG_PATH}"

    if source == "manual":
        poses = build_manual_camera_poses(limit=limit)
        return poses, "manual POSITIONS list"

    if source != "preset":
        print(
            f"Unknown POSE_SEQUENCE_SOURCE '{source}', using preset gimbal photo list."
        )

    poses = build_preset_camera_poses(limit=limit)
    return poses, "preset gimbal photo list (deduped 1:1)"

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

def request_controller_state(
    sock: socket.socket, context: str
) -> Optional[Tuple[Dict[str, float], str]]:
    """Send a GET command and return the parsed response and raw text."""
    try:
        sock.sendall(b"GET\n")
    except OSError as exc:
        print(f"Failed to request {context}: {exc}")
        return None

    try:
        resp_bytes = sock.recv(1024)
    except OSError as exc:
        print(f"Failed to read response for {context}: {exc}")
        return None

    try:
        resp = resp_bytes.decode().strip()
    except UnicodeDecodeError:
        print(f"Failed to decode response for {context}: {resp_bytes!r}")
        return None

    if not resp:
        print(f"Empty response received when requesting {context}")
        return None

    data = parse_response(resp)
    return data, resp

def fetch_position(sock: socket.socket) -> Optional[Dict[str, float]]:
    """Request the current drone position from the remote controller."""
    state = request_controller_state(sock, "position data")
    if state is None:
        return None
    data, raw = state
    lat = data.get("LAT") or data.get("LATITUDE")
    lon = data.get("LON") or data.get("LONGITUDE")
    if lat is None or lon is None:
        print("Coordinates missing in response:", raw)
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
        if idx >= len(image_bytes) or image_bytes[idx] != 0xFF:
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

# ===== NEW: count trucks (keep your original API too) =====
def count_trucks(model: YOLO, frame, conf_threshold: Optional[float] = None) -> int:
    """Return number of 'truck' detections above confidence threshold."""
    th = TRUCK_CONF_THRESHOLD if conf_threshold is None else conf_threshold
    results = model(frame, verbose=False)[0]
    if results.boxes is None or results.boxes.cls is None:
        return 0
    names = results.names or {}
    cls_list = results.boxes.cls.tolist()
    conf_list = results.boxes.conf.tolist() if results.boxes.conf is not None else [1.0] * len(cls_list)
    count = 0
    for cls_idx, conf in zip(cls_list, conf_list):
        if names.get(int(cls_idx)) == "truck" and float(conf) >= th:
            count += 1
    return count

def detect_truck(model: YOLO, frame) -> bool:
    """Compat wrapper: 'truck present' means ≥ TRUCK_MIN_COUNT."""
    return count_trucks(model, frame) >= TRUCK_MIN_COUNT

# NEW: 1:1 mapping — NO division by 2 anymore.
def get_linked_coord_for_pose(pose_index: int) -> Optional[Tuple[float, float]]:
    """Return the lat/lon for this pose index in 1:1 mapping."""
    if 0 <= pose_index < len(PHOTO_COORDS):
        return PHOTO_COORDS[pose_index]
    return None

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

def apply_zoom_commands(sock: socket.socket, zoom_value: float) -> None:
    """Send one or more ZOOM commands to enforce the requested magnification."""
    command = f"ZOOM {zoom_value:.2f}\n".encode()
    repeats = ZOOM_COMMAND_REPEAT
    for attempt in range(repeats):
        sock.sendall(command)
        if attempt + 1 < repeats and ZOOM_COMMAND_INTERVAL > 0:
            time.sleep(ZOOM_COMMAND_INTERVAL)

# ---------- ONLY THIS FUNCTION CHANGED (robust GStreamer path) ----------
def _open_capture():
    """Open the RTSP stream via GStreamer (recommended) or FFmpeg."""
    if USE_GSTREAMER:
        # Choose depay/parse/decoder by codec
        if FORCE_HEVC:
            depay = "rtph265depay ! h265parse ! avdec_h265"
        else:
            depay = "rtph264depay ! h264parse ! avdec_h264"

        # Larger latency buffers and reorders RTP; drop-on-latency prevents stalls
        gst = (
            f"rtspsrc location={RTSP_URL} protocols=tcp latency=350 drop-on-latency=true ! "
            f"{depay} ! videoconvert ! "
            "appsink sync=false emit-signals=false max-buffers=1 drop=true"
        )
        return cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    else:
        # FFmpeg fallback—small buffer helps keep it live
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
# -----------------------------------------------------------------------

def _read_stable_frame(cap, warmup=60):
    """Throw away initial frames to reach a keyframe/steady state."""
    ok = False
    frame = None
    for _ in range(warmup):
        ok, frame = cap.read()
        if not ok:
            continue
    return ok, frame

def process_camera_poses(poses: Iterable[CameraPose]) -> None:
    """Automatic sweep: go through each pose, detect, and publish with gating."""
    poses = list(poses)
    if not poses:
        print("No camera poses available to process.")
        return

    detections: List[DetectionResult] = []   # store per-pose detection result (if any)
    has_trucks: List[bool] = []              # per-pose gate result (≥ TRUCK_MIN_COUNT)
    result_index_by_pose: List[Optional[int]] = []  # map pose -> detections[]

    cap = None
    try:
        with socket.create_connection((HOST, PORT)) as sock:
            cap = _open_capture()
            if not cap.isOpened():
                print("Failed to open RTSP stream")
                return

            # Initial warmup
            _read_stable_frame(cap, warmup=60)

            # Helper: reconnect on broken frames
            def reopen_capture(current_zoom: float):
                nonlocal cap
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(0.5)
                cap = _open_capture()
                _read_stable_frame(cap, warmup=60)
                apply_zoom_commands(sock, current_zoom)
                if ZOOM_SETTLE_SECONDS > 0:
                    time.sleep(ZOOM_SETTLE_SECONDS)
                _read_stable_frame(cap, warmup=30)

            for index, pose in enumerate(poses):
                aircraft_yaw: Optional[float] = None
                aircraft_heading: Optional[float] = None
                raw_attitude: Optional[str] = None
                if pose.absolute_yaw is not None:
                    state = request_controller_state(sock, "aircraft attitude")
                    if state is None:
                        print(
                            f"Pose {index}: failed to fetch aircraft yaw; "
                            "using stored relative yaw."
                        )
                    else:
                        attitude_data, raw_attitude = state
                        aircraft_yaw = attitude_data.get("YAW")
                        if aircraft_yaw is None:
                            heading_value = attitude_data.get("HEADING") or attitude_data.get("HDG")
                            if heading_value is not None:
                                aircraft_yaw = wrap_to_180(heading_value)
                        if aircraft_yaw is not None:
                            aircraft_heading = normalise_heading(aircraft_yaw)
                        else:
                            print(
                                f"Pose {index}: controller response missing yaw/heading; "
                                f"raw='{raw_attitude}'. Using stored relative yaw."
                            )

                command_text, command_yaw, zoom_value = pose.command(
                    DEFAULT_ZOOM, aircraft_yaw
                )
                sock.sendall(command_text.encode())

                desired_heading = (
                    normalise_heading(pose.absolute_yaw)
                    if pose.absolute_yaw is not None
                    else None
                )
                if desired_heading is not None:
                    if aircraft_heading is not None:
                        print(
                            f"Pose {index}: yaw={command_yaw:.2f} pitch={pose.pitch:.2f} "
                            f"zoom={zoom_value:.2f} (abs={desired_heading:.2f}°, "
                            f"aircraft={aircraft_heading:.2f}°)"
                        )
                    else:
                        print(
                            f"Pose {index}: yaw={command_yaw:.2f} pitch={pose.pitch:.2f} "
                            f"zoom={zoom_value:.2f} (abs={desired_heading:.2f}°, aircraft=unknown)"
                        )
                else:
                    print(
                        f"Pose {index}: yaw={command_yaw:.2f} pitch={pose.pitch:.2f} "
                        f"zoom={zoom_value:.2f}"
                    )

                # Wait after gimbal move, then enforce zoom and discard initial frames
                time.sleep(POSE_SETTLE_SECONDS)
                apply_zoom_commands(sock, zoom_value)
                if ZOOM_SETTLE_SECONDS > 0:
                    time.sleep(ZOOM_SETTLE_SECONDS)
                _read_stable_frame(cap, warmup=60)

                last_frame = None
                pose_has_trucks = False
                pose_detection_idx: Optional[int] = None

                start_time = time.time()
                while time.time() - start_time <= DETECTION_TIMEOUT_SECONDS:
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        print("Failed to read frame from RTSP stream; reopening…")
                        reopen_capture(zoom_value)
                        ret, frame = cap.read()
                        if not ret or frame is None or frame.size == 0:
                            print("Still failing after reopen.")
                            break

                    last_frame = frame

                    # Require ≥ TRUCK_MIN_COUNT trucks
                    n_trucks = count_trucks(model, frame)
                    if n_trucks >= TRUCK_MIN_COUNT:
                        measurement_time = datetime.now(timezone.utc)

                        # ===== Use linked coordinates (1:1). Fallback to controller.
                        coord = get_linked_coord_for_pose(index)
                        range_value: Optional[float] = None
                        if coord is not None:
                            lat, lon = coord
                        else:
                            position = fetch_position(sock)
                            if position is None:
                                print(
                                    "Trucks detected but coordinates unavailable; retrying position fetch."
                                )
                                continue
                            lat = position["latitude"]
                            lon = position["longitude"]
                            range_value = position.get("RANGE")

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
                            yaw=command_yaw,
                            pitch=pose.pitch,
                            zoom=zoom_value,
                            absolute_yaw=pose.absolute_yaw,
                            range_m=range_value,
                            image_path=image_path,
                        )
                        detections.append(detection)
                        pose_detection_idx = len(detections) - 1
                        pose_has_trucks = True
                        print(
                            f"Detected {n_trucks} trucks:",
                            f"lat {lat} lon {lon}",
                            f"zoom {zoom_value:.2f}",
                            f"image {image_path}",
                        )
                        break  # positive; no need to keep scanning

                    if DISPLAY_PREVIEW:
                        cv2.imshow("H20 Stream", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                if DISPLAY_PREVIEW:
                    cv2.waitKey(1)

                # Save a 'no truck' frame for record keeping
                if not pose_has_trucks:
                    if last_frame is not None:
                        position = fetch_position(sock)
                        if position is None:
                            print(
                                f"No trucks detected at pose {index} (zoom={zoom_value:.2f}) and "
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
                                f"No trucks detected at pose {index}; "
                                f"zoom={zoom_value:.2f}; saved image {image_path}"
                            )
                    else:
                        print(
                            f"No trucks detected at pose {index} "
                            f"(zoom={zoom_value:.2f}, no frame captured)"
                        )

                # Track per-pose result for the gating logic
                has_trucks.append(pose_has_trucks)
                result_index_by_pose.append(pose_detection_idx)

                # Publish gate — publish pose i-2 if i-1 and i have NO trucks
                if index >= 2:
                    if has_trucks[index - 2] and not has_trucks[index - 1] and not has_trucks[index]:
                        idx_to_publish = result_index_by_pose[index - 2]
                        if idx_to_publish is not None:
                            det = detections[idx_to_publish]
                            if not det.published:
                                ok = publish_detection(det)
                                det.published = ok

                time.sleep(BETWEEN_POSE_PAUSE_SECONDS)
    finally:
        if 'cap' in locals() and cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if DISPLAY_PREVIEW:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

def process_camera_poses_interactive(poses: Iterable[CameraPose]) -> None:
    """
    Interactive mode: type a pose number (e.g., 33) to aim the gimbal to that pose.
    Type 'q' or 'quit' to exit. Press Enter with no input to repeat the last pose.
    Saves truck/no-truck frames exactly like the automatic mode.
    """
    poses = list(poses)
    if not poses:
        print("No camera poses available to process (interactive).")
        return

    detections: List[DetectionResult] = []

    cap = None
    try:
        with socket.create_connection((HOST, PORT)) as sock:
            cap = _open_capture()
            if not cap.isOpened():
                print("Failed to open RTSP stream (interactive)")
                return

            # small warmup
            _read_stable_frame(cap, warmup=60)

            def reopen_capture(current_zoom: float):
                nonlocal cap
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(0.5)
                cap = _open_capture()
                _read_stable_frame(cap, warmup=60)
                apply_zoom_commands(sock, current_zoom)
                if ZOOM_SETTLE_SECONDS > 0:
                    time.sleep(ZOOM_SETTLE_SECONDS)
                _read_stable_frame(cap, warmup=30)

            print(
                f"[Interactive] Loaded {len(poses)} poses. "
                "Type a pose number (0-based or 1-based), Enter to repeat last, or 'q' to quit."
            )
            last_index: Optional[int] = None

            while True:
                try:
                    raw = input("Pose #: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting interactive mode.")
                    break

                if raw in {"q", "quit", "exit"}:
                    print("Goodbye.")
                    break

                if raw == "":
                    if last_index is None:
                        print("No previous pose to repeat.")
                        continue
                    index = last_index
                else:
                    try:
                        n = int(raw)
                    except ValueError:
                        print("Please type an integer pose index or 'q' to quit.")
                        continue

                    # accept both 0-based and 1-based
                    if 0 <= n < len(poses):
                        index = n
                    elif 1 <= n <= len(poses):
                        index = n - 1
                        print(f"(Interpreting {n} as 1-based → using index {index})")
                    else:
                        print(f"Index out of range. Valid: 0..{len(poses)-1} (or 1..{len(poses)})")
                        continue

                pose = poses[index]
                last_index = index

                # ---- resolve aircraft yaw if absolute heading provided ----
                aircraft_yaw: Optional[float] = None
                aircraft_heading: Optional[float] = None
                raw_attitude: Optional[str] = None
                if pose.absolute_yaw is not None:
                    state = request_controller_state(sock, "aircraft attitude")
                    if state is not None:
                        attitude_data, raw_attitude = state
                        aircraft_yaw = attitude_data.get("YAW")
                        if aircraft_yaw is None:
                            heading_value = attitude_data.get("HEADING") or attitude_data.get("HDG")
                            if heading_value is not None:
                                aircraft_yaw = wrap_to_180(heading_value)
                        if aircraft_yaw is not None:
                            aircraft_heading = normalise_heading(aircraft_yaw)

                # ---- build and send gimbal command ----
                command_text, command_yaw, zoom_value = pose.command(DEFAULT_ZOOM, aircraft_yaw)
                try:
                    sock.sendall(command_text.encode())
                except OSError as exc:
                    print(f"Failed to send SET command: {exc}")
                    continue

                desired_heading = (
                    normalise_heading(pose.absolute_yaw)
                    if pose.absolute_yaw is not None
                    else None
                )
                if desired_heading is not None:
                    if aircraft_heading is not None:
                        print(
                            f"[Pose {index}] yaw={command_yaw:.2f} pitch={pose.pitch:.2f} "
                            f"zoom={zoom_value:.2f} (abs={desired_heading:.2f}°, "
                            f"aircraft={aircraft_heading:.2f}°)"
                        )
                    else:
                        print(
                            f"[Pose {index}] yaw={command_yaw:.2f} pitch={pose.pitch:.2f} "
                            f"zoom={zoom_value:.2f} (abs={desired_heading:.2f}°, aircraft=unknown)"
                        )
                else:
                    print(
                        f"[Pose {index}] yaw={command_yaw:.2f} pitch={pose.pitch:.2f} "
                        f"zoom={zoom_value:.2f}"
                    )

                # settle + enforce zoom
                time.sleep(POSE_SETTLE_SECONDS)
                apply_zoom_commands(sock, zoom_value)
                if ZOOM_SETTLE_SECONDS > 0:
                    time.sleep(ZOOM_SETTLE_SECONDS)
                _read_stable_frame(cap, warmup=60)

                # ---- grab frames, run detector, save images like before ----
                last_frame = None
                pose_has_trucks = False

                start_time = time.time()
                while time.time() - start_time <= DETECTION_TIMEOUT_SECONDS:
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        print("Failed to read frame from RTSP stream; reopening…")
                        reopen_capture(zoom_value)
                        ret, frame = cap.read()
                        if not ret or frame is None or frame.size == 0:
                            print("Still failing after reopen.")
                            break

                    last_frame = frame

                    n_trucks = count_trucks(model, frame)
                    if n_trucks >= TRUCK_MIN_COUNT:
                        measurement_time = datetime.now(timezone.utc)

                        coord = get_linked_coord_for_pose(index)
                        range_value: Optional[float] = None
                        if coord is not None:
                            lat, lon = coord
                        else:
                            position = fetch_position(sock)
                            if position is None:
                                print("Trucks detected but coordinates unavailable; retrying…")
                                continue
                            lat = position["latitude"]
                            lon = position["longitude"]
                            range_value = position.get("RANGE")

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
                            yaw=command_yaw,
                            pitch=pose.pitch,
                            zoom=zoom_value,
                            absolute_yaw=pose.absolute_yaw,
                            range_m=range_value,
                            image_path=image_path,
                        )
                        detections.append(detection)
                        pose_has_trucks = True
                        print(
                            f"[Pose {index}] Detected {n_trucks} trucks:",
                            f"lat {lat} lon {lon}",
                            f"zoom {zoom_value:.2f}",
                            f"image {image_path}",
                        )
                        # Interactive mode: publish immediately (no 3-pose gate)
                        if publish_detection(detection):
                            detection.published = True
                        break

                    if DISPLAY_PREVIEW:
                        cv2.imshow("H20 Stream", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Preview quit requested.")
                            break

                if DISPLAY_PREVIEW:
                    cv2.waitKey(1)

                # Save a 'no truck' frame as record (same as automatic)
                if not pose_has_trucks:
                    if last_frame is not None:
                        position = fetch_position(sock)
                        if position is None:
                            print(
                                f"[Pose {index}] No trucks and coordinates unavailable; "
                                "skipping save."
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
                                f"[Pose {index}] No trucks; zoom={zoom_value:.2f}; "
                                f"saved image {image_path}"
                            )
                    else:
                        print(
                            f"[Pose {index}] No trucks (zoom={zoom_value:.2f}, no frame captured)"
                        )

    finally:
        if 'cap' in locals() and cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if DISPLAY_PREVIEW:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

def main() -> None:
    limit = MAX_POSES if MAX_POSES > 0 else None
    poses, description = resolve_camera_pose_sequence(limit)
    print(f"Loaded {len(poses)} camera poses from {description}")

    # Choose interactive-by-number or the old automatic sweep via env var
    # INTERACTIVE_INPUT=1 (default) -> manual by typing pose numbers
    # INTERACTIVE_INPUT=0 -> old automatic process
    if os.getenv("INTERACTIVE_INPUT", "1") == "1":
        process_camera_poses_interactive(poses)
    else:
        process_camera_poses(poses)

if __name__ == "__main__":
    main()
