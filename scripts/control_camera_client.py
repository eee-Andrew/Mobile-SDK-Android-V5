import json
import math
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import cv2
from ultralytics import YOLO

# Positions as (zoom, pitch, yaw)
POSITIONS = [
    (20.00, -7.00, 79.80),
    (56.00, -6.50, 79.40),
    (200.00, -6.50, 79.40),
    (20.00, -6.50, 79.40),
    (56.00, -6.50, 79.40),
    (200.00, -5.70, 78.70),
    (20.00, -5.40, 78.40),
    (80.00, -5.40, 78.40),
    (200.00, -5.00, 78.00),
    (20.00, -5.00, 78.00),
    (80.00, -5.00, 78.00),
    (200.00, -4.60, 77.40),
    (14.00, -4.60, 77.40),
    (80.00, -4.40, 77.20),
    (200.00, -4.40, 77.20),
    (12.00, -4.20, 76.90),
    (80.00, -4.20, 76.90),
    (200.00, -3.90, 76.50),
    (10.00, -3.90, 76.50),
    (96.00, -3.80, 76.10),
    (200.00, -3.80, 76.10),
    (7.00, -3.60, 75.90),
    (96.00, -3.40, 75.50),
    (200.00, -3.30, 75.20),
    (7.00, -3.30, 75.20),
    (96.00, -3.20, 75.00),
    (200.00, -3.20, 75.00),
    (6.00, -3.10, 74.70),
    (112.00, -3.10, 74.50),
    (200.00, -3.00, 74.30),
    (5.00, -3.00, 74.30),
    (112.00, -2.90, 74.10),
    (200.00, -2.90, 74.10),
    (4.00, -2.80, 73.80),
    (112.00, -2.80, 73.60),
    (200.00, -2.70, 73.30),
    (4.00, -2.70, 73.30),
    (112.00, -2.60, 73.10),
    (200.00, -2.60, 73.10),
    (3.00, -2.60, 73.10),
    (128.00, -2.60, 73.10),
    (200.00, -2.50, 72.70),
    (2.00, -2.50, 72.70),
    (128.00, -2.40, 72.30),
    (200.00, -2.40, 72.30),
    (2.00, -2.40, 72.10),
    (128.00, -2.40, 72.10),
    (200.00, -2.40, 72.10),
    (2.00, -2.30, 71.70),
    (144.00, -2.20, 71.40),
    (200.00, -2.20, 71.30),
    (2.00, -2.10, 71.10),
    (144.00, -2.10, 71.10),
    (200.00, -2.00, 71.00),
    (3.00, -2.00, 71.00),
    (144.00, -2.00, 71.00),
    (200.00, -2.00, 71.00),
    (3.00, -1.90, 71.00),
    (160.00, -1.80, 70.90),
    (200.00, -1.80, 70.90),
    (4.00, -1.80, 70.90),
    (168.00, -1.80, 70.90),
    (200.00, -1.80, 71.00),
    (6.00, -1.70, 71.10),
    (168.00, -1.60, 71.40),
    (200.00, -1.60, 71.50),
    (6.00, -1.60, 71.60),
    (168.00, -1.60, 71.60),
    (200.00, -1.50, 71.80),
    (7.00, -1.50, 71.90),
    (168.00, -1.50, 71.90),
    (200.00, -1.50, 72.00),
    (8.00, -1.50, 72.00),
    (168.00, -1.50, 72.10),
    (200.00, -1.50, 72.10),
    (8.00, -1.50, 72.10),
    (168.00, -1.40, 72.20),
    (200.00, -1.40, 72.30),
    (12.00, -1.40, 72.30),
    (168.00, -1.40, 72.40),
    (200.00, -1.40, 72.40),
    (14.00, -1.40, 72.40),
    (168.00, -1.40, 72.50),
    (200.00, -1.40, 72.60),
    (24.00, -1.40, 72.60),
    (168.00, -1.30, 72.70),
    (200.00, -1.30, 72.70),
    (24.00, -1.30, 72.80),
    (176.00, -1.30, 72.80),
    (200.00, -1.20, 72.90),
    (36.00, -1.20, 72.90),
    (176.00, -1.20, 73.00),
    (200.00, -1.20, 73.10),
    (36.00, -1.20, 73.20),
    (176.00, -1.10, 73.30),
    (200.00, -1.10, 73.40),
    (36.00, -1.10, 73.50),
    (176.00, -1.10, 73.60),
    (200.00, -1.10, 73.70),
    (36.00, -1.10, 73.90),
    (200.00, -1.10, 74.00),
    (200.00, -1.10, 74.00),
    (40.00, -1.10, 74.00),
    (200.00, -1.00, 74.10),
    (200.00, -1.00, 74.20),
    (56.00, -1.00, 74.20),
]

HOST = "192.168.0.161"  # IP of the RC Plus device
PORT = 8989
RTSP_URL = "rtsp://user:192.168.0.160@192.168.0.161:8554/streaming/live/1"

IMAGE_ROOT = Path("captured_images")
METADATA_FILE = IMAGE_ROOT / "capture_log.json"
MM_OFFSET_AT_RANGE = 0.001  # 1 mm
REFERENCE_RANGE_M = 10000.0
ANGLE_OFFSET_DEG = math.degrees(math.atan(MM_OFFSET_AT_RANGE / REFERENCE_RANGE_M))
SETTLE_TIME_SEC = 1.0
FRAMES_PER_VARIANT = 5
FRAME_DELAY_SEC = 0.1

model = YOLO("best.pt")


def parse_response(resp: str) -> Dict[str, float]:
    """Parse server response into a dictionary."""
    tokens = resp.split()
    data = {}
    for i in range(0, len(tokens) - 1, 2):
        key = tokens[i]
        try:
            value = float(tokens[i + 1])
        except ValueError:
            continue
        data[key] = value
    return data


def generate_variants(zoom: float, pitch: float, yaw: float) -> Generator[Tuple[str, float, float, float], None, None]:
    """Yield (label, zoom, pitch, yaw) variants with +/- offsets."""
    offsets = [(-ANGLE_OFFSET_DEG, "minus"), (0.0, "center"), (ANGLE_OFFSET_DEG, "plus")]
    for offset, label in offsets:
        yield label, zoom, pitch + offset, yaw + offset


def ensure_image_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_metadata(entries: List[Dict[str, object]]) -> None:
    ensure_image_dir(IMAGE_ROOT)
    with METADATA_FILE.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)


def main():
    ensure_image_dir(IMAGE_ROOT)
    metadata: List[Dict[str, object]] = []
    last_detection = None
    sock = socket.create_connection((HOST, PORT))
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Failed to open RTSP stream")
        sock.close()
        return

    user_abort = False
    try:
        for idx, (zoom, pitch, yaw) in enumerate(POSITIONS):
            position_dir = IMAGE_ROOT / f"position_{idx:03d}"
            ensure_image_dir(position_dir)
            for label, z, p, y in generate_variants(zoom, pitch, yaw):
                variant_dir = position_dir / label
                ensure_image_dir(variant_dir)
                cmd = f"SET {y:.6f} {p:.6f} {z:.2f}\n"
                sock.sendall(cmd.encode())
                time.sleep(SETTLE_TIME_SEC)

                frames: List[Dict[str, object]] = []
                truck_detected = False
                for frame_idx in range(FRAMES_PER_VARIANT):
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame from RTSP stream")
                        break

                    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S_%f")
                    image_path = variant_dir / f"{timestamp}_f{frame_idx:02d}.jpg"
                    cv2.imwrite(str(image_path), frame)

                    result = model(frame, verbose=False)[0]
                    labels = [result.names[int(cls_id)] for cls_id in result.boxes.cls]
                    if any(name == "truck" for name in labels):
                        truck_detected = True

                    frames.append(
                        {
                            "frame_index": frame_idx,
                            "image_path": str(image_path),
                            "timestamp_utc": timestamp,
                            "detections": labels,
                        }
                    )

                    cv2.imshow("H20 Stream", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        user_abort = True
                        break

                    time.sleep(FRAME_DELAY_SEC)

                sock.sendall(b"GET\n")
                resp = sock.recv(1024).decode().strip()
                data = parse_response(resp)

                entry = {
                    "position_index": idx,
                    "variant": label,
                    "command": {"zoom": z, "pitch": p, "yaw": y},
                    "frames": frames,
                    "gimbal_status": data,
                    "gimbal_status_raw": resp,
                    "truck_detected": truck_detected,
                }
                metadata.append(entry)

                if truck_detected:
                    last_detection = entry
                    print(
                        f"Detected truck at position {idx} variant {label}: "
                        f"range {data.get('RANGE', -1)} m lat {data.get('LAT', 0)} lon {data.get('LON', 0)}"
                    )

                if user_abort:
                    break

            if user_abort:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        save_metadata(metadata)

    if last_detection is not None:
        status = last_detection["gimbal_status"]
        print(
            "Last truck detection:",
            f"position {last_detection['position_index']} variant {last_detection['variant']} "
            f"range {status.get('RANGE', -1)} m lat {status.get('LAT', 0)} lon {status.get('LON', 0)}",
        )
    else:
        print("No trucks detected")

if __name__ == '__main__':
    main()
