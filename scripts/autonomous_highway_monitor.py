#!/usr/bin/env python3
"""Autonomous highway monitoring workflow for RTSP stream + gimbal control.

The script coordinates three subsystems:

1.  ``GimbalClient`` communicates with the TCP control server that ships with the
    DJI Android sample app.  It is responsible for orienting the camera, issuing
    zoom commands and capturing telemetry (range finder, GPS, etc.).
2.  ``RoadDetector`` wraps a CLIPSeg segmentation model so the camera can stay
    aligned with the highway.  Once a road is detected, the class derives the
    road axis and splits it into analysis segments.
3.  ``TruckDetector`` (YOLO based) scans each segment for trucks.

The whole workflow is orchestrated by :class:`HighwayMonitor` which implements
all of the steps described in the prompt: orienting the gimbal, rotating to the
highway start, zooming, scanning segment-by-segment, remembering the most recent
truck, and finally taking a photo with the proper framing.

Example usage::

    python scripts/autonomous_highway_monitor.py \
        --host 192.168.0.161 \
        --port 8989 \
        --rtsp rtsp://user:192.168.0.160@192.168.0.161:8554/streaming/live/1 \
        --yolo-weights best.pt \
        --log-file highway_log.json

The implementation logs every decision so you can audit detections afterwards.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import socket
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Utility dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RoadDetectionResult:
    mask: np.ndarray
    confidence: float
    axis_info: Optional[Dict]

    @property
    def found(self) -> bool:
        return self.axis_info is not None and self.confidence >= 0.35


@dataclasses.dataclass
class TruckDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    zoom_level: float
    segment_index: int

    def to_json(self) -> Dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": float(self.confidence),
            "zoom": float(self.zoom_level),
            "segment": int(self.segment_index),
        }


@dataclasses.dataclass
class SegmentLog:
    index: int
    zoom_commands: List[float]
    trucks: List[TruckDetection]
    status: str = "UNPROCESSED"

    def to_json(self) -> Dict:
        return {
            "index": self.index,
            "zoom_commands": self.zoom_commands,
            "status": self.status,
            "truck_detections": [t.to_json() for t in self.trucks],
        }


# ---------------------------------------------------------------------------
# Socket helper
# ---------------------------------------------------------------------------


class GimbalClient:
    """Tiny helper that speaks the TCP control protocol from the sample app."""

    def __init__(self, host: str, port: int, timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.current_zoom = 1.0
        self.current_pitch = 90.0
        self.current_yaw = 0.0

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Command helpers
    # ------------------------------------------------------------------
    def send_command(self, cmd: str) -> str:
        logging.debug("TX: %s", cmd.strip())
        self.sock.sendall(cmd.encode("ascii"))
        resp = self.sock.recv(1024).decode("ascii", errors="ignore").strip()
        logging.debug("RX: %s", resp)
        return resp

    def set_orientation(self, yaw: float, pitch: float, zoom: float) -> None:
        self.current_yaw = yaw
        self.current_pitch = pitch
        self.current_zoom = zoom
        self.send_command(f"SET {yaw:.2f} {pitch:.2f} {zoom:.2f}\n")

    def scan_left(self, degrees: float, step: float = 2.0) -> None:
        target = self.current_yaw - degrees
        while self.current_yaw > target:
            self.current_yaw -= step
            self.send_command(f"SET {self.current_yaw:.2f} {self.current_pitch:.2f} {self.current_zoom:.2f}\n")
            time.sleep(0.25)

    def set_zoom(self, zoom: float) -> None:
        self.current_zoom = zoom
        self.send_command(f"SET {self.current_yaw:.2f} {self.current_pitch:.2f} {zoom:.2f}\n")

    def capture_photo(self) -> str:
        return self.send_command("TAKE_PHOTO\n")

    def read_telemetry(self) -> Dict[str, float]:
        resp = self.send_command("GET\n")
        tokens = resp.split()
        data: Dict[str, float] = {}
        for idx in range(0, len(tokens) - 1, 2):
            key = tokens[idx]
            try:
                data[key] = float(tokens[idx + 1])
            except ValueError:
                continue
        return data


# ---------------------------------------------------------------------------
# Road detection (CLIPSeg wrapper)
# ---------------------------------------------------------------------------


class ClipSegRoad:
    def __init__(self, prompts: Sequence[str] = ("Highway",)) -> None:
        self.prompts = list(prompts)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(self.device)
        if self.device.type == "cuda":
            self.model.half()
        self.model.eval()

    @torch.no_grad()
    def predict(self, roi: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=self.prompts, images=[pil] * len(self.prompts), return_tensors="pt").to(self.device)
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(**inputs).logits
        else:
            logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits)
        prob = torch.max(probs, dim=0).values.detach().float().cpu().numpy()
        prob = cv2.resize(prob, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)
        return prob


class RoadDetector:
    def __init__(
        self,
        smooth_window: int = 5,
        threshold: float = 0.35,
        ignore_border_pct: float = 0.1,
        frame_skip: int = 2,
    ) -> None:
        self.threshold = threshold
        self.ignore_border_pct = ignore_border_pct
        self.frame_skip = frame_skip
        self.mask_history: Deque[np.ndarray] = deque(maxlen=smooth_window)
        self.model = ClipSegRoad()
        self.frame_index = 0
        self.last_roi_mask: Optional[np.ndarray] = None
        self.last_prob: Optional[np.ndarray] = None

    def _extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        border_w = int(w * self.ignore_border_pct)
        border_h = int(h * self.ignore_border_pct)
        x0, y0, x1, y1 = border_w, border_h, w - border_w, h - border_h
        return frame[y0:y1, x0:x1], (x0, y0, x1, y1)

    def detect(self, frame: np.ndarray) -> RoadDetectionResult:
        roi, roi_box = self._extract_roi(frame)
        run_model = (self.frame_index % (self.frame_skip + 1) == 0) or self.last_roi_mask is None
        if run_model:
            prob = self.model.predict(roi)
            mask = (prob >= self.threshold).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            self.mask_history.append(mask)
            self.last_prob = prob
        if self.mask_history:
            smooth_mask = (np.mean(np.stack(self.mask_history, axis=0), axis=0) > 127).astype(np.uint8) * 255
            self.last_roi_mask = smooth_mask
        else:
            smooth_mask = np.zeros_like(roi[:, :, 0])

        axis = self._fit_axis(self.last_roi_mask) if self.last_roi_mask is not None else None
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x0, y0, x1, y1 = roi_box
        full_mask[y0:y1, x0:x1] = self.last_roi_mask if self.last_roi_mask is not None else 0
        conf = float(np.mean((self.last_prob if run_model else self.last_roi_mask / 255.0)[smooth_mask > 0])) if smooth_mask.any() else 0.0
        info_full = None
        if axis is not None:
            info_full = self._map_axis(axis, roi_box)
        self.frame_index += 1
        return RoadDetectionResult(mask=full_mask, confidence=conf, axis_info=info_full)

    @staticmethod
    def _fit_axis(mask: np.ndarray) -> Optional[Dict]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 50:
            return None
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        w, h = max(w, 1.0), max(h, 1.0)
        theta = math.radians(angle)
        if h > w:
            theta += math.pi / 2.0
        vx, vy = math.cos(theta), math.sin(theta)
        long_len = max(w, h)
        p1 = (cx - 0.5 * long_len * vx, cy - 0.5 * long_len * vy)
        p2 = (cx + 0.5 * long_len * vx, cy + 0.5 * long_len * vy)
        return {
            "center": (cx, cy),
            "axis": (p1, p2),
            "width": min(w, h),
            "length": long_len,
            "rect": rect,
            "angle_deg": float((math.degrees(theta) + 360) % 360),
        }

    @staticmethod
    def _map_axis(axis_info: Dict, roi_box: Tuple[int, int, int, int]) -> Dict:
        x0, y0, _, _ = roi_box
        (p1, p2) = axis_info["axis"]
        rect = axis_info["rect"]
        return {
            "center": (axis_info["center"][0] + x0, axis_info["center"][1] + y0),
            "axis": ((p1[0] + x0, p1[1] + y0), (p2[0] + x0, p2[1] + y0)),
            "width": axis_info["width"],
            "length": axis_info["length"],
            "angle_deg": axis_info["angle_deg"],
            "rect": ((rect[0][0] + x0, rect[0][1] + y0), rect[1], rect[2]),
        }

    def segment_highway(self, frame_shape: Tuple[int, int, int], axis_info: Dict, segments: int) -> List[np.ndarray]:
        (x1, y1), (x2, y2) = axis_info["axis"]
        dx, dy = (x2 - x1), (y2 - y1)
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        seg_len = length / segments
        width = axis_info["width"] * 1.4
        half_wx, half_wy = -uy * width / 2.0, ux * width / 2.0
        masks = []
        h, w = frame_shape[:2]
        for idx in range(segments):
            start_x = x1 + ux * seg_len * idx
            start_y = y1 + uy * seg_len * idx
            end_x = start_x + ux * seg_len
            end_y = start_y + uy * seg_len
            poly = np.array(
                [
                    (start_x + half_wx, start_y + half_wy),
                    (start_x - half_wx, start_y - half_wy),
                    (end_x - half_wx, end_y - half_wy),
                    (end_x + half_wx, end_y + half_wy),
                ],
                dtype=np.float32,
            )
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
            masks.append(mask)
        return masks


# ---------------------------------------------------------------------------
# Truck detection wrapper
# ---------------------------------------------------------------------------


class TruckDetector:
    def __init__(self, weights: str, conf: float = 0.25) -> None:
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame: np.ndarray, segment_mask: Optional[np.ndarray] = None) -> List[Tuple[Tuple[int, int, int, int], float]]:
        if segment_mask is not None:
            ys, xs = np.where(segment_mask > 0)
            if len(xs) and len(ys):
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                crop = frame[y_min:y_max + 1, x_min:x_max + 1]
                offset = (x_min, y_min)
            else:
                return []
        else:
            crop = frame
            offset = (0, 0)
        results = self.model(crop, conf=self.conf, verbose=False)[0]
        detections: List[Tuple[Tuple[int, int, int, int], float]] = []
        for box, cls, conf in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
        ):
            cls_name = results.names[int(cls)]
            if cls_name.lower() != "truck":
                continue
            x1, y1, x2, y2 = box
            detections.append(
                (
                    (
                        int(x1 + offset[0]),
                        int(y1 + offset[1]),
                        int(x2 + offset[0]),
                        int(y2 + offset[1]),
                    ),
                    float(conf),
                )
            )
        return detections


# ---------------------------------------------------------------------------
# Highway monitor orchestrator
# ---------------------------------------------------------------------------


class HighwayMonitor:
    def __init__(
        self,
        gimbal: GimbalClient,
        road_detector: RoadDetector,
        truck_detector: TruckDetector,
        stream: cv2.VideoCapture,
        log_file: Path,
        segments: int = 6,
        zoom_levels: Sequence[float] = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 12.0, 14.0, 16.0, 18.0, 20.0), #maximun digital zoom of DJO H20 camera
    ) -> None:
        self.gimbal = gimbal
        self.road_detector = road_detector
        self.truck_detector = truck_detector
        self.stream = stream
        self.log_file = log_file
        self.segments = segments
        self.zoom_levels = zoom_levels
        self.segment_logs: List[SegmentLog] = [SegmentLog(idx, [], []) for idx in range(segments)]
        self.last_truck: Optional[TruckDetection] = None
        self.final_frame: Optional[np.ndarray] = None
        self.telemetry: Dict[str, float] = {}

    def run(self) -> None:
        logging.info("Orienting gimbal to nadir view")
        self.gimbal.set_orientation(yaw=0.0, pitch=90.0, zoom=1.0)
        frame = self._wait_for_frame()
        detection = self._wait_for_highway(frame)
        self._rotate_to_start(detection)
        self._analyze_segments()
        self._finalize()
        self._write_log()

    def _wait_for_frame(self) -> np.ndarray:
        ok, frame = self.stream.read()
        if not ok:
            raise RuntimeError("Unable to read from RTSP stream")
        return frame

    def _wait_for_highway(self, initial_frame: np.ndarray) -> RoadDetectionResult:
        frame = initial_frame
        while True:
            detection = self.road_detector.detect(frame)
            logging.info("Road detection confidence %.2f", detection.confidence)
            if detection.found:
                logging.info("Highway detected")
                return detection
            time.sleep(0.1)
            ok, frame = self.stream.read()
            if not ok:
                raise RuntimeError("Video stream ended before detecting highway")

    def _rotate_to_start(self, detection: RoadDetectionResult) -> None:
        target_threshold = 0.1 * detection.mask.shape[1]
        attempts = 0
        while attempts < 8:
            axis = detection.axis_info["axis"] if detection.axis_info else None
            if axis:
                (p1, _) = axis
                if p1[0] <= target_threshold:
                    logging.info("Reached left-most view of highway")
                    return
            self.gimbal.scan_left(degrees=5.0)
            frame = self._wait_for_frame()
            detection = self.road_detector.detect(frame)
            attempts += 1
        logging.warning("Unable to confirm highway start; proceeding with current framing")

    def _analyze_segments(self) -> None:
        ok, frame = self.stream.read()
        if not ok:
            raise RuntimeError("Video stream unavailable during segment analysis")
        detection = self.road_detector.detect(frame)
        if not detection.found:
            raise RuntimeError("Road lost before segment analysis")
        masks = self.road_detector.segment_highway(frame.shape, detection.axis_info, self.segments)
        for idx, mask in enumerate(masks):
            segment_log = self.segment_logs[idx]
            segment_log.status = "SCANNING"
            for zoom in self.zoom_levels:
                logging.info("Segment %d zoom %.1f", idx, zoom)
                self.gimbal.set_zoom(zoom)
                segment_log.zoom_commands.append(zoom)
                ok, frame = self.stream.read()
                if not ok:
                    raise RuntimeError("RTSP stream interrupted")
                detections = self.truck_detector.detect(frame, mask)
                for bbox, conf in detections:
                    truck = TruckDetection(bbox=bbox, confidence=conf, zoom_level=zoom, segment_index=idx)
                    segment_log.trucks.append(truck)
                    self.last_truck = truck
                    self.final_frame = frame.copy()
                if detections:
                    segment_log.status = "TRUCK_FOUND"
                    break
            if not segment_log.trucks:
                segment_log.status = "NO_TRUCK"

    def _finalize(self) -> None:
        if not self.last_truck:
            logging.warning("No trucks detected anywhere; skipping photo")
            return
        if self._needs_last_truck_estimation():
            logging.info("Last two segments empty; estimating last truck position")
            self._estimate_last_truck()
        self._zoom_for_truck()
        self.telemetry = self.gimbal.read_telemetry()
        photo_resp = self.gimbal.capture_photo()
        logging.info("Photo command response: %s", photo_resp)
        if self.final_frame is not None:
            out_path = self.log_file.with_suffix(".jpg")
            cv2.imwrite(str(out_path), self.final_frame)
            logging.info("Saved final frame to %s", out_path)

    def _needs_last_truck_estimation(self) -> bool:
        trailing = self.segment_logs[-2:]
        return all(not seg.trucks for seg in trailing)

    def _estimate_last_truck(self) -> None:
        for seg in reversed(self.segment_logs[:-2]):
            if seg.trucks:
                candidate = seg.trucks[-1]
                self.last_truck = candidate
                return
        logging.warning("No truck available for estimation")

    def _zoom_for_truck(self) -> None:
        if not self.last_truck or self.final_frame is None:
            return
        frame_h, frame_w = self.final_frame.shape[:2]
        bbox = self.last_truck.bbox
        box_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        frame_area = frame_w * frame_h
        scale = math.sqrt((0.8 * frame_area) / box_area)
        new_zoom = min(self.zoom_levels[-1], self.gimbal.current_zoom * scale)
        logging.info("Adjusting zoom for 80%% framing: %.2f", new_zoom)
        self.gimbal.set_zoom(new_zoom)

    def _write_log(self) -> None:
        payload = {
            "segments": [seg.to_json() for seg in self.segment_logs],
            "last_truck": self.last_truck.to_json() if self.last_truck else None,
            "telemetry": self.telemetry,
            "log_created": time.time(),
        }
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.write_text(json.dumps(payload, indent=2))
        logging.info("Wrote log to %s", self.log_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous highway gimbal control")
    parser.add_argument("--host", required=True, help="IP of the RC Plus device")
    parser.add_argument("--port", type=int, default=8989, help="TCP port of control server")
    parser.add_argument("--rtsp", required=True, help="RTSP url of the zoom camera stream")
    parser.add_argument("--yolo-weights", required=True, help="Path to YOLO weights")
    parser.add_argument("--log-file", type=Path, default=Path("highway_log.json"), help="JSON file for run log")
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--zoom-levels", type=float, nargs="+", default=(2.0, 4.0, 6.0, 8.0))
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    stream = cv2.VideoCapture(args.rtsp)
    if not stream.isOpened():
        raise RuntimeError("Failed to open RTSP stream")
    gimbal = GimbalClient(args.host, args.port)
    road_detector = RoadDetector()
    truck_detector = TruckDetector(args.yolo_weights)
    monitor = HighwayMonitor(
        gimbal=gimbal,
        road_detector=road_detector,
        truck_detector=truck_detector,
        stream=stream,
        log_file=args.log_file,
        segments=args.segments,
        zoom_levels=args.zoom_levels,
    )
    try:
        monitor.run()
    finally:
        stream.release()
        gimbal.close()


if __name__ == "__main__":
    main()
