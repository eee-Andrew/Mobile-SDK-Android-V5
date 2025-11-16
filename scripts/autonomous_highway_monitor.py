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
    """Container for the road/highway detection result.

    Attributes
    ----------
    mask:
        Binary mask (H x W, uint8) in *full-frame coordinates* indicating
        highway pixels (255) vs background (0).
    confidence:
        Average confidence of the model over pixels considered as road
        in the final smoothed mask.
    axis_info:
        Dictionary describing the estimated main axis of the highway, or None
        if no reliable road was found. See RoadDetector._fit_axis() for
        the structure of this dictionary.
    """

    mask: np.ndarray
    confidence: float
    axis_info: Optional[Dict]

    @property
    def found(self) -> bool:
        """Return True if a road was successfully detected.

        The condition is:
        - axis_info is not None (we managed to fit a geometric axis)
        - confidence is above a fixed threshold (0.35 by default)

        This hides low-quality detections from the rest of the pipeline.
        """
        return self.axis_info is not None and self.confidence >= 0.35


@dataclasses.dataclass
class TruckDetection:
    """Simple struct for a single truck detection.

    Attributes
    ----------
    bbox:
        Bounding box in full-frame pixel coordinates (x1, y1, x2, y2).
    confidence:
        YOLO detection confidence for this truck.
    zoom_level:
        Camera zoom level at the time this truck was detected. Used later
        to reason about framing and for logging.
    segment_index:
        Index of the highway segment in which the truck was found. The
        highway is discretized into a fixed number of segments along its axis.
    """

    bbox: Tuple[int, int, int, int]
    confidence: float
    zoom_level: float
    segment_index: int

    def to_json(self) -> Dict:
        """Convert detection to JSON-serializable dict."""
        x1, y1, x2, y2 = self.bbox
        return {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": float(self.confidence),
            "zoom": float(self.zoom_level),
            "segment": int(self.segment_index),
        }


@dataclasses.dataclass
class SegmentLog:
    """Per-segment logging structure.

    Each highway segment keeps track of:
    - which zoom commands were issued while scanning this segment
    - all trucks detected in this segment
    - its final status ("UNPROCESSED", "SCANNING", "TRUCK_FOUND", "NO_TRUCK")
    """

    index: int
    zoom_commands: List[float]
    trucks: List[TruckDetection]
    status: str = "UNPROCESSED"

    def to_json(self) -> Dict:
        """Convert log to JSON-serializable dict."""
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
    """Tiny helper that speaks the TCP control protocol from the sample app.

    This wrapper assumes a very simple line-based text protocol:

    - "SET yaw pitch zoom" to command the gimbal orientation and optical/digital zoom.
    - "TAKE_PHOTO" to trigger a still image capture on the payload.
    - "GET" to read telemetry (e.g. altitude, GPS). The payload is assumed to
      respond with key-value pairs such as "alt 120.0 lat 37.1 lon 22.9 ...".

    The exact details depend on the OEM sample server running on the RC Plus
    Android device. This class only deals with TCP I/O and keeps an internal
    mirror of yaw/pitch/zoom so higher-level code can reason about state.
    """

    def __init__(self, host: str, port: int, timeout: float = 5.0) -> None:
        """Create a TCP connection to the control server.

        Parameters
        ----------
        host:
            IP address or hostname of the control server (RC Plus device).
        port:
            TCP port where the sample app listens (e.g. 8989).
        timeout:
            Socket timeout in seconds for both connect and read operations.
        """
        self.host = host
        self.port = port

        # Open TCP connection immediately so we fail fast if unreachable.
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)

        # Local state cache. These are continuously updated when we send commands.
        self.current_zoom = 1.0
        self.current_pitch = 90.0  # 90° = nadir view (camera pointing down)
        self.current_yaw = 0.0

    def close(self) -> None:
        """Close the TCP connection to the gimbal control server."""
        try:
            self.sock.close()
        except OSError:
            # Socket may already be closed; ignore errors on shutdown.
            pass

    # ------------------------------------------------------------------
    # Low-level command helpers
    # ------------------------------------------------------------------
    def send_command(self, cmd: str) -> str:
        """Send a single command line and return the raw response string.

        The protocol is assumed to be:
        - ASCII text
        - One command per line, terminated with '\n'
        - Response fits into 1024 bytes (small simple protocol)

        All traffic is logged at DEBUG level for post-mortem analysis.
        """
        logging.debug("TX: %s", cmd.strip())
        self.sock.sendall(cmd.encode("ascii"))
        resp = self.sock.recv(1024).decode("ascii", errors="ignore").strip()
        logging.debug("RX: %s", resp)
        return resp

    def set_orientation(self, yaw: float, pitch: float, zoom: float) -> None:
        """Set yaw, pitch and zoom in a single atomic command.

        This is used for coarse positioning of the payload (e.g. initial nadir view).
        """
        self.current_yaw = yaw
        self.current_pitch = pitch
        self.current_zoom = zoom
        self.send_command(f"SET {yaw:.2f} {pitch:.2f} {zoom:.2f}\n")

    def scan_left(self, degrees: float, step: float = 2.0) -> None:
        """Perform a coarse leftward scan by a given number of degrees.

        The method steps the yaw in small increments (default 2 degrees) and
        sleeps briefly between steps to allow the gimbal to physically move.

        This is used when the system tries to find the "start" of the highway
        by scanning left until the road axis is near the left edge of the frame.
        """
        target = self.current_yaw - degrees
        while self.current_yaw > target:
            self.current_yaw -= step
            self.send_command(
                f"SET {self.current_yaw:.2f} {self.current_pitch:.2f} {self.current_zoom:.2f}\n"
            )
            time.sleep(0.25)

    def set_zoom(self, zoom: float) -> None:
        """Update only the zoom parameter while keeping yaw/pitch constant."""
        self.current_zoom = zoom
        self.send_command(
            f"SET {self.current_yaw:.2f} {self.current_pitch:.2f} {zoom:.2f}\n"
        )

    def capture_photo(self) -> str:
        """Trigger a photo capture and return the raw response.

        The actual storage location (SD card path) is usually encoded in the
        response depending on the vendor protocol.
        """
        return self.send_command("TAKE_PHOTO\n")

    def read_telemetry(self) -> Dict[str, float]:
        """Query the payload for telemetry values.

        Returns
        -------
        Dict[str, float]
            A dictionary mapping telemetry key names to float values, e.g.
            {"alt": 110.5, "lat": 37.1234, "lon": 22.9876, ...}.
        """
        resp = self.send_command("GET\n")
        tokens = resp.split()

        data: Dict[str, float] = {}
        # Expect key value key value ... layout from the sample app.
        for idx in range(0, len(tokens) - 1, 2):
            key = tokens[idx]
            try:
                data[key] = float(tokens[idx + 1])
            except ValueError:
                # If parsing fails for a given value, ignore that pair.
                continue
        return data


# ---------------------------------------------------------------------------
# Road detection (CLIPSeg wrapper)
# ---------------------------------------------------------------------------


class ClipSegRoad:
    """Thin wrapper around CLIPSeg for text-prompted road segmentation.

    This class encapsulates:
    - Model and processor loading
    - Device selection (CPU vs CUDA)
    - Mixed precision on GPU for speed
    - Normalization of the model output into a probability map per pixel
    """

    def __init__(self, prompts: Sequence[str] = ("Highway",)) -> None:
        """Initialize CLIPSeg with a given list of text prompts.

        Parameters
        ----------
        prompts:
            Text queries describing the class of interest. By default we use
            a single prompt "Highway", but the list could include synonyms
            (e.g. "road", "asphalt", "street") for better robustness.
        """
        self.prompts = list(prompts)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # HuggingFace processor handles preprocessing (resize, normalize, etc).
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

        # Load the CLIPSeg model and move it to the selected device.
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(self.device)

        # If running on GPU, use half precision to speed up inference.
        if self.device.type == "cuda":
            self.model.half()
        self.model.eval()

    @torch.no_grad()
    def predict(self, roi: np.ndarray) -> np.ndarray:
        """Return a probability map for 'road' over the given ROI.

        Parameters
        ----------
        roi:
            Region of interest in BGR format (OpenCV image) where the model
            will search for highway pixels.

        Returns
        -------
        np.ndarray
            Single-channel float32 array with values in [0, 1], resized to
            the original ROI size.
        """
        # Convert BGR (OpenCV) to RGB and wrap into PIL for the processor.
        pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # Duplicate the image for each text prompt; CLIPSeg expects one image
        # per prompt for joint embedding.
        inputs = self.processor(
            text=self.prompts,
            images=[pil] * len(self.prompts),
            return_tensors="pt",
        ).to(self.device)

        # Optional autocast for GPU to gain speed with half precision.
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(**inputs).logits
        else:
            logits = self.model(**inputs).logits

        # Apply sigmoid to convert logits to probabilities.
        probs = torch.sigmoid(logits)

        # Take the maximum probability across all prompts for each pixel.
        prob = torch.max(probs, dim=0).values.detach().float().cpu().numpy()

        # Resize probability map back to ROI resolution.
        prob = cv2.resize(
            prob, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC
        )
        return prob


class RoadDetector:
    """High-level road detection and segmentation logic.

    Responsibilities
    ----------------
    - Maintain a smoothed mask history to stabilize the road shape over time.
    - Ignore a percentage of the frame border to avoid UI/HUD artifacts.
    - Perform inference at a configurable frame rate (frame skipping).
    - Derive the main highway axis (direction and extent) using contour analysis.
    - Split the detected highway into rectangular segments for downstream truck
      analysis.
    """

    def __init__(
        self,
        smooth_window: int = 5,
        threshold: float = 0.35,
        ignore_border_pct: float = 0.1,
        frame_skip: int = 2,
    ) -> None:
        """Create a RoadDetector backed by CLIPSeg.

        Parameters
        ----------
        smooth_window:
            Number of past masks to average for temporal smoothing. Larger
            values result in more stable but slower-reacting masks.
        threshold:
            Probability threshold used to convert CLIPSeg output into a
            binary road mask.
        ignore_border_pct:
            Percentage of image width/height to ignore along the borders.
            This removes top/bottom HUD overlays and avoids spurious detections.
        frame_skip:
            Number of frames to skip between model inferences. For example,
            frame_skip=2 means "run CLIPSeg every 3rd frame" and reuse the
            last mask in between for speed.
        """
        self.threshold = threshold
        self.ignore_border_pct = ignore_border_pct
        self.frame_skip = frame_skip

        # Past masks used for temporal smoothing.
        self.mask_history: Deque[np.ndarray] = deque(maxlen=smooth_window)

        # Underlying CLIPSeg model wrapper.
        self.model = ClipSegRoad()

        # Counter of processed frames (used for frame skipping).
        self.frame_index = 0

        # Last ROI mask and probability map, used when we reuse old results.
        self.last_roi_mask: Optional[np.ndarray] = None
        self.last_prob: Optional[np.ndarray] = None

    def _extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Crop the frame to the central region of interest.

        The border percentage is defined by self.ignore_border_pct. We remove
        the same proportion from all sides to obtain a central ROI.

        Returns
        -------
        roi:
            Cropped frame region (H_roi x W_roi x 3).
        roi_box:
            Tuple (x0, y0, x1, y1) describing ROI coordinates in full-frame space.
        """
        h, w = frame.shape[:2]
        border_w = int(w * self.ignore_border_pct)
        border_h = int(h * self.ignore_border_pct)
        x0, y0, x1, y1 = border_w, border_h, w - border_w, h - border_h
        return frame[y0:y1, x0:x1], (x0, y0, x1, y1)

    def detect(self, frame: np.ndarray) -> RoadDetectionResult:
        """Run the road detection pipeline on a single frame.

        Steps
        -----
        1. Extract ROI to avoid borders.
        2. Decide whether to run the CLIPSeg model (based on frame_skip).
        3. Threshold and morphologically clean the probability map.
        4. Temporally smooth masks using a sliding window.
        5. Fit the main highway axis using the largest contour.
        6. Map ROI mask and axis back to full-frame coordinates.
        7. Estimate a global confidence score.

        Returns
        -------
        RoadDetectionResult
            Includes full-frame mask, confidence and axis_info.
        """
        roi, roi_box = self._extract_roi(frame)

        # Decide if we should run the model or reuse the previous result.
        run_model = (
            self.frame_index % (self.frame_skip + 1) == 0
        ) or self.last_roi_mask is None

        if run_model:
            # Step 1: Get per-pixel probability map from CLIPSeg.
            prob = self.model.predict(roi)

            # Step 2: Threshold to binary mask.
            mask = (prob >= self.threshold).astype(np.uint8) * 255

            # Step 3: Morphological operations to clean up the mask.
            # Opening removes small blobs; closing fills small gaps.
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

            # Store for temporal smoothing.
            self.mask_history.append(mask)
            self.last_prob = prob

        # Smooth the mask over the past `smooth_window` frames.
        if self.mask_history:
            smooth_mask = (
                np.mean(np.stack(self.mask_history, axis=0), axis=0) > 127
            ).astype(np.uint8) * 255
            self.last_roi_mask = smooth_mask
        else:
            smooth_mask = np.zeros_like(roi[:, :, 0])

        # Fit a line / axis through the road region if we have a valid mask.
        axis = (
            self._fit_axis(self.last_roi_mask) if self.last_roi_mask is not None else None
        )

        # Reconstruct a full-frame mask (zero outside the ROI).
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x0, y0, x1, y1 = roi_box
        full_mask[y0:y1, x0:x1] = (
            self.last_roi_mask if self.last_roi_mask is not None else 0
        )

        # Estimate a scalar confidence measure: mean probability over the
        # area that is considered road in the smoothed mask.
        if smooth_mask.any():
            # If we just ran the model, use last_prob; otherwise approximate
            # with the binary mask normalized to [0,1].
            prob_source = (
                self.last_prob if run_model else self.last_roi_mask / 255.0
            )
            conf = float(np.mean(prob_source[smooth_mask > 0]))
        else:
            conf = 0.0

        # Map axis info into full-frame coordinates.
        info_full = None
        if axis is not None:
            info_full = self._map_axis(axis, roi_box)

        self.frame_index += 1
        return RoadDetectionResult(mask=full_mask, confidence=conf, axis_info=info_full)

    @staticmethod
    def _fit_axis(mask: np.ndarray) -> Optional[Dict]:
        """Fit the main highway axis using the largest contour.

        Using the binary road mask:
        - Find all contours.
        - Select the largest one by area.
        - Fit a minimum-area rectangle (cv2.minAreaRect) around it.
        - Construct an oriented line segment along the long side of the rectangle.

        Returns
        -------
        dict or None
            A dictionary describing the road axis or None if no reliable road
            contour exists.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Use the contour with maximum area as the road candidate.
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 50:
            # Ignore tiny blobs (likely noise).
            return None

        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect

        # Avoid degenerate zeros.
        w, h = max(w, 1.0), max(h, 1.0)

        theta = math.radians(angle)
        # Ensure the axis follows the long side of the rectangle.
        if h > w:
            theta += math.pi / 2.0

        vx, vy = math.cos(theta), math.sin(theta)
        long_len = max(w, h)

        # Endpoints of the axis segment.
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
        """Convert axis coordinates from ROI space back to full-frame space.

        Parameters
        ----------
        axis_info:
            Output from _fit_axis() in ROI coordinates.
        roi_box:
            (x0, y0, x1, y1) describing ROI position in full-frame coordinates.

        Returns
        -------
        dict
            Same structure as axis_info but with coordinates shifted by ROI offset.
        """
        x0, y0, _, _ = roi_box
        (p1, p2) = axis_info["axis"]
        rect = axis_info["rect"]

        return {
            "center": (axis_info["center"][0] + x0, axis_info["center"][1] + y0),
            "axis": ((p1[0] + x0, p1[1] + y0), (p2[0] + x0, p2[1] + y0)),
            "width": axis_info["width"],
            "length": axis_info["length"],
            "angle_deg": axis_info["angle_deg"],
            # Shift the center of the rectangle but keep size and angle.
            "rect": ((rect[0][0] + x0, rect[0][1] + y0), rect[1], rect[2]),
        }

    def segment_highway(
        self,
        frame_shape: Tuple[int, int, int],
        axis_info: Dict,
        segments: int,
    ) -> List[np.ndarray]:
        """Split the detected highway into longitudinal segments.

        The road is approximated as a long oriented rectangle defined by:
        - a central axis segment (p1 -> p2)
        - an effective width (axis_info["width"])

        We then construct N smaller rectangles along this axis, each
        covering a piece of the road. Each rectangle is converted into a
        binary mask so we can restrict truck detection to that portion.

        Parameters
        ----------
        frame_shape:
            Shape tuple of the full frame (H, W, C).
        axis_info:
            Dictionary from _map_axis(), describing the road in full-frame coordinates.
        segments:
            Number of segments to generate.

        Returns
        -------
        List[np.ndarray]
            List of binary masks (H x W, uint8) for each segment.
        """
        (x1, y1), (x2, y2) = axis_info["axis"]

        # Direction vector along the highway axis.
        dx, dy = (x2 - x1), (y2 - y1)
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length

        # Each segment has equal length along the axis.
        seg_len = length / segments

        # Effective road width is slightly expanded to ensure coverage.
        width = axis_info["width"] * 1.4

        # Half-width vector perpendicular to the axis (rotate by +90°).
        half_wx, half_wy = -uy * width / 2.0, ux * width / 2.0

        masks = []
        h, w = frame_shape[:2]

        for idx in range(segments):
            # Define segment start and end points along the main axis.
            start_x = x1 + ux * seg_len * idx
            start_y = y1 + uy * seg_len * idx
            end_x = start_x + ux * seg_len
            end_y = start_y + uy * seg_len

            # Build a quadrilateral representing this road slice.
            poly = np.array(
                [
                    (start_x + half_wx, start_y + half_wy),
                    (start_x - half_wx, start_y - half_wy),
                    (end_x - half_wx, end_y - half_wy),
                    (end_x + half_wx, end_y + half_wy),
                ],
                dtype=np.float32,
            )

            # Rasterize polygon into a binary mask.
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
            masks.append(mask)

        return masks


# ---------------------------------------------------------------------------
# Truck detection wrapper
# ---------------------------------------------------------------------------


class TruckDetector:
    """Wrapper around a YOLO model specialized for truck detection.

    The model is assumed to be trained or configured with a 'truck' class.
    This wrapper:
    - Handles input cropping to a given road segment.
    - Translates detection boxes back to full-frame coordinates.
    - Filters out all classes except 'truck'.
    """

    def __init__(self, weights: str, conf: float = 0.25) -> None:
        """Load YOLO model from a weights file.

        Parameters
        ----------
        weights:
            Path to a YOLO weights file compatible with ultralytics.YOLO.
        conf:
            Global confidence threshold used for inference.
        """
        self.model = YOLO(weights)
        self.conf = conf

    def detect(
        self,
        frame: np.ndarray,
        segment_mask: Optional[np.ndarray] = None,
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Detect trucks in a frame, optionally restricted to a segment mask.

        If a segment mask is provided:
        - We find the bounding box around non-zero pixels of the mask.
        - Crop the frame to that bounding box to reduce computation.
        - Run YOLO on the crop and then translate detections back to
          full-frame coordinates using the offset.

        Parameters
        ----------
        frame:
            Full frame (H x W x 3) in BGR format.
        segment_mask:
            Binary mask of the region we want to inspect (same H x W). If None,
            the full frame is used.

        Returns
        -------
        List[(bbox, conf)]
            Each bbox is (x1, y1, x2, y2) in full-frame coordinates; conf is
            the YOLO confidence for that detection.
        """
        if segment_mask is not None:
            ys, xs = np.where(segment_mask > 0)
            if len(xs) and len(ys):
                # Tight bounding box around the active segment region.
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                crop = frame[y_min : y_max + 1, x_min : x_max + 1]
                offset = (x_min, y_min)
            else:
                # Segment is empty; nothing to detect.
                return []
        else:
            crop = frame
            offset = (0, 0)

        # Run YOLO detection on the cropped region.
        results = self.model(crop, conf=self.conf, verbose=False)[0]

        detections: List[Tuple[Tuple[int, int, int, int], float]] = []
        for box, cls, conf in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
        ):
            cls_name = results.names[int(cls)]
            # Filter out everything that is not labeled as 'truck'.
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
    """Main orchestrator for the autonomous highway monitoring mission.

    This class ties together:
    - Gimbal control (orientation, zoom, telemetry).
    - Road detection using RoadDetector.
    - Truck detection using TruckDetector.
    - A procedural workflow that mimics how a human operator might search
      along a highway for trucks, zoom in, and capture a final image.

    High-level workflow
    -------------------
    1. Nadir orientation: point camera straight down with minimal zoom.
    2. Wait until the highway is reliably detected in the nadir view.
    3. Rotate left slowly until the "start" of the highway comes into view.
    4. Split the detected road axis into N segments.
    5. For each segment, sweep through predefined zoom levels and run YOLO
       to search for trucks.
    6. Keep track of the latest truck detection and store the frame where it
       appeared.
    7. After scanning all segments, adjust zoom for aesthetically good framing
       around the last truck and trigger a photo capture.
    8. Write a detailed JSON log with all decisions and detections.
    """

    def __init__(
        self,
        gimbal: GimbalClient,
        road_detector: RoadDetector,
        truck_detector: TruckDetector,
        stream: cv2.VideoCapture,
        log_file: Path,
        segments: int = 6,
        zoom_levels: Sequence[float] = (
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
        ),  # maximun digital zoom of DJI H20 camera
    ) -> None:
        """Construct the monitor with its core components.

        Parameters
        ----------
        gimbal:
            Instance of GimbalClient providing camera control.
        road_detector:
            Instance of RoadDetector to locate highways and segments.
        truck_detector:
            Instance of TruckDetector to locate trucks.
        stream:
            OpenCV VideoCapture object for the RTSP camera stream.
        log_file:
            Path where JSON log (and optional final JPEG) will be written.
        segments:
            Number of highway segments to consider along the detected axis.
        zoom_levels:
            Sequence of zoom values to iterate through when scanning each
            segment. Typically starts from low zoom and goes up to maximum.
        """
        self.gimbal = gimbal
        self.road_detector = road_detector
        self.truck_detector = truck_detector
        self.stream = stream
        self.log_file = log_file
        self.segments = segments
        self.zoom_levels = zoom_levels

        # One SegmentLog object per highway segment.
        self.segment_logs: List[SegmentLog] = [
            SegmentLog(idx, [], []) for idx in range(segments)
        ]

        # Last detected truck (across all segments) and corresponding frame.
        self.last_truck: Optional[TruckDetection] = None
        self.final_frame: Optional[np.ndarray] = None

        # Telemetry at the moment of final photo capture.
        self.telemetry: Dict[str, float] = {}

    def run(self) -> None:
        """Execute the entire autonomous monitoring mission.

        This is the main entry point when deploying this script. It assumes
        the gimbal connection and RTSP stream are already open.
        """
        logging.info("Orienting gimbal to nadir view")
        self.gimbal.set_orientation(yaw=0.0, pitch=90.0, zoom=1.0)

        # 1) Wait for the first usable frame.
        frame = self._wait_for_frame()

        # 2) Wait until the road/highway is confidently detected.
        detection = self._wait_for_highway(frame)

        # 3) Rotate left to find the "start" of the road in the frame.
        self._rotate_to_start(detection)

        # 4) Segment the highway and scan each segment for trucks.
        self._analyze_segments()

        # 5) Once scanning is complete, refine framing and capture photo.
        self._finalize()

        # 6) Persist everything for offline analysis.
        self._write_log()

    def _wait_for_frame(self) -> np.ndarray:
        """Block until we receive a single frame from the RTSP stream.

        Raises
        ------
        RuntimeError
            If the stream fails to deliver any frame.
        """
        ok, frame = self.stream.read()
        if not ok:
            raise RuntimeError("Unable to read from RTSP stream")
        return frame

    def _wait_for_highway(self, initial_frame: np.ndarray) -> RoadDetectionResult:
        """Loop until a reliable highway detection is obtained.

        The method keeps running road detection on incoming frames until
        RoadDetectionResult.found is True.

        Parameters
        ----------
        initial_frame:
            First frame already fetched from the stream.

        Returns
        -------
        RoadDetectionResult
            The first result that passes the 'found' test.
        """
        frame = initial_frame
        while True:
            detection = self.road_detector.detect(frame)
            logging.info("Road detection confidence %.2f", detection.confidence)

            if detection.found:
                logging.info("Highway detected")
                return detection

            # If not found, wait briefly and grab the next frame.
            time.sleep(0.1)
            ok, frame = self.stream.read()
            if not ok:
                raise RuntimeError("Video stream ended before detecting highway")

    def _rotate_to_start(self, detection: RoadDetectionResult) -> None:
        """Rotate the gimbal to reach the 'start' of the highway.

        The notion of 'start' is defined heuristically: we try to rotate left
        until one endpoint of the axis gets close to the left border of the
        frame (10% of frame width). We perform a small number of attempts
        (max 8) to avoid endless scanning if the logic fails.

        Parameters
        ----------
        detection:
            A RoadDetectionResult representing the initial road detection.
        """
        # Threshold in pixels from the left edge under which we consider that the
        # road "starts" near the left border of the frame.
        target_threshold = 0.1 * detection.mask.shape[1]
        attempts = 0

        while attempts < 8:
            axis = detection.axis_info["axis"] if detection.axis_info else None
            if axis:
                (p1, _) = axis
                if p1[0] <= target_threshold:
                    logging.info("Reached left-most view of highway")
                    return

            # Rotate left by a few degrees and re-check.
            self.gimbal.scan_left(degrees=5.0)
            frame = self._wait_for_frame()
            detection = self.road_detector.detect(frame)
            attempts += 1

        logging.warning(
            "Unable to confirm highway start; proceeding with current framing"
        )

    def _analyze_segments(self) -> None:
        """Scan each highway segment for trucks at different zoom levels.

        Procedure
        ---------
        1. Grab a fresh frame and confirm the road is still visible.
        2. Compute segment masks along the road axis.
        3. For each segment:
           - Mark status as SCANNING.
           - Iterate over predefined zoom levels:
             - Set the gimbal zoom.
             - Grab a new frame.
             - Run TruckDetector inside the segment mask.
             - If any truck is found:
               - Record it in the segment log and update last_truck/final_frame.
               - Mark status as TRUCK_FOUND and stop scanning this segment.
           - If no truck is found after all zoom levels, mark segment as NO_TRUCK.
        """
        ok, frame = self.stream.read()
        if not ok:
            raise RuntimeError("Video stream unavailable during segment analysis")

        detection = self.road_detector.detect(frame)
        if not detection.found:
            raise RuntimeError("Road lost before segment analysis")

        # Create a mask for each segment along the detected road axis.
        masks = self.road_detector.segment_highway(
            frame.shape, detection.axis_info, self.segments
        )

        for idx, mask in enumerate(masks):
            segment_log = self.segment_logs[idx]
            segment_log.status = "SCANNING"

            for zoom in self.zoom_levels:
                logging.info("Segment %d zoom %.1f", idx, zoom)

                # Change zoom level and log it.
                self.gimbal.set_zoom(zoom)
                segment_log.zoom_commands.append(zoom)

                ok, frame = self.stream.read()
                if not ok:
                    raise RuntimeError("RTSP stream interrupted")

                # Run truck detection restricted to this segment.
                detections = self.truck_detector.detect(frame, mask)

                for bbox, conf in detections:
                    truck = TruckDetection(
                        bbox=bbox,
                        confidence=conf,
                        zoom_level=zoom,
                        segment_index=idx,
                    )
                    segment_log.trucks.append(truck)

                    # Always keep the most recent truck as the candidate
                    # for final framing.
                    self.last_truck = truck
                    self.final_frame = frame.copy()

                if detections:
                    # Once we find at least one truck in this segment at
                    # a given zoom, we consider this segment "covered".
                    segment_log.status = "TRUCK_FOUND"
                    break

            if not segment_log.trucks:
                segment_log.status = "NO_TRUCK"

    def _finalize(self) -> None:
        """Finalize the mission: adjust zoom, capture photo, and save frame.

        This method is called after scanning all segments. It:

        - Checks whether any trucks were found at all. If not, it simply
          logs a warning and returns.
        - Optionally adjusts the last_truck if we suspect that the very last
          frames were empty and we want to fall back to an earlier detection.
        - Computes a zoom level so the truck occupies roughly 80% of the frame.
        - Requests telemetry data and triggers a photo capture.
        - Saves the final frame to disk next to the log file.
        """
        if not self.last_truck:
            logging.warning("No trucks detected anywhere; skipping photo")
            return

        if self._needs_last_truck_estimation():
            logging.info("Last two segments empty; estimating last truck position")
            self._estimate_last_truck()

        # Adjust zoom for final framing and command the payload.
        self._zoom_for_truck()

        # Read telemetry exactly at the moment of final photo.
        self.telemetry = self.gimbal.read_telemetry()

        photo_resp = self.gimbal.capture_photo()
        logging.info("Photo command response: %s", photo_resp)

        # Optionally save the last frame to disk for human inspection.
        if self.final_frame is not None:
            out_path = self.log_file.with_suffix(".jpg")
            cv2.imwrite(str(out_path), self.final_frame)
            logging.info("Saved final frame to %s", out_path)

    def _needs_last_truck_estimation(self) -> bool:
        """Return True if the last two segments have no detections.

        This heuristic suggests that the final frames might not contain the
        most recent truck (for example, if the gimbal moved too far after
        the last detection). In that case we try to fall back to an earlier
        segment with detections.
        """
        trailing = self.segment_logs[-2:]
        return all(not seg.trucks for seg in trailing)

    def _estimate_last_truck(self) -> None:
        """Fallback strategy: pick the last truck from previous segments.

        We scan segments from the end (excluding the last two) and choose
        the most recent truck detection to serve as the final candidate.
        """
        for seg in reversed(self.segment_logs[:-2]):
            if seg.trucks:
                candidate = seg.trucks[-1]
                self.last_truck = candidate
                return
        logging.warning("No truck available for estimation")

    def _zoom_for_truck(self) -> None:
        """Compute and apply a zoom level that gives 80% framing around the truck.

        The idea is:
        - Estimate the ratio between frame area and truck bbox area.
        - Compute a zoom factor so that the truck would fill ~80% of the frame.
        - Clamp zoom to the maximum allowed by zoom_levels.
        """
        if not self.last_truck or self.final_frame is None:
            return

        frame_h, frame_w = self.final_frame.shape[:2]
        bbox = self.last_truck.bbox

        box_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        frame_area = frame_w * frame_h

        # Desired scaling factor for approximate 80% area coverage.
        scale = math.sqrt((0.8 * frame_area) / box_area)

        new_zoom = min(self.zoom_levels[-1], self.gimbal.current_zoom * scale)
        logging.info("Adjusting zoom for 80%% framing: %.2f", new_zoom)
        self.gimbal.set_zoom(new_zoom)

    def _write_log(self) -> None:
        """Serialize the entire run into a JSON log file.

        The log includes:
        - Per-segment zoom sequences and truck detections.
        - The last_truck metadata (if any).
        - Telemetry at capture time.
        - Timestamp of log creation.

        The JSON is intended for offline analysis and replay of decisions made
        during the autonomous mission.
        """
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
    """Parse command-line arguments for the autonomous highway monitor.

    Important parameters
    --------------------
    --host:
        IP of the RC Plus device (Android controller running the sample app).
    --port:
        TCP port for the control server on the RC Plus.
    --rtsp:
        RTSP URL of the zoom camera stream (H20 etc.).
    --yolo-weights:
        Path to YOLO weights file used for truck detection.
    --log-file:
        Output JSON log path; a JPEG with the final frame is stored next to it.
    --segments:
        Number of discrete highway segments to scan.
    --zoom-levels:
        Sequence of zoom values to iterate through per segment.
    --verbose:
        Enable debug-level logging (very chatty, useful for research).
    """
    parser = argparse.ArgumentParser(description="Autonomous highway gimbal control")
    parser.add_argument("--host", required=True, help="IP of the RC Plus device")
    parser.add_argument(
        "--port", type=int, default=8989, help="TCP port of control server"
    )
    parser.add_argument(
        "--rtsp", required=True, help="RTSP url of the zoom camera stream"
    )
    parser.add_argument(
        "--yolo-weights", required=True, help="Path to YOLO weights"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("highway_log.json"),
        help="JSON file for run log",
    )
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument(
        "--zoom-levels", type=float, nargs="+", default=(2.0, 4.0, 6.0, 8.0)
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the script.

    This function wires together:
    - Parsing CLI arguments.
    - Opening the RTSP video stream.
    - Connecting to the gimbal control server.
    - Initializing RoadDetector and TruckDetector.
    - Constructing and running the HighwayMonitor orchestrator.
    - Cleaning up resources on exit (even when exceptions occur).
    """
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Open video stream from the payload camera.
    stream = cv2.VideoCapture(args.rtsp)
    if not stream.isOpened():
        raise RuntimeError("Failed to open RTSP stream")

    # Create the low-level control and perception components.
    gimbal = GimbalClient(args.host, args.port)
    road_detector = RoadDetector()
    truck_detector = TruckDetector(args.yolo_weights)

    # Instantiate the high-level mission controller.
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
        # Always release hardware resources.
        stream.release()
        gimbal.close()


if __name__ == "__main__":
    main()
