import argparse
import logging
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from transformers import TimesformerConfig, TimesformerForVideoClassification
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from PIL import Image as _PILImage
    from facenet_pytorch import MTCNN as FaceMTCNN
    _MTCNN_AVAILABLE = True
except Exception:  # ImportError or torchvision/torch ABI mismatch at import
    _MTCNN_AVAILABLE = False

_log = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent.parent

from mas_engagement.config import (  # noqa: E402
    CAMERA_INDEX,
    CAMERA_SCAN_RANGE,
    CAPTURE_FPS,
    FRAME_SIZE,
    INFERENCE_INTERVAL,
    MODEL_CONFIG_DIR,
    MODEL_WEIGHTS_PATH,
    MTCNN_MARGIN_PX,
    NUM_CLASSES,
    NUM_FRAMES,
    TEMPERATURE_SCALE,
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if _MTCNN_AVAILABLE:
    _mtcnn = FaceMTCNN(
        image_size=FRAME_SIZE,
        margin=MTCNN_MARGIN_PX,
        keep_all=False,
        device="cpu",
        post_process=False,
    )
else:
    _mtcnn = None


def _crop_face(frame_rgb_np: np.ndarray) -> np.ndarray:
    """Detect a face in an RGB uint8 frame and return a 224x224 RGB crop.

    Falls back to a centre square crop if no face is found (or MTCNN is
    unavailable), and emits one warning per call so the rate of fallbacks is
    visible in the logs.
    """
    h, w = frame_rgb_np.shape[:2]

    boxes = None
    if _mtcnn is not None:
        try:
            boxes, _ = _mtcnn.detect(_PILImage.fromarray(frame_rgb_np))
        except Exception as exc:  # pragma: no cover — defensive
            _log.warning("MTCNN detect failed: %s", exc)
            boxes = None

    if boxes is not None and len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0]
        m = MTCNN_MARGIN_PX
        x1 = int(max(0, x1 - m))
        y1 = int(max(0, y1 - m))
        x2 = int(min(w, x2 + m))
        y2 = int(min(h, y2 + m))
        if x2 > x1 and y2 > y1:
            crop = frame_rgb_np[y1:y2, x1:x2]
            if _CV2_AVAILABLE:
                return _cv2.resize(crop, (FRAME_SIZE, FRAME_SIZE))

    _log.warning('{"agent":"detection","event":"no_face_fallback"}')
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    centre = frame_rgb_np[y0:y0 + side, x0:x0 + side]
    if _CV2_AVAILABLE:
        return _cv2.resize(centre, (FRAME_SIZE, FRAME_SIZE))
    return centre


def _open_camera(hint: int) -> Optional[object]:
    """Try hint index first, then scan 0‥CAMERA_SCAN_RANGE-1 for OBS Virtual Camera."""
    if not _CV2_AVAILABLE:
        return None
    candidates = [hint] + [i for i in range(CAMERA_SCAN_RANGE - 1, -1, -1) if i != hint]

    for idx in candidates:
        cap = _cv2.VideoCapture(idx, _cv2.CAP_DSHOW)
        if cap.isOpened():
            _log.info("Camera opened on index %d", idx)
            return cap
        cap.release()
        _log.debug("Camera index %d unavailable", idx)
    return None


class DetectionAgent:
    """Engagement detection agent; runs in mock or TimeSformer inference mode."""

    def __init__(self, use_mock: bool = False):
        torch.set_num_threads(os.cpu_count() or 1)
        self.use_mock = use_mock
        self._model: Optional[object] = None
        self._buffer: deque = deque(maxlen=CAPTURE_FPS * 10)
        self._stop_event = threading.Event()

        if not use_mock:
            self._try_load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _try_load_model(self) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            _log.warning("transformers not installed – switching to mock mode")
            self.use_mock = True
            return

        try:
            cfg = (
                TimesformerConfig.from_pretrained(str(MODEL_CONFIG_DIR))
                if MODEL_CONFIG_DIR.is_dir()
                else TimesformerConfig(num_labels=NUM_CLASSES)
            )
        except Exception as exc:
            _log.warning("Config load failed (%s) – using default TimesformerConfig", exc)
            cfg = TimesformerConfig(num_labels=NUM_CLASSES)

        try:
            model = TimesformerForVideoClassification(cfg)
            if not MODEL_WEIGHTS_PATH.is_file():
                _log.warning("Weights not found at %s – switching to mock mode", MODEL_WEIGHTS_PATH)
                self.use_mock = True
                return
            state = torch.load(str(MODEL_WEIGHTS_PATH), map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            self._model = model
            _log.info("Loaded TimeSformer weights from %s", MODEL_WEIGHTS_PATH)
        except Exception as exc:
            _log.warning("Model load failed (%s) – switching to mock mode", exc)
            self.use_mock = True

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, blackboard, logger) -> None:
        """Blocking capture-and-inference loop; returns when stop() is called."""
        if not self.use_mock and not _CV2_AVAILABLE:
            _log.warning("cv2 not available – switching to mock mode")
            self.use_mock = True

        cap = None
        if not self.use_mock:
            cap = _open_camera(CAMERA_INDEX)
            if cap is None:
                _log.warning("No camera found on indices 0-%d – switching to mock mode",
                             CAMERA_SCAN_RANGE - 1)
                self.use_mock = True

        frame_interval = 1.0 / CAPTURE_FPS
        last_inference: float = 0.0

        try:
            while not self._stop_event.is_set(): #camera
                if cap is not None:
                    ret, frame = cap.read()
                    if ret:
                        self._buffer.append(frame)
                else:
                    time.sleep(frame_interval)

                now = time.time()
                if now - last_inference >= INFERENCE_INTERVAL:
                    last_inference = now
                    level, confidence, softmax = self._infer()
                    # Re-stamp AFTER inference: _infer() can take several seconds
                    # on CPU, but `now` above was captured before it. Using the
                    # stale pre-inference `now` makes the record older than the
                    # orchestrator's _last_poll_time (which advances at 1 Hz), so
                    # its `timestamp > last_poll` filter would drop every record
                    # after the first. Stamp when the evidence is actually ready.
                    now = time.time()
                    blackboard.append_engagement(level, confidence, now, softmax=softmax)
                    # Mark when evidence finished being written, so Delivery can
                    # measure end-to-end latency to the resulting notification.
                    blackboard.set_last_evidence_ts(now)
                    logger.log(
                        "detection",
                        "engagement_posted",
                        level=level,
                        confidence=confidence,
                        softmax=softmax,
                        mock=self.use_mock,
                    )
        finally:
            if cap is not None:
                cap.release()

    def stop(self) -> None:
        """Signal the run loop to exit on the next iteration."""
        self._stop_event.set()

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer(self) -> Tuple[int, float, List[float]]:
        """Return (level, confidence, softmax). level=argmax, confidence=max(softmax)."""
        return self._mock_infer() if self.use_mock else self._model_infer()

    def _mock_infer(self) -> Tuple[int, float, List[float]]:
        # Boosted Dirichlet-like sample: one class peaks, others share the remainder.
        raw = np.random.gamma(shape=2.0, scale=1.0, size=NUM_CLASSES)
        raw[random.randint(0, NUM_CLASSES - 1)] *= random.uniform(3.0, 8.0)
        sm = raw / raw.sum()
        softmax = [round(float(p), 6) for p in sm]
        level = int(np.argmax(sm))
        confidence = round(float(sm[level]), 4)
        return level, confidence, softmax

    def _model_infer(self) -> Tuple[int, float, List[float]]:
        frames = list(self._buffer)
        if len(frames) < NUM_FRAMES:
            return self._mock_infer()

        indices = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
        clip = np.stack([self._preprocess(frames[i]) for i in indices])
        tensor = torch.from_numpy(clip).unsqueeze(0)  # (1, T, C, H, W)

        with torch.no_grad():
            logits = self._model(pixel_values=tensor).logits
        # Temperature scaling: soften over-confident logits before softmax.
        logits = logits / TEMPERATURE_SCALE
        probs = torch.softmax(logits, dim=-1)[0]
        softmax = [round(float(p), 6) for p in probs]
        level = int(probs.argmax())
        confidence = round(float(probs[level]), 4)
        return level, confidence, softmax

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        cropped = _crop_face(rgb)
        normalized = (cropped.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        return normalized.transpose(2, 0, 1)  # (C, H, W)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DetectionAgent standalone runner")
    p.add_argument("--mock", action="store_true", help="Force mock simulation mode")
    p.add_argument("--duration", type=float, default=None,
                   help="Auto-stop after N seconds (default: run until Ctrl-C)")
    return p.parse_args()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
    args = _parse_args()

    sys.path.insert(0, str(_PKG_ROOT.parent))
    from mas_engagement.blackboard import SharedState
    from mas_engagement.logger import JsonlLogger
    from mas_engagement.config import LOG_DIR

    bb = SharedState()
    log = JsonlLogger(log_dir=str(LOG_DIR))
    agent = DetectionAgent(use_mock=args.mock)

    _log.info("Starting DetectionAgent (mock=%s, duration=%s)",
              agent.use_mock, args.duration)

    if args.duration is not None:
        worker = threading.Thread(target=agent.run, args=(bb, log), daemon=True)
        worker.start()
        worker.join(timeout=args.duration)
        agent.stop()
        worker.join(timeout=3.0)
        _log.info("Auto-stopped after %.1fs. Log: %s", args.duration, log.path)
    else:
        try:
            agent.run(bb, log)
        except KeyboardInterrupt:
            agent.stop()
            _log.info("Stopped. Log: %s", log.path)
