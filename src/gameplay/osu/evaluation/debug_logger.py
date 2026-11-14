"""
Visual & Gameplay Debug Logger for Shiori AI osu!
Provides real-time annotation of game frames, detection forensics,
color sample logging, and debug image exports.
"""

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("Shiori.Osu.DebugLogger")


class VisualDebugLogger:
    """Handles visual inspection, debug frame generation, and forensics logging."""

    def __init__(self, output_dir: str = "data/osu_debug", enable_disk_saving: bool = False):
        self.output_dir = output_dir
        self.enable_disk_saving = enable_disk_saving
        if self.enable_disk_saving:
            os.makedirs(self.output_dir, exist_ok=True)
        self.latest_frame_annotated: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self.last_save_time = 0.0
        self.save_interval_s = 0.5  # Max 2 saves per second if disk saving enabled

    def clear_debug_files(self) -> int:
        """Purge temporary debug images from disk."""
        removed = 0
        if os.path.isdir(self.output_dir):
            for fname in os.listdir(self.output_dir):
                if fname.endswith((".png", ".jpg", ".jpeg")):
                    try:
                        os.remove(os.path.join(self.output_dir, fname))
                        removed += 1
                    except Exception:
                        pass
        return removed

    def annotate_and_record(
        self,
        frame_rgb: np.ndarray,
        detected_objects: List[Any],
        mode: str,
        drum_line_x: Optional[float] = None,
        accuracy: float = 100.0,
        combo: int = 0,
        offset_ms: float = 0.0,
        recent_action: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Draw detection overlays, bounding circles, annular color metrics,
        and hit target line onto the frame.
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return None

        try:
            import cv2
            # Create a copy in BGR for OpenCV drawing and saving
            annotated = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            h, w = annotated.shape[:2]

            # 1. Draw Taiko target drum line
            if drum_line_x is not None and 0 < drum_line_x < w:
                dl_x = int(drum_line_x)
                cv2.line(annotated, (dl_x, 0), (dl_x, h), (0, 255, 255), 2)
                cv2.putText(
                    annotated,
                    f"HIT LINE (X={dl_x})",
                    (dl_x + 5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # 2. Draw detected objects
            for det in detected_objects:
                cx = int(det.x)
                cy = int(det.y)
                r = int(det.radius)
                is_katsu = getattr(det, "is_katsu", False)
                is_don = getattr(det, "is_don", True)
                ring_r = getattr(det, "ring_r", 0.0)
                ring_b = getattr(det, "ring_b", 0.0)

                # Color: Blue for Katsu, Red for Don (BGR format: Blue=[255, 120, 0], Red=[30, 30, 240])
                if is_katsu:
                    color = (255, 140, 0)
                    label = f"KATSU B={ring_b:.0f} R={ring_r:.0f}"
                else:
                    color = (30, 30, 240)
                    label = f"DON R={ring_r:.0f} B={ring_b:.0f}"

                # Draw outer circle
                cv2.circle(annotated, (cx, cy), max(3, r), color, 2)
                # Draw center crosshair
                cv2.circle(annotated, (cx, cy), 2, (255, 255, 255), -1)

                # Draw annular sampling ring
                r_ring = int(max(2, r * 0.6))
                cv2.circle(annotated, (cx, cy), r_ring, (200, 200, 200), 1, cv2.LINE_AA)

                # Text label
                cv2.putText(
                    annotated,
                    label,
                    (cx - 30, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # 3. Top HUD / Telemetry Banner
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

            hud_text = f"Mode: {mode.upper()} | Combo: {combo}x | Acc: {accuracy:.1f}% | Offset: {offset_ms:+.1f}ms | Notes: {len(detected_objects)}"
            if recent_action:
                hud_text += f" | Last Key: {recent_action}"

            cv2.putText(
                annotated,
                hud_text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Store in-memory and periodically write to disk
            with self._lock:
                self.latest_frame_annotated = annotated

            if self.enable_disk_saving:
                now = time.time()
                if now - self.last_save_time >= self.save_interval_s:
                    self.last_save_time = now
                    os.makedirs(self.output_dir, exist_ok=True)
                    save_path = os.path.join(self.output_dir, "live_perception.png")
                    cv2.imwrite(save_path, annotated)

            return annotated

        except Exception as e:
            logger.debug(f"Error generating debug annotation: {e}")
            return None

    def get_latest_jpeg_bytes(self, quality: int = 85, size: Optional[Tuple[int, int]] = None) -> Optional[bytes]:
        """Encode the latest annotated frame into fast JPEG bytes for real-time streaming (< 2ms)."""
        with self._lock:
            if self.latest_frame_annotated is None:
                return None
            try:
                import cv2
                img = self.latest_frame_annotated
                if size is not None and (img.shape[1] != size[0] or img.shape[0] != size[1]):
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(100, quality))]
                success, buf = cv2.imencode(".jpg", img, encode_param)
                if success:
                    return buf.tobytes()
            except Exception:
                pass
        return None

    def get_latest_png_bytes(self, size: Optional[Tuple[int, int]] = None) -> Optional[bytes]:
        """Encode the latest annotated frame into PNG bytes for REST / MCP."""
        with self._lock:
            if self.latest_frame_annotated is None:
                return None
            try:
                import cv2
                img = self.latest_frame_annotated
                if size is not None and (img.shape[1] != size[0] or img.shape[0] != size[1]):
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                success, buf = cv2.imencode(".png", img)
                if success:
                    return buf.tobytes()
            except Exception:
                pass
        return None


visual_debugger = VisualDebugLogger()
