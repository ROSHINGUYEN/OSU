"""
Multi-Object Tracker across sequential video frames.
Estimates object velocity and approach circle shrinking rates for visual timing.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.gameplay.osu.perception.detector import DetectedObject


@dataclass
class TrackedObject:
    track_id: str
    current_x: float
    current_y: float
    radius: float
    approach_scale: float
    obj_type: str
    confidence: float
    first_seen_ns: int
    last_seen_ns: int
    positions: List[Tuple[float, float, int]] = field(default_factory=list)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    is_don: bool = True
    is_katsu: bool = False
    is_big: bool = False
    ring_r: float = 0.0
    ring_b: float = 0.0
    hit_consumed: bool = False
    has_approach_ring: bool = True

    def update(self, detected: DetectedObject, t_ns: int) -> None:
        dt_s = (t_ns - self.last_seen_ns) / 1_000_000_000.0
        if dt_s > 0.001:
            self.velocity_x = (detected.x - self.current_x) / dt_s
            self.velocity_y = (detected.y - self.current_y) / dt_s

        self.current_x = detected.x
        self.current_y = detected.y
        self.radius = detected.radius
        self.approach_scale = detected.approach_scale
        self.confidence = detected.confidence
        self.is_don = getattr(detected, "is_don", True)
        self.is_katsu = getattr(detected, "is_katsu", False)
        self.is_big = getattr(detected, "is_big", False)
        self.ring_r = getattr(detected, "ring_r", 0.0)
        self.ring_b = getattr(detected, "ring_b", 0.0)
        self.has_approach_ring = getattr(detected, "has_approach_ring", True)
        self.last_seen_ns = t_ns
        self.positions.append((detected.x, detected.y, t_ns))
        if len(self.positions) > 30:
            self.positions.pop(0)


class ObjectTracker:
    """Tracks active objects across video frames."""

    def __init__(self, max_distance: float = 50.0, max_track_age_ns: int = 300_000_000):
        self._max_distance = max_distance
        self._max_track_age_ns = max_track_age_ns
        self._tracks: Dict[str, TrackedObject] = {}
        self._next_track_id = 1
        self._conveyor_velocity_x: float = -800.0  # Shared scroll velocity for Taiko belt

    @property
    def active_tracks(self) -> List[TrackedObject]:
        return list(self._tracks.values())

    @property
    def conveyor_velocity_x(self) -> float:
        return self._conveyor_velocity_x

    def update(self, detections: List[DetectedObject], timestamp_ns: int) -> List[TrackedObject]:
        """Match detections with existing tracks or create new ones."""
        unmatched_detections = list(detections)
        matched_track_ids = set()

        for det in detections:
            best_track_id: Optional[str] = None
            best_dist = float("inf")

            for t_id, track in self._tracks.items():
                if t_id in matched_track_ids:
                    continue
                # Motion-compensated position prediction
                dt_s = max(0.0, (timestamp_ns - track.last_seen_ns) / 1_000_000_000.0)
                pred_x = track.current_x + (track.velocity_x * dt_s if abs(track.velocity_x) > 10.0 else 0.0)
                pred_y = track.current_y + (track.velocity_y * dt_s if abs(track.velocity_y) > 10.0 else 0.0)
                dist = math.hypot(det.x - pred_x, det.y - pred_y)
                effective_max = self._max_distance * 1.5 if abs(track.velocity_x) > 10.0 else self._max_distance

                if dist < effective_max and dist < best_dist:
                    best_dist = dist
                    best_track_id = t_id

            if best_track_id is not None:
                self._tracks[best_track_id].update(det, timestamp_ns)
                matched_track_ids.add(best_track_id)
                v_x = self._tracks[best_track_id].velocity_x
                if v_x < -50.0:
                    # Exponential moving average of the conveyor scroll velocity
                    self._conveyor_velocity_x = 0.85 * self._conveyor_velocity_x + 0.15 * v_x
                if det in unmatched_detections:
                    unmatched_detections.remove(det)

        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            t_id = f"trk_{self._next_track_id}"
            self._next_track_id += 1
            new_track = TrackedObject(
                track_id=t_id,
                current_x=det.x,
                current_y=det.y,
                radius=det.radius,
                approach_scale=det.approach_scale,
                obj_type=det.obj_type,
                confidence=det.confidence,
                first_seen_ns=timestamp_ns,
                last_seen_ns=timestamp_ns,
                positions=[(det.x, det.y, timestamp_ns)],
                velocity_x=self._conveyor_velocity_x if getattr(det, "obj_type", "") == "taiko_note" else 0.0,
                is_don=getattr(det, "is_don", True),
                is_katsu=getattr(det, "is_katsu", False),
                is_big=getattr(det, "is_big", False),
                ring_r=getattr(det, "ring_r", 0.0),
                ring_b=getattr(det, "ring_b", 0.0),
            )
            self._tracks[t_id] = new_track

        # Prune dead tracks
        dead_keys = [
            k for k, track in self._tracks.items()
            if (timestamp_ns - track.last_seen_ns) > self._max_track_age_ns
        ]
        for k in dead_keys:
            del self._tracks[k]

        return list(self._tracks.values())

    def clear(self) -> None:
        self._tracks.clear()
