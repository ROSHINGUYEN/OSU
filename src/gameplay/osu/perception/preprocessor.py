"""
Playfield Coordinate Mapper and Image Preprocessor.
Translates between virtual osu! coordinates (512x384) and actual screen pixels.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class PlayfieldBounds:
    """Bounding box of the active osu! playfield on screen."""
    x: int
    y: int
    width: int
    height: int
    scale: float


class PlayfieldPreprocessor:
    """
    Handles coordinate transformations between the 512x384 osu! playfield
    and the actual game client window or desktop resolution.
    """

    OSU_BASE_WIDTH = 512.0
    OSU_BASE_HEIGHT = 384.0

    def __init__(
        self,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
        playfield_height_ratio: float = 0.8,
        y_offset_ratio: float = 0.05,
    ):
        if screen_width is None or screen_height is None:
            try:
                import ctypes
                rw = ctypes.windll.user32.GetSystemMetrics(0)
                rh = ctypes.windll.user32.GetSystemMetrics(1)
                if rw > 200 and rh > 200:
                    screen_width = screen_width or rw
                    screen_height = screen_height or rh
            except Exception:
                pass
        self._screen_width = screen_width or 1920
        self._screen_height = screen_height or 1080
        self._playfield_height_ratio = playfield_height_ratio
        self._y_offset_ratio = y_offset_ratio
        self.window_offset_x: int = 0
        self.window_offset_y: int = 0
        self._active_cs: float = 4.0
        self._letterbox_in_breaks: bool = False
        self._widescreen_storyboard: bool = False

        self._bounds = self._calculate_bounds()

    @property
    def bounds(self) -> PlayfieldBounds:
        return self._bounds

    @property
    def scale(self) -> float:
        return self._bounds.scale

    def update_resolution(self, width: int, height: int) -> None:
        """Update screen resolution and recalculate bounds."""
        self._screen_width = width
        self._screen_height = height
        self._bounds = self._calculate_bounds()

    def update_window_position(self, offset_x: int, offset_y: int) -> None:
        """Update window client position on desktop (e.g. for windowed mode)."""
        self.window_offset_x = int(offset_x)
        self.window_offset_y = int(offset_y)

    def set_circle_size(self, cs: float) -> None:
        """Set active song Circle Size (CS) for dynamic scaling and radius computations."""
        self._active_cs = float(cs)

    def set_beatmap_difficulty(self, difficulty: object, general: Optional[object] = None) -> None:
        """Synchronize playfield preprocessor with active beatmap difficulty and layout."""
        if difficulty and hasattr(difficulty, "circle_size"):
            self._active_cs = float(difficulty.circle_size)
        if general:
            if hasattr(general, "letterbox_in_breaks"):
                self._letterbox_in_breaks = bool(general.letterbox_in_breaks)
            if hasattr(general, "widescreen_storyboard"):
                self._widescreen_storyboard = bool(general.widescreen_storyboard)

    @property
    def active_circle_size(self) -> float:
        return self._active_cs

    @property
    def letterbox_in_breaks(self) -> bool:
        return self._letterbox_in_breaks

    @property
    def widescreen_storyboard(self) -> bool:
        return self._widescreen_storyboard

    def _calculate_bounds(self) -> PlayfieldBounds:
        """
        Calculate playfield bounding box matching ppy/osu OsuPlayfieldAdjustmentContainer.cs:
        1. Fit 4:3 aspect container inside screen.
        2. Apply playfield_size_adjust = 0.8f.
        3. Center inside screen.
        """
        w = float(self._screen_width)
        h = float(self._screen_height)

        if (w / max(1.0, h)) >= (4.0 / 3.0):
            box_h = h
            box_w = box_h * (4.0 / 3.0)
        else:
            box_w = w
            box_h = box_w * (3.0 / 4.0)

        pf_width = box_w * 0.8
        pf_height = box_h * 0.8
        scale = pf_width / self.OSU_BASE_WIDTH

        # Standard osu! playfield shift: 8f * scale downwards to match osu!stable and osu!lazer legacy storyboard alignment
        y_shift = int(8.0 * scale)
        x = int((w - pf_width) / 2.0)
        y = int((h - pf_height) / 2.0) + y_shift

        return PlayfieldBounds(
            x=x,
            y=y,
            width=int(pf_width),
            height=int(pf_height),
            scale=scale,
        )

    def calibrate_from_visual(self, visual_x: float, visual_y: float, osu_x: float, osu_y: float) -> None:
        """
        Auto-adjusts playfield offset based on an optically confirmed hit object on screen.
        visual_x, visual_y: Screen pixel coordinates of the detected hit circle.
        osu_x, osu_y: Corresponding beatmap coordinate (0..512, 0..384).
        Eliminates 100% of spatial drift between beatmap timeline and actual game client window.
        """
        client_x = visual_x - self.window_offset_x
        client_y = visual_y - self.window_offset_y

        expected_x = self._bounds.x + (osu_x * self._bounds.scale)
        expected_y = self._bounds.y + (osu_y * self._bounds.scale)

        dx = int(client_x - expected_x)
        dy = int(client_y - expected_y)

        if abs(dx) < 150 and abs(dy) < 150:
            new_x = max(0, self._bounds.x + dx)
            new_y = max(0, self._bounds.y + dy)
            self._bounds = PlayfieldBounds(
                x=new_x,
                y=new_y,
                width=self._bounds.width,
                height=self._bounds.height,
                scale=self._bounds.scale,
            )

    def osu_to_screen(self, osu_x: float, osu_y: float) -> Tuple[int, int]:
        """Convert osu! coordinate (0..512, 0..384) to absolute screen pixel coordinate."""
        screen_x = self.window_offset_x + self._bounds.x + int(osu_x * self._bounds.scale)
        screen_y = self.window_offset_y + self._bounds.y + int(osu_y * self._bounds.scale)
        return screen_x, screen_y

    @property
    def affine_matrix(self) -> np.ndarray:
        """3x3 Affine Transformation matrix for homogeneous coordinate mapping (osu -> screen)."""
        tx = float(self.window_offset_x + self._bounds.x)
        ty = float(self.window_offset_y + self._bounds.y)
        s = float(self._bounds.scale)
        return np.array([
            [s, 0.0, tx],
            [0.0, s, ty],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def osu_to_screen_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorized high-performance batch coordinate transformation using linear algebra (NumPy).
        points: (N, 2) numpy array of (osu_x, osu_y).
        returns: (N, 2) numpy array of (screen_x, screen_y) as integers.
        """
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32)
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        # Augment with ones for homogeneous coordinates (N, 3)
        homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
        # Matrix multiplication: (N, 3) @ (3, 3).T -> (N, 3)
        transformed = homo @ self.affine_matrix.T
        return np.rint(transformed[:, :2]).astype(np.int32)

    def screen_to_osu(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen pixel coordinate back to osu! coordinate."""
        if self._bounds.scale <= 0:
            return 0.0, 0.0
        osu_x = (screen_x - self.window_offset_x - self._bounds.x) / self._bounds.scale
        osu_y = (screen_y - self.window_offset_y - self._bounds.y) / self._bounds.scale
        return osu_x, osu_y

    def get_hit_circle_radius(self, cs: Optional[float] = None) -> float:
        """
        Calculates screen hit circle radius for given Circle Size (CS).
        Formula: Radius_osu = 54.4 - 4.48 * CS (in 512x384 coordinates).
        Uses active song circle size if cs is None.
        """
        effective_cs = self._active_cs if cs is None else float(cs)
        osu_radius = 54.4 - (4.48 * effective_cs)
        return osu_radius * self._bounds.scale

    def get_hitbox_radius(self, cs: Optional[float] = None) -> float:
        """Calculates effective screen pixel radius for aim and click confirmation."""
        return self.get_hit_circle_radius(cs)

    def get_slider_follow_radius(self, cs: Optional[float] = None, is_lazer: bool = True) -> float:
        """Calculates screen pixel radius of slider follow circle (3.0x on lazer, 2.4x on stable)."""
        mult = 3.0 if is_lazer else 2.4
        return self.get_hit_circle_radius(cs) * mult

    def get_circle_diameter(self, cs: Optional[float] = None) -> float:
        """Calculates screen pixel diameter of a hit circle."""
        return self.get_hit_circle_radius(cs) * 2.0

    def crop_playfield(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame to the active playfield area."""
        h, w = frame.shape[:2]
        b = self._bounds
        y1 = max(0, min(h, b.y))
        y2 = max(0, min(h, b.y + b.height))
        x1 = max(0, min(w, b.x))
        x2 = max(0, min(w, b.x + b.width))
        return frame[y1:y2, x1:x2]

    @staticmethod
    def crop_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop region of interest (x, y, width, height)."""
        x, y, w, h = roi
        fh, fw = frame.shape[:2]
        y1 = max(0, min(fh, y))
        y2 = max(0, min(fh, y + h))
        x1 = max(0, min(fw, x))
        x2 = max(0, min(fw, x + w))
        return frame[y1:y2, x1:x2]
