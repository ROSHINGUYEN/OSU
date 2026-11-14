"""
High-Performance Screen Capture Engine for Shiori AI osu! Gameplay.
Supports DirectX Desktop Duplication (dxcam), MSS fallback, and Mock backend for CI.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.data_channels import frame_channel
from src.gameplay.osu.core.event_bus import Event, event_bus

logger = logging.getLogger("Shiori.Osu.Capture")


@dataclass(frozen=True)
class ScreenFrame:
    """Represents a single captured video frame with nanosecond timing."""
    data: np.ndarray
    timestamp_ns: int
    sequence_id: int
    width: int
    height: int
    latency_ms: float = 0.0


def find_osu_window_hwnd() -> Optional[int]:
    """Auto-detect active osu! game window HWND on Windows using pure Win32 API with ranked scoring."""
    try:
        import ctypes
        import ctypes.wintypes
        desk = ctypes.windll.user32.OpenInputDesktop(0, False, 0x01FF)
        if desk:
            ctypes.windll.user32.SetThreadDesktop(desk)

        target_candidates: List[Tuple[int, int]] = []  # (score, hwnd)
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def _enum_cb(hwnd, _):
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                # Reject console, terminal, and command prompt windows immediately
                c_buf = ctypes.create_unicode_buffer(256)
                ctypes.windll.user32.GetClassNameW(hwnd, c_buf, 256)
                classname = c_buf.value.lower()
                if any(c in classname for c in ("console", "cascadia", "pseudoconsole")):
                    return True

                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                    title = buff.value.lower()

                    # Strictly ignore companion, discord, IDE, and terminal windows
                    if any(ign in title for ign in ("discord", "code", "visual studio", "shiori", "terminal", "powershell", "cmd.exe", "python")):
                        return True

                    if ("osu!" in title or title.startswith("osu")):
                        score = 10
                        if "sdl_app" in classname:
                            score += 500  # Official osu!lazer SDL window
                        elif "windowsforms" in classname:
                            score += 500  # Official osu!stable WinForms window

                        if " - " in title and "[" in title:
                            score += 100  # Active map or song select
                        if title.startswith("osu!") or title == "osu!":
                            score += 50
                        if not ctypes.windll.user32.IsIconic(hwnd):
                            score += 30
                        rect = ctypes.wintypes.RECT()
                        ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
                        cw = rect.right - rect.left
                        ch = rect.bottom - rect.top
                        if cw >= 640 and ch >= 480:
                            score += 40
                        if ctypes.windll.user32.GetForegroundWindow() == hwnd:
                            score += 25
                        target_candidates.append((score, hwnd))
            return True

        if desk:
            ctypes.windll.user32.EnumDesktopWindows(desk, WNDENUMPROC(_enum_cb), 0)
        if not target_candidates:
            ctypes.windll.user32.EnumWindows(WNDENUMPROC(_enum_cb), 0)

        if target_candidates:
            target_candidates.sort(key=lambda item: item[0], reverse=True)
            return target_candidates[0][1]
    except Exception as e:
        logger.debug(f"Could not find osu! HWND: {e}")
    return None


def find_osu_window_region() -> Optional[Tuple[int, int, int, int]]:
    """
    Auto-detect active osu! game window client region on Windows using pure Win32 API.
    Returns (left, top, right, bottom) in screen coordinates or None.
    Rejects minimized windows (IsIconic) and invalid negative coordinates (-32000).
    Auto-restores minimized windows (SW_RESTORE) if game was minimized.
    """
    try:
        import ctypes
        import ctypes.wintypes
        desk = ctypes.windll.user32.OpenInputDesktop(0, False, 0x01FF)
        if desk:
            ctypes.windll.user32.SetThreadDesktop(desk)
        hwnd = find_osu_window_hwnd()
        if hwnd:
            # Check window placement to detect minimization even if IsIconic is bypassed
            class WINDOWPLACEMENT(ctypes.Structure):
                _fields_ = [
                    ('length', ctypes.c_uint),
                    ('flags', ctypes.c_uint),
                    ('showCmd', ctypes.c_uint),
                    ('ptMinPosition', ctypes.wintypes.POINT),
                    ('ptMaxPosition', ctypes.wintypes.POINT),
                    ('rcNormalPosition', ctypes.wintypes.RECT)
                ]
            wp = WINDOWPLACEMENT()
            wp.length = ctypes.sizeof(WINDOWPLACEMENT)
            ctypes.windll.user32.GetWindowPlacement(hwnd, ctypes.byref(wp))
            if wp.showCmd == 2 or ctypes.windll.user32.IsIconic(hwnd):
                # Window is minimized! Auto-restore so capture and coordinates are valid
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                import time
                time.sleep(0.05)

            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
            p1 = ctypes.wintypes.POINT(rect.left, rect.top)
            p2 = ctypes.wintypes.POINT(rect.right, rect.bottom)
            ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(p1))
            ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(p2))

            if p1.x > -10000 and p1.y > -10000 and p2.x > p1.x and p2.y > p1.y:
                region = (int(p1.x), int(p1.y), int(p2.x), int(p2.y))
                if getattr(find_osu_window_region, "_last_region", None) != region:
                    find_osu_window_region._last_region = region
                    logger.info(f"Auto-detected osu! client window region: {region} (HWND: {hwnd})")
                return region
    except Exception as e:
        logger.debug(f"Could not auto-detect osu! window region: {e}")
    return None


def capture_osu_window_printwindow(hwnd: int, width: int, height: int) -> Optional[np.ndarray]:
    """Capture pure osu! client area directly via PrintWindow (PW_RENDERFULLCONTENT = 2)."""
    try:
        import ctypes
        import cv2
        desk = ctypes.windll.user32.OpenInputDesktop(0, False, 0x01FF)
        if desk:
            ctypes.windll.user32.SetThreadDesktop(desk)

        hdc_screen = ctypes.windll.user32.GetDC(hwnd)
        if not hdc_screen:
            return None
        hdc_mem = ctypes.windll.gdi32.CreateCompatibleDC(hdc_screen)
        hbm = ctypes.windll.gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
        ctypes.windll.gdi32.SelectObject(hdc_mem, hbm)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32), ("biHeight", ctypes.c_int32),
                ("biPlanes", ctypes.c_uint16), ("biBitCount", ctypes.c_uint16),
                ("biCompression", ctypes.c_uint32), ("biSizeImage", ctypes.c_uint32),
                ("biXPelsPerMeter", ctypes.c_int32), ("biYPelsPerMeter", ctypes.c_int32),
                ("biClrUsed", ctypes.c_uint32), ("biClrImportant", ctypes.c_uint32),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = width
        bmi.biHeight = -height
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0
        buf = (ctypes.c_ubyte * (width * height * 4))()

        res = ctypes.windll.user32.PrintWindow(hwnd, hdc_mem, 2)
        if res == 1:
            ctypes.windll.gdi32.GetDIBits(hdc_mem, hbm, 0, height, ctypes.byref(buf), ctypes.byref(bmi), 0)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        else:
            rgb = None

        ctypes.windll.gdi32.DeleteObject(hbm)
        ctypes.windll.gdi32.DeleteDC(hdc_mem)
        ctypes.windll.user32.ReleaseDC(hwnd, hdc_screen)
        return rgb
    except Exception:
        return None


def focus_osu_window() -> bool:
    """
    An toàn: Kiểm tra cửa sổ osu! mà không cưỡng bức giật tiêu điểm (no aggressive foreground stealing),
    để người dùng tự do Alt+Tab, thu nhỏ cửa sổ hoặc tắt game bất cứ lúc nào.
    """
    try:
        hwnd = find_osu_window_hwnd()
        if not hwnd:
            return False
        import ctypes
        return ctypes.windll.user32.GetForegroundWindow() == hwnd
    except Exception as e:
        logger.debug(f"Could not check osu! window focus: {e}")
    return False


class BaseCaptureBackend(ABC):
    """Abstract interface for screen capture backends."""

    @abstractmethod
    def start(self) -> bool:
        """Initialize and start capture backend."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop and release backend resources."""
        pass

    @abstractmethod
    def grab(self) -> Optional[np.ndarray]:
        """Grab a single raw frame as a numpy ndarray (RGB or BGR)."""
        pass

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Check if backend is currently running."""
        pass

    def update_region(self, region: Optional[Tuple[int, int, int, int]]) -> None:
        """Update the capture bounding box dynamically (left, top, right, bottom)."""
        pass


class DXCamCaptureBackend(BaseCaptureBackend):
    """DirectX Desktop Duplication API (DDA) backend using dxcam (ETS2 architecture: 30-60 FPS, frame cache)."""

    def __init__(self, device_idx: int = 0, target_fps: int = 60, region: Optional[Tuple[int, int, int, int]] = None):
        self._device_idx = device_idx
        self._target_fps = target_fps
        self._region = region  # (left, top, right, bottom)
        self._cam = None
        self._active = False
        self._consecutive_fails = 0
        self._cached_frame: Optional[np.ndarray] = None
        self._cached_time: float = 0.0
        self._osu_hwnd: Optional[int] = None

    def start(self) -> bool:
        try:
            import dxcam
            self._osu_hwnd = find_osu_window_hwnd()
            # Clean up any stale camera reference first
            if self._cam is not None:
                try:
                    self._cam.stop()
                except Exception:
                    pass
                del self._cam
                self._cam = None

            if self._region is None:
                self._region = find_osu_window_region()

            self._cam = dxcam.create(device_idx=self._device_idx, output_color="RGB")
            if self._cam is None:
                logger.error("dxcam.create() returned None. Check display connection.")
                return False
            self._cam.start(target_fps=self._target_fps, video_mode=True, region=self._region)
            self._active = True
            self._consecutive_fails = 0
            self._cached_frame = None
            self._cached_time = 0.0
            logger.info(f"DXCam capture started on device {self._device_idx} at {self._target_fps} FPS.")
            return True
        except Exception as e:
            logger.warning(f"Failed to start DXCam backend: {e}. Fallback recommended.")
            self._active = False
            return False

    def stop(self) -> None:
        if self._cam is not None:
            try:
                if self._active:
                    self._cam.stop()
            except Exception as e:
                logger.warning(f"Error stopping DXCam: {e}")
            try:
                del self._cam
            except Exception:
                pass
            self._cam = None
        self._active = False
        self._consecutive_fails = 0
        self._cached_frame = None

    def grab(self) -> Optional[np.ndarray]:
        if not self._active or self._cam is None:
            return None
        now = time.time()
        # Fast cache check within 8ms (prevents hammering DXGI buffer if called faster than monitor refresh)
        if self._cached_frame is not None and (now - self._cached_time) < 0.008:
            return self._cached_frame

        try:
            frame = self._cam.get_latest_frame()
            if frame is not None:
                self._consecutive_fails = 0
                self._cached_frame = frame
                self._cached_time = now
                return frame

            # If video_mode buffer is starved or hung, try synchronous grab fallback
            self._consecutive_fails += 1
            if self._consecutive_fails >= 3:
                try:
                    direct = self._cam.grab(region=self._region)
                    if direct is not None:
                        self._consecutive_fails = 0
                        self._cached_frame = direct
                        self._cached_time = now
                        return direct
                except Exception:
                    pass

            if self._consecutive_fails > 10:
                # Signal freeze so CaptureService can auto-recover
                logger.warning("DXCam frozen (10 consecutive None frames). Signaling recovery...")
                self._active = False
                return None
        except Exception as e:
            logger.debug(f"DXCam grab exception: {e}")
            self._consecutive_fails += 1
            if self._consecutive_fails > 10:
                self._active = False
        return None

    @property
    def is_active(self) -> bool:
        return self._active

    def update_region(self, region: Optional[Tuple[int, int, int, int]]) -> None:
        if self._region == region:
            return
        self._region = region
        if self._active and self._cam is not None:
            try:
                self._cam.stop()
                self._cam.start(target_fps=self._target_fps, video_mode=True, region=self._region)
                self._cached_frame = None
                logger.info(f"DXCam capture restarted with updated region: {self._region}")
            except Exception as e:
                logger.warning(f"Failed to restart DXCam with updated region: {e}")


class MSSCaptureBackend(BaseCaptureBackend):
    """Multi-Screen Shot (MSS) backend for cross-platform/reliable fallback."""

    def __init__(self, monitor_idx: int = 1, region: Optional[Tuple[int, int, int, int]] = None):
        self._monitor_idx = monitor_idx
        self._region = region  # (left, top, width, height)
        self._sct = None
        self._active = False
        self._monitor = {}

    def start(self) -> bool:
        try:
            import mss
            self._sct = mss.mss()
            monitors = self._sct.monitors
            if self._monitor_idx < len(monitors):
                base_mon = monitors[self._monitor_idx]
            else:
                base_mon = monitors[0]

            if self._region:
                r_left, r_top, r_right, r_bottom = self._region
                if r_right > r_left and r_bottom > r_top:
                    w = r_right - r_left
                    h = r_bottom - r_top
                else:
                    w = r_right
                    h = r_bottom
                self._monitor = {
                    "left": int(r_left),
                    "top": int(r_top),
                    "width": int(w),
                    "height": int(h),
                }
            else:
                self._monitor = base_mon

            self._active = True
            logger.info(f"MSS capture started on monitor {self._monitor_idx}: {self._monitor}")
            return True
        except Exception as e:
            logger.error(f"Failed to start MSS capture backend: {e}")
            self._active = False
            return False

    def stop(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
        self._sct = None
        self._active = False

    def grab(self) -> Optional[np.ndarray]:
        if not self._active or self._sct is None:
            return None
        try:
            import ctypes
            desk = ctypes.windll.user32.OpenInputDesktop(0, False, 0x01FF)
            if desk:
                ctypes.windll.user32.SetThreadDesktop(desk)
        except Exception:
            pass

        try:
            raw = self._sct.grab(self._monitor)
            # Robust conversion from BGRA to RGB
            import cv2
            frame_bgra = np.array(raw, dtype=np.uint8)
            return cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2RGB)
        except Exception as e:
            logger.debug(f"MSS grab failed: {e}")
            return None

    @property
    def is_active(self) -> bool:
        return self._active

    def update_region(self, region: Optional[Tuple[int, int, int, int]]) -> None:
        self._region = region
        if not region:
            if self._sct and self._monitor_idx < len(self._sct.monitors):
                self._monitor = self._sct.monitors[self._monitor_idx]
            return
        r_left, r_top, r_right, r_bottom = region
        w = r_right - r_left if r_right > r_left else r_right
        h = r_bottom - r_top if r_bottom > r_top else r_bottom
        self._monitor = {
            "left": int(r_left),
            "top": int(r_top),
            "width": int(w),
            "height": int(h),
        }


class Win32GDICaptureBackend(BaseCaptureBackend):
    """
    Win32 GDI BitBlt persistent capture backend.
    Reuses HDC / HBITMAP objects to avoid per-frame allocation overhead.
    Reliable fallback when DXCam / MSS fail (no DirectX privilege required).
    Benchmark: ~15ms per grab (≈60 FPS) at 1280x720 using a pre-allocated DC.
    """

    def __init__(self, region: Optional[Tuple[int, int, int, int]] = None):
        self._region = region  # (left, top, right, bottom)
        self._active = False
        self._user32 = None
        self._gdi32 = None
        self._l = self._t = self._w = self._h = 0
        self._hdc_screen = None
        self._hdc_mem = None
        self._hbm = None
        self._old_bm = None
        self._buf = None
        self._bmi = None
        self._hwnd: Optional[int] = None

    def start(self) -> bool:
        try:
            import ctypes
            import ctypes.wintypes

            self._user32 = ctypes.windll.user32
            self._gdi32 = ctypes.windll.gdi32
            self._hwnd = find_osu_window_hwnd()

            desk = self._user32.OpenInputDesktop(0, False, 0x01FF)
            if desk:
                self._user32.SetThreadDesktop(desk)

            if self._region:
                l, t, r, b = self._region
                self._l, self._t = l, t
                self._w, self._h = r - l, b - t
            else:
                # Auto-detect osu window
                region = find_osu_window_region()
                if region:
                    self._l, self._t = region[0], region[1]
                    self._w, self._h = region[2] - region[0], region[3] - region[1]
                else:
                    self._l, self._t = 0, 0
                    self._w = self._user32.GetSystemMetrics(0)
                    self._h = self._user32.GetSystemMetrics(1)

            if self._w <= 0 or self._h <= 0:
                logger.error(f"Win32GDI: invalid capture size {self._w}x{self._h}")
                return False

            # Allocate persistent GDI resources
            self._hdc_screen = self._user32.GetDC(0)

            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32), ("biHeight", ctypes.c_int32),
                    ("biPlanes", ctypes.c_uint16), ("biBitCount", ctypes.c_uint16),
                    ("biCompression", ctypes.c_uint32), ("biSizeImage", ctypes.c_uint32),
                    ("biXPelsPerMeter", ctypes.c_int32), ("biYPelsPerMeter", ctypes.c_int32),
                    ("biClrUsed", ctypes.c_uint32), ("biClrImportant", ctypes.c_uint32),
                ]

            self._bmi = BITMAPINFOHEADER()
            self._bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            self._bmi.biWidth = self._w
            self._bmi.biHeight = -self._h  # top-down DIB
            self._bmi.biPlanes = 1
            self._bmi.biBitCount = 32
            self._bmi.biCompression = 0  # BI_RGB

            self._hdc_mem = self._gdi32.CreateCompatibleDC(self._hdc_screen)
            self._hbm = self._gdi32.CreateCompatibleBitmap(self._hdc_screen, self._w, self._h)
            self._old_bm = self._gdi32.SelectObject(self._hdc_mem, self._hbm)
            self._buf = (ctypes.c_ubyte * (self._w * self._h * 4))()

            self._active = True
            logger.info(f"Win32GDI capture started: region ({self._l},{self._t}) size {self._w}x{self._h}")
            return True
        except Exception as e:
            logger.warning(f"Win32GDI capture start failed: {e}")
            self._active = False
            return False

    def stop(self) -> None:
        try:
            import ctypes
            if self._hdc_mem and self._old_bm:
                self._gdi32.SelectObject(self._hdc_mem, self._old_bm)
            if self._hbm:
                self._gdi32.DeleteObject(self._hbm)
            if self._hdc_mem:
                self._gdi32.DeleteDC(self._hdc_mem)
            if self._hdc_screen:
                self._user32.ReleaseDC(0, self._hdc_screen)
        except Exception:
            pass
        self._active = False
        self._hdc_screen = self._hdc_mem = self._hbm = self._old_bm = None

    def grab(self) -> Optional[np.ndarray]:
        if not self._active:
            return None
        try:
            import ctypes
            # Re-acquire desktop thread input if needed
            desk = self._user32.OpenInputDesktop(0, False, 0x01FF)
            if desk:
                self._user32.SetThreadDesktop(desk)

            # Prioritize direct PrintWindow (PW_RENDERFULLCONTENT = 2) when osu window is available
            # Immune to overlapping windows (Vision GUI / Discord) and eliminates recursive mirror loop
            if self._hwnd and self._user32.IsWindow(self._hwnd) and not self._user32.IsIconic(self._hwnd):
                if self._user32.PrintWindow(self._hwnd, self._hdc_mem, 2) == 1:
                    self._gdi32.GetDIBits(
                        self._hdc_mem, self._hbm, 0, self._h,
                        self._buf, ctypes.byref(self._bmi), 0,  # DIB_RGB_COLORS
                    )
                    bgra = np.frombuffer(self._buf, dtype=np.uint8).reshape((self._h, self._w, 4))
                    import cv2
                    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)

            self._gdi32.BitBlt(
                self._hdc_mem, 0, 0, self._w, self._h,
                self._hdc_screen, self._l, self._t,
                0x00CC0020,  # SRCCOPY
            )
            self._gdi32.GetDIBits(
                self._hdc_mem, self._hbm, 0, self._h,
                self._buf, ctypes.byref(self._bmi), 0,  # DIB_RGB_COLORS
            )
            bgra = np.frombuffer(self._buf, dtype=np.uint8).reshape((self._h, self._w, 4))
            # SIMD AVX2 BGRA → RGB conversion (6x faster than numpy slicing)
            import cv2
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        except Exception as e:
            logger.debug(f"Win32GDI grab failed: {e}")
            return None

    @property
    def is_active(self) -> bool:
        return self._active

    def update_region(self, region: Optional[Tuple[int, int, int, int]]) -> None:
        """Update capture region and reallocate persistent GDI buffers if size changed."""
        if not region:
            return
        l, t, r, b = region
        new_w = int(r - l)
        new_h = int(b - t)
        if new_w <= 0 or new_h <= 0:
            return
        self._region = region
        self._l, self._t = int(l), int(t)

        if self._active and self._gdi32 and self._hdc_screen:
            if new_w != self._w or new_h != self._h or self._hbm is None:
                try:
                    import ctypes
                    self._w = new_w
                    self._h = new_h

                    if self._hdc_mem and self._old_bm:
                        self._gdi32.SelectObject(self._hdc_mem, self._old_bm)
                    if self._hbm:
                        self._gdi32.DeleteObject(self._hbm)

                    self._bmi.biWidth = self._w
                    self._bmi.biHeight = -self._h
                    self._hbm = self._gdi32.CreateCompatibleBitmap(self._hdc_screen, self._w, self._h)
                    self._old_bm = self._gdi32.SelectObject(self._hdc_mem, self._hbm)
                    self._buf = (ctypes.c_ubyte * (self._w * self._h * 4))()
                    logger.debug(f"Win32GDICaptureBackend reallocated buffer to {self._w}x{self._h} at ({self._l}, {self._t})")
                except Exception as e:
                    logger.warning(f"Win32GDICaptureBackend update_region failed: {e}")
        else:
            self._w = new_w
            self._h = new_h


class MockCaptureBackend(BaseCaptureBackend):
    """Deterministic mock backend producing synthetic frames for unit tests & simulation."""

    def __init__(self, width: int = 1280, height: int = 720, color: Tuple[int, int, int] = (0, 0, 0)):
        self._width = width
        self._height = height
        self._color = color
        self._active = False
        self._frame_count = 0

    def start(self) -> bool:
        self._active = True
        self._frame_count = 0
        return True

    def stop(self) -> None:
        self._active = False

    def grab(self) -> Optional[np.ndarray]:
        if not self._active:
            return None
        self._frame_count += 1
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        frame[:, :] = self._color
        # Add a moving synthetic dot to verify frame changes
        x = (self._frame_count * 10) % self._width
        y = (self._frame_count * 10) % self._height
        frame[max(0, y - 5):min(self._height, y + 5), max(0, x - 5):min(self._width, x + 5)] = [255, 255, 255]
        return frame

    @property
    def is_active(self) -> bool:
        return self._active

    def update_region(self, region: Optional[Tuple[int, int, int, int]]) -> None:
        self._region = region
        if region:
            l, t, r, b = region
            self._width = max(1, int(r - l))
            self._height = max(1, int(b - t))


class CaptureService:
    """
    High-level screen capture service.
    Orchestrates capture threads, pushes to DataChannel, and tracks FPS/dropped frames.
    """

    def __init__(self, backend_type: Optional[str] = None, fps: int = 60, region: Optional[Tuple[int, int, int, int]] = None):
        if backend_type is None:
            self._backend_type = "mock"  # Camera capture eliminated. Default to mock backend for 0ms memory engine.
        else:
            self._backend_type = backend_type.lower()
        self._target_fps = fps
        self._region = region
        self._backend: Optional[BaseCaptureBackend] = None

        # Persistent backend used by grab_single_frame (avoids start/stop per call)
        self._single_backend: Optional[BaseCaptureBackend] = None
        self._single_backend_type: str = ""

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._sequence_id = 0
        self._frames_captured = 0
        self._dropped_frames = 0
        self._actual_fps = 0.0
        self._last_fps_calc_time = 0.0
        self._fps_counter = 0

    @property
    def is_running(self) -> bool:
        return self._backend is not None and self._backend.is_active and not self._stop_event.is_set()

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    @property
    def frames_captured(self) -> int:
        return self._frames_captured

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def update_region(self, region: Optional[Tuple[int, int, int, int]]) -> None:
        """Dynamically update capture region across active and single-grab backends."""
        if not region:
            return
        self._region = region
        if self._backend is not None and hasattr(self._backend, "update_region"):
            self._backend.update_region(region)
        if self._single_backend is not None and hasattr(self._single_backend, "update_region"):
            self._single_backend.update_region(region)
        frame_channel.clear()

    def _create_backend(self) -> BaseCaptureBackend:
        if self._backend_type == "dxcam":
            return DXCamCaptureBackend(target_fps=self._target_fps, region=self._region)
        elif self._backend_type == "mss":
            return MSSCaptureBackend(region=self._region)
        elif self._backend_type == "win32gdi":
            return Win32GDICaptureBackend(region=self._region)
        elif self._backend_type == "mock":
            return MockCaptureBackend(width=1280, height=720)
        else:
            raise ValueError(f"Unknown capture backend: {self._backend_type}")

    def start(self) -> bool:
        """Start the capture service and background polling thread."""
        if self.is_running:
            return True

        # Auto-detect osu! window region if not manually specified and not in mock test mode
        if self._region is None and self._backend_type != "mock":
            auto_region = find_osu_window_region()
            if auto_region:
                self._region = auto_region
                logger.info(f"CaptureService automatically bound to osu! window region: {self._region}")

        self._backend = self._create_backend()
        success = self._backend.start()

        # If DXCam fails, auto-fallback to MSS
        if not success and self._backend_type == "dxcam":
            logger.warning("DXCam initialization failed, falling back to Win32GDI backend...")
            self._backend_type = "win32gdi"
            self._backend = self._create_backend()
            success = self._backend.start()

        # If Win32GDI/MSS fails, auto-fallback to MSS then Mock
        if not success and self._backend_type in ("win32gdi", "mss"):
            logger.warning(f"{self._backend_type} initialization failed, falling back to Win32GDI/Mock capture backend...")
            if self._backend_type == "win32gdi":
                self._backend_type = "mss"
                self._backend = self._create_backend()
                success = self._backend.start()
            if not success:
                self._backend_type = "mock"
                self._backend = self._create_backend()
                success = self._backend.start()

        if not success:
            logger.error("Failed to initialize any capture backend.")
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="OsuCaptureLoop")
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the capture service and release resources."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        if self._backend is not None:
            self._backend.stop()
            self._backend = None
        logger.info(f"CaptureService stopped. Total captured: {self._frames_captured}, Dropped: {self._dropped_frames}")

    def grab_single_frame(self) -> Optional[ScreenFrame]:
        """
        Synchronously grab a single frame.
        Uses the running background backend if available.
        Otherwise, creates and persists a Win32GDI backend (avoids start/stop-per-call).
        """
        if self._backend is not None and self._backend.is_active:
            # Read from the active ring channel directly to prevent thread collisions on backend
            latest = frame_channel.get_latest()
            if latest is not None and latest.data is not None:
                return latest
            frame_arr = self._backend.grab()
        else:
            if self._backend_type != "mock":
                auto_reg = find_osu_window_region()
                if auto_reg and auto_reg != self._region:
                    self.update_region(auto_reg)

            # Lazily create a persistent single-grab backend
            if not hasattr(self, "_single_backend") or self._single_backend is None or not self._single_backend.is_active:
                if self._backend_type == "mock":
                    self._single_backend = self._create_backend()
                    self._single_backend.start()
                else:
                    if self._region is None:
                        auto_region = find_osu_window_region()
                        if auto_region:
                            self._region = auto_region

                    win32_backend = Win32GDICaptureBackend(region=self._region)
                    if win32_backend.start():
                        self._single_backend = win32_backend
                        logger.debug("grab_single_frame: using persistent Win32GDI backend")
                    else:
                        fallback = self._create_backend()
                        if fallback.start():
                            self._single_backend = fallback
                        else:
                            logger.warning("grab_single_frame: all backends failed")
                            return None

            frame_arr = self._single_backend.grab()

        if frame_arr is None:
            return None

        t_ns = clock.now_ns()
        self._sequence_id += 1
        h, w = frame_arr.shape[:2]
        return ScreenFrame(
            data=frame_arr,
            timestamp_ns=t_ns,
            sequence_id=self._sequence_id,
            width=w,
            height=h,
        )

    def _capture_loop(self) -> None:
        """Continuous background capture loop with target frame pacing and auto-recovery."""
        frame_interval_ns = int(1_000_000_000 / self._target_fps)
        next_deadline_ns = clock.now_ns()
        self._last_fps_calc_time = time.time()
        self._fps_counter = 0
        consecutive_drops = 0

        while not self._stop_event.is_set():
            t_start_ns = clock.now_ns()
            raw_frame = self._backend.grab() if self._backend else None

            if raw_frame is not None:
                consecutive_drops = 0
                t_captured_ns = clock.now_ns()
                self._sequence_id += 1
                self._frames_captured += 1
                self._fps_counter += 1

                h, w = raw_frame.shape[:2]
                latency_ms = (t_captured_ns - t_start_ns) / 1_000_000.0

                frame = ScreenFrame(
                    data=raw_frame,
                    timestamp_ns=t_captured_ns,
                    sequence_id=self._sequence_id,
                    width=w,
                    height=h,
                    latency_ms=latency_ms,
                )

                # Push to high-speed circular ring buffer channel
                frame_channel.push(frame)

                # FPS calculation
                now = time.time()
                if now - self._last_fps_calc_time >= 1.0:
                    self._actual_fps = self._fps_counter / (now - self._last_fps_calc_time)
                    self._fps_counter = 0
                    self._last_fps_calc_time = now
            else:
                self._dropped_frames += 1
                consecutive_drops += 1
                # Auto-recovery: If DXCam freezes or returns None for 20 frames, switch to Win32GDI seamlessly
                if consecutive_drops >= 20 and self._backend_type == "dxcam":
                    logger.warning("CaptureService: DXCam frozen (20 dropped frames). Auto-recovering to Win32GDI...")
                    try:
                        if self._backend:
                            self._backend.stop()
                    except Exception:
                        pass
                    self._backend_type = "win32gdi"
                    self._backend = self._create_backend()
                    if self._backend.start():
                        consecutive_drops = 0
                        logger.info("CaptureService: successfully recovered and switched to Win32GDI!")
                elif consecutive_drops >= 30 and self._backend_type == "win32gdi":
                    # Re-acquire desktop input for Win32GDI if screen locked
                    if self._backend and hasattr(self._backend, "start"):
                        self._backend.stop()
                        self._backend.start()
                    consecutive_drops = 0

            # Deadline scheduling for next frame
            next_deadline_ns += frame_interval_ns
            if clock.now_ns() > next_deadline_ns:
                # Catch up if running behind
                next_deadline_ns = clock.now_ns()
            else:
                clock.sleep_until_ns(next_deadline_ns)


# Global default capture service instance
capture_service = CaptureService(fps=240)
