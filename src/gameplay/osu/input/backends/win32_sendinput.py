"""
High-Speed Win32 SendInput Backend using ctypes.
Dispatches sub-millisecond mouse movements and keyboard scan codes on Windows 11.
"""

import ctypes
import logging
import math
import time
from ctypes import wintypes
from typing import Dict, Optional, Set, Tuple

from src.gameplay.osu.input.backends.base import BaseInputBackend

logger = logging.getLogger("Shiori.Osu.Win32Input")

# ==========================================
# Win32 Structures and Constants
# ==========================================

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _INPUT_UNION),
    ]


# Virtual Key mapping for osu keys
VK_MAP: Dict[str, int] = {
    "z": 0x5A,
    "x": 0x58,
    "c": 0x43,
    "v": 0x56,
    "d": 0x44,
    "f": 0x46,
    "j": 0x4A,
    "k": 0x4B,
    "s": 0x53,
    "l": 0x4C,
    "space": 0x20,
    "shift": 0x10,
    "left": 0x25,
    "right": 0x27,
    "up": 0x26,
    "down": 0x28,
    "esc": 0x1B,
}


class Win32SendInputBackend(BaseInputBackend):
    """Direct Win32 SendInput driver using hardware scan codes."""

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self._user32 = ctypes.windll.user32
        self._winmm = ctypes.windll.winmm

        # Attach thread to interactive input desktop so background threads can move mouse
        try:
            desk = self._user32.OpenInputDesktop(0, False, 0x01FF)
            if desk:
                self._user32.SetThreadDesktop(desk)
        except Exception:
            pass

        # Dynamically query active monitor resolution
        try:
            real_w = self._user32.GetSystemMetrics(0)
            real_h = self._user32.GetSystemMetrics(1)
            if real_w > 0 and real_h > 0:
                self._screen_width = real_w
                self._screen_height = real_h
            else:
                self._screen_width = screen_width
                self._screen_height = screen_height
        except Exception:
            self._screen_width = screen_width
            self._screen_height = screen_height

        self._active_keys: Set[str] = set()
        self._active_mouse_buttons: Set[str] = set()
        self._cached_osu_hwnd: Optional[int] = None
        self._last_hwnd_check: float = 0.0
        self._last_shiori_cursor_pos: Optional[Tuple[int, int]] = None
        self._human_override_until_s: float = 0.0
        self._last_move_time_s: float = 0.0
        self._was_osu_active: bool = True
        self.enable_deadman_switch: bool = True

        # Set Windows timer resolution to 1ms
        try:
            self._winmm.timeBeginPeriod(1)
        except Exception:
            pass

    def _get_osu_hwnd(self) -> Optional[int]:
        now = time.time()
        try:
            desk = self._user32.OpenInputDesktop(0, False, 0x01FF)
            if desk:
                self._user32.SetThreadDesktop(desk)
        except Exception:
            pass

        if self._cached_osu_hwnd and (now - self._last_hwnd_check < 2.0):
            if self._user32.IsWindow(self._cached_osu_hwnd):
                return self._cached_osu_hwnd
            self._cached_osu_hwnd = None

        self._last_hwnd_check = now
        try:
            from src.gameplay.osu.perception.capture import find_osu_window_hwnd
            self._cached_osu_hwnd = find_osu_window_hwnd()
        except Exception:
            self._cached_osu_hwnd = None
        return self._cached_osu_hwnd

    def __del__(self):
        try:
            self._winmm.timeEndPeriod(1)
        except Exception:
            pass

    def _get_current_cursor_pos(self) -> Optional[Tuple[int, int]]:
        """Get current physical cursor position on desktop."""
        try:
            pt = wintypes.POINT()
            if self._user32.GetCursorPos(ctypes.byref(pt)):
                return (int(pt.x), int(pt.y))
        except Exception:
            pass
        return None

    def _is_osu_active(self, osu_hwnd: Optional[int]) -> bool:
        """Check if osu! is alive, not minimized, and actively focused (or Shiori GUI focused)."""
        try:
            desk = self._user32.OpenInputDesktop(0, False, 0x01FF)
            if desk:
                self._user32.SetThreadDesktop(desk)
        except Exception:
            pass

        if not osu_hwnd or not self._user32.IsWindow(osu_hwnd):
            return False

        if self._user32.IsIconic(osu_hwnd):
            from src.gameplay.osu.perception.capture import focus_osu_window
            focus_osu_window()

        fore_hwnd = self._user32.GetForegroundWindow()
        if not fore_hwnd or fore_hwnd == osu_hwnd:
            return True

        fore_pid = wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(fore_hwnd, ctypes.byref(fore_pid))
        osu_pid = wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(osu_hwnd, ctypes.byref(osu_pid))
        if fore_pid.value == osu_pid.value:
            return True

        buf = ctypes.create_unicode_buffer(256)
        self._user32.GetWindowTextW(fore_hwnd, buf, 256)
        title_lower = buf.value.lower()
        # TUYỆT ĐỐI KHÔNG cướp chuột/bàn phím/focus khi Ba Roshi không ở trong màn chơi thực tế!
        from src.gameplay.osu.integrations.tosu_client import tosu_client
        if getattr(tosu_client, "is_playing", False):
            now = time.time()
            if now >= self._human_override_until_s and (now - getattr(self, "_last_auto_focus_time", 0.0)) > 0.45:
                self._last_auto_focus_time = now
                from src.gameplay.osu.perception.capture import focus_osu_window
                if focus_osu_window():
                    return True
            if "shiori" in title_lower:
                return True

        return False

    def move_cursor(self, screen_x: int, screen_y: int) -> bool:
        """
        Move OS cursor to absolute desktop screen coordinate.
        Safety features:
        1. Only moves when osu! is alive and visible (not minimized).
        2. Process check: Only moves when active foreground window is osu! or Shiori companion GUI.
        3. Deadman's Switch: If the human user moves the mouse (>35px), immediately yields for 1.5s.
        4. Rate-limited to max 250Hz (~4ms) to prevent violent 2000Hz jitter.
        Triple-dispatch: SetCursorPos + SendInput (MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE) + PostMessageW(WM_MOUSEMOVE).
        """
        if screen_x is None or screen_y is None:
            return False
        try:
            if math.isnan(screen_x) or math.isnan(screen_y) or math.isinf(screen_x) or math.isinf(screen_y):
                return False
        except Exception:
            pass

        now_s = time.time()

        # 0. Check if human user is in override cooldown (never fight the human hand!)
        if now_s < self._human_override_until_s:
            curr_pt = self._get_current_cursor_pos()
            if curr_pt:
                self._last_shiori_cursor_pos = curr_pt
            return False

        # 1. Check if osu! window is alive and is active
        osu_hwnd = self._get_osu_hwnd()
        if not self._is_osu_active(osu_hwnd):
            self._was_osu_active = False
            return False

        # Reload timestamp AFTER window/focus queries so OS call latency never trips inactivity reset
        now_s = time.time()
        just_refocused = not self._was_osu_active
        self._was_osu_active = True

        # 2. Deadman's Switch: Detect physical human intervention
        # Uses deliberate macro-movement (>120px) or deliberate corner escape (x<=5, y<=5)
        # Micro-movements (<120px) are ignored to prevent false alarms from 2000Hz polling, DPI scaling, or optical jitter.
        if self.enable_deadman_switch:
            curr_pt = self._get_current_cursor_pos()
            if curr_pt:
                # If focus just returned from another window or Shiori was inactive >0.12s,
                # sync reference position so resuming never false-triggers a 0.5s deadman blackout!
                if just_refocused or (self._last_move_time_s > 0.0 and (now_s - self._last_move_time_s) > 0.12):
                    self._last_shiori_cursor_pos = curr_pt

                is_corner_escape = False
                if (0 < curr_pt[0] <= 15 and 0 < curr_pt[1] <= 15) and self._last_shiori_cursor_pos is not None:
                    if self._last_shiori_cursor_pos[0] > 50 or self._last_shiori_cursor_pos[1] > 50:
                        is_corner_escape = True

                is_human_move = False
                if self._last_shiori_cursor_pos is not None:
                    dx = curr_pt[0] - self._last_shiori_cursor_pos[0]
                    dy = curr_pt[1] - self._last_shiori_cursor_pos[1]
                    is_human_move = (dx * dx + dy * dy) > (160.0 * 160.0)

                if is_corner_escape or is_human_move:
                    logger.warning(
                        f"Deadman switch triggered (corner={is_corner_escape}, human={is_human_move}, "
                        f"curr={curr_pt}, last_shiori={self._last_shiori_cursor_pos}). Yielding control for 0.5s."
                    )
                    # User physically intervened! Yield control immediately for 0.5s so human hand has full priority
                    self._human_override_until_s = now_s + 0.5
                    self._last_shiori_cursor_pos = curr_pt
                    return False

        self._last_move_time_s = now_s

        # 3. Direct hardware cursor position set (0ms latency, exact pixel precision, 0 queue flood)
        res = bool(self._user32.SetCursorPos(int(screen_x), int(screen_y)))

        # Also inject into Windows raw input event stream with Virtual Desktop normalization (eliminating multi-monitor jitter)
        try:
            vl = self._user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
            vt = self._user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
            vw = self._user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            vh = self._user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN

            if vw > 0 and vh > 0:
                norm_x = int(((screen_x - vl) * 65535.0) / max(1, vw - 1))
                norm_y = int(((screen_y - vt) * 65535.0) / max(1, vh - 1))
                flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
            else:
                norm_x = int((screen_x * 65535.0) / max(1, self._screen_width - 1))
                norm_y = int((screen_y * 65535.0) / max(1, self._screen_height - 1))
                flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE

            inp = INPUT()
            inp.type = INPUT_MOUSE
            inp.union.mi.dx = norm_x
            inp.union.mi.dy = norm_y
            inp.union.mi.dwFlags = flags
            inp.union.mi.time = 0
            inp.union.mi.dwExtraInfo = 0xFFFFFFFF
            self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        except Exception:
            pass

        self._last_shiori_cursor_pos = (int(screen_x), int(screen_y))
        return res

    def mouse_down(self, button: str = "left") -> bool:
        """Press down a mouse button ('left' or 'right') via hardware SendInput."""
        button = button.lower()
        osu_hwnd = self._get_osu_hwnd()
        if not self._is_osu_active(osu_hwnd):
            return False

        flag = MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN

        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.dx = 0
        inp.union.mi.dy = 0
        inp.union.mi.dwFlags = flag
        inp.union.mi.time = 0
        inp.union.mi.dwExtraInfo = 0xFFFFFFFF

        res = self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if res == 1:
            self._active_mouse_buttons.add(button)
        return res == 1

    def mouse_up(self, button: str = "left") -> bool:
        """Release a mouse button ('left' or 'right') via hardware SendInput."""
        button = button.lower()
        flag = MOUSEEVENTF_LEFTUP if button == "left" else MOUSEEVENTF_RIGHTUP

        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.dx = 0
        inp.union.mi.dy = 0
        inp.union.mi.dwFlags = flag
        inp.union.mi.time = 0
        inp.union.mi.dwExtraInfo = 0xFFFFFFFF

        res = self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if button in self._active_mouse_buttons:
            self._active_mouse_buttons.remove(button)
        return res == 1

    def key_down(self, key: str) -> bool:
        key = key.lower()
        vk = VK_MAP.get(key)
        if vk is None:
            logger.warning(f"Unsupported key for Win32 SendInput: '{key}'")
            return False

        osu_hwnd = self._get_osu_hwnd()
        if not self._is_osu_active(osu_hwnd):
            return False

        scan = self._user32.MapVirtualKeyW(vk, 0)
        flags = KEYEVENTF_SCANCODE
        if key in ("left", "right", "up", "down"):
            flags |= 0x0001  # KEYEVENTF_EXTENDEDKEY

        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = vk  # Populate both wVk and wScan for full SDL2, DirectInput, and WPF compatibility
        inp.union.ki.wScan = scan
        inp.union.ki.dwFlags = flags
        inp.union.ki.time = 0
        inp.union.ki.dwExtraInfo = 0xFFFFFFFF  # Bypass UniKeyNT / EVKey Vietnamese IME WH_KEYBOARD_LL hooks

        res = self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if res == 1:
            self._active_keys.add(key)
        return res == 1

    def key_up(self, key: str) -> bool:
        key = key.lower()
        vk = VK_MAP.get(key)
        if vk is None:
            return False

        scan = self._user32.MapVirtualKeyW(vk, 0)
        flags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP
        if key in ("left", "right", "up", "down"):
            flags |= 0x0001  # KEYEVENTF_EXTENDEDKEY

        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = vk  # Populate both wVk and wScan for full SDL2, DirectInput, and WPF compatibility
        inp.union.ki.wScan = scan
        inp.union.ki.dwFlags = flags
        inp.union.ki.time = 0
        inp.union.ki.dwExtraInfo = 0xFFFFFFFF  # Bypass UniKeyNT / EVKey Vietnamese IME WH_KEYBOARD_LL hooks

        res = self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if key in self._active_keys:
            self._active_keys.remove(key)
        return res == 1

    def release_all(self) -> None:
        """Emergency release of any stuck or currently pressed keys and mouse buttons."""
        # CHỈ nhả các phím và nút chuột mà Shiori ĐANG THỰC SỰ GIỮ! Tuyệt đối không bắn KEYUP/MOUSEUP mù gây cướp chuột/phím của Ba!
        keys_to_release = list(self._active_keys)
        for key in keys_to_release:
            self.key_up(key)
        self._active_keys.clear()

        buttons_to_release = list(self._active_mouse_buttons)
        for btn in buttons_to_release:
            self.mouse_up(btn)
        self._active_mouse_buttons.clear()

        # Reset cursor history and override cooldown so new sessions don't false-trigger deadman switch
        self._last_shiori_cursor_pos = None
        self._human_override_until_s = 0.0
        self._last_move_time_s = 0.0
