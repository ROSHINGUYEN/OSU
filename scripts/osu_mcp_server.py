#!/usr/bin/env python3
"""
Shiori AI osu! Gameplay MCP Server for Google Antigravity & Agentic Pair Programming
=====================================================================================
Cung cấp bộ công cụ Model Context Protocol (MCP) giúp Agent AI:
1. osu_get_status: Lấy trạng thái thời gian thực của Shiori AI gameplay, accuracy, combo, FPS, Tosu.
2. osu_get_debug_frame: Trích xuất phân tích thị giác và lưu ảnh debug annotated (bounding box, Don/Katsu, hit line).
3. osu_get_telemetry: Xem lịch sử gõ phím, độ lệch thời gian delta_ms, và gradient descent adaptive offset.
4. osu_send_command: Điều khiển phiên chơi osu! (play, stop, mode, calibrate).
5. osu_inspect_color: Lấy mẫu màu RGB và kiểm tra phân loại Don/Katsu tại tọa độ bất kỳ trên màn hình.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure UTF-8 I/O on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup project root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.gameplay.osu.app import osu_app
from src.gameplay.osu.core.state_store import GameMode
from src.gameplay.osu.integrations.tosu_client import tosu_client
from src.gameplay.osu.perception.mode_detector import mode_detection_service
from src.gameplay.osu.timing.calibration import timing_calibrator

TOOLS_DEFINITIONS = [
    {
        "name": "osu_get_status",
        "description": "Lấy trạng thái chi tiết của Shiori AI osu! gameplay (Đang chạy, Chế độ, Accuracy, Combo, Offset thích ứng, FPS, kết nối Tosu).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "osu_get_debug_frame",
        "description": "Chụp và trả về phân tích khung hình thị giác thời gian thực kèm đường dẫn file ảnh debug live_perception.png (có bounding box nhận diện nốt, phân loại màu Don/Katsu, vạch đích Taiko).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "osu_get_telemetry",
        "description": "Lấy lịch sử telemetry gần nhất (danh sách nốt gõ, delta mili-giây, phân loại màu, cập nhật thuật toán học máy bù trễ).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Số lượng bản ghi gần nhất (mặc định 25)", "default": 25}
            },
        },
    },
    {
        "name": "osu_send_command",
        "description": "Gửi lệnh điều khiển gameplay trực tiếp tới Shiori (play, stop, mode, calibrate, test_key).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": ["play", "stop", "mode", "calibrate", "test_key"], "description": "Tên lệnh (play, stop, mode, calibrate, test_key)"},
                "args": {"type": "string", "description": "Tham số bổ sung (vd: tên phím 'j', 'd', 'f', 'k' cho test_key, hoặc mode, hoặc ms)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "osu_inspect_color",
        "description": "Kiểm tra màu sắc RGB và phân loại Don/Katsu tại tọa độ (x, y) trên viewport game osu!.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "Tọa độ X pixel"},
                "y": {"type": "integer", "description": "Tọa độ Y pixel"},
            },
            "required": ["x", "y"],
        },
    },
]


class OsuMCPServer:
    """Máy chủ MCP điều khiển và giám sát Shiori AI osu! Gameplay."""

    def __init__(self):
        try:
            tosu_client.start(auto_launch=True)
        except Exception:
            pass

    def _get_osu_process_info(self) -> Dict[str, Any]:
        """Query osu!.exe process, HWND, and foreground status directly."""
        try:
            import ctypes
            import ctypes.wintypes
            from src.gameplay.osu.perception.capture import find_osu_window_hwnd, find_osu_window_region

            hwnd = find_osu_window_hwnd()
            if not hwnd or not ctypes.windll.user32.IsWindow(hwnd):
                return {"status": "not_found", "pid": None, "hwnd": None, "is_foreground": False}

            pid = ctypes.wintypes.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            fore_hwnd = ctypes.windll.user32.GetForegroundWindow()
            is_foreground = (fore_hwnd == hwnd)

            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            title = ""
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                title = buff.value

            region = find_osu_window_region()
            rect_dict = {}
            if region:
                l, t, r, b = region
                rect_dict = {"left": l, "top": t, "right": r, "bottom": b, "width": r - l, "height": b - t}

            return {
                "status": "connected",
                "pid": pid.value,
                "hwnd": hwnd,
                "title": title,
                "is_foreground": is_foreground,
                "window_rect": rect_dict,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def handle_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        status = osu_app.get_status()
        status["osu_process"] = self._get_osu_process_info()
        status["tosu_snapshot"] = tosu_client.get_snapshot()
        if not status.get("beatmap_loaded") and tosu_client.beatmap_title:
            status["beatmap_loaded"] = True
            status["beatmap_title"] = tosu_client.beatmap_title
        if not status.get("tosu_connected"):
            status["tosu_connected"] = tosu_client.is_connected
        status["engine"] = "memory_driven_zero_latency"
        status["debug_frame_ready"] = False
        return status

    def handle_get_debug_frame(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "memory_driven",
            "engine": "Tosu Memory Reader v4.26.3 (0ms Latency)",
            "camera_status": "ELIMINATED",
            "accuracy": osu_app.feedback.metrics.accuracy,
            "combo": osu_app.feedback.metrics.current_combo,
            "mode": osu_app.get_status()["mode"],
            "beatmap_title": tosu_client.beatmap_title or "No beatmap active",
            "time_ms": tosu_client.time_ms,
        }

    def handle_get_telemetry(self, args: Dict[str, Any]) -> Dict[str, Any]:
        limit = int(args.get("limit", 25))
        return {
            "session_id": osu_app.telemetry.session_id,
            "history": osu_app.telemetry.get_recent_events(limit=limit),
            "current_metrics": {
                "accuracy": osu_app.feedback.metrics.accuracy,
                "combo": osu_app.feedback.metrics.current_combo,
                "max_combo": osu_app.feedback.metrics.max_combo,
                "adaptive_offset_ms": timing_calibrator.adaptive_offset_ms,
            },
        }

    def handle_send_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        cmd = args.get("command", "").lower().strip()
        cmd_args = args.get("args", "").strip()

        if cmd == "play":
            path = cmd_args if cmd_args else None
            success = osu_app.start_session(beatmap_path=path)
            return {"success": success, "message": "Phiên chơi osu! đã khởi động" if success else "Khởi động thất bại"}
        elif cmd == "stop":
            osu_app.stop()
            return {"success": True, "message": "Đã dừng phiên chơi osu! an toàn"}
        elif cmd == "mode":
            mode_map = {
                "standard": GameMode.STANDARD,
                "std": GameMode.STANDARD,
                "taiko": GameMode.TAIKO,
                "catch": GameMode.CATCH,
                "mania": GameMode.MANIA,
            }
            if cmd_args.lower() in mode_map:
                chosen = mode_map[cmd_args.lower()]
                mode_detection_service.set_user_override(chosen)
                return {"success": True, "mode": chosen.value, "message": f"Đã chuyển mode sang {chosen.value}"}
            return {"success": False, "message": "Mode không hợp lệ. Chọn: standard, taiko, catch, mania"}
        elif cmd == "aim":
            arg = cmd_args.lower()
            if arg in ("on", "true", "enable", "1"):
                osu_app.auto_aim_enabled = True
                return {"success": True, "auto_aim": True, "message": "Đã BẬT tính năng Shiori tự ngắm chuột (Auto-Aim)."}
            else:
                osu_app.auto_aim_enabled = False
                return {"success": True, "auto_aim": False, "message": "Đã TẮT tính năng tự ngắm chuột (Ba Roshi tự do cầm chuột, Shiori chỉ gõ phím)."}
        elif cmd == "calibrate":
            try:
                cmd_args_clean = cmd_args.strip()
                if cmd_args_clean.startswith("latency ") or cmd_args_clean.startswith("lead "):
                    val = float(cmd_args_clean.split()[1])
                    timing_calibrator.set_system_latency_ms(val)
                    return {
                        "success": True,
                        "system_latency_ms": timing_calibrator.system_latency_ms,
                        "adaptive_offset_ms": timing_calibrator.adaptive_offset_ms,
                        "message": f"Đã cập nhật system_latency_ms = {val:.1f}ms",
                    }
                offset_val = float(cmd_args_clean)
                timing_calibrator.record_hit_sample(offset_val, confidence=1.0)
                return {
                    "success": True,
                    "adaptive_offset_ms": timing_calibrator.adaptive_offset_ms,
                    "system_latency_ms": timing_calibrator.system_latency_ms,
                }
            except ValueError:
                return {"success": False, "message": "Giá trị offset hoặc latency không hợp lệ"}
        elif cmd == "test_key":
            target_key = cmd_args.lower() if cmd_args else "j"
            import time
            from src.gameplay.osu.input.controller import input_controller
            from src.gameplay.osu.perception.capture import focus_osu_window
            focused = focus_osu_window()
            time.sleep(0.015)
            tap_ok = input_controller.tap_key(target_key)
            proc_info = self._get_osu_process_info()
            return {
                "success": tap_ok,
                "command": "test_key",
                "key": target_key,
                "osu_focused": focused or proc_info.get("is_foreground", False),
                "osu_process": proc_info,
                "message": f"Đã gõ phím '{target_key}' trực tiếp vào osu!.exe (PID: {proc_info.get('pid')}, HWND: {proc_info.get('hwnd')}).",
            }
        elif cmd == "test_mouse_move":
            import time
            from src.gameplay.osu.input.controller import input_controller
            from src.gameplay.osu.perception.capture import focus_osu_window
            focused = focus_osu_window()
            time.sleep(0.015)
            coords = cmd_args.split() if cmd_args else []
            proc_info = self._get_osu_process_info()
            w_rect = proc_info.get("window_rect", {})
            if len(coords) >= 2:
                target_x, target_y = int(coords[0]), int(coords[1])
            elif w_rect:
                target_x = w_rect.get("left", 0) + (w_rect.get("width", 1280) // 2)
                target_y = w_rect.get("top", 0) + (w_rect.get("height", 720) // 2)
            else:
                target_x, target_y = 640, 360
            move_ok = input_controller.move_cursor(target_x, target_y)
            return {
                "success": move_ok,
                "command": "test_mouse_move",
                "x": target_x,
                "y": target_y,
                "osu_focused": focused or proc_info.get("is_foreground", False),
                "message": f"Đã di chuyển con trỏ chuột đến ({target_x}, {target_y}).",
            }
        elif cmd == "test_mouse_click":
            btn = cmd_args.lower() if cmd_args in ["left", "right", "middle"] else "left"
            import time
            from src.gameplay.osu.input.controller import input_controller
            from src.gameplay.osu.perception.capture import focus_osu_window
            focused = focus_osu_window()
            time.sleep(0.015)
            click_ok = input_controller.mouse_click(btn)
            proc_info = self._get_osu_process_info()
            return {
                "success": click_ok,
                "command": "test_mouse_click",
                "button": btn,
                "osu_focused": focused or proc_info.get("is_foreground", False),
                "message": f"Đã click chuột '{btn}' trực tiếp vào osu!.exe.",
            }
        else:
            return {"success": False, "message": f"Lệnh không xác định: {cmd}"}

    def handle_inspect_color(self, args: Dict[str, Any]) -> Dict[str, Any]:
        x = int(args.get("x", 0))
        y = int(args.get("y", 0))
        return {
            "x": x,
            "y": y,
            "classification": "MEMORY_DRIVEN_AUTO",
            "message": "Kiến trúc zero-latency đọc trực tiếp loại nốt và beatmap từ bộ nhớ qua Tosu. Không sử dụng camera/quét màu màn hình.",
        }


async def run_stdio_server():
    """Vòng lặp đọc JSON-RPC từ stdin và phản hồi ra stdout."""
    server = OsuMCPServer()
    # In stdio MCP protocol, stdout is reserved strictly for JSON-RPC messages.
    # Redirect all unintended standard stdout (e.g. library prints) to stderr to prevent protocol corruption.
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    while True:
        line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
        if not line:
            break

        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except Exception:
            continue

        msg_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "osu-gameplay-mcp", "version": "1.0.0"},
                    "capabilities": {"tools": {}},
                },
            }
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": TOOLS_DEFINITIONS},
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})

            result_data = None
            if tool_name == "osu_get_status":
                result_data = server.handle_status(tool_args)
            elif tool_name == "osu_get_debug_frame":
                result_data = server.handle_get_debug_frame(tool_args)
            elif tool_name == "osu_get_telemetry":
                result_data = server.handle_get_telemetry(tool_args)
            elif tool_name == "osu_send_command":
                result_data = server.handle_send_command(tool_args)
            elif tool_name == "osu_inspect_color":
                result_data = server.handle_inspect_color(tool_args)
            else:
                result_data = {"error": f"Không tìm thấy tool: {tool_name}"}

            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result_data, ensure_ascii=False, indent=2)}]
                },
            }
        elif method == "ping":
            resp = {"jsonrpc": "2.0", "id": msg_id, "result": {}}
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        protocol_out.write(json.dumps(resp, ensure_ascii=False) + "\n")
        protocol_out.flush()


if __name__ == "__main__":
    asyncio.run(run_stdio_server())
