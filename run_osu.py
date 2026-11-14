#!/usr/bin/env python3
"""
=============================================================================
OSU! AUTONOMOUS PLAY & REAL-TIME RHYTHM COMPANION (ZERO-LATENCY ENGINE)
Repository: https://github.com/ROSHINGUYEN/OSU
=============================================================================
- Zero-Latency Memory Architecture: Đọc trực tiếp bộ nhớ tiến trình (Port 24050).
- Pure Gameplay Engine: Không webcam, không thị giác máy tính, 0% CPU rác.
- Minimum-Jerk Polynomial Kinematics & 1D Bayesian Kalman Filter.
- Biometric Anti-Ban & Rank Surge Protection (Gaussian Timing & Spatial Dispersion).
=============================================================================
"""

import os
import sys
import time
import signal
import socket
import logging
import threading
import subprocess
from pathlib import Path

# Đường dẫn gốc dự án
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Nạp biến môi trường nếu có
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except Exception:
    pass

# Đảm bảo UTF-8 an toàn trên Windows Console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Osu.Companion")

from src.gameplay.osu.app import osu_app
from src.gameplay.osu.core.state_store import GameMode, state_store
from src.gameplay.osu.api.server import start_osu_api_server
from src.gameplay.osu.perception.capture import find_osu_window_hwnd, focus_osu_window
from src.gameplay.osu.integrations.tosu_client import tosu_client
from src.gameplay.osu.safety.humanizer import humanizer

# Biến cờ chạy
is_running = True
tosu_proc = None


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.4)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def stop_tosu():
    """Tự động tắt tosu.exe khi đóng game hoặc dừng chương trình."""
    global tosu_proc
    logger.info("[TOSU] Đang tự động đóng tiến trình Tosu Memory Reader...")
    if tosu_proc and tosu_proc.poll() is None:
        try:
            tosu_proc.terminate()
            tosu_proc.wait(timeout=1.5)
        except Exception:
            try:
                tosu_proc.kill()
            except Exception:
                pass
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "tosu.exe"], capture_output=True, timeout=2.0)
        except Exception:
            pass


def ensure_tosu_running():
    """Tự động kiểm tra và khởi động tosu.exe nếu chưa chạy."""
    global tosu_proc
    if is_port_in_use(24050):
        logger.info("[TOSU] Cổng 24050 đã sẵn sàng (Tosu Memory Reader trực tuyến).")
        return True

    tosu_exe = ROOT_DIR / "tosu" / "tosu.exe"
    if not tosu_exe.is_file():
        logger.warning(f"[TOSU] Không tìm thấy file {tosu_exe}. Hãy khởi động Tosu thủ công.")
        return False

    logger.info("[TOSU] Đang tự động khởi chạy Tosu Memory Reader nền...")
    try:
        tosu_proc = subprocess.Popen(
            [str(tosu_exe)],
            cwd=str(tosu_exe.parent),
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
        )
        for _ in range(8):
            time.sleep(0.5)
            if is_port_in_use(24050):
                logger.info("[TOSU] Tosu Memory Reader đã mở cổng 24050 thành công!")
                return True
        logger.info("[TOSU] Đã gửi lệnh chạy tosu.exe. Đang đợi kết nối...")
        return True
    except Exception as e:
        logger.warning(f"[TOSU] Lỗi khi mở tosu.exe: {e}")
        return False


def set_console_title(title: str) -> None:
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW(title)
    except Exception:
        pass


def _signal_handler(sig, frame):
    global is_running
    logger.info("Nhận tín hiệu dừng (Ctrl+C). Đang ngắt kết nối an toàn...")
    is_running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def interactive_keyboard_listener():
    """Lắng nghe phím tắt điều khiển từ console."""
    global is_running
    try:
        import msvcrt
        while is_running:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    char = ch.decode("utf-8", errors="ignore").lower()
                except Exception:
                    char = ""

                if char == "q":
                    logger.info("Đã nhấn 'Q'. Đang dừng tiến trình...")
                    is_running = False
                    break
                elif char == "a":
                    osu_app.auto_aim_enabled = not osu_app.auto_aim_enabled
                    state_text = "BẬT (Tự rê chuột & nhấp nốt)" if osu_app.auto_aim_enabled else "TẮT (Tự rê chuột thủ công, bot tự bấm)"
                    print(f"\n[CẬP NHẬT] Tự động ngắm (Auto-Aim): {state_text}")
                elif char == "s":
                    humanizer.set_enabled(not humanizer.is_enabled)
                    s_state = "BẬT (Gaussian UR: 65-85, Phân tán chuột, Chống Spike PP)" if humanizer.is_enabled else "TẮT (Chế độ Chính xác Tuyệt đối 0ms)"
                    print(f"\n[ANTI-BAN] Chế độ Tàng hình & Sinh trắc học (Stealth): {s_state}")
                elif char == "[":
                    from src.gameplay.osu.timing.calibration import timing_calibrator
                    new_lead = timing_calibrator.adjust_offset(2.0)
                    print(f"\n[OFFSET] Tăng lead-time: đánh sớm hơn 2ms (Tổng lead: {new_lead:.1f}ms)")
                elif char == "]":
                    from src.gameplay.osu.timing.calibration import timing_calibrator
                    new_lead = timing_calibrator.adjust_offset(-2.0)
                    print(f"\n[OFFSET] Giảm lead-time: đánh trễ hơn 2ms (Tổng lead: {new_lead:.1f}ms)")
                elif char == "h":
                    print_help()

            time.sleep(0.05)
    except Exception:
        pass


def print_banner():
    banner = """
=====================================================================
   OSU! AUTONOMOUS PLAY & REAL-TIME RHYTHM COMPANION
   GitHub: https://github.com/ROSHINGUYEN/OSU
=====================================================================
[*] Động cơ: ZERO-LATENCY MEMORY (Tosu Reader - Port 24050)
[*] Cổng API / MCP Gateway: Port 8003
[*] Chế độ chơi: TỰ ĐỘNG (Standard, Taiko, Catch, Mania)
[*] Tự động ngắm (Auto-Aim): BẬT (Minimum-Jerk Kinematics)
[*] Chế độ Tàng hình (Anti-Ban / Stealth): BẬT (UR 65-85, Chống Spike PP)
---------------------------------------------------------------------
Phím tắt điều khiển:
  [A] - Bật/Tắt Tự động ngắm (Auto-Aim)
  [S] - Bật/Tắt Anti-Ban / Stealth Mode (Humanized Biometrics)
  [[] / []] - Tinh chỉnh bù độ trễ âm thanh (+2ms / -2ms lead-time)
  [Q] - Dừng và thoát an toàn
=====================================================================
"""
    print(banner)


def print_help():
    print("""
[HƯỚNG DẪN ĐIỀU KHIỂN]:
  A - Bật/Tắt Auto-Aim (Mặc định: BẬT)
  S - Bật/Tắt Anti-Ban / Stealth Mode (Giả lập UR 65-85, chống tăng hạng đột biến)
  [ - Đánh sớm hơn 2ms (bù trễ nốt rơi trễ)
  ] - Đánh trễ hơn 2ms (bù trễ nốt rơi sớm)
  Q - Thoát chương trình
""")


def main():
    global is_running
    print_banner()

    # Đảm bảo Tosu đang chạy
    ensure_tosu_running()

    # Cấu hình gameplay thuần túy
    osu_app.enable_vtuber_bridge = False
    osu_app.auto_aim_enabled = True
    humanizer.set_enabled(True)

    # 1. Khởi động API Server Port 8003
    logger.info("Khởi động Osu Gameplay API Server trên cổng 8003...")
    try:
        start_osu_api_server(port=8003)
        logger.info("API Server sẵn sàng tại cổng 8003")
    except Exception as e:
        logger.warning(f"Cảnh báo khi chạy API Server: {e}")

    # 2. Khởi động lõi OsuGameplayApp
    from src.gameplay.osu.core.gpu_engine import gpu_engine
    gpu_hw = gpu_engine.get_hardware_status()
    gpu_badge = f"GPU: {gpu_hw.get('device_name', 'Accelerated')}" if gpu_hw.get("gpu_accelerated") else "CPU SIMD"
    logger.info(f"Khởi động lõi OsuGameplayApp (Tăng tốc: {gpu_badge})...")
    started = osu_app.start()
    if not started:
        osu_app.is_running = True

    # 3. Chạy luồng đọc phím tương tác
    key_thread = threading.Thread(target=interactive_keyboard_listener, daemon=True, name="KeyboardListener")
    key_thread.start()

    logger.info("Hệ thống đã sẵn sàng! Đang lắng nghe bộ nhớ game qua Tosu...")

    # 4. Vòng lặp hiển thị trạng thái định kỳ
    last_print_time = 0.0
    last_logged_signature = None
    last_play_log_time = 0.0
    was_hwnd_present = False
    is_tty = bool(sys.stdout and hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
    try:
        while is_running:
            now = time.time()
            if now - last_print_time >= 1.5:
                last_print_time = now
                hwnd = find_osu_window_hwnd()
                if hwnd and not was_hwnd_present:
                    was_hwnd_present = True
                    if not getattr(osu_app, "_is_running", False):
                        osu_app.start()
                elif not hwnd and was_hwnd_present:
                    logger.info("\n[OSU] Phát hiện game osu! đã đóng. Đang tự động tắt Tosu và kết thúc...")
                    is_running = False
                    break
                elif not hwnd:
                    was_hwnd_present = False

                st = osu_app.get_status()
                mode = st.get("mode", "UNKNOWN").upper()
                combo = st.get("combo", 0)
                acc = st.get("accuracy", 100.0)
                aim = "ON" if st.get("auto_aim_enabled", True) else "OFF"
                tosu_status = f"LIVE ({int(tosu_client.time_ms)}ms)" if tosu_client.is_connected else "ĐANG ĐỢI TOSU..."
                osu_status = f"HWND: {hwnd}" if hwnd else "Chưa thấy cửa sổ osu!"

                bm_title = tosu_client.beatmap_title or st.get("beatmap_title") or "Chưa chọn map"
                if len(bm_title) > 28:
                    bm_title = bm_title[:25] + "..."

                full_map = tosu_client.beatmap_title or st.get("beatmap_title") or "Đang chờ bài"
                gpu_tag = "GPU" if gpu_hw.get("gpu_accelerated") else "CPU"
                set_console_title(f"OSU! Companion — {full_map} [{mode}] | Acc: {acc:.1f}% | Combo: {combo}x | {gpu_tag}")

                stealth_tag = "ON" if humanizer.is_enabled else "OFF"
                actively_in_game = bool(st.get("is_running") and hwnd and (tosu_client.is_playing or combo > 0))

                if not actively_in_game:
                    status_line = f"[STATUS] Chế độ CHỜ | {osu_status} | {tosu_status} | Stealth: {stealth_tag} | Thoát: 'Q'"
                else:
                    status_line = f"[STATUS] Đang chơi | Màn: {bm_title:<22} | Mode: {mode:<8} | Acc: {acc:5.1f}% | Combo: {combo:3d}x | {gpu_tag} | Aim: {aim} | Stealth: {stealth_tag}"

                if is_tty:
                    sys.stdout.write(f"\r{status_line:<110}")
                    sys.stdout.flush()
                else:
                    sig = (bool(hwnd), bool(tosu_client.is_connected), actively_in_game, full_map if actively_in_game else "")
                    if sig != last_logged_signature or (actively_in_game and (now - last_play_log_time) >= 8.0):
                        last_logged_signature = sig
                        last_play_log_time = now
                        logger.info(status_line)

            time.sleep(0.1)

    except (KeyboardInterrupt, SystemExit):
        logger.info("\nĐang dừng tiến trình...")
    finally:
        logger.info("\nĐang ngắt kết nối và giải phóng toàn bộ phím bấm...")
        try:
            osu_app.stop()
        except Exception:
            pass
        stop_tosu()
        logger.info("Phiên chơi kết thúc an toàn. Tosu đã tự động tắt.")


if __name__ == "__main__":
    main()
