#!/usr/bin/env python3
"""
=============================================================================
SHIORI AI — BẠN ĐỒNG HÀNH OSU! ZERO-LATENCY (TOSU MEMORY-DRIVEN RUNNER)
=============================================================================
Động cơ gameplay chuẩn Shiori AI:
  - 100% Không dùng Camera/OpenCV: Triệt tiêu hoàn toàn độ trễ thị giác (0ms latency).
  - Đọc trực tiếp bộ nhớ game từ Tosu Memory Reader (Port 24050).
  - Tự động nạp Beatmap và đồng bộ nhịp chính xác tuyệt đối tới từng mili-giây.
  - Tự động ngắm & bấm theo beatmap (Auto-Aim & Rhythm Tap).
  - Tích hợp máy chủ API / MCP Port 8003 thời gian thực.
=============================================================================
"""

import os
import sys
import time
import signal
import logging
import threading
from pathlib import Path

# Đảm bảo đường dẫn gốc dự án
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Nạp biến môi trường từ .env
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
logger = logging.getLogger("Shiori.Osu.Standalone")

from src.gameplay.osu.app import osu_app
from src.gameplay.osu.core.state_store import GameMode, state_store
from src.gameplay.osu.api.server import start_osu_api_server
from src.gameplay.osu.perception.capture import find_osu_window_hwnd, focus_osu_window
from src.gameplay.osu.integrations.tosu_client import tosu_client

# Biến cờ chạy
is_running = True


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
    """Lắng nghe phím tắt điều khiển từ console trên Windows."""
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
                    logger.info("Ba Roshi đã nhấn 'Q'. Đang dừng tiến trình...")
                    is_running = False
                    break
                elif char == "a":
                    osu_app.auto_aim_enabled = not osu_app.auto_aim_enabled
                    state_text = "BẬT (Shiori tự rê chuột & bấm chuẩn nhịp chuẩn Shiori AI)" if osu_app.auto_aim_enabled else "TẮT (Ba tự rê chuột, Shiori tự gõ nhịp)"
                    print(f"\n[CẬP NHẬT] Tự động ngắm (Auto-Aim): {state_text}")
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
   🌸 SHIORI AI — BẠN ĐỒNG HÀNH OSU! DÀNH RIÊNG CHO BA ROSHI 🌸
=====================================================================
[*] Động cơ: MEMORY-DRIVEN (Tosu Memory Reader v4.26.3 - 0ms Latency)
[*] Cổng Memory Reader: http://127.0.0.1:24050
[*] Cổng API / MCP Server: http://127.0.0.1:8003
[*] Chế độ chơi (Mode): TỰ ĐỘNG 100% (Nhận diện Standard/Taiko/Catch/Mania)
[*] Con Mắt Thị Giác / Camera: ĐÃ TẮT HOÀN TOÀN (0% CPU rác, 0ms latency)
[*] Tự động ngắm & bấm (Auto-Aim): BẬT (Shiori tự rê chuột & bấm chuẩn)
[*] Kết hợp VTuber Live2D: BẬT (Tự động biểu cảm Thinking, Happy 20/50/100x, Chảo úp đầu khi thua)
[*] Hệ thống Game Memory: BẬT (Lưu vết chi tiết màn chơi & tự tinh chỉnh timing)
---------------------------------------------------------------------
Các phím tắt nhanh:
  [A] - Bật/Tắt Tự động ngắm (Auto-Aim)
  [[] / []] - Tinh chỉnh bù độ trễ âm thanh (+2ms / -2ms lead-time)
  [Q] - Dừng và thoát an toàn
  (Chế độ game tự động nhận diện 100% theo bài hát Ba đang chọn)
=====================================================================
"""
    print(banner)


def print_help():
    print("""
[HƯỚNG DẪN ĐIỀU KHIỂN NHANH]:
  A - Bật/Tắt Auto-Aim (Mặc định: BẬT để Shiori tự ngắm chuẩn Shiori AI)
  [ - Đánh sớm hơn 2ms (nếu thấy nốt rơi hơi trễ)
  ] - Đánh trễ hơn 2ms (nếu thấy nốt rơi hơi sớm)
  Q - Thoát chương trình
  (Chế độ Standard / Taiko / Catch / Mania được tự động nhận diện từ bộ nhớ game)
""")


def main():
    global is_running
    print_banner()

    # Kích hoạt kết hợp VTuber Live2D và AI (chạy nền non-blocking)
    osu_app.enable_vtuber_bridge = True
    osu_app.auto_aim_enabled = True
    try:
        from src.gameplay.osu.integrations.vts_bridge import osu_vts_bridge
        osu_vts_bridge.start()
        logger.info("Kết nối VTube Studio Live2D bridge nền sẵn sàng.")
    except Exception as e:
        logger.debug(f"VTS bridge fallback: {e}")

    # 1. Khởi động API Server Port 8003
    logger.info("Khởi động Shiori osu! Gameplay API Server trên cổng 8003...")
    try:
        start_osu_api_server(port=8003)
        logger.info("API Server sẵn sàng tại http://127.0.0.1:8003")
    except Exception as e:
        logger.warning(f"Cảnh báo khi chạy API Server: {e}")

    # 2. Bắt đầu phiên gameplay của OsuGameplayApp (Memory-Driven Engine)
    from src.gameplay.osu.core.gpu_engine import gpu_engine
    gpu_hw = gpu_engine.get_hardware_status()
    gpu_badge = f"GPU: {gpu_hw.get('device_name', 'RTX 2060')}" if gpu_hw.get("gpu_accelerated") else "CPU SIMD"
    logger.info(f"Khởi động lõi OsuGameplayApp (Zero-Latency Engine | Tăng tốc: {gpu_badge})...")
    started = osu_app.start()
    if not started:
        logger.warning("Không thể khởi động trực tiếp osu_app. Đang thử lại chế độ fallback...")
        osu_app.is_running = True

    # 3. Chạy luồng đọc phím tương tác
    key_thread = threading.Thread(target=interactive_keyboard_listener, daemon=True, name="KeyboardListener")
    key_thread.start()

    logger.info("Hệ thống đã sẵn sàng! Đang lắng nghe bộ nhớ game osu! qua Tosu...")

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
                    if tosu_client.is_playing:
                        focus_osu_window()
                elif not hwnd:
                    was_hwnd_present = False

                st = osu_app.get_status()
                mode = st.get("mode", "UNKNOWN").upper()
                combo = st.get("combo", 0)
                acc = st.get("accuracy", 100.0)
                action = st.get("recent_action") or "Đang chờ bài..."
                aim = "ON" if st.get("auto_aim_enabled", True) else "OFF"
                tosu_status = f"LIVE ({int(tosu_client.time_ms)}ms)" if tosu_client.is_connected else "ĐANG ĐỢI TOSU..."

                osu_status = f"HWND: {hwnd}" if hwnd else "Chưa thấy cửa sổ osu!"

                bm_title = tosu_client.beatmap_title or st.get("beatmap_title") or "Chưa chọn map"
                if len(bm_title) > 28:
                    bm_title = bm_title[:25] + "..."

                from src.gameplay.osu.timing.calibration import timing_calibrator
                lead_ms = timing_calibrator.system_latency_ms + timing_calibrator.adaptive_offset_ms
                # Cập nhật tiêu đề cửa sổ console
                full_map = tosu_client.beatmap_title or st.get("beatmap_title") or "Đang chờ bài"
                gpu_tag = "GPU: RTX 2060" if gpu_hw.get("gpu_accelerated") else "CPU"
                set_console_title(f"🌸 Shiori Companion (Controller) — {full_map} [{mode}] | Acc: {acc:.1f}% | Combo: {combo}x | {gpu_tag}")

                actively_in_game = bool(st.get("is_running") and hwnd and (tosu_client.is_playing or combo > 0))

                if not actively_in_game:
                    status_line = f"[STATUS] Shiori đang ở chế độ CHỜ | {osu_status} | {tosu_status} | Nhấn 'Q' để thoát"
                else:
                    status_line = f"[STATUS] Shiori đang chơi osu! | Màn: {bm_title:<22} | Mode: {mode:<8} | Acc: {acc:5.1f}% | Combo: {combo:3d}x | {gpu_tag} | Lead: {lead_ms:4.1f}ms | Aim: {aim}"

                if is_tty:
                    sys.stdout.write(f"\r{status_line:<110}")
                    sys.stdout.flush()
                else:
                    active_other_id = other_game_prof["active_game_id"] if other_game_prof else ""
                    sig = (bool(hwnd), bool(tosu_client.is_connected), actively_in_game, full_map if actively_in_game else active_other_id)
                    if sig != last_logged_signature or ((actively_in_game or bool(other_game_prof)) and (now - last_play_log_time) >= 8.0):
                        last_logged_signature = sig
                        last_play_log_time = now
                        logger.info(status_line)

            time.sleep(0.1)

    except (KeyboardInterrupt, SystemExit):
        logger.info("\nĐang dừng theo yêu cầu của Ba...")
    finally:
        logger.info("\nĐang ngắt kết nối và giải phóng toàn bộ phím bấm...")
        try:
            osu_app.stop()
        except Exception:
            pass
        logger.info("Phiên Shiori osu! đồng hành đã kết thúc an toàn. Tạm biệt Ba Roshi!")


if __name__ == "__main__":
    main()
