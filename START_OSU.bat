@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set "TEMP=%~dp0.cache\tmp"
set "TMP=%~dp0.cache\tmp"
if not exist "%TEMP%" mkdir "%TEMP%"

title OSU! Autonomous Play & Real-Time Rhythm Companion

echo =====================================================================
echo    OSU! AUTONOMOUS PLAY & REAL-TIME RHYTHM COMPANION
echo    GitHub: https://github.com/ROSHINGUYEN/OSU
echo =====================================================================
echo.

:: 1. Xác định môi trường Python
set "PY_EXE=python"
if exist "..\Project_AI\.venv\Scripts\python.exe" (
    set "PY_EXE=..\Project_AI\.venv\Scripts\python.exe"
    echo [OK] Đã tìm thấy Python Environment.
) else if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
    echo [OK] Đã tìm thấy Local Python Environment.
) else (
    echo [INFO] Sử dụng Python hệ thống: %PY_EXE%
)

:: 2. Kiểm tra bộ đọc bộ nhớ Tosu Memory Reader (Cổng 24050)
echo.
echo [*] Đang kiểm tra Tosu Memory Reader (Port 24050)...
netstat -ano | findstr ":24050" | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Tosu Memory Reader đã trực tuyến trên cổng 24050.
) else (
    if exist "%~dp0tosu\tosu.exe" (
        echo [*] Đang tự động khởi chạy Tosu Memory Reader ngầm...
        start "Tosu Memory Reader" /min "%~dp0tosu\tosu.exe"
        timeout /t 2 /nobreak >nul
        echo [OK] Đã kích hoạt Tosu Memory Reader.
    ) else (
        echo [CẢNH BÁO] Không tìm thấy tosu\tosu.exe. Hãy mở Tosu thủ công!
    )
)

:: 3. Kiểm tra tiến trình game osu!
echo.
echo [*] Đang kiểm tra tiến trình game osu!...
tasklist /FI "IMAGENAME eq osu!.exe" 2>NUL | find /I /N "osu!.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] Game osu! đang chạy.
) else (
    echo [INFO] osu! chưa chạy. Bạn hãy mở game osu! để bắt đầu chơi nha!
)

:: 4. Khởi chạy Động cơ Osu Gameplay Engine
echo.
echo =====================================================================
echo [*] Đang khởi chạy Osu Gameplay Engine ^& Anti-Ban Stealth...
echo =====================================================================
echo.

"%PY_EXE%" "%~dp0run_osu.py" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Ứng dụng kết thúc với mã lỗi %ERRORLEVEL%.
    pause
)
