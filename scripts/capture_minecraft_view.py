# -*- coding: utf-8 -*-
"""
scripts/capture_minecraft_view.py (Bridge)
------------------------------------------
Chuyển tiếp lệnh chụp ảnh Minecraft tới games_companion.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_SCRIPT = PROJECT_ROOT.parent / "Project_AI_Extensions" / "games_companion" / "scripts" / "capture_minecraft_view.py"

if not TARGET_SCRIPT.is_file():
    alt = Path(r"E:\Project_AI_Extensions\games_companion\scripts\capture_minecraft_view.py")
    if alt.is_file():
        TARGET_SCRIPT = alt

if not TARGET_SCRIPT.is_file():
    print("Snapshot failed: capture_minecraft_view.py not found in games_companion")
    sys.exit(1)

cmd = [sys.executable, str(TARGET_SCRIPT)] + sys.argv[1:]
res = subprocess.run(cmd, cwd=str(TARGET_SCRIPT.parent.parent))
sys.exit(res.returncode)

