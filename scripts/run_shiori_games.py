# -*- coding: utf-8 -*-
"""
scripts/run_shiori_games.py (Bridge)
------------------------------------
Chuyển tiếp lệnh khởi động Shiori Multi-Game tới games_companion.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_SCRIPT = PROJECT_ROOT.parent / "Project_AI_Extensions" / "games_companion" / "scripts" / "run_shiori_games.py"

if not TARGET_SCRIPT.is_file():
    alt = Path(r"E:\Project_AI_Extensions\games_companion\scripts\run_shiori_games.py")
    if alt.is_file():
        TARGET_SCRIPT = alt

if not TARGET_SCRIPT.is_file():
    print("Error: games_companion/scripts/run_shiori_games.py not found")
    sys.exit(1)

cmd = [sys.executable, str(TARGET_SCRIPT)] + sys.argv[1:]
res = subprocess.run(cmd, cwd=str(TARGET_SCRIPT.parent.parent))
sys.exit(res.returncode)

