# -*- coding: utf-8 -*-
"""
scripts/manage_game.py (Bridge)
-------------------------------
Chuyển tiếp lệnh quản lý game tới Project_AI_Extensions/games_companion/scripts/manage_game.py.
"""

import sys
import subprocess
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_SCRIPT = PROJECT_ROOT.parent / "Project_AI_Extensions" / "games_companion" / "scripts" / "manage_game.py"

if not TARGET_SCRIPT.is_file():
    alt = Path(r"E:\Project_AI_Extensions\games_companion\scripts\manage_game.py")
    if alt.is_file():
        TARGET_SCRIPT = alt

if not TARGET_SCRIPT.is_file():
    print('{"status": "error", "message": "Không tìm thấy games_companion/scripts/manage_game.py"}')
    sys.exit(1)

cmd = [sys.executable, str(TARGET_SCRIPT)] + sys.argv[1:]
res = subprocess.run(cmd, cwd=str(TARGET_SCRIPT.parent.parent))
sys.exit(res.returncode)

