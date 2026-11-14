"""
Discord Game Log & Alarm Reporter for Shiori AI osu! Gameplay.
==============================================================
Dispatches formatted match summaries, combo milestone celebrations,
match start announcements, and fail-safe alarms directly to Discord channel 1563921667556773969.

Supports multi-channel delivery:
  1. Direct Discord Webhook (DISCORD_OSU_WEBHOOK_URL / DISCORD_WEBHOOK_URL)
  2. Direct Discord REST API v10 via DISCORD_BOT_TOKEN to target channel
  3. WebSocket broadcast to Shiori Node.js Bot Bridge ('output:osu:game_log')
"""

import asyncio
import json
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
    else:
        load_dotenv()
except Exception:
    pass

logger = logging.getLogger("Shiori.Osu.DiscordReporter")

DEFAULT_TARGET_CHANNEL = "1563921667556773969"
SHIORI_AVATAR_URL = "https://cdn.discordapp.com/embed/avatars/0.png"


class OsuDiscordReporter:
    """Dispatches beautiful anime-styled gameplay embeds, milestones, and alarms to Discord."""

    def __init__(self, target_channel_id: str = DEFAULT_TARGET_CHANNEL):
        self.target_channel_id = os.environ.get("DISCORD_OSU_GAME_LOG_CHANNEL_ID", target_channel_id)
        self._bot_token: Optional[str] = os.environ.get("DISCORD_BOT_TOKEN") or os.environ.get("DISCORD_TOKEN")
        self._webhook_url: Optional[str] = (
            os.environ.get("DISCORD_OSU_WEBHOOK_URL") or os.environ.get("DISCORD_WEBHOOK_URL")
        )
        self._last_report_time: float = 0.0

    # =========================================================================
    # PUBLIC DISPATCH METHODS
    # =========================================================================

    def send_match_start(
        self,
        title: str,
        artist: str,
        difficulty: str,
        mode: str,
        bpm: float = 0.0,
        stars: float = 0.0,
        commentary: str = "",
    ) -> None:
        """Suppressed: only send summary when match ends with results per user request."""
        return

    def send_combo_milestone(
        self,
        title: str,
        mode: str,
        milestone: int,
        accuracy: float,
        current_combo: int,
        commentary: str = "",
    ) -> None:
        """Suppressed: only send summary when match ends with results per user request."""
        return

    def send_game_summary(
        self,
        title: str,
        artist: str,
        difficulty: str,
        mode: str,
        accuracy: float,
        combo: int,
        max_combo: int,
        hits: Dict[str, int],
        commentary: str = "",
        is_failed: bool = False,
        status: str = "completed",
        lead_time_ms: float = 12.0,
        memory_offset_ms: float = 0.0,
        emotion: str = "neutral",
        image_path: Optional[str] = None,
    ) -> None:
        """Asynchronously send game match summary with full stats & GPU telemetry to Discord."""
        from src.gameplay.osu.core.gpu_engine import gpu_engine
        gpu_info = gpu_engine.get_hardware_status()

        misses = hits.get("miss", 0)
        rank = "F" if is_failed else self._calculate_rank(accuracy, misses)

        data = {
            "channelId": self.target_channel_id,
            "type": "summary",
            "title": title or "Không rõ tên bài",
            "artist": artist or "Various Artists",
            "difficulty": difficulty or "Normal",
            "mode": mode.upper(),
            "rank": rank,
            "is_failed": is_failed,
            "status": status,
            "accuracy": round(accuracy, 2),
            "combo": combo,
            "max_combo": max_combo,
            "hits": hits,
            "commentary": commentary,
            "lead_time_ms": lead_time_ms,
            "memory_offset_ms": memory_offset_ms,
            "emotion": emotion,
            "gpu_device": gpu_info.get("device_name", "NVIDIA GeForce RTX 2060"),
            "gpu_backend": gpu_info.get("backend", "NVIDIA CUDA Hardware"),
            "timestamp": time.time(),
            "image_path": image_path if image_path and os.path.exists(image_path) else None,
        }
        self._dispatch_async(data, name="OsuDiscordSummaryThread")

    def send_game_alarm(self, reason: str, details: str = "") -> None:
        """Asynchronously send failure/error alarm when Shiori cannot play."""
        data = {
            "channelId": self.target_channel_id,
            "type": "alarm",
            "reason": reason,
            "details": details,
            "timestamp": time.time(),
        }
        self._dispatch_async(data, name="OsuDiscordAlarmThread")

    # =========================================================================
    # INTERNAL HELPERS & EMBED BUILDERS
    # =========================================================================

    def _dispatch_async(self, payload: Dict[str, Any], name: str = "OsuDiscordThread") -> None:
        # TEST ISOLATION GUARD: Never send real Discord webhooks/REST requests during automated pytest runs
        if "PYTEST_CURRENT_TEST" in os.environ or os.environ.get("SHIORI_TESTING") == "1":
            logger.debug(f"[DiscordReporter] Suppressed dispatch in test environment: {payload.get('type')}")
            return

        threading.Thread(
            target=self._dispatch_report_worker,
            args=(payload,),
            daemon=True,
            name=name,
        ).start()

    @staticmethod
    def _render_accuracy_bar(accuracy: float, length: int = 10) -> str:
        """Creates a visual emoji progress bar for accuracy: [████████░░]."""
        clamped = max(0.0, min(100.0, accuracy))
        filled = int(round((clamped / 100.0) * length))
        bar = "█" * filled + "░" * (length - filled)
        return f"`[{bar}]` **{clamped:.2f}%**"

    def _calculate_rank(self, accuracy: float, misses: int) -> str:
        if accuracy >= 100.0:
            return "SS"
        if accuracy >= 95.0 and misses == 0:
            return "S"
        if accuracy >= 90.0:
            return "A"
        if accuracy >= 80.0:
            return "B"
        if accuracy >= 70.0:
            return "C"
        return "D"

    def _build_embed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs a modern, rich, color-coded Discord Embed based on event type."""
        event_type = payload.get("type", "summary")
        iso_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if event_type == "start":
            mode = payload.get("mode", "STANDARD")
            bpm_text = f" • **BPM:** `{payload.get('bpm')}`" if payload.get("bpm") else ""
            sr_text = f" • **Stars:** `★{payload.get('stars')}`" if payload.get("stars") else ""

            return {
                "title": f"🎮 SHIORI AI — BẮT ĐẦU MÀN CHƠI OSU! [{mode}]",
                "description": (
                    f"### 🎵 {payload.get('title')}\n"
                    f"**Nghệ sĩ:** `{payload.get('artist')}` • **Độ khó:** `{payload.get('difficulty')}`\n"
                    f"**Chế độ:** `{mode}`{bpm_text}{sr_text}"
                ),
                "color": 0x00E5FF,  # Electric Cyan
                "fields": [
                    {
                        "name": "⚡ Lõi Tăng Tốc Phần Cứng",
                        "value": (
                            f"🚀 **Thiết bị:** `{payload.get('gpu_device', 'NVIDIA GeForce RTX 2060')}`\n"
                            f"🧠 **Chế độ:** `CUDA GPU Acceleration (2000Hz Loop)`\n"
                            f"💾 **VRAM Khả dụng:** `{payload.get('free_vram_mb', 0):.0f} MB`"
                        ),
                        "inline": False,
                    },
                    {
                        "name": "💬 Suy Nghĩ Của Shiori",
                        "value": f"*{payload.get('commentary')}*",
                        "inline": False,
                    },
                ],
                "footer": {"text": "Shiori AI Autonomous osu! Engine • Kênh 1563921667556773969"},
                "timestamp": iso_now,
            }

        elif event_type == "milestone":
            milestone = payload.get("milestone", 100)
            acc = payload.get("accuracy", 100.0)
            return {
                "title": f"🔥 SHIORI AI — ĐẠT MỐC COMBO KHỦNG: {milestone}x COMBO!",
                "description": f"### 🎵 {payload.get('title')}\nĐang chơi chế độ: **`{payload.get('mode')}`**",
                "color": 0xFF4081,  # Hot Pink
                "fields": [
                    {"name": "🔥 Combo Hiện Tại", "value": f"**`{payload.get('current_combo')}x`**", "inline": True},
                    {"name": "🎯 Accuracy", "value": f"**`{acc:.2f}%`**", "inline": True},
                    {"name": "🎭 Live2D Biểu Cảm", "value": "`happy / excited (脸红)`", "inline": True},
                    {
                        "name": "💬 Tiếng Reo Hò",
                        "value": f"*{payload.get('commentary')}*",
                        "inline": False,
                    },
                ],
                "footer": {"text": "Shiori AI Autonomous osu! Engine • Kênh 1563921667556773969"},
                "timestamp": iso_now,
            }

        elif event_type == "alarm":
            return {
                "title": "⚠️ SHIORI AI — CẢNH BÁO SỰ CỐ KHÔNG THỂ CHƠI GAME",
                "description": f"**Lý do:** {payload.get('reason')}\n{payload.get('details', '')}",
                "color": 0xFF3D00,  # Vivid Orange Red
                "footer": {"text": "Shiori AI Fail-Safe Guard • Kênh 1563921667556773969"},
                "timestamp": iso_now,
            }

        else:
            # Summary Event
            is_failed = bool(payload.get("is_failed", False))
            rank = payload.get("rank", "A")
            acc = float(payload.get("accuracy", 100.0))
            max_combo = int(payload.get("max_combo", 0))
            hits = payload.get("hits", {})
            h300 = hits.get("300", 0)
            h100 = hits.get("100", 0)
            h50 = hits.get("50", 0)
            miss = hits.get("miss", 0)

            # Color scheme by grade
            if is_failed:
                color = 0xFF1744  # Crimson Red
                title_prefix = "🍳 SHIORI AI — THẤT BẠI & CHẢO ÚP ĐẦU [FAILED]"
            elif rank in ("SS", "S"):
                color = 0xFFD700 if rank == "SS" else 0x00E676  # Gold or Emerald
                title_prefix = f"🌸 SHIORI AI — TỔNG KẾT XUẤT SẮC [{rank}]"
            elif rank == "A":
                color = 0x00B0FF  # Cyan
                title_prefix = f"🌸 SHIORI AI — TỔNG KẾT MÀN CHƠI OSU! [HẠNG {rank}]"
            elif rank == "B":
                color = 0x7C4DFF  # Purple
                title_prefix = f"🌸 SHIORI AI — TỔNG KẾT MÀN CHƠI OSU! [HẠNG {rank}]"
            else:
                color = 0xFF9100  # Amber
                title_prefix = f"🌸 SHIORI AI — TỔNG KẾT MÀN CHƠI OSU! [HẠNG {rank}]"

            acc_bar = self._render_accuracy_bar(acc)

            fields = [
                {"name": "🎯 Độ Chính Xác (Accuracy)", "value": acc_bar, "inline": False},
                {"name": "🔥 Max Combo", "value": f"**`{max_combo}x`**", "inline": True},
                {"name": "🏆 Xếp Hạng", "value": f"**`Rank {rank}`**", "inline": True},
                {"name": "🎭 Live2D", "value": f"`{payload.get('emotion', 'neutral')}`", "inline": True},
                {
                    "name": "📊 Thống Kê Hit Judgements",
                    "value": f"🟢 300: **`{h300}`** | 🟡 100: **`{h100}`**\n🟠 50: **`{h50}`** | 🔴 Miss: **`{miss}`**",
                    "inline": True,
                },
                {
                    "name": "⚡ Lõi Xử Lý & Tự Tinh Chỉnh",
                    "value": (
                        f"🚀 **GPU:** `{payload.get('gpu_device', 'RTX 2060')}`\n"
                        f"⏱️ **Lead-time:** `{payload.get('lead_time_ms', 12.0):.1f}ms` "
                        f"(Offset: `{payload.get('memory_offset_ms', 0.0):+.1f}ms`)"
                    ),
                    "inline": True,
                },
            ]

            commentary = payload.get("commentary", "")
            if commentary:
                fields.append({
                    "name": "💬 Lời Nhắn Từ Shiori AI",
                    "value": f"*{commentary}*",
                    "inline": False,
                })

            return {
                "title": title_prefix,
                "description": (
                    f"### 🎵 {payload.get('title')} [{payload.get('difficulty')}]\n"
                    f"**Nghệ sĩ:** `{payload.get('artist')}` • **Chế độ:** `{payload.get('mode')}`"
                ),
                "color": color,
                "fields": fields,
                "footer": {"text": "Shiori AI Autonomous osu! Engine • Kênh 1563921667556773969"},
                "timestamp": iso_now,
            }

    # =========================================================================
    # WORKER DELIVERY (WEBHOOK -> BOT REST -> WEBSOCKET BRIDGE)
    # =========================================================================

    def _dispatch_report_worker(self, payload: Dict[str, Any]) -> None:
        """Dual-delivery engine with automatic failover."""
        now = time.time()
        # Rate-limiting guard: prevent Discord HTTP 429
        if (now - self._last_report_time) < 0.5:
            time.sleep(0.5 - (now - self._last_report_time))

        embed = self._build_embed(payload)
        delivered = False

        # 1. Primary: Direct Discord Webhook if configured
        if self._webhook_url:
            delivered = self._send_via_webhook(embed)

        # 2. Secondary: Direct Discord REST API v10 via Bot Token
        if not delivered and self._bot_token:
            delivered = self._send_via_bot_rest(embed, payload.get("channelId", self.target_channel_id))

        if delivered:
            self._last_report_time = time.time()

        # 3. Tertiary: Broadcast via Shiori WebSocket Bridge to Discord Bot client
        try:
            self._broadcast_via_bridge_sync(payload)
        except Exception as e:
            logger.debug(f"[DiscordReporter] Bridge broadcast: {e}")

    def _send_via_webhook(self, embed: Dict[str, Any]) -> bool:
        """Sends rich embed directly via Discord Webhook URL with retry on timeout."""
        if not self._webhook_url:
            return False
        body = json.dumps({"embeds": [embed]}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ShioriDiscordLogger/2.0 (Windows NT 10.0; Win64; x64)",
        }
        for attempt in range(2):
            try:
                req = urllib.request.Request(self._webhook_url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    if resp.status in (200, 204):
                        logger.info("[DiscordReporter] Successfully delivered embed via Discord Webhook.")
                        return True
            except Exception as e:
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                logger.warning(f"[DiscordReporter] Webhook delivery failed: {e}")
        return False

    def _send_via_bot_rest(self, embed: Dict[str, Any], channel_id: str) -> bool:
        """Direct Discord REST API v10 POST request using DISCORD_BOT_TOKEN with retry on timeout."""
        token = self._bot_token or os.environ.get("DISCORD_BOT_TOKEN") or os.environ.get("DISCORD_TOKEN")
        if not token:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                token = os.environ.get("DISCORD_BOT_TOKEN") or os.environ.get("DISCORD_TOKEN")
                if token:
                    self._bot_token = token
            except Exception:
                pass
        if not token:
            return False

        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        body = json.dumps({"embeds": [embed]}).encode("utf-8")
        headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "ShioriBot/2.0 (Windows NT 10.0; Win64; x64)",
        }
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    if resp.status in (200, 201):
                        logger.info(f"[DiscordReporter] Successfully posted embed to Discord channel {channel_id}.")
                        return True
            except Exception as e:
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                logger.warning(f"[DiscordReporter] Failed to send via REST API to channel {channel_id}: {e}")
                return False
        return False

    def _broadcast_via_bridge_sync(self, payload: Dict[str, Any]) -> None:
        """Broadcasts payload to ShioriBridge if active."""
        try:
            import sys
            bridge = getattr(sys.modules.get("src.integrations.shiori_bridge"), "global_bridge", None)
            if bridge and hasattr(bridge, "broadcast_event"):
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    pass
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        bridge.broadcast_event("output:osu:game_log", payload), loop
                    )
                else:
                    asyncio.run(bridge.broadcast_event("output:osu:game_log", payload))
        except Exception:
            pass


osu_discord_reporter = OsuDiscordReporter()
