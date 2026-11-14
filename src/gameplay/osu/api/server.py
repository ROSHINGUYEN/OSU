"""
FastAPI Server instance for Shiori AI osu! Gameplay API.
Can be run standalone or spawned in background thread by Shiori runtime.
"""

import threading
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.gameplay.osu.api.routes import router as osu_router

app = FastAPI(
    title="Shiori AI osu! Gameplay API",
    description="Real-time gameplay telemetry, debug frame extraction, and MCP interface.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(osu_router)

_server_thread = None


def start_osu_api_server(host: str = "127.0.0.1", port: int = 8003) -> None:
    """Start the osu! API server in a background daemon thread."""
    global _server_thread
    if _server_thread is not None and _server_thread.is_alive():
        return

    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config=config)

    _server_thread = threading.Thread(target=server.run, daemon=True, name="OsuApiServerThread")
    _server_thread.start()


if __name__ == "__main__":
    uvicorn.run("src.gameplay.osu.api.server:app", host="127.0.0.1", port=8003, reload=True)
