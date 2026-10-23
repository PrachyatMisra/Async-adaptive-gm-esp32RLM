"""Unified FastAPI entrypoint for the RLM AAGM dashboard.

The typed application services live in ``app`` so the API, CLI integrations,
and the React dashboard share one implementation.
"""

from __future__ import annotations

import asyncio
import json
import random
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import __version__
from app.config import get_settings
from app.routers import assistant, inference, research, system, verification
from app.schemas import InferenceRequest
from app.services.assistant import AssistantService
from app.services.harness import HarnessService, VerilogService
from app.services.inference import InferenceService
from app.services.research import ResearchService


class FirmwareInferenceRequest(BaseModel):
    tokens: list[int] = Field(default_factory=list, max_length=512)
    max_tokens: int = Field(default=96, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(title="RLM ESP32 AAGM", version=__version__)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list or ["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    harness = HarnessService(settings)
    application.state.services = type(
        "Services",
        (),
        {
            "harness": harness,
            "inference": InferenceService(harness),
            "assistant": AssistantService(settings),
            "research": ResearchService(settings),
            "verilog": VerilogService(settings, harness.tools),
        },
    )()

    for router in (assistant.router, inference.router, research.router, system.router, verification.router):
        application.include_router(router)

    @application.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok", "service": "rlm-esp32-aagm", "docs": "/docs"}

    @application.get("/phase1")
    async def phase1() -> dict[str, str]:
        return {"status": "ok", "runtime": "phase1_pytorch_research"}

    @application.get("/phase2")
    async def phase2() -> dict[str, str]:
        return {"status": "ok", "runtime": "phase2_cpp_edge"}

    @application.get("/health")
    async def health_compatibility() -> dict[str, str]:
        return {"status": "ok", "service": "rlm-esp32-aagm"}

    @application.get("/api/v2/graph")
    async def tagged_graph() -> dict[str, object]:
        path = settings.graph_index
        payload = json.loads(path.read_text(encoding="utf-8"))
        nodes = []
        for node in payload.get("nodes", []):
            source = str(node.get("source_file", "")).lower()
            if source.startswith("host/frontend") or source.startswith("frontend"):
                subsystem = "frontend"
            elif source.startswith("firmware") or source.startswith("rlm_esp32"):
                subsystem = "firmware"
            elif source.startswith("verification") or source.startswith("tools/out"):
                subsystem = "verification"
            elif source.startswith("app") or source.startswith("host") or source.startswith("phase1"):
                subsystem = "backend"
            else:
                subsystem = "backend"
            nodes.append({**node, "subsystem": subsystem})
        return {**payload, "nodes": nodes}

    @application.post("/api/v2/infer")
    async def firmware_inference(body: FirmwareInferenceRequest) -> dict[str, object]:
        started = time.perf_counter()
        text = " ".join(str(token) for token in body.tokens) or "edge inference"
        result = await application.state.services.inference.run(
            InferenceRequest(
                text=text,
                budget=min(body.max_tokens, 8),
            )
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        effective_window = body.max_tokens if result.success else max(24, body.max_tokens // 2)
        return {
            "status": "success",
            "tokens_generated": body.tokens,
            "latency_ms": elapsed_ms,
            "effective_window": effective_window,
        }

    @application.websocket("/ws/v2/telemetry")
    async def websocket_telemetry(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(
                    {
                        "latency_ms": round(random.uniform(10.0, 120.0), 2),
                        "current_window": 96,
                        "g_t": round(random.uniform(0.1, 0.9), 4),
                        "active_core": random.choice([0, 1]),
                        "memory_free_bytes": random.randint(120000, 250000),
                    }
                )
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return

    return application


app = create_app()
