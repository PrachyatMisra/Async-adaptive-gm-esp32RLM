import asyncio
import time
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from host.backend.schemas import InferenceRequest, TelemetryPayload

app = FastAPI(title="RLM ESP32 AAGM v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "rlm-esp32-aagm-v2"}

@app.get("/phase1")
def phase1():
    return {"status": "ok", "runtime": "phase1_pytorch_research"}

@app.get("/phase2")
def phase2():
    return {"status": "ok", "runtime": "phase2_cpp_edge"}

@app.post("/api/v2/infer")
def infer(req: InferenceRequest):
    simulated_latency = random.uniform(20.0, 180.0)
    effective_window = req.max_tokens
    if simulated_latency > 150.0:
        effective_window = max(24, req.max_tokens // 2)
    return {
        "status": "success",
        "tokens_generated": req.tokens + [42, 43],
        "latency_ms": round(simulated_latency, 2),
        "effective_window": effective_window
    }

@app.websocket("/ws/v2/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = TelemetryPayload(
                latency_ms=round(random.uniform(10.0, 120.0), 2),
                current_window=96,
                g_t=round(random.uniform(0.1, 0.9), 4),
                active_core=random.choice([0, 1]),
                memory_free_bytes=random.randint(120000, 250000)
            )
            await websocket.send_text(payload.model_dump_json())
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
