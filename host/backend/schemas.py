from pydantic import BaseModel, Field

class InferenceRequest(BaseModel):
    tokens: list[int]
    max_tokens: int = Field(default=96, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class TelemetryPayload(BaseModel):
    latency_ms: float
    current_window: int
    g_t: float
    active_core: int
    memory_free_bytes: int
    execution_source: str = "host_native"

class FallbackEvent(BaseModel):
    timestamp: float
    previous_window: int
    new_window: int
    trigger_reason: str
