"""Pydantic v2 request and response models for the versioned API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .provenance import ExecutionSource

Profile = Literal["PERF", "BAL", "ECO"]
HaltReason = Literal["act_halt", "budget_exhausted", "aborted", "none"]


class Provenance(BaseModel):
    """Provenance block attached to every API response."""

    model_config = ConfigDict(extra="forbid")

    execution_source: ExecutionSource
    note: str = ""
    simulator: str | None = None
    hardware: str = "none"


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str | None = None
    execution_source: ExecutionSource = ExecutionSource.HOST_NATIVE


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4096)
    profile: Profile = "PERF"
    budget: int | None = Field(default=None, ge=2, le=8)
    batt_mv: float = Field(default=4000.0, ge=2500.0, le=4500.0)
    tstf: bool = False

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("text must contain at least one non-space character")
        return cleaned


class GateTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gates: list[float] = Field(default_factory=list)
    shadows: list[float] = Field(default_factory=list)
    halting_mass: float = 0.0
    depth: float = 0.0
    prearm_hints: int = 0
    prearm_confirmed: int = 0
    frozen_tokens: int = 0
    macs_saved: int = 0


class EnergyEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_source: ExecutionSource = ExecutionSource.HOST_NATIVE
    note: str = (
        "Analytic model of energy drawn at the stated clock and current. "
        "Not a power measurement on hardware."
    )
    current_ma: float
    rail_v: float = 3.3
    energy_uj: float
    baseline_energy_uj: float
    saved_pct: float
    steps: int
    baseline_steps: int


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_source: ExecutionSource = ExecutionSource.HOST_NATIVE
    note: str = ""

    prediction: int
    logits: list[float]
    verdict: Literal["POSITIVE", "NEGATIVE"]
    confidence_pct: float
    steps: int
    budget: int
    halt_reason: HaltReason
    aborted: bool
    success: bool = Field(
        description="True only when the run halted for the ACT reason."
    )

    tokens: int
    profile: Profile
    cpu_mhz: int
    batt_mv: float = Field(
        description="Measured terminal voltage under load, in millivolts."
    )
    ocv_mv: float = Field(
        description="Open-circuit voltage estimated by adding the droop back."
    )
    droop_mv: float
    batt_src: str = Field(
        description="Where the battery reading came from (sim, adc, ...)."
    )

    host_latency_us: int = Field(
        description="Wall-clock time of the native engine on this host. "
        "Not ESP32 latency."
    )
    trace: GateTrace
    energy: EnergyEstimate
    tstf: bool = False


# ----------------------------------------------------------------------
# Serial console
# ----------------------------------------------------------------------
class SerialRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(default="STAT", min_length=1, max_length=512)

    @field_validator("command")
    @classmethod
    def _single_line(cls, value: str) -> str:
        cleaned = value.strip()
        if "\n" in cleaned or "\r" in cleaned:
            raise ValueError("command must be a single line")
        return cleaned


class SerialResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_source: ExecutionSource = ExecutionSource.HOST_NATIVE
    note: str = "Host harness speaking the firmware UART protocol. No board attached."
    command: str
    device_output: str
    parsed: dict[str, Any] | None = None


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------
class GoldenSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    golden: int = 0
    passed: int = 0
    skipped: int = 0
    max_logit_err: float | None = None
    max_gate_err: float | None = None
    max_shadow_err: float | None = None
    max_depth_err: float | None = None
    mean_steps: float | None = None
    mean_prearm_hints: float | None = None
    workspace_bytes: int | None = None
    max_tokens: int | None = None
    passed_flag: bool = False


class GoldenResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_source: ExecutionSource = ExecutionSource.HOST_NATIVE
    note: str = (
        "Numerical parity of the native C++ engine against the approved "
        "golden vectors. These are regression vectors, not a correctness "
        "oracle."
    )
    status: Literal["pass", "fail", "error"]
    summary: GoldenSummary
    samples: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class RtlResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_source: ExecutionSource
    note: str
    status: Literal["pass", "fail", "error"]
    simulator: str
    output: str
    table: list[dict[str, Any]] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Research data
# ----------------------------------------------------------------------
class DatasetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_source: ExecutionSource
    note: str
    dataset: str
    sources: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class ProjectDataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_source: ExecutionSource
    note: str
    model_config_payload: dict[str, Any] = Field(default_factory=dict)
    software_results: dict[str, Any] = Field(default_factory=dict)
    cognitive_benchmark: dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------------------------
# Assistant
# ----------------------------------------------------------------------
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=16)
    mode: Literal["chat", "analyze"] = "chat"
    profile: Profile = "PERF"


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_source: ExecutionSource = ExecutionSource.HOST_NATIVE
    note: str
    reply: str
    kind: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    engine: str
    analysis: dict[str, Any] | None = None


class AssistantStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_source: ExecutionSource = ExecutionSource.HOST_NATIVE
    note: str
    backend: str
    model: str
    host: str
    label: str
    ollama_online: bool
    model_installed: bool
    ollama_error: str | None = None


# ----------------------------------------------------------------------
# System
# ----------------------------------------------------------------------
class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_source: ExecutionSource = ExecutionSource.HOST_NATIVE
    status: Literal["ok", "degraded"]
    version: str
    harness_built: bool
    iverilog_available: bool
    vvp_available: bool
    verilator_available: bool
    platformio_available: bool
    wokwi_available: bool
    hardware_attached: bool = False
    note: str = (
        "Capability probe. No physical ESP32 is attached to this service; "
        "all inference is host-native."
    )
