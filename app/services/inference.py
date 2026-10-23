"""Turns raw harness telemetry into the typed inference response.

The energy figures produced here are an ANALYTIC MODEL, not a measurement:
they combine the measured host latency with a board-level current estimate
and a nominal 3.3 V rail. The response says so in the `note` field of the
energy block, and the whole response is tagged `host_native` rather than
`physical_device`.
"""

from __future__ import annotations

import math
from typing import Any

from ..schemas import (
    EnergyEstimate,
    GateTrace,
    InferenceRequest,
    InferenceResponse,
)
from .harness import HarnessService

#: Board-level current estimates, matching the firmware's `load_current_ma`.
#: These are estimates of the SoC's supply current at each clock point, not
#: measurements taken from a board.
CURRENT_MA = {"PERF": 50.0, "BAL": 36.0, "ECO": 27.0}

#: The rail the model is nominally powered from.
RAIL_V = 3.3

#: The fixed-depth baseline the adaptive scheme is compared against: the
#: engine's maximum recursion depth.
BASELINE_STEPS = 8

PROFILE_BUDGET = {"PERF": 8, "BAL": 6, "ECO": 4}


def _confidence(logits: list[float], prediction: int) -> float:
    if len(logits) != 2:
        return 0.0
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    if total <= 0:
        return 0.0
    index = prediction if prediction in (0, 1) else 0
    return round(100.0 * exps[index] / total, 2)


def build_response(
    request: InferenceRequest, payload: dict[str, Any]
) -> InferenceResponse:
    """Map a raw harness payload onto the typed response."""
    logits = [float(value) for value in payload.get("logits", [0.0, 0.0])]
    prediction = int(payload.get("pred", 0))
    steps = int(payload.get("steps", 0))
    budget = int(payload.get("budget", request.budget or PROFILE_BUDGET[request.profile]))
    halt_reason = str(payload.get("halt_reason", "none"))
    aborted = bool(payload.get("aborted", 0))
    host_us = int(payload.get("us", 0))
    profile = str(payload.get("profile", request.profile))
    current_ma = CURRENT_MA.get(profile, 50.0)

    # Energy model: I * V * t, with t taken from the measured host latency.
    energy_uj = round(current_ma * RAIL_V * (host_us / 1000.0), 1)
    per_step_ms = (host_us / max(steps, 1)) / 1000.0
    baseline_uj = round(current_ma * RAIL_V * (per_step_ms * BASELINE_STEPS), 1)
    saved_pct = (
        round(max(0.0, (1.0 - energy_uj / baseline_uj) * 100.0), 1)
        if baseline_uj > 0
        else 0.0
    )

    return InferenceResponse(
        execution_source=payload.get("_execution_source", "host_native"),
        note=(
            "Executed by the native C++ engine on this workstation. The "
            "latency is host wall time, not ESP32 latency, and the energy "
            "figures are an analytic model, not a power measurement."
        ),
        prediction=prediction,
        logits=logits,
        verdict="POSITIVE" if prediction == 1 else "NEGATIVE",
        confidence_pct=_confidence(logits, prediction),
        steps=steps,
        budget=budget,
        halt_reason=halt_reason,  # type: ignore[arg-type]
        aborted=aborted,
        success=halt_reason == "act_halt",
        tokens=int(payload.get("tokens", 0)),
        profile=profile,  # type: ignore[arg-type]
        cpu_mhz=int(payload.get("cpu_mhz", 0)),
        batt_mv=float(payload.get("batt_mv", 0.0)),
        ocv_mv=float(payload.get("ocv_mv", 0.0)),
        droop_mv=float(payload.get("droop_mv", 0.0)),
        batt_src=str(payload.get("batt_src", "sim")),
        host_latency_us=host_us,
        trace=GateTrace(
            gates=[float(value) for value in payload.get("gates", [])],
            shadows=[float(value) for value in payload.get("shadows", [])],
            halting_mass=float(payload.get("mass", 0.0)),
            depth=float(payload.get("depth", 0.0)),
            prearm_hints=int(payload.get("hints", 0)),
            prearm_confirmed=int(payload.get("prearm_confirmed", 0)),
            frozen_tokens=int(payload.get("frozen", 0)),
            macs_saved=int(payload.get("macs_saved", 0)),
        ),
        energy=EnergyEstimate(
            current_ma=current_ma,
            rail_v=RAIL_V,
            energy_uj=energy_uj,
            baseline_energy_uj=baseline_uj,
            saved_pct=saved_pct,
            steps=steps,
            baseline_steps=BASELINE_STEPS,
        ),
        tstf=bool(payload.get("tstf", 0)),
    )


class InferenceService:
    """Runs an inference through the native harness."""

    def __init__(self, harness: HarnessService) -> None:
        self._harness = harness

    async def run(self, request: InferenceRequest) -> InferenceResponse:
        budget = request.budget or PROFILE_BUDGET.get(request.profile, 8)
        payload = await self._harness.infer(
            text=request.text,
            budget=budget,
            profile=request.profile,
            batt_mv=request.batt_mv,
            tstf=request.tstf,
        )
        return build_response(request, payload)
