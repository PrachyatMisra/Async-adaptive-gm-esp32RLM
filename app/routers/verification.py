"""Versioned verification routes: golden parity, RTL checks, waveform."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..provenance import ExecutionSource
from ..schemas import GoldenResponse, GoldenSummary, RtlResponse
from ..services.harness import HarnessError, HarnessTimeout

router = APIRouter(prefix="/api/v1/verification", tags=["verification"])


def _services(request: Request):
    return request.app.state.services


@router.get(
    "/golden",
    response_model=GoldenResponse,
    summary="Numerical parity of the native engine against the golden vectors",
)
async def golden(request: Request) -> GoldenResponse:
    services = _services(request)
    try:
        payload = await services.harness.golden_check()
    except HarnessTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except HarnessError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    summary = payload.get("summary", {}) or {}
    passed = bool(summary.get("PASS", False))
    return GoldenResponse(
        status="pass" if passed else "fail",
        summary=GoldenSummary(
            golden=int(summary.get("golden", 0)),
            passed=int(summary.get("passed", 0)),
            skipped=int(summary.get("skipped", 0)),
            max_logit_err=summary.get("max_logit_err"),
            max_gate_err=summary.get("max_gate_err"),
            max_shadow_err=summary.get("max_shadow_err"),
            max_depth_err=summary.get("max_depth_err"),
            mean_steps=summary.get("mean_steps"),
            mean_prearm_hints=summary.get("mean_prearm_hints"),
            workspace_bytes=summary.get("workspace_bytes"),
            max_tokens=summary.get("max_tokens"),
            passed_flag=passed,
        ),
        samples=payload.get("samples", []) or [],
        raw=payload,
    )


@router.get(
    "/rtl",
    response_model=RtlResponse,
    summary="Run the RTL control-policy verification",
)
async def rtl(request: Request) -> RtlResponse:
    services = _services(request)
    result = await services.verilog.check()
    source = services.verilog.execution_source
    simulator = (
        "icarus"
        if source is ExecutionSource.RTL_SIMULATION
        else "python-reference-model"
    )
    return RtlResponse(
        execution_source=source,
        note=services.verilog.note,
        status=result["status"],  # type: ignore[arg-type]
        simulator=simulator,
        output=result["output"],
        table=result.get("table", []),
    )
