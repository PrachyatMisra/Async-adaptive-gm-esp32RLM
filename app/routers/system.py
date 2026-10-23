"""System routes: health, capability probe, and the code graph."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from .. import __version__
from ..schemas import HealthResponse
from ..services.research import ArtifactUnavailable

router = APIRouter(prefix="/api/v1", tags=["system"])


def _services(request: Request):
    return request.app.state.services


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    services = _services(request)
    tools = services.harness.tools
    degraded = not services.harness.is_built
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=__version__,
        harness_built=services.harness.is_built,
        iverilog_available=tools.iverilog,
        vvp_available=tools.vvp,
        verilator_available=tools.verilator,
        platformio_available=tools.platformio,
        wokwi_available=tools.wokwi,
    )


@router.get("/graph", summary="Repository code and pipeline graph index")
async def graph(request: Request):
    services = _services(request)
    try:
        return await services.research.graph_index()
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
