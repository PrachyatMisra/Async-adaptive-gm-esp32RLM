"""Versioned research-data routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..provenance import ExecutionSource
from ..schemas import DatasetResponse, ProjectDataResponse
from ..services.research import ArtifactUnavailable

router = APIRouter(prefix="/api/v1/research", tags=["research"])


def _services(request: Request):
    return request.app.state.services


@router.get(
    "/project-data",
    response_model=ProjectDataResponse,
    summary="Aggregate research data for the overview page",
)
async def project_data(request: Request) -> ProjectDataResponse:
    services = _services(request)
    try:
        payload = await services.research.project_data()
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    source = services.research.software_results_source(
        payload.get("software_results", {})
    )
    return ProjectDataResponse(
        execution_source=source,
        note=(
            "Recorded artifact produced by the local training/export "
            "pipeline. It is not a live measurement."
        ),
        model_config_payload=payload.get("model_config", {}),
        software_results=payload.get("software_results", {}),
        cognitive_benchmark=payload.get("cognitive_benchmark", {}),
    )


@router.get(
    "/dataset/training",
    response_model=DatasetResponse,
    summary="Training-pipeline results artifact",
)
async def training_dataset(request: Request) -> DatasetResponse:
    services = _services(request)
    try:
        payload = await services.research.software_results()
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DatasetResponse(
        execution_source=services.research.software_results_source(payload),
        note=(
            "Output of the local training/export pipeline. Reproducible only "
            "for a fixed corpus and seed on the same machine."
        ),
        dataset=str(payload.get("dataset", "unknown")),
        sources=["artifacts/esp32_software_results.json"],
        payload=payload,
    )


@router.get(
    "/dataset/device",
    response_model=DatasetResponse,
    summary="Device-side parity and telemetry artifacts",
)
async def device_dataset(request: Request) -> DatasetResponse:
    services = _services(request)
    try:
        parity = await services.research.parity()
        telemetry = await services.research.telemetry()
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DatasetResponse(
        execution_source=ExecutionSource.HOST_NATIVE,
        note=(
            "Parity and telemetry records captured from the host harness "
            "speaking the firmware protocol. No physical ESP32 was attached."
        ),
        dataset="ESP32-RLM host simulation",
        sources=[
            "artifacts/esp32_parity.json",
            "artifacts/esp32_eval_telemetry.json",
        ],
        payload={"parity": parity, "telemetry": telemetry},
    )
