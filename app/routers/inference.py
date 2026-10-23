"""Versioned inference and serial-console routes."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request

from ..schemas import (
    InferenceRequest,
    InferenceResponse,
    SerialRequest,
    SerialResponse,
)
from ..services.harness import HarnessError, HarnessTimeout

router = APIRouter(prefix="/api/v1", tags=["inference"])


def _services(request: Request):
    return request.app.state.services


@router.post(
    "/inference",
    response_model=InferenceResponse,
    summary="Run a sentiment inference through the native C++ engine",
)
async def run_inference(
    body: InferenceRequest, request: Request
) -> InferenceResponse:
    services = _services(request)
    try:
        return await services.inference.run(body)
    except HarnessTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except HarnessError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/serial",
    response_model=SerialResponse,
    summary="Send one command to the mock UART and return the device reply",
)
async def serial(body: SerialRequest, request: Request) -> SerialResponse:
    services = _services(request)
    try:
        output = await services.harness.serial(body.command)
    except HarnessTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except HarnessError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    parsed: dict | None = None
    for line in reversed(output.splitlines()):
        candidate = line.strip()
        if candidate.startswith("{"):
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue

    return SerialResponse(
        command=body.command,
        device_output=output,
        parsed=parsed,
    )
