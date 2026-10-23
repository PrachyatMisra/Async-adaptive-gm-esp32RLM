"""Versioned assistant routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from ..schemas import AssistantStatusResponse, ChatRequest, ChatResponse
from ..services.harness import HarnessError, HarnessTimeout

router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])


def _services(request: Request):
    return request.app.state.services


@router.get("/status", response_model=AssistantStatusResponse)
async def status(request: Request) -> AssistantStatusResponse:
    services = _services(request)
    payload = await services.assistant.status()
    return AssistantStatusResponse(
        note=(
            "The assistant is entirely local. It may consult a loopback "
            "Ollama server if one is running; otherwise it answers from "
            "indexed repository files."
        ),
        backend=str(payload.get("backend", "local")),
        model=str(payload.get("model", "")),
        host=str(payload.get("host", "")),
        label=str(payload.get("label", "")),
        ollama_online=bool(payload.get("ollama_online", False)),
        model_installed=bool(payload.get("model_installed", False)),
        ollama_error=payload.get("ollama_error"),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    services = _services(request)
    history = [message.model_dump() for message in body.history]

    def _inference(text: str, budget: int, profile: str, batt_mv: float):
        from ..schemas import InferenceRequest

        result = asyncio.run(
            services.inference.run(
                InferenceRequest(
                    text=text,
                    profile=profile,  # type: ignore[arg-type]
                    budget=budget,
                    batt_mv=batt_mv,
                )
            )
        )
        payload = result.model_dump()
        payload.update(
            {
                "pred": result.prediction,
                "mass": result.trace.halting_mass,
                "us": result.host_latency_us,
            }
        )
        return payload

    try:
        payload = await services.assistant.respond(
            text=body.text,
            history=history,
            mode=body.mode,
            profile=body.profile,
            inference=_inference,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (HarnessTimeout, HarnessError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(
        note="Local repository assistant with optional loopback Ollama inference.",
        reply=str(payload.get("reply", "")),
        kind=str(payload.get("kind", "chat")),
        sources=payload.get("sources", []) or [],
        engine=str(payload.get("engine", "Local")),
        analysis=payload.get("analysis"),
    )
