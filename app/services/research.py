"""Loaders for the recorded research artifacts.

Every artifact this module returns was produced by a tool in `tools/` or
`host/` and is tagged with the provenance it actually carries. Where an
artifact's own schema is ambiguous, the ambiguity is surfaced rather than
papered over.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..config import Settings
from ..provenance import ExecutionSource


class ArtifactUnavailable(RuntimeError):
    """Raised when a recorded artifact is missing or unreadable."""


def _read_json(path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactUnavailable(f"{path.name} is not present in artifacts/") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactUnavailable(f"{path.name} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ArtifactUnavailable(f"{path.name} could not be read: {exc}") from exc


class ResearchService:
    """Read-only access to the recorded artifacts."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def software_results(self) -> dict[str, Any]:
        return await asyncio.to_thread(
            _read_json, self._settings.artifacts_dir / "esp32_software_results.json"
        )

    async def cognitive_benchmark(self) -> dict[str, Any]:
        return await asyncio.to_thread(
            _read_json,
            self._settings.artifacts_dir / "cognitive_reasoning_benchmark.json",
        )

    async def parity(self) -> dict[str, Any]:
        return await asyncio.to_thread(
            _read_json, self._settings.artifacts_dir / "esp32_parity.json"
        )

    async def telemetry(self) -> dict[str, Any]:
        return await asyncio.to_thread(
            _read_json,
            self._settings.artifacts_dir / "esp32_eval_telemetry.json",
        )

    async def project_data(self) -> dict[str, Any]:
        """The aggregate view used by the research page."""

        def _load() -> dict[str, Any]:
            return {
                "software_results": _read_json(
                    self._settings.artifacts_dir / "esp32_software_results.json"
                ),
                "cognitive_benchmark": _read_json(
                    self._settings.artifacts_dir
                    / "cognitive_reasoning_benchmark.json"
                ),
                "model_config": {
                    "min_recursion": 2,
                    "max_recursion": 8,
                    "halt_threshold": 0.9,
                },
            }

        try:
            return await asyncio.to_thread(_load)
        except ArtifactUnavailable as exc:
            raise ArtifactUnavailable(str(exc)) from exc

    async def graph_index(self) -> dict[str, Any]:
        path = self._settings.graph_index
        if not path.exists():
            raise ArtifactUnavailable(
                "graphify-out/graph.json is not present; run the Graphify indexer"
            )
        return await asyncio.to_thread(_read_json, path)

    @staticmethod
    def software_results_source(payload: dict[str, Any]) -> ExecutionSource:
        """Classify the software-results artifact.

        The artifact is the output of the local training/export pipeline, so
        it is `trained_model`. If it instead carries a manuscript marker, it
        is reported as `paper_reported` instead.
        """
        if payload.get("paper_reported") or payload.get("source") == "paper":
            return ExecutionSource.PAPER_REPORTED
        return ExecutionSource.TRAINED_MODEL
