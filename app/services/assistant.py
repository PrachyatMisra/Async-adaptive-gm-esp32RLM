"""Adapter around the repository-grounded chat assistant.

The assistant is optional and entirely local. It may consult a loopback
Ollama server; when none is reachable it answers from indexed repository
files. It never calls a hosted service.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Callable

from ..config import Settings


def _load_chat_agent(repo_root: Path):
    """Import `host/chat_agent.py` without requiring it to be a package."""
    host_dir = str(repo_root / "host")
    if host_dir not in sys.path:
        sys.path.insert(0, host_dir)
    import chat_agent  # type: ignore

    return chat_agent


class AssistantService:
    """Thin async wrapper around the synchronous chat agent."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._agent = _load_chat_agent(settings.repo_root)

    async def status(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._agent.status)

    async def respond(
        self,
        text: str,
        history: list[dict[str, str]],
        mode: str,
        profile: str,
        inference: Callable[..., dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()

        def _call() -> dict[str, Any]:
            return self._agent.respond(
                text=text,
                history=history,
                mode=mode,
                profile=profile,
                inference=inference,
            )

        return await loop.run_in_executor(None, _call)
