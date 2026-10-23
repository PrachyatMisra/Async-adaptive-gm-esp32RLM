"""Runtime configuration for the local API.

All settings are environment-driven with safe local-first defaults:

  RLM_HOST                 bind address (default 127.0.0.1, loopback only)
  RLM_PORT                 bind port (default 8000)
  RLM_CORS_ORIGINS         comma separated explicit origins; empty by default
  RLM_REQUEST_MAX_BYTES    maximum accepted request body (default 65536)
  RLM_ENGINE_TIMEOUT_S     per-inference timeout (default 30)
  RLM_VERILOG_TIMEOUT_S    per RTL-check timeout (default 300)
  RLM_MAX_ENGINE_CONCURRENCY  concurrent harness processes (default 1)

The bind address defaults to loopback. Exposing the server on a LAN address
is an explicit, documented choice: set RLM_HOST=0.0.0.0 to do it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class Settings(BaseSettings):
    """Local-only server settings."""

    model_config = SettingsConfigDict(
        env_prefix="RLM_",
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)

    # Explicit allow-list. The wildcard "*" is deliberately not representable:
    # a browser on another origin must never be able to drive this API.
    cors_origins: str = Field(default="")

    request_max_bytes: int = Field(default=65536, ge=1024, le=1_048_576)
    engine_timeout_s: float = Field(default=30.0, gt=0)
    verilog_timeout_s: float = Field(default=300.0, gt=0)
    max_engine_concurrency: int = Field(default=1, ge=1, le=8)

    # Repository layout. Overridable so tests can point at a scratch copy.
    repo_root: Path = Field(default=REPO_ROOT)
    frontend_dist: Path = Field(default=REPO_ROOT / "web" / "dist")

    @field_validator("host")
    @classmethod
    def _known_host(cls, value: str) -> str:
        value = value.strip() or "127.0.0.1"
        if value not in {"127.0.0.1", "localhost", "0.0.0.0", "::"}:
            # Any other address is allowed but must have been asked for.
            return value
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if not raw:
            return []
        origins = [item.strip() for item in raw.split(",") if item.strip()]
        # A wildcard would allow any web page to drive the local API.
        return [origin for origin in origins if origin != "*"]

    @property
    def harness_path(self) -> Path:
        return self.repo_root / "firmware" / "test" / "host_harness"

    @property
    def harness_dir(self) -> Path:
        return self.repo_root / "firmware" / "test"

    @property
    def golden_path(self) -> Path:
        return self.repo_root / "tools" / "out" / "golden_vectors.txt"

    @property
    def verilog_dir(self) -> Path:
        return self.repo_root / "verification" / "verilog"

    @property
    def artifacts_dir(self) -> Path:
        return self.repo_root / "artifacts"

    @property
    def graph_index(self) -> Path:
        return self.repo_root / "graphify-out" / "graph.json"

    @property
    def legacy_html_dir(self) -> Path:
        return self.repo_root / "host"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (cached)."""
    return Settings()
