"""Typed local API for the Edge-RLM ESP32 AAGM project.

The package exposes a FastAPI application with a versioned, typed surface
(`/api/v1/...`) plus thin compatibility wrappers for the historical
unversioned routes (`/api/infer`, `/api/verify-golden`, ...).

Every response carries an `execution_source` field. The value states where
the numbers actually came from, so a host simulation is never presented as
an RTL simulation and a paper-reported figure is never presented as a
measurement.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
