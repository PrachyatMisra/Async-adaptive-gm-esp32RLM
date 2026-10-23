"""Service wrappers around the native C++ engine and the Verilog checkers.

Two rules govern this module:

  * The engine is non-reentrant. Every inference therefore runs in its own
    short-lived process, and the number of concurrent engine processes is
    bounded (default 1) so the host is not oversubscribed.
  * Subprocesses never block the event loop. They are awaited through
    `asyncio.to_thread`, and every call has a timeout.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings
from ..provenance import ExecutionSource


class HarnessError(RuntimeError):
    """Raised when the native harness cannot be built or run."""


class HarnessTimeout(HarnessError):
    """Raised when a harness invocation exceeds its timeout."""


@dataclass(frozen=True)
class ToolAvailability:
    iverilog: bool
    vvp: bool
    verilator: bool
    platformio: bool
    wokwi: bool

    @property
    def rtl_simulation_possible(self) -> bool:
        return self.iverilog and self.vvp


def probe_tools() -> ToolAvailability:
    """Report which external simulators are installed.

    Absence of a tool is not an error: the API degrades to the reference
    tier and says so in `execution_source`.
    """
    return ToolAvailability(
        iverilog=shutil.which("iverilog") is not None,
        vvp=shutil.which("vvp") is not None,
        verilator=shutil.which("verilator") is not None,
        platformio=shutil.which("pio") is not None,
        wokwi=shutil.which("wokwi-cli") is not None,
    )


class HarnessService:
    """Build-once, run-bounded wrapper around `firmware/test/host_harness`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._build_lock = asyncio.Lock()
        self._build_ok = False
        self._build_error: str | None = None
        self._semaphore = asyncio.Semaphore(settings.max_engine_concurrency)
        self._tools = probe_tools()

    # -- lifecycle -----------------------------------------------------
    @property
    def tools(self) -> ToolAvailability:
        return self._tools

    @property
    def is_built(self) -> bool:
        return self._build_ok and self._settings.harness_path.exists()

    async def ensure_built(self) -> None:
        """Build the native harness once. Safe to call repeatedly.

        The build is performed with `make`, which is itself incremental, so
        repeated calls are cheap once the binary exists.
        """
        if self.is_built:
            return
        async with self._build_lock:
            if self.is_built:
                return
            try:
                await asyncio.to_thread(self._build)
            except HarnessError as exc:
                self._build_ok = False
                self._build_error = str(exc)
                raise
            self._build_ok = True
            self._build_error = None

    def _build(self) -> None:
        result = subprocess.run(
            ["make", "-C", str(self._settings.harness_dir), "host_harness"],
            capture_output=True,
            text=True,
            timeout=self._settings.engine_timeout_s * 4,
        )
        if result.returncode != 0:
            raise HarnessError(
                "Failed to build the native harness: "
                + (result.stderr.strip() or result.stdout.strip() or "unknown error")
            )

    # -- execution -----------------------------------------------------
    async def _run(self, argv: list[str], timeout: float | None = None) -> str:
        """Run the harness with a timeout, off the event loop."""
        await self.ensure_built()
        limit = timeout if timeout is not None else self._settings.engine_timeout_s

        def _invoke() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [str(self._settings.harness_path), *argv],
                capture_output=True,
                text=True,
                timeout=limit,
            )

        try:
            result = await asyncio.to_thread(_invoke)
        except subprocess.TimeoutExpired as exc:
            raise HarnessTimeout(
                f"The native harness did not finish within {limit:.0f}s."
            ) from exc
        if result.returncode != 0:
            raise HarnessError(
                (result.stderr.strip() or result.stdout.strip() or "harness failed")
                .strip()
            )
        return result.stdout.strip()

    async def infer(
        self,
        text: str,
        budget: int,
        profile: str,
        batt_mv: float,
        tstf: bool = False,
    ) -> dict[str, Any]:
        """Run a single inference and return the raw harness payload."""
        argv = [
            "--infer",
            text,
            "--budget",
            str(budget),
            "--profile",
            profile,
        ]
        if tstf:
            argv.append("--tstf")
        async with self._semaphore:
            stdout = await self._run(argv)
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"Harness returned invalid JSON: {exc}") from exc
        if "err" in payload:
            raise HarnessError(str(payload.get("err")))
        return payload

    async def golden_check(self, bench: int = 20) -> dict[str, Any]:
        """Run the golden-vector parity check."""
        argv = [str(self._settings.golden_path), "--bench", str(bench)]
        async with self._semaphore:
            stdout = await self._run(argv, timeout=self._settings.engine_timeout_s * 3)
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"Harness returned invalid JSON: {exc}") from exc

    async def serial(self, command: str) -> str:
        """Send one command to the mock UART and return the device output."""
        script = f"{command}\nQUIT\n"

        def _invoke() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [str(self._settings.harness_path), "--interactive"],
                input=script,
                capture_output=True,
                text=True,
                timeout=self._settings.engine_timeout_s,
            )

        try:
            await self.ensure_built()
            result = await asyncio.to_thread(_invoke)
        except subprocess.TimeoutExpired as exc:
            raise HarnessTimeout("The mock UART did not respond in time.") from exc
        lines = [
            line
            for line in result.stdout.splitlines()
            if line.strip() and not line.startswith(">RLM-AAGM")
        ]
        return "\n".join(lines)

    async def memory_report(self) -> dict[str, Any]:
        """Report the engine's static footprint (host compile-time constant)."""
        stdout = await self._run(["--memory"])
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"Harness returned invalid JSON: {exc}") from exc


class VerilogService:
    """Runs the RTL verification tiers and reports their provenance."""

    def __init__(self, settings: Settings, tools: ToolAvailability) -> None:
        self._settings = settings
        self._tools = tools

    @property
    def execution_source(self) -> ExecutionSource:
        if self._tools.rtl_simulation_possible:
            return ExecutionSource.RTL_SIMULATION
        return ExecutionSource.REFERENCE_SIMULATION

    @property
    def note(self) -> str:
        if self._tools.rtl_simulation_possible:
            return (
                "Icarus Verilog elaborated and simulated the repository's "
                "Verilog sources."
            )
        return (
            "Icarus Verilog is not installed, so the Python reference model "
            "of the control policy was used. This is NOT an RTL simulation."
        )

    async def check(self) -> dict[str, Any]:
        """Run `make check` in the Verilog directory and parse the result."""
        timeout = self._settings.verilog_timeout_s

        def _invoke() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["make", "-C", str(self._settings.verilog_dir), "check"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        try:
            result = await asyncio.to_thread(_invoke)
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "output": f"RTL check exceeded {timeout:.0f}s and was aborted.",
                "table": [],
            }
        combined = (result.stdout or "") + (result.stderr or "")
        passed = (
            "AAGM_VERILOG_PASS" in combined
            and "AAGM_COPROCESSOR_VERILOG_PASS" in combined
            and result.returncode == 0
        )
        status = "pass" if passed else "fail"
        if result.returncode != 0 and not passed:
            status = "fail"
        return {
            "status": status,
            "output": combined.strip(),
            "table": await self.waveform_table(),
        }

    async def waveform_table(self) -> list[dict[str, Any]]:
        """Parse the Icarus waveform, if one exists, into a table."""
        vcd = self._settings.verilog_dir / "aagm_coprocessor.vcd"
        if not vcd.exists():
            return []

        def _parse() -> list[dict[str, Any]]:
            sys.path.insert(0, str(self._settings.repo_root / "tools"))
            try:
                from vcd_table_viewer import parse_vcd_table  # type: ignore

                return parse_vcd_table(vcd)
            except Exception:
                return []

        return await asyncio.to_thread(_parse)
