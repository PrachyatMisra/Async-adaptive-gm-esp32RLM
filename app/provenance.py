"""Execution provenance.

Every number the API returns is tagged with where it came from. The tag is a
closed enum so that "paper reported", "reproduced locally", "synthetic" and
"measured on hardware" can never be confused with one another.

The rules the rest of the project relies on:

  * `host_native` is a real execution of the shipping C++ engine on the
    workstation. It is NOT an ESP32 measurement.
  * `reference_simulation` is a Python model of the RTL. It is NOT an RTL
    simulation and must never be reported as one.
  * `rtl_simulation` is Icarus Verilog elaborating the actual `.v` sources.
  * `wokwi_simulation` is the Wokwi web simulator.
  * `physical_device` is a measurement taken from real ESP32 hardware.
  * `paper_reported` is a figure copied from the manuscript. It was not
    reproduced here.
  * `trained_model` is an output of the training pipeline in `tools/`.
"""

from __future__ import annotations

from enum import Enum


class ExecutionSource(str, Enum):
    """Where a given result was produced."""

    PAPER_REPORTED = "paper_reported"
    TRAINED_MODEL = "trained_model"
    HOST_NATIVE = "host_native"
    REFERENCE_SIMULATION = "reference_simulation"
    RTL_SIMULATION = "rtl_simulation"
    WOKWI_SIMULATION = "wokwi_simulation"
    PHYSICAL_DEVICE = "physical_device"


#: Human-readable caveat appended to every response body.
SOURCE_NOTES: dict[ExecutionSource, str] = {
    ExecutionSource.PAPER_REPORTED: (
        "Figure reported in the project manuscript. Not reproduced by this "
        "service."
    ),
    ExecutionSource.TRAINED_MODEL: (
        "Produced by the local training pipeline in tools/. Reproducible "
        "only for a fixed corpus and seed on the same machine."
    ),
    ExecutionSource.HOST_NATIVE: (
        "Executed by the shipping C++ engine compiled natively on this "
        "workstation. Not an ESP32 measurement."
    ),
    ExecutionSource.REFERENCE_SIMULATION: (
        "Python reference model of the control policy. This is not an RTL "
        "simulation and is not hardware."
    ),
    ExecutionSource.RTL_SIMULATION: (
        "Icarus Verilog elaborated and simulated the repository's Verilog "
        "sources."
    ),
    ExecutionSource.WOKWI_SIMULATION: (
        "Wokwi web simulation of the ESP32 firmware."
    ),
    ExecutionSource.PHYSICAL_DEVICE: (
        "Measured on a physical ESP32 board."
    ),
}
