#!/usr/bin/env python3
"""Reference model of the AAGM hardware arbiter coprocessor.

EXECUTION SOURCE: reference_simulation.

This is NOT an RTL simulation. It is a Python re-implementation of the
register-level behaviour of aagm_arbiter_coprocessor.v, used only as a
fallback when Icarus Verilog is not installed.

The model mirrors the RTL contract exactly:

  * the terminal-event policy is shared with sim_aagm_gate.py, so the two
    cannot disagree about what a budget exhaustion or an abort means;
  * the mass accumulator is 18 bits wide (unsigned Q3.15) and saturates
    rather than wrapping, so a cumulative mass above 1.0 is representable;
  * gates are clamped to 1.0, matching the firmware, and an out-of-range
    gate contributes zero mass instead of underflowing the accumulator;
  * the recursion budget is clamped to 1..8, matching RLM_MAX_RECURSION;
  * the abort request is a one-shot level and is exclusive of a pending
    halt;
  * the speculative pre-arm hint is opt-in through the spec_en bit.

The waveform is written to aagm_coprocessor_ref.vcd so that it can never
overwrite the genuine Icarus Verilog waveform.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_aagm_gate import classify  # noqa: E402  (shared policy, single source of truth)

# --- Constants (mirrors aagm_arbiter_coprocessor.v) -------------------------
ONE_Q15 = 32768
THRESHOLD_Q15 = 29491  # 0.900
TAU_LO_Q15 = 11370     # 0.347
MASS_MAX = (1 << 18) - 1
MIN_STEPS = 2
MAX_BUDGET = 8

ADDR_CTRL = 0x00
ADDR_STATUS = 0x04
ADDR_GATE_IN = 0x08
ADDR_SHADOW_IN = 0x0C
ADDR_MASS_ACC = 0x10
ADDR_BUDGET = 0x14
ADDR_STEPS = 0x18
ADDR_TAU_LO = 0x1C
ADDR_HALT_REASON = 0x1E


class CoprocessorModel:
    """Register-level model of aagm_arbiter_coprocessor.v."""

    def __init__(self) -> None:
        self.ctrl = 0
        self.busy = 0
        self.halt = 0
        self.budget_flag = 0
        self.abort = 0
        self.prearm = 0
        self.gate_reg = 0
        self.shadow_reg = 0
        self.mass = 0
        self.budget = MAX_BUDGET
        self.steps = 0
        self.tau_lo = TAU_LO_Q15
        self.reason = 0
        self.gate_oob = 0
        # Deferred abort, mirroring the one-cycle latency of the register
        # write that requests it.
        self._abort_pending = 0

    # -- bus ----------------------------------------------------------------
    def write(self, addr: int, data: int) -> None:
        data = data & 0xFFFFFFFF
        if addr == ADDR_CTRL:
            self.ctrl = data & 0xF
            if data & 0x1:  # start
                self.busy = 1
                self.halt = 0
                self.budget_flag = 0
                self.abort = 0
                self.prearm = 0
                self.mass = 0
                self.steps = 0
                self.reason = 0
                self.gate_oob = 0
        elif addr == ADDR_GATE_IN:
            self.gate_reg = data & 0xFFFF
            if self.busy and not self.halt and not self.budget_flag and not self.abort:
                gate = data & 0xFFFF
                in_range = gate <= ONE_Q15
                clamped = gate if in_range else ONE_Q15
                mass_next = min(self.mass + (ONE_Q15 - clamped), MASS_MAX)
                steps_next = self.steps + 1
                halt, halt_budget, _abort, reason = classify(
                    1,
                    1 if steps_next >= MIN_STEPS else 0,
                    1 if steps_next >= self.budget else 0,
                    0,
                    mass_next,
                )
                self.steps = steps_next
                self.mass = mass_next
                if not in_range:
                    self.gate_oob = 1
                if halt:
                    self.halt = 1
                    self.busy = 0
                    self.reason = reason
                elif halt_budget:
                    self.budget_flag = 1
                    self.busy = 0
                    self.reason = reason
        elif addr == ADDR_SHADOW_IN:
            self.shadow_reg = data & 0xFFFF
            if self.ctrl & 0x8:  # spec_en
                self.prearm = 1 if (data & 0xFFFF) < self.tau_lo else 0
            else:
                self.prearm = 0
        elif addr == ADDR_BUDGET:
            b = data & 0xF
            if b == 0:
                self.budget = 1
            elif b > MAX_BUDGET:
                self.budget = MAX_BUDGET
            else:
                self.budget = b
        elif addr == ADDR_TAU_LO:
            self.tau_lo = data & 0xFFFF

    def read(self, addr: int) -> int:
        if addr == ADDR_CTRL:
            return self.ctrl
        if addr == ADDR_STATUS:
            return (self.busy | (self.halt << 1) | (self.abort << 2) |
                    (self.prearm << 3) | (self.budget_flag << 4) |
                    (self.gate_oob << 5))
        if addr == ADDR_GATE_IN:
            return self.gate_reg
        if addr == ADDR_SHADOW_IN:
            return self.shadow_reg
        if addr == ADDR_MASS_ACC:
            return self.mass
        if addr == ADDR_BUDGET:
            return self.budget
        if addr == ADDR_STEPS:
            return self.steps
        if addr == ADDR_TAU_LO:
            return self.tau_lo
        if addr == ADDR_HALT_REASON:
            return (self.reason & 0x3) | (self.gate_oob << 2)
        return 0

    # -- clock --------------------------------------------------------------
    def tick(self) -> None:
        """Advance one clock cycle."""
        if self.ctrl & 0x2:  # soft reset
            self.busy = 0
            self.halt = 0
            self.budget_flag = 0
            self.abort = 0
            self.prearm = 0
            self.mass = 0
            self.steps = 0
            self.reason = 0
            self.gate_oob = 0
            self.ctrl &= ~0x2
            return
        if self.ctrl & 0x4:  # one-shot async abort, exclusive of a halt
            self.abort = 1
            self.halt = 0
            self.budget_flag = 0
            self.busy = 0
            self.reason = 3
            self.ctrl &= ~0x4


def write_vcd(vcd_path: Path, trace: list) -> None:
    with open(vcd_path, "w") as f:
        f.write("$date\n   generated by sim_aagm_arbiter_coprocessor.py (reference model)\n$end\n")
        f.write("$version\n   AAGM reference model (NOT a Verilog simulation)\n$end\n")
        f.write("$timescale\n   1ns\n$end\n")
        f.write("$scope module tb_aagm_arbiter_coprocessor $end\n")
        f.write("$var wire 1 ! busy $end\n")
        f.write("$var wire 4 \" steps [3:0] $end\n")
        f.write("$var wire 18 # mass_acc [17:0] $end\n")
        f.write("$var wire 1 $ halt_irq $end\n")
        f.write("$var wire 1 % budget_irq $end\n")
        f.write("$var wire 1 & abort_irq $end\n")
        f.write("$var wire 1 ( speculative_prearm $end\n")
        f.write("$var wire 2 ) halt_reason [1:0] $end\n")
        f.write("$upscope $end\n")
        f.write("$enddefinitions $end\n")
        for t, s in trace:
            f.write(f"#{t}\n")
            f.write(f"{s['busy']}!\n")
            f.write(f"b{s['steps']:04b} \"\n")
            f.write(f"b{s['mass_acc']:018b} #\n")
            f.write(f"{s['halt_irq']}$\n")
            f.write(f"{s['budget_irq']}%\n")
            f.write(f"{s['abort_irq']}&\n")
            f.write(f"{s['speculative_prearm']}(\n")
            f.write(f"b{s['halt_reason']:02b} )\n")


def simulate() -> int:
    print("=" * 72)
    print("   AAGM Hardware Arbiter Coprocessor - REFERENCE MODEL")
    print("   execution_source=reference_simulation (NOT an RTL simulation)")
    print("=" * 72)
    print("DUT: aagm_arbiter_coprocessor.v contract | APB-style register interface")
    print(f"Halt threshold: Q1.15 >= {THRESHOLD_Q15} (0.900) | tau_lo: {TAU_LO_Q15} (0.347)")
    print("-" * 72)

    failures = 0
    trace = []
    time_ns = 0
    dut = CoprocessorModel()

    def snapshot(label: str) -> None:
        nonlocal time_ns
        time_ns += 10
        trace.append((time_ns, {
            "busy": dut.busy,
            "steps": dut.steps,
            "mass_acc": dut.mass,
            "halt_irq": dut.halt,
            "budget_irq": dut.budget_flag,
            "abort_irq": dut.abort,
            "speculative_prearm": dut.prearm,
            "halt_reason": dut.reason,
        }))
        print(f"#{time_ns:<5} {label:<58} busy={dut.busy} steps={dut.steps} "
              f"mass={dut.mass} halt={dut.halt} budget={dut.budget_flag} "
              f"abort={dut.abort} reason={dut.reason}")

    def check(label: str, got, expected) -> None:
        nonlocal failures
        if got != expected:
            failures += 1
            print(f"  --> FAIL: {label}: got {got}, expected {expected}")

    # 1. Start an inference with the speculative pre-arm enabled.
    dut.write(ADDR_CTRL, 0x09)
    dut.tick()
    snapshot("CTRL <= 0x09 (START | SPEC_EN)")

    # 2. Step 1: g_1 = 0.6058 (19851), s_2 = 0.5466 (17911)
    dut.write(ADDR_SHADOW_IN, 17911)
    dut.tick()
    dut.write(ADDR_GATE_IN, 19851)
    dut.tick()
    snapshot("GATE_IN <= 19851 (g_1 = 0.6058)")
    check("no halt at step 1", dut.halt, 0)
    check("steps after step 1", dut.steps, 1)
    check("mass after step 1", dut.mass, 12917)

    # 3. Step 2: s_3 = 0.2800 (9175) < tau_lo -> pre-arm; then the gate hits
    #    the ACT threshold.
    dut.write(ADDR_SHADOW_IN, 9175)
    dut.tick()
    snapshot("SHADOW_IN <= 9175 (s_3 = 0.2800 < tau_lo)")
    check("prearm asserted", dut.prearm, 1)
    dut.write(ADDR_GATE_IN, 12330)
    dut.tick()
    snapshot("GATE_IN <= 12330 (g_2 = 0.3763)")
    check("ACT halt", dut.halt, 1)
    check("not a budget halt", dut.budget_flag, 0)
    check("reason is ACT", dut.reason, 1)
    check("mass after 2 steps", dut.mass, 33355)

    # 4. Budget exhaustion is a termination, not a success.
    dut.write(ADDR_BUDGET, 2)
    dut.write(ADDR_CTRL, 0x01)
    dut.tick()
    dut.write(ADDR_GATE_IN, 29491)
    dut.tick()
    dut.write(ADDR_GATE_IN, 29491)
    dut.tick()
    snapshot("BUDGET <= 2, two gates of 29491")
    check("budget: not an ACT halt", dut.halt, 0)
    check("budget: budget halt", dut.budget_flag, 1)
    check("budget: reason is BUDGET", dut.reason, 2)

    # 5. One-shot abort, exclusive of a pending halt.
    dut.write(ADDR_BUDGET, 8)
    dut.write(ADDR_CTRL, 0x01)
    dut.tick()
    dut.write(ADDR_GATE_IN, 19851)
    dut.tick()
    dut.write(ADDR_CTRL, 0x04)
    dut.tick()
    snapshot("CTRL <= 0x04 (ASYNC_ABORT)")
    check("abort irq", dut.abort, 1)
    check("abort clears busy", dut.busy, 0)
    check("abort reason", dut.reason, 3)
    check("abort request consumed", dut.ctrl & 0x4, 0)

    # 6. An out-of-range gate is clamped, not underflowed.
    dut.write(ADDR_CTRL, 0x01)
    dut.tick()
    dut.write(ADDR_GATE_IN, 40000)
    dut.tick()
    dut.write(ADDR_GATE_IN, 40000)
    dut.tick()
    snapshot("GATE_IN <= 40000 twice (g = 1.22, out of range)")
    check("oob: no halt", dut.halt, 0)
    check("oob: mass unchanged", dut.mass, 0)
    check("oob: flag raised", dut.gate_oob, 1)

    # 7. The accumulator does not wrap past 2.0.
    dut.write(ADDR_CTRL, 0x01)
    dut.tick()
    dut.write(ADDR_GATE_IN, 0)
    dut.tick()
    dut.write(ADDR_GATE_IN, 0)
    dut.tick()
    snapshot("GATE_IN <= 0 twice (mass would wrap a 16-bit accumulator)")
    check("wrap: ACT halt at step 2", dut.halt, 1)
    check("wrap: mass is 2.0, not 0", dut.mass, 65536)

    vcd_out = Path(__file__).resolve().parent / "aagm_coprocessor_ref.vcd"
    write_vcd(vcd_out, trace)
    print("-" * 72)
    print(f"Reference VCD -> {vcd_out.name}  (does not overwrite the Icarus waveform)")

    if failures == 0:
        print("AAGM_COPROCESSOR_REFERENCE_PASS")
        return 0
    print(f"AAGM_COPROCESSOR_REFERENCE_FAIL count={failures}")
    return 1


if __name__ == "__main__":
    sys.exit(simulate())
