# RTL verification for the AAGM control policy.
#
# Two tiers are available and they are NOT equivalent:
#
#   rtl_simulation      Icarus Verilog (iverilog + vvp) elaborates and
#                       simulates the real .v sources. This is the only tier
#                       that may be reported as an RTL simulation.
#
#   reference_simulation
#                       The Python reference model, used only when Icarus
#                       Verilog is not installed. It re-implements the intent
#                       of the RTL in Python; it does not read, parse or
#                       execute any Verilog. It must never be reported as an
#                       RTL simulation.
#
# The reference tier writes its waveform to a separate file so that it can
# never overwrite a genuine Icarus VCD.
#
# A compile error or a failing testbench fails the build. The testbenches
# use $fatal, so vvp exits non-zero on a failed check.

IVERILOG ?= $(shell which iverilog 2>/dev/null)
VVP      ?= $(shell which vvp 2>/dev/null)
PYTHON   ?= python3

all: check

# `sim` is a compatibility alias for `check`. It does not produce anything
# that `check` does not already produce.
sim: check

check: check-gate check-coprocessor

# Cross-check: the Python reference model emits its expectations, and the
# real RTL is then replayed against them. A disagreement between the two
# independent implementations of the contract fails the build.
#
# This is only meaningful when Icarus Verilog is available; without it the
# reference model has nothing to be checked against.
crosscheck:
	@if [ -n "$(IVERILOG)" ] && [ -n "$(VVP)" ]; then \
		echo "== AAGM Gate | execution_source=rtl_simulation | cross-check against reference model"; \
		$(PYTHON) sim_aagm_gate.py --emit-vectors aagm_gate_vectors.txt && \
		$(IVERILOG) -g2012 -o aagm_gate.vvp aagm_gate.v tb_aagm_gate.v && \
		$(VVP) aagm_gate.vvp +vectors=aagm_gate_vectors.txt; \
	else \
		echo "SKIP: cross-check needs Icarus Verilog (iverilog + vvp)"; \
		echo "      The reference model alone cannot validate the RTL."; \
	fi

check-gate:
	@if [ -n "$(IVERILOG)" ] && [ -n "$(VVP)" ]; then \
		echo "== AAGM Gate | execution_source=rtl_simulation | simulator=icarus"; \
		$(IVERILOG) -g2012 -o aagm_gate.vvp aagm_gate.v tb_aagm_gate.v && \
		$(VVP) aagm_gate.vvp; \
	else \
		echo "== AAGM Gate | execution_source=reference_simulation | simulator=none (Icarus Verilog not found)"; \
		$(PYTHON) sim_aagm_gate.py; \
	fi

check-coprocessor:
	@if [ -n "$(IVERILOG)" ] && [ -n "$(VVP)" ]; then \
		echo "== AAGM Arbiter Coprocessor | execution_source=rtl_simulation | simulator=icarus"; \
		$(IVERILOG) -g2012 -o aagm_coprocessor.vvp aagm_arbiter_coprocessor.v aagm_gate.v tb_aagm_arbiter_coprocessor.v && \
		$(VVP) aagm_coprocessor.vvp; \
	else \
		echo "== AAGM Arbiter Coprocessor | execution_source=reference_simulation | simulator=none (Icarus Verilog not found)"; \
		$(PYTHON) sim_aagm_arbiter_coprocessor.py; \
	fi

clean:
	rm -f aagm_gate.vvp aagm_gate.vcd aagm_coprocessor.vvp aagm_coprocessor.vcd
	rm -f aagm_gate_ref.vcd aagm_coprocessor_ref.vcd aagm_gate_vectors.txt

.PHONY: all sim check crosscheck check-gate check-coprocessor clean
