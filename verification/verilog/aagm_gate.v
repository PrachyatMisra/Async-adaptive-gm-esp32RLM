// aagm_gate.v
// Single source of truth for the AAGM terminal-event policy.
//
// This module is a pure combinational function: it classifies the terminal
// event of a recursion step. It is deliberately stateless so that it can be
// instantiated both as a standalone reference policy and inside the
// register-backed arbiter coprocessor (aagm_arbiter_coprocessor.v), which
// guarantees that both agree on every decision.
//
// CONTROL CONTRACT (mirrors firmware/rlm_esp32/src/rlm_engine.h:RlmHaltReason)
//
//   Terminal events are mutually exclusive and are reported through
//   halt_reason, not through a single overloaded "halt" bit:
//
//     REASON_NONE   (2'b00) - no terminal event; the recursion continues.
//     REASON_ACT    (2'b01) - the ACT halting mass reached 1 - RLM_HALT_EPS
//                             (0.9). This is the ONLY successful outcome.
//     REASON_BUDGET (2'b10) - the recursion budget was exhausted before the
//                             ACT threshold was reached. The run terminated
//                             but did NOT succeed.
//     REASON_ABORT  (2'b11) - a cooperative abort was requested. Exclusive:
//                             the run did not complete, so no other reason is
//                             reported alongside it.
//
//   Note that budget exhaustion and abort are terminations, not successes.
//   The firmware reports them as RLM_HALT_BUDGET and RLM_HALT_ABORT and
//   never as RLM_HALT_ACT; this module encodes exactly that distinction.
//
// Fixed point: "unsigned Q1.15", i.e. value = raw / 32768, range [0, 1.99997).
module aagm_gate (
    input  wire        step_valid,         // a recursion step is being presented
    input  wire        min_steps_reached,  // k >= K_min has been reached
    input  wire        budget_exhausted,   // the recursion budget is spent
    input  wire        async_abort,        // cooperative abort requested
    input  wire [17:0] halting_mass_q15,   // cumulative ACT mass (unsigned Q3.15)
    output wire        halt,               // ACT halt: the successful outcome
    output wire        halt_budget,        // budget exhausted: terminated
    output wire        abort,              // abort requested
    output wire [1:0]  halt_reason         // REASON_* code
);

    // 0.9 in unsigned Q1.15, matching RLM_HALT_EPS = 0.1. The mass input is
    // 18 bits wide (unsigned Q3.15) so that a cumulative mass above 1.0 can
    // be represented without wrapping.
    localparam [17:0] THRESHOLD_Q15 = 18'd29491;

    localparam [1:0] REASON_NONE   = 2'b00;
    localparam [1:0] REASON_ACT    = 2'b01;
    localparam [1:0] REASON_BUDGET = 2'b10;
    localparam [1:0] REASON_ABORT  = 2'b11;

    // An abort request is a level that is honoured independently of whether a
    // step happens to be presented in the same cycle.
    assign abort = async_abort;

    // The ACT halt requires all of: a valid step, the minimum step count,
    // budget still available, no abort pending, and enough accumulated mass.
    assign halt = step_valid && min_steps_reached && !budget_exhausted &&
                  !async_abort && (halting_mass_q15 >= THRESHOLD_Q15);

    // Budget exhaustion terminates the run but is not a success, so it is
    // reported on its own output rather than through "halt".
    assign halt_budget = step_valid && budget_exhausted && !async_abort;

    assign halt_reason = async_abort ? REASON_ABORT :
                         halt          ? REASON_ACT   :
                         halt_budget   ? REASON_BUDGET : REASON_NONE;

endmodule
