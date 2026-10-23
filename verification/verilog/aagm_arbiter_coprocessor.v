// aagm_arbiter_coprocessor.v
// Hardware coprocessor for the Asynchronous Adaptive Gating Mechanism (AAGM).
//
// Implements an APB-style memory-mapped register interface, a hardware
// halting-mass accumulator, a calibrated shadow forecaster comparator, a
// dynamic recursion budget counter, and a one-shot asynchronous abort.
//
// The terminal-event policy is NOT re-implemented here. This module
// instantiates aagm_gate.v, which is the single source of truth for the
// contract described in that file. Consequently the two RTL modules cannot
// disagree about what a budget exhaustion or an abort means.
//
// CONTROL CONTRACT (see aagm_gate.v and
// firmware/rlm_esp32/src/rlm_engine.h:RlmHaltReason):
//
//   irq_halt    - ACT halt. The model chose to stop. The only success.
//   irq_budget  - The recursion budget was spent. The run terminated but did
//                 not succeed.
//   irq_abort   - A cooperative abort was requested. The run did not finish.
//   irq_reason  - Latched reason code: 0 none, 1 ACT, 2 budget, 3 abort.
//
// FIXED POINT
//
//   The mass accumulator is 18 bits wide in unsigned Q3.15 (value = raw/32768,
//   range 0..7.996) so that a cumulative mass above 1.0 - which is reachable
//   whenever the gate is small - cannot wrap. The accumulator saturates at
//   18'h3FFFF rather than wrapping. ADDR_MASS_ACC exposes the full 18-bit
//   value in bits [17:0]; bits [15:0] therefore still carry the Q1.15 value
//   for every mass below 2.0.
//
//   Gates are clamped to 1.0 exactly as the firmware does, so an
//   out-of-range gate contributes zero mass instead of underflowing the
//   accumulator. The occurrence is recorded in STATUS[5] / HALT_REASON[2].

`timescale 1ns / 1ps

module aagm_arbiter_coprocessor (
    input  wire        clk,
    input  wire        rst_n,

    // APB-style bus interface
    input  wire        p_sel,
    input  wire        p_enable,
    input  wire        p_write,
    input  wire [4:0]  p_addr,
    input  wire [31:0] p_wdata,
    output reg  [31:0] p_rdata,
    output wire        p_ready,

    // Hardware interrupt & control flags
    output wire        irq_halt,
    output wire        irq_budget,
    output wire        irq_abort,
    output wire [1:0]  irq_reason,
    output wire        speculative_prearm
);

    // Zero wait states: the slave is always able to complete a transfer.
    assign p_ready = 1'b1;

    // Register Map Offsets
    localparam ADDR_CTRL         = 5'h00; // [0]=start, [1]=soft_reset, [2]=async_abort, [3]=spec_en
    localparam ADDR_STATUS       = 5'h04; // [0]=busy, [1]=halted, [2]=aborted, [3]=prearmed,
                                         // [4]=budget_halted, [5]=gate_oob
    localparam ADDR_GATE_IN      = 5'h08; // [15:0]=g_k in Q1.15
    localparam ADDR_SHADOW_IN    = 5'h0C; // [15:0]=s_{k+1} in Q1.15
    localparam ADDR_MASS_ACC     = 5'h10; // [17:0]=cumulative halting mass in Q3.15
    localparam ADDR_BUDGET       = 5'h14; // [3:0]=max recursion budget (clamped to 1..8)
    localparam ADDR_STEPS        = 5'h18; // [3:0]=current step count
    localparam ADDR_TAU_LO       = 5'h1C; // [15:0]=shadow threshold (0.347 in Q1.15 = 11370)
    localparam ADDR_HALT_REASON  = 5'h1E; // [1:0]=reason, [2]=gate_oob

    // Golden Constants
    localparam [17:0] THRESHOLD_Q15 = 18'd29491; // 0.900 in Q1.15
    localparam [15:0] ONE_Q15       = 16'd32768; // 1.000 in Q1.15
    localparam [15:0] DEFAULT_TAU   = 16'd11370; // 0.347 in Q1.15
    localparam [17:0] MASS_MAX      = 18'h3FFFF; // accumulator saturation limit
    localparam [3:0]  MIN_STEPS     = 4'd2;      // RLM_MIN_RECURSION
    localparam [3:0]  MAX_BUDGET    = 4'd8;      // RLM_MAX_RECURSION

    // Internal Registers
    reg [3:0]  ctrl_reg;
    reg        busy_flag;
    reg        halt_flag;
    reg        budget_flag;
    reg        abort_flag;
    reg        prearm_flag;
    reg [15:0] gate_reg;
    reg [15:0] shadow_reg;
    reg [17:0] mass_acc_reg;
    reg [3:0]  budget_reg;
    reg [3:0]  steps_reg;
    reg [15:0] tau_lo_reg;
    reg [1:0]  reason_reg;
    reg        gate_oob_flag;

    // Combinational Outputs
    assign irq_halt   = halt_flag;
    assign irq_budget = budget_flag;
    assign irq_abort  = abort_flag;
    assign irq_reason = reason_reg;
    assign speculative_prearm = prearm_flag;

    // ---------------------------------------------------------------------
    // Current-step datapath
    // ---------------------------------------------------------------------
    wire gate_write = p_sel && p_enable && p_write && (p_addr == ADDR_GATE_IN);

    // The firmware clamps the gate to [0, 1] before use. The hardware does
    // the same, and records that the input was out of range.
    wire gate_in_range = (p_wdata[15:0] <= ONE_Q15);
    wire [15:0] gate_clamped = gate_in_range ? p_wdata[15:0] : ONE_Q15;
    wire [17:0] gate_delta = {2'b0, ONE_Q15} - {2'b0, gate_clamped};

    wire [17:0] mass_sum  = mass_acc_reg + gate_delta;
    wire [17:0] mass_next = (mass_sum > MASS_MAX) ? MASS_MAX : mass_sum;

    // A step is accepted only while the run is live.
    wire step_accept = gate_write && busy_flag && !halt_flag &&
                       !budget_flag && !abort_flag;

    wire min_steps_reached = (steps_reg + 1'b1 >= MIN_STEPS);
    wire budget_exhausted  = (steps_reg + 1'b1 >= {14'b0, budget_reg});
    wire async_abort       = ctrl_reg[2];

    // ---------------------------------------------------------------------
    // The policy itself: instantiated, never re-implemented.
    // ---------------------------------------------------------------------
    wire       halt_act;
    wire       halt_budget;
    wire       abort_now;
    wire [1:0] reason_now;

    aagm_gate u_policy (
        .step_valid       (step_accept),
        .min_steps_reached(min_steps_reached),
        .budget_exhausted (budget_exhausted),
        .async_abort      (async_abort),
        .halting_mass_q15 (mass_next),
        .halt             (halt_act),
        .halt_budget      (halt_budget),
        .abort            (abort_now),
        .halt_reason      (reason_now)
    );

    // Bus Read Operation. APB: the data phase is qualified by the access
    // phase, and reads are never qualified by p_write.
    always @(*) begin
        p_rdata = 32'b0;
        if (p_sel && p_enable && !p_write) begin
            case (p_addr)
                ADDR_CTRL:        p_rdata = {28'b0, ctrl_reg};
                ADDR_STATUS:      p_rdata = {26'b0, gate_oob_flag, budget_flag,
                                             prearm_flag, abort_flag, halt_flag, busy_flag};
                ADDR_GATE_IN:     p_rdata = {16'b0, gate_reg};
                ADDR_SHADOW_IN:   p_rdata = {16'b0, shadow_reg};
                ADDR_MASS_ACC:    p_rdata = {14'b0, mass_acc_reg};
                ADDR_BUDGET:      p_rdata = {28'b0, budget_reg};
                ADDR_STEPS:       p_rdata = {28'b0, steps_reg};
                ADDR_TAU_LO:      p_rdata = {16'b0, tau_lo_reg};
                ADDR_HALT_REASON: p_rdata = {29'b0, gate_oob_flag, 2'b00, reason_reg};
                default:          p_rdata = 32'b0;
            endcase
        end
    end

    // Bus Write & Sequential Coprocessor Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_reg     <= 4'b0;
            busy_flag    <= 1'b0;
            halt_flag    <= 1'b0;
            budget_flag  <= 1'b0;
            abort_flag   <= 1'b0;
            prearm_flag  <= 1'b0;
            gate_reg     <= 16'b0;
            shadow_reg   <= 16'b0;
            mass_acc_reg <= 18'b0;
            budget_reg   <= MAX_BUDGET;
            steps_reg    <= 4'b0;
            tau_lo_reg   <= DEFAULT_TAU;
            reason_reg   <= 2'b00;
            gate_oob_flag<= 1'b0;
        end else begin
            // A single if / else-if chain, so that no two branches ever
            // assign to the same register in the same cycle.
            //
            // 1. Soft reset: clears the flow-control state but leaves the
            //    configuration registers (budget, tau_lo) untouched.
            if (ctrl_reg[1]) begin
                busy_flag     <= 1'b0;
                halt_flag     <= 1'b0;
                budget_flag   <= 1'b0;
                abort_flag    <= 1'b0;
                prearm_flag   <= 1'b0;
                mass_acc_reg  <= 18'b0;
                steps_reg     <= 4'b0;
                reason_reg    <= 2'b00;
                gate_oob_flag <= 1'b0;
                ctrl_reg[1]   <= 1'b0;
            end
            // 2. One-shot asynchronous abort. The request level in
            //    ctrl_reg[2] is consumed here, so it does not re-assert on
            //    later cycles. An abort is exclusive: it clears the ACT and
            //    budget terminations because the run did not complete.
            else if (abort_now) begin
                abort_flag  <= 1'b1;
                halt_flag   <= 1'b0;
                budget_flag <= 1'b0;
                busy_flag   <= 1'b0;
                reason_reg  <= reason_now; // REASON_ABORT
                ctrl_reg[2] <= 1'b0;
            end
            // 3. Bus writes.
            else if (p_sel && p_enable && p_write) begin
                case (p_addr)
                    ADDR_CTRL: begin
                        ctrl_reg <= p_wdata[3:0];
                        if (p_wdata[0]) begin // Start new inference
                            busy_flag     <= 1'b1;
                            halt_flag     <= 1'b0;
                            budget_flag   <= 1'b0;
                            abort_flag    <= 1'b0;
                            prearm_flag   <= 1'b0;
                            mass_acc_reg  <= 18'b0;
                            steps_reg     <= 4'b0;
                            reason_reg    <= 2'b00;
                            gate_oob_flag <= 1'b0;
                        end
                    end
                    ADDR_GATE_IN: begin
                        gate_reg <= p_wdata[15:0];
                        if (step_accept) begin
                            steps_reg    <= steps_reg + 1'b1;
                            mass_acc_reg <= mass_next;
                            if (!gate_in_range) gate_oob_flag <= 1'b1;
                            if (halt_act) begin
                                halt_flag   <= 1'b1;
                                busy_flag   <= 1'b0;
                                reason_reg  <= reason_now; // REASON_ACT
                            end else if (halt_budget) begin
                                budget_flag <= 1'b1;
                                busy_flag   <= 1'b0;
                                reason_reg  <= reason_now; // REASON_BUDGET
                            end
                        end
                    end
                    ADDR_SHADOW_IN: begin
                        shadow_reg <= p_wdata[15:0];
                        // The speculative pre-arm hint is opt-in (spec_en).
                        if (ctrl_reg[3]) begin
                            prearm_flag <= (p_wdata[15:0] < tau_lo_reg);
                        end else begin
                            prearm_flag <= 1'b0;
                        end
                    end
                    ADDR_BUDGET: begin
                        // Clamped to 1..MAX_BUDGET, matching the firmware.
                        if (p_wdata[3:0] == 4'd0)          budget_reg <= 4'd1;
                        else if (p_wdata[3:0] > MAX_BUDGET) budget_reg <= MAX_BUDGET;
                        else                               budget_reg <= p_wdata[3:0];
                    end
                    ADDR_TAU_LO: begin
                        tau_lo_reg <= p_wdata[15:0];
                    end
                endcase
            end
        end
    end

endmodule
