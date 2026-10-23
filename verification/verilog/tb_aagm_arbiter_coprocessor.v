// tb_aagm_arbiter_coprocessor.v
// Testbench for the AAGM hardware arbiter coprocessor.
//
// The bus is driven on the falling edge and sampled on the rising edge, so
// there is no race between the testbench and the DUT.
//
// Coverage: reset values, budget clamping, register read-back, unmapped
// reads, the ACT halt, budget exhaustion, abort (standalone and concurrent
// with a pending halt), out-of-range gate clamping, accumulator overflow
// behaviour, soft reset, and the opt-in speculative pre-arm.

`timescale 1ns / 1ps

module tb_aagm_arbiter_coprocessor;
    reg         clk;
    reg         rst_n;
    reg         p_sel;
    reg         p_enable;
    reg         p_write;
    reg  [4:0]  p_addr;
    reg  [31:0] p_wdata;
    wire [31:0] p_rdata;
    wire        p_ready;
    wire        irq_halt;
    wire        irq_budget;
    wire        irq_abort;
    wire [1:0]  irq_reason;
    wire        speculative_prearm;

    integer failures;
    reg  [31:0] rd;

    localparam [4:0] ADDR_CTRL        = 5'h00;
    localparam [4:0] ADDR_STATUS      = 5'h04;
    localparam [4:0] ADDR_GATE_IN     = 5'h08;
    localparam [4:0] ADDR_SHADOW_IN   = 5'h0C;
    localparam [4:0] ADDR_MASS_ACC    = 5'h10;
    localparam [4:0] ADDR_BUDGET      = 5'h14;
    localparam [4:0] ADDR_STEPS       = 5'h18;
    localparam [4:0] ADDR_TAU_LO      = 5'h1C;
    localparam [4:0] ADDR_HALT_REASON = 5'h1E;

    aagm_arbiter_coprocessor dut (
        .clk(clk),
        .rst_n(rst_n),
        .p_sel(p_sel),
        .p_enable(p_enable),
        .p_write(p_write),
        .p_addr(p_addr),
        .p_wdata(p_wdata),
        .p_rdata(p_rdata),
        .p_ready(p_ready),
        .irq_halt(irq_halt),
        .irq_budget(irq_budget),
        .irq_abort(irq_abort),
        .irq_reason(irq_reason),
        .speculative_prearm(speculative_prearm)
    );

    always #5 clk = ~clk; // 100 MHz clock

    task apb_write;
        input [4:0]  addr;
        input [31:0] data;
        begin
            @(negedge clk);
            p_sel   = 1'b1;
            p_enable= 1'b0;
            p_write = 1'b1;
            p_addr  = addr;
            p_wdata = data;
            @(negedge clk);
            p_enable= 1'b1;
            @(negedge clk);
            p_sel   = 1'b0;
            p_enable= 1'b0;
            p_write = 1'b0;
        end
    endtask

    task apb_read;
        input  [4:0]  addr;
        output [31:0] data;
        begin
            @(negedge clk);
            p_sel   = 1'b1;
            p_enable= 1'b0;
            p_write = 1'b0;
            p_addr  = addr;
            @(negedge clk);
            p_enable= 1'b1;
            #1;
            data = p_rdata;
            @(negedge clk);
            p_sel   = 1'b0;
            p_enable= 1'b0;
        end
    endtask

    task check_eq;
        input [8*40-1:0] label;
        input [31:0]     got;
        input [31:0]     expected;
        begin
            if (got !== expected) begin
                $display("FAIL %0s: got=%0d (0x%08x), expected=%0d (0x%08x)",
                         label, got, got, expected, expected);
                failures = failures + 1;
            end
        end
    endtask

    task check_bit;
        input [8*40-1:0] label;
        input             got;
        input             expected;
        begin
            if (got !== expected) begin
                $display("FAIL %0s: got=%b, expected=%b", label, got, expected);
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("aagm_coprocessor.vcd");
        $dumpvars(0, tb_aagm_arbiter_coprocessor);
        failures = 0;
        clk = 0;
        rst_n = 0;
        p_sel = 0;
        p_enable = 0;
        p_write = 0;
        p_addr = 0;
        p_wdata = 0;

        #20;
        rst_n = 1;
        $display("[TIME %0tps] RST_N deasserted.", $time);
        #20;

        // ---- 1. Reset values -------------------------------------------------
        $display("[TIME %0tps] CHECK 1: reset register values", $time);
        apb_read(ADDR_BUDGET, rd);      check_eq("reset budget", rd, 32'd8);
        apb_read(ADDR_TAU_LO, rd);      check_eq("reset tau_lo", rd, 32'd11370);
        apb_read(ADDR_MASS_ACC, rd);    check_eq("reset mass", rd, 32'd0);
        apb_read(ADDR_STEPS, rd);       check_eq("reset steps", rd, 32'd0);
        apb_read(ADDR_HALT_REASON, rd); check_eq("reset reason", rd, 32'd0);
        apb_read(ADDR_STATUS, rd);      check_eq("reset status", rd, 32'd0);

        // ---- 2. Budget clamping ---------------------------------------------
        $display("[TIME %0tps] CHECK 2: budget clamped to 1..8", $time);
        apb_write(ADDR_BUDGET, 32'd0);
        apb_read(ADDR_BUDGET, rd);      check_eq("budget clamp low", rd, 32'd1);
        apb_write(ADDR_BUDGET, 32'd15);
        apb_read(ADDR_BUDGET, rd);      check_eq("budget clamp high", rd, 32'd8);
        apb_write(ADDR_BUDGET, 32'd8);

        // ---- 3. Bus decoding -------------------------------------------------
        $display("[TIME %0tps] CHECK 3: unmapped read returns 0, reads are qualified", $time);
        apb_read(5'h1F, rd);            check_eq("unmapped read", rd, 32'd0);
        @(negedge clk);
        p_sel = 1'b0; p_enable = 1'b0;
        #1;
        check_eq("unqualified read", p_rdata, 32'd0);

        // ---- 4. ACT halt on the mass threshold -------------------------------
        $display("[TIME %0tps] CHECK 4: ACT halt (g1=0.6058, g2=0.3763)", $time);
        apb_write(ADDR_CTRL, 32'h09);            // start + spec_en
        apb_write(ADDR_SHADOW_IN, 32'd17911);    // s_2 = 0.5466
        apb_write(ADDR_GATE_IN, 32'd19851);      // g_1 = 0.6058
        @(posedge clk); #1;
        check_bit("step 1 no halt", irq_halt, 1'b0);
        apb_read(ADDR_STEPS, rd);   check_eq("step 1 steps", rd, 32'd1);
        apb_read(ADDR_MASS_ACC, rd); check_eq("step 1 mass", rd, 32'd12917);
        apb_write(ADDR_SHADOW_IN, 32'd9175);     // s_3 = 0.2800 < tau_lo
        @(posedge clk); #1;
        check_bit("prearm asserted", speculative_prearm, 1'b1);
        apb_write(ADDR_GATE_IN, 32'd12330);      // g_2 = 0.3763
        @(posedge clk); #1;
        check_bit("ACT halt", irq_halt, 1'b1);
        check_bit("ACT: not a budget halt", irq_budget, 1'b0);
        check_eq("ACT reason", irq_reason, 2'd1);
        apb_read(ADDR_MASS_ACC, rd); check_eq("mass after 2 steps", rd, 32'd33355);
        apb_read(ADDR_STEPS, rd);    check_eq("steps after 2", rd, 32'd2);
        apb_read(ADDR_HALT_REASON, rd); check_eq("halt reason register", rd, 32'd1);

        // ---- 5. Budget exhaustion is a termination, not a halt ---------------
        $display("[TIME %0tps] CHECK 5: budget exhaustion", $time);
        apb_write(ADDR_BUDGET, 32'd2);
        apb_write(ADDR_CTRL, 32'h01);            // start
        apb_write(ADDR_GATE_IN, 32'd29491);      // g = 0.9 -> delta 3277
        @(posedge clk); #1;
        check_bit("budget: no halt at step 1", irq_halt, 1'b0);
        check_bit("budget: not exhausted at step 1", irq_budget, 1'b0);
        apb_write(ADDR_GATE_IN, 32'd29491);
        @(posedge clk); #1;
        check_bit("budget: NOT an ACT halt", irq_halt, 1'b0);
        check_bit("budget: budget halt", irq_budget, 1'b1);
        check_eq("budget reason", irq_reason, 2'd2);
        apb_read(ADDR_MASS_ACC, rd); check_eq("budget mass", rd, 32'd6554);
        apb_read(ADDR_STEPS, rd);    check_eq("budget steps", rd, 32'd2);

        // ---- 6. Abort during a live run ---------------------------------------
        $display("[TIME %0tps] CHECK 6: asynchronous abort", $time);
        apb_write(ADDR_BUDGET, 32'd8);
        apb_write(ADDR_CTRL, 32'h01);            // start
        apb_write(ADDR_GATE_IN, 32'd29491);
        @(posedge clk); #1;
        check_bit("abort: busy before abort", dut.busy_flag, 1'b1);
        apb_write(ADDR_CTRL, 32'h04);            // async_abort
        @(posedge clk); #1;
        check_bit("abort irq", irq_abort, 1'b1);
        check_bit("abort clears busy", dut.busy_flag, 1'b0);
        check_eq("abort reason", irq_reason, 2'd3);
        apb_read(ADDR_CTRL, rd); check_eq("abort request is one-shot", rd, 32'd0);

        // ---- 7. Abort is exclusive of a pending halt --------------------------
        $display("[TIME %0tps] CHECK 7: abort is exclusive of a pending ACT halt", $time);
        apb_write(ADDR_CTRL, 32'h01);            // start
        apb_write(ADDR_GATE_IN, 32'd19851);
        apb_write(ADDR_GATE_IN, 32'd12330);
        @(posedge clk); #1;
        check_bit("exclusive: halt before abort", irq_halt, 1'b1);
        apb_write(ADDR_CTRL, 32'h04);            // async_abort
        @(posedge clk); #1;
        check_bit("exclusive: abort irq", irq_abort, 1'b1);
        check_bit("exclusive: halt cleared", irq_halt, 1'b0);
        check_eq("exclusive: reason is abort", irq_reason, 2'd3);

        // ---- 8. Out-of-range gate is clamped, not underflowed ------------------
        $display("[TIME %0tps] CHECK 8: out-of-range gate is clamped", $time);
        apb_write(ADDR_BUDGET, 32'd8);
        apb_write(ADDR_CTRL, 32'h01);            // start
        apb_write(ADDR_GATE_IN, 32'd40000);      // g = 1.22 > 1.0
        @(posedge clk); #1;
        check_bit("oob: no halt", irq_halt, 1'b0);
        apb_write(ADDR_GATE_IN, 32'd40000);
        @(posedge clk); #1;
        check_bit("oob: still no halt", irq_halt, 1'b0);
        check_bit("oob: no budget halt", irq_budget, 1'b0);
        apb_read(ADDR_MASS_ACC, rd); check_eq("oob: mass unchanged", rd, 32'd0);
        apb_read(ADDR_STATUS, rd);    check_eq("oob: status flag", rd[5], 1'b1);

        // ---- 9. The mass accumulator does not wrap ----------------------------
        $display("[TIME %0tps] CHECK 9: mass accumulator does not wrap past 2.0", $time);
        apb_write(ADDR_CTRL, 32'h01);            // start
        apb_write(ADDR_GATE_IN, 32'd0);          // g = 0 -> delta 32768
        @(posedge clk); #1;
        check_bit("wrap: no halt at step 1", irq_halt, 1'b0);
        apb_read(ADDR_MASS_ACC, rd); check_eq("wrap: mass after 1 step", rd, 32'd32768);
        apb_write(ADDR_GATE_IN, 32'd0);
        @(posedge clk); #1;
        check_bit("wrap: ACT halt at step 2", irq_halt, 1'b1);
        apb_read(ADDR_MASS_ACC, rd); check_eq("wrap: mass after 2 steps", rd, 32'd65536);

        // ---- 10. Soft reset clears flow state, keeps configuration --------------
        $display("[TIME %0tps] CHECK 10: soft reset", $time);
        apb_write(ADDR_BUDGET, 32'd4);
        apb_write(ADDR_TAU_LO, 32'd5000);
        apb_write(ADDR_CTRL, 32'h01);            // start
        apb_write(ADDR_GATE_IN, 32'd19851);
        @(posedge clk); #1;
        apb_write(ADDR_CTRL, 32'h02);            // soft_reset
        @(posedge clk); #1;
        apb_read(ADDR_STEPS, rd);     check_eq("soft reset steps", rd, 32'd0);
        apb_read(ADDR_MASS_ACC, rd);  check_eq("soft reset mass", rd, 32'd0);
        apb_read(ADDR_STATUS, rd);    check_eq("soft reset busy", rd[0], 1'b0);
        apb_read(ADDR_BUDGET, rd);    check_eq("soft reset keeps budget", rd, 32'd4);
        apb_read(ADDR_TAU_LO, rd);    check_eq("soft reset keeps tau_lo", rd, 32'd5000);

        // ---- 11. Speculative pre-arm is opt-in --------------------------------
        $display("[TIME %0tps] CHECK 11: speculative pre-arm requires spec_en", $time);
        apb_write(ADDR_CTRL, 32'h01);            // start, spec_en = 0
        apb_write(ADDR_SHADOW_IN, 32'd9175);     // s = 0.28 < tau_lo
        @(posedge clk); #1;
        check_bit("prearm disabled", speculative_prearm, 1'b0);
        apb_write(ADDR_CTRL, 32'h09);            // start + spec_en
        apb_write(ADDR_SHADOW_IN, 32'd17911);    // s = 0.5466 > tau_lo
        @(posedge clk); #1;
        check_bit("prearm above threshold", speculative_prearm, 1'b0);

        #20;
        if (failures == 0) $display(">>> AAGM_COPROCESSOR_VERILOG_PASS (All hardware checks verified)");
        else $display(">>> AAGM_COPROCESSOR_VERILOG_FAIL count=%0d", failures);
        if (failures != 0) $fatal(1, "AAGM coprocessor Verilog checks failed");
        $finish;
    end

endmodule
