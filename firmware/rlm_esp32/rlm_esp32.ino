/*
 * rlm_esp32.ino - Recursive Language Model with Asynchronous Adaptive
 * Gating Mechanism (AAGM) on ESP32 (Arduino core).
 *
 * Hardware-level AAGM (dual-core FreeRTOS):
 *   Core 1 (APP_CPU) - compute task: tokenizer -> recursive block steps ->
 *                      head. Pure math, never handles I/O. Runs INFER/CHAT/BENCH
 *                      jobs taken from a request mailbox.
 *   Core 0 (PRO_CPU) - AAGM arbiter task: evaluates ALL gates (authoritative
 *                      gate + shadow forecaster) offloaded via a zero-copy
 *                      mailbox; samples the battery and compensates the
 *                      internal-resistance droop; applies the DVFS energy
 *                      policy (PERF 240MHz/B8, BAL 160MHz/B6, ECO 80MHz/B4);
 *                      and parses the async serial commands. A mid-inference
 *                      budget shrink or a brownout warning raises a
 *                      cooperative abort checkpoint.
 *
 * CROSS-CORE SYNCHRONISATION
 *   Every cross-core hand-off uses a FreeRTOS primitive, never a bare
 *   `volatile`:
 *     * g_req_q    - gate/shadow requests, compute -> arbiter.
 *     * g_shadow_q - shadow results, arbiter -> compute.
 *     * g_gate_q   - gate results, arbiter -> compute.
 *     * g_cmd_q    - serial jobs, arbiter -> compute.
 *     * g_abort_mux - a spinlock protecting the abort sequence counter.
 *   A FreeRTOS queue also provides the memory barrier that makes the
 *   zero-copy `pooled` pointer safe: the arbiter observes the updated
 *   activation vector because the queue publish/consume orders the writes.
 *
 * BATTERY MODEL
 *   g_batt_mv is always the MEASURED terminal voltage under the present
 *   load: what the divider on GPIO34 reads, or the value set with
 *   `BATT <mv>`. The open-circuit voltage is *estimated* from it by adding
 *   back the droop I*R_int, and the state-of-charge policy uses that
 *   estimate. The droop is never subtracted twice.
 *
 * Serial protocol (115200 baud, newline-terminated ASCII):
 *   CHAT <text>                 -> conversational RLM response with reasoning
 *   INFER <text>                -> raw JSON telemetry line
 *   MODE PERF|BAL|ECO|AUTO      (AUTO derives the profile from the battery)
 *   BATT <millivolts>|AUTO      (AUTO: GPIO34 100k/100k divider x2)
 *   BENCH <n>                   -> n self-timing runs of a fixed review
 *   ABORT                       -> request a cooperative abort of the running job
 *   TSTF <0|1>                  -> enable/disable the experimental TSTF path
 *   STAT                        -> heap/freq/board state
 *   PING                        -> {"pong":1,...}
 */
#include <Arduino.h>
#include <string.h>
#include "src/rlm_engine.h"

/* ---------------- energy context / profiles --------------------------- */
enum Profile : uint8_t { PROF_PERF = 0, PROF_BAL = 1, PROF_ECO = 2 };
static const uint8_t  PROF_BUDGET[3] = { 8, 6, 4 };
static const uint32_t PROF_MHZ[3]    = { 240, 160, 80 };
static const char*    PROF_NAME[3]   = { "PERF", "BAL", "ECO" };

/* Where the current battery reading came from. */
enum BattSource : uint8_t { BATT_SRC_SIM = 0, BATT_SRC_ADC = 1, BATT_SRC_ADC_FALLBACK = 2 };
static const char* BATT_SRC_NAME[3] = { "sim", "adc34", "adc34-unavailable" };

static volatile uint8_t  g_profile   = PROF_PERF;
static volatile bool     g_auto_mode = false;
static volatile uint8_t  g_budget    = RLM_MAX_RECURSION;
static volatile uint8_t  g_tstf      = 0;   /* experimental, off by default */

static volatile bool     g_batt_auto = false;
static float             g_batt_mv   = 4000.0f; /* measured terminal voltage */
static volatile uint8_t  g_batt_src  = BATT_SRC_SIM;
static const int         BATT_ADC_PIN = 34;     /* 100k/100k divider */

/* Battery internal resistance droop compensation (R_int = 200 mOhm) */
static const float BATT_R_INT = 0.200f;

/* ---------------- cross-core abort flag ------------------------------- */
/* A bare `volatile` is not a cross-core synchronisation primitive: it gives
 * no ordering guarantee and no atomicity for a read-modify-write. The
 * abort is therefore a monotonically increasing sequence counter guarded by
 * a spinlock. The compute core snapshots it when a job starts and aborts
 * when it changes, which avoids a lost-abort race entirely. */
static portMUX_TYPE   g_abort_mux = portMUX_INITIALIZER_UNLOCKED;
static volatile uint32_t g_abort_seq = 0;

static uint32_t abort_seq_snapshot(void) {
    portENTER_CRITICAL(&g_abort_mux);
    uint32_t v = g_abort_seq;
    portEXIT_CRITICAL(&g_abort_mux);
    return v;
}
static void request_abort(void) {
    portENTER_CRITICAL(&g_abort_mux);
    g_abort_seq++;
    portEXIT_CRITICAL(&g_abort_mux);
}

/* ---------------- async gate mailbox (core 1 -> core 0) ---------------- */
enum GateKind : uint8_t { GATE_KIND_SHADOW = 0, GATE_KIND_GATE = 1 };

typedef struct {
    const float* pooled;   /* zero-copy into the engine's workspace */
    uint8_t      kind;
} GateReq;

typedef struct { float value; } GateRsp;

static QueueHandle_t g_req_q    = nullptr;
static QueueHandle_t g_shadow_q = nullptr;
static QueueHandle_t g_gate_q   = nullptr;

static TaskHandle_t g_compute_task = nullptr;
static TaskHandle_t g_arbiter_task = nullptr;

static float g_shadow_val = 0.0f;
static float g_gate_val   = 0.0f;

/* The abort sequence observed when the current job started. */
static uint32_t g_job_abort_base = 0;

/* gate-provider callbacks, executed on the compute core */
static void fw_post(const float* pooled, uint8_t kind) {
    GateReq req = { pooled, kind };
    xQueueSend(g_req_q, &req, portMAX_DELAY);
}
static void fw_prime(void*, const float* pooled_embed) {
    fw_post(pooled_embed, GATE_KIND_SHADOW);
    GateRsp rsp;
    xQueueReceive(g_shadow_q, &rsp, portMAX_DELAY);
    g_shadow_val = rsp.value;
}
static void fw_launch(void*, const float* pooled, uint8_t) {
    fw_post(pooled, GATE_KIND_GATE);
}
static float fw_wait_gate(void*) {
    GateRsp rsp;
    xQueueReceive(g_gate_q, &rsp, portMAX_DELAY);
    g_gate_val = rsp.value;
    return g_gate_val;
}
static float fw_wait_shadow(void*) { return g_shadow_val; }
static int   fw_should_abort(void*) { return abort_seq_snapshot() != g_job_abort_base; }

static RlmGateProvider FW_PROVIDER = {
    fw_prime, fw_launch, fw_wait_gate, fw_wait_shadow, fw_should_abort, nullptr
};

/* ---------------- battery policy with droop compensation --------------- */
/* Load current drawn by the SoC, as a function of the CPU frequency. These
 * are board-level estimates, not measurements: see docs/MEASUREMENT_PROTOCOL.md. */
static float load_current_ma(uint32_t cpu_mhz) {
    return (cpu_mhz >= 240) ? 50.0f : ((cpu_mhz >= 160) ? 36.0f : 27.0f);
}
/* Voltage drop across the battery's internal resistance. */
static float droop_mv(uint32_t cpu_mhz) {
    return (load_current_ma(cpu_mhz) / 1000.0f) * BATT_R_INT * 1000.0f;
}
/* Estimate the open-circuit voltage from the measured terminal voltage. */
static float ocv_from_measured(float measured_mv, uint32_t cpu_mhz) {
    return measured_mv + droop_mv(cpu_mhz);
}

static float read_batt_mv(void) {
    if (!g_batt_auto) return g_batt_mv;
    uint32_t mv = analogReadMilliVolts(BATT_ADC_PIN);
    if (mv < 200) {
        /* No divider fitted / pin floating. Report the simulated value and
         * say so, rather than pretending to have measured the battery. */
        g_batt_src = BATT_SRC_ADC_FALLBACK;
        return 4000.0f;
    }
    g_batt_src = BATT_SRC_ADC;
    return mv * 2.0f;                           /* 100k/100k divider */
}

static void apply_profile(uint8_t prof) {
    uint8_t new_budget = PROF_BUDGET[prof];
    if (new_budget < g_budget) request_abort();   /* shrink -> async abort */
    g_profile = prof;
    g_budget  = new_budget;
    setCpuFrequencyMhz(PROF_MHZ[prof]);
}

static uint8_t profile_for_ocv(float ocv_mv) {
    float pct = constrain((ocv_mv - 3300.0f) / 900.0f * 100.0f, 0.0f, 100.0f);
    if (pct < 25.0f) return PROF_ECO;
    if (pct < 50.0f) return PROF_BAL;
    return PROF_PERF;
}

static void energy_tick(void) {                    /* runs on the arbiter core */
    float measured = read_batt_mv();
    g_batt_mv = measured;
    float ocv = ocv_from_measured(measured, PROF_MHZ[g_profile]);

    /* Brownout protection: the chip sees the MEASURED terminal voltage, so
     * that is what is compared against the cutoff. */
    if (measured < 3350.0f) {
        apply_profile(PROF_ECO);
        request_abort();                 /* leave the run before a brownout */
    } else if (g_auto_mode) {
        apply_profile(profile_for_ocv(ocv));
    }
}

/* ---------------- job mailbox (arbiter -> compute core) ---------------- */
enum CmdType : uint8_t { CMD_INFER = 0, CMD_CHAT = 1, CMD_BENCH = 2, CMD_STAT = 3 };

#define REQ_TEXT_MAX 512

typedef struct {
    uint8_t type;
    int32_t value;
    char    text[REQ_TEXT_MAX];
} CmdJob;

static QueueHandle_t g_cmd_q = nullptr;
static SemaphoreHandle_t g_serial_mtx;

/* ---------------- JSON helpers (compute + arbiter cores) -------------- */
static void json_lock(void)   { xSemaphoreTake(g_serial_mtx, portMAX_DELAY); }
static void json_unlock(void) { xSemaphoreGive(g_serial_mtx); }

/* No command echoes the operator's own text into the telemetry, so there
 * is no free-form text to escape. If a future field ever carries
 * operator-supplied text it MUST be escaped (quotes, backslashes and
 * control characters) before it is interpolated into a JSON line; raw
 * interpolation of a serial line is a JSON-injection bug. */

static const char* halt_reason_name(uint8_t reason) {
    switch (reason) {
        case RLM_HALT_ACT:    return "act_halt";
        case RLM_HALT_BUDGET: return "budget_exhausted";
        case RLM_HALT_ABORT:  return "aborted";
        default:              return "none";
    }
}

/* ---------------- inference workspace --------------------------------- */
/* The workspace is ~136 KiB, so it must live in static DRAM, never on the
 * stack. It is owned exclusively by the compute core. */
static RlmWorkspace g_ws;

/* ---------------- inference (compute core) ---------------------------- */
static void do_infer(const char* text) {
    uint16_t ids[RLM_MAX_TOKENS];
    uint8_t n_real = rlm_tokenize(text, ids);
    if (n_real == 0) {
        json_lock();
        Serial.println("{\"err\":\"empty\"}");
        json_unlock();
        return;
    }
    g_job_abort_base = abort_seq_snapshot();
    uint32_t t0 = micros();
    RlmResult r;
    rlm_engine_infer(ids, n_real, g_budget, &FW_PROVIDER, &g_ws, &r);
    uint32_t us = micros() - t0;

    float measured = g_batt_mv;
    float droop    = droop_mv(PROF_MHZ[g_profile]);
    float ocv      = ocv_from_measured(measured, PROF_MHZ[g_profile]);

    json_lock();
    Serial.printf("{\"pred\":%u,\"logits\":[%.4f,%.4f],\"steps\":%u,"
                  "\"depth\":%.4f,\"mass\":%.4f,\"hints\":%u,"
                  "\"prearm_confirmed\":%u,\"aborted\":%u,\"halt_reason\":\"%s\","
                  "\"tokens\":%u,\"frozen\":%u,\"tstf\":%u,\"macs_saved\":%lu,"
                  "\"gates\":[",
                  r.pred, r.logits[0], r.logits[1],
                  r.trace.steps, r.trace.depth, r.trace.halting_mass,
                  r.trace.prearm_hints, r.trace.prearm_confirmed, r.trace.aborted,
                  halt_reason_name(r.trace.halt_reason), n_real,
                  r.trace.frozen_tokens, r.trace.tstf_enabled,
                  (unsigned long)r.trace.ffn_macs_saved);
    for (uint8_t i = 0; i < r.trace.steps; ++i)
        Serial.printf(i ? ",%.4f" : "%.4f", r.trace.gates[i]);
    Serial.print("],\"shadows\":[");
    for (uint8_t i = 0; i < r.trace.steps; ++i)
        Serial.printf(i ? ",%.4f" : "%.4f", r.trace.shadows[i]);
    Serial.printf("],\"us\":%lu,\"budget\":%u,\"profile\":\"%s\","
                  "\"batt_mv\":%.0f,\"ocv_mv\":%.0f,\"droop_mv\":%.1f,"
                  "\"batt_src\":\"%s\",\"cpu_mhz\":%lu,\"heap\":%u}\n",
                  (unsigned long)us, g_budget, PROF_NAME[g_profile],
                  (double)measured, (double)ocv, (double)droop,
                  BATT_SRC_NAME[g_batt_src],
                  (unsigned long)getCpuFrequencyMhz(), ESP.getFreeHeap());
    json_unlock();
}

static void do_chat(const char* text) {
    uint16_t ids[RLM_MAX_TOKENS];
    uint8_t n_real = rlm_tokenize(text, ids);
    if (n_real == 0) {
        json_lock();
        Serial.println("{\"reply\":\"Please provide text for me to analyze.\"}");
        json_unlock();
        return;
    }
    g_job_abort_base = abort_seq_snapshot();
    uint32_t t0 = micros();
    RlmResult r;
    rlm_engine_infer(ids, n_real, g_budget, &FW_PROVIDER, &g_ws, &r);
    uint32_t us = micros() - t0;

    float max_l = max(r.logits[0], r.logits[1]);
    float e0 = expf(r.logits[0] - max_l);
    float e1 = expf(r.logits[1] - max_l);
    float conf = (r.pred == 1 ? e1 : e0) / (e0 + e1) * 100.0f;
    const char* verdict = r.pred == 1 ? "POSITIVE" : "NEGATIVE";

    /* The compute saving is only meaningful when the run terminated on the
     * ACT threshold. For a budget exhaustion or an abort there was no
     * saving, and saying so would be dishonest. */
    uint8_t saved = (r.trace.halt_reason == RLM_HALT_ACT && g_budget > r.trace.steps)
                        ? (uint8_t)(g_budget - r.trace.steps) : 0;
    float saved_pct = g_budget ? (float)saved / (float)g_budget * 100.0f : 0.0f;

    /* The user's own text is NOT echoed into the telemetry: it may be a
     * review or other private content. */
    json_lock();
    Serial.print("{\"chat_reply\":\"I evaluated your input. Verdict: ");
    Serial.print(verdict);
    Serial.printf(" (Confidence: %.1f%%). Reasoned in %u recursive steps "
                  "(Mass: %.3f/0.900). ", conf, r.trace.steps, r.trace.halting_mass);
    if (r.trace.halt_reason == RLM_HALT_ACT)
        Serial.printf("Saved %u steps (%.0f%% compute reduction). ", saved, saved_pct);
    else if (r.trace.halt_reason == RLM_HALT_BUDGET)
        Serial.print("The recursion budget was exhausted before the model "
                     "converged, so this result is not a converged prediction. ");
    else
        Serial.print("The run was aborted, so this result is incomplete. ");
    Serial.printf("Latency: %lu us.\",", (unsigned long)us);
    Serial.printf("\"pred\":%u,\"verdict\":\"%s\",\"confidence\":%.1f,"
                  "\"steps\":%u,\"saved_steps\":%u,\"halt_reason\":\"%s\","
                  "\"tokens\":%u,\"us\":%lu,\"profile\":\"%s\"}\n",
                  r.pred, verdict, conf, r.trace.steps, saved,
                  halt_reason_name(r.trace.halt_reason), n_real,
                  (unsigned long)us, PROF_NAME[g_profile]);
    json_unlock();
}

static void do_bench(int n) {
    const char* text = "the first act is clumsy and uneven meanwhile the cast "
                       "shines, yet the final scene feels magnificent";
    uint16_t ids[RLM_MAX_TOKENS];
    uint8_t n_real = rlm_tokenize(text, ids);
    uint32_t best = UINT32_MAX, worst = 0, steps_sum = 0;
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        g_job_abort_base = abort_seq_snapshot();
        uint32_t t0 = micros();
        RlmResult r;
        rlm_engine_infer(ids, n_real, g_budget, &FW_PROVIDER, &g_ws, &r);
        uint32_t us = micros() - t0;
        best = min(best, us); worst = max(worst, us);
        sum += us; steps_sum += r.trace.steps;
    }
    json_lock();
    Serial.printf("{\"bench_n\":%d,\"us_avg\":%.1f,\"us_min\":%lu,\"us_max\":%lu,"
                  "\"mean_steps\":%.3f,\"budget\":%u,\"cpu_mhz\":%lu,"
                  "\"profile\":\"%s\",\"heap\":%u}\n",
                  n, sum / n, (unsigned long)best, (unsigned long)worst,
                  (double)steps_sum / n, g_budget,
                  (unsigned long)getCpuFrequencyMhz(), PROF_NAME[g_profile],
                  ESP.getFreeHeap());
    json_unlock();
}

static void do_stat() {
    float measured = g_batt_mv;
    float ocv = ocv_from_measured(measured, PROF_MHZ[g_profile]);
    json_lock();
    Serial.printf("{\"board\":\"%s\",\"cpu_mhz\":%lu,\"budget\":%u,"
                  "\"profile\":\"%s\",\"auto_mode\":%s,\"batt_mv\":%.0f,"
                  "\"ocv_mv\":%.0f,\"batt_src\":\"%s\",\"heap\":%u,"
                  "\"min_heap\":%u,\"sketch_kb\":%u,\"flash_mb\":%u,"
                  "\"max_tokens\":%u,\"tstf\":%u}\n",
                  ARDUINO_VARIANT, (unsigned long)getCpuFrequencyMhz(), g_budget,
                  PROF_NAME[g_profile], g_auto_mode ? "true" : "false",
                  (double)measured, (double)ocv, BATT_SRC_NAME[g_batt_src],
                  ESP.getFreeHeap(), ESP.getMinFreeHeap(),
                  ESP.getSketchSize() / 1024,
                  ESP.getFlashChipSize() / (1024 * 1024),
                  RLM_MAX_TOKENS, g_tstf);
    json_unlock();
}

/* ---------------- command handling (arbiter core) ---------------------- */
static void enqueue_cmd(const CmdJob& job) {
    if (xQueueSend(g_cmd_q, &job, 0) != pdTRUE) {
        json_lock();
        Serial.println("{\"err\":\"busy\"}");
        json_unlock();
    }
}

static void handle_line(char* line) {
    if (!strncmp(line, "CHAT ", 5) || !strncmp(line, "INFER ", 6)) {
        bool is_chat = (line[0] == 'C');
        const char* body = is_chat ? line + 5 : line + 6;
        CmdJob job;
        memset(&job, 0, sizeof(job));
        job.type = is_chat ? CMD_CHAT : CMD_INFER;
        strncpy(job.text, body, REQ_TEXT_MAX - 1);
        job.text[REQ_TEXT_MAX - 1] = 0;
        enqueue_cmd(job);
    }
    else if (!strncmp(line, "BENCH ", 6)) {
        int n = atoi(line + 6);
        if (n < 1) n = 1;
        if (n > 1000) n = 1000;
        CmdJob job;
        memset(&job, 0, sizeof(job));
        job.type = CMD_BENCH;
        job.value = n;
        enqueue_cmd(job);
    }
    else if (!strcmp(line, "STAT")) {
        CmdJob job;
        memset(&job, 0, sizeof(job));
        job.type = CMD_STAT;
        enqueue_cmd(job);
    }
    else if (!strcmp(line, "ABORT")) {
        request_abort();
        json_lock();
        Serial.println("{\"abort_requested\":1}");
        json_unlock();
    }
    else if (!strcmp(line, "PING")) {
        json_lock();
        Serial.printf("{\"pong\":1,\"uptime_ms\":%lu}\n", (unsigned long)millis());
        json_unlock();
    }
    else if (!strncmp(line, "MODE ", 5)) {
        const char* m = line + 5;
        json_lock();
        if      (!strcmp(m, "PERF")) { g_auto_mode = false; apply_profile(PROF_PERF); }
        else if (!strcmp(m, "BAL"))  { g_auto_mode = false; apply_profile(PROF_BAL); }
        else if (!strcmp(m, "ECO"))  { g_auto_mode = false; apply_profile(PROF_ECO); }
        else if (!strcmp(m, "AUTO")) { g_auto_mode = true; energy_tick(); }
        else { Serial.println("{\"err\":\"bad mode\"}"); json_unlock(); return; }
        Serial.printf("{\"mode\":\"%s\",\"budget\":%u,\"cpu_mhz\":%lu}\n",
                      PROF_NAME[g_profile], g_budget,
                      (unsigned long)getCpuFrequencyMhz());
        json_unlock();
    }
    else if (!strncmp(line, "BATT ", 5)) {
        const char* b = line + 5;
        if (!strcmp(b, "AUTO")) {
            g_batt_auto = true;
        } else {
            float mv = atof(b);
            if (mv < 2000.0f || mv > 5000.0f) {
                json_lock();
                Serial.println("{\"err\":\"batt out of range (2000..5000 mV)\"}");
                json_unlock();
                return;
            }
            g_batt_auto = false;
            g_batt_mv = mv;
            g_batt_src = BATT_SRC_SIM;
        }
        energy_tick();
        json_lock();
        Serial.printf("{\"batt_mv\":%.0f,\"batt_src\":\"%s\",\"profile\":\"%s\","
                      "\"budget\":%u}\n", (double)g_batt_mv, BATT_SRC_NAME[g_batt_src],
                      PROF_NAME[g_profile], g_budget);
        json_unlock();
    }
    else if (!strncmp(line, "TSTF ", 5)) {
        int on = atoi(line + 5);
        g_tstf = (on != 0) ? 1 : 0;
        json_lock();
        Serial.printf("{\"tstf\":%u,\"note\":\"experimental, opt-in\"}\n", g_tstf);
        json_unlock();
    }
    else {
        json_lock();
        Serial.println("{\"err\":\"unknown cmd\"}");
        json_unlock();
    }
}

/* ---------------- arbiter task (core 0) -------------------------------- */
static void arbiter_loop(void*) {
    char line[REQ_TEXT_MAX + 1];
    size_t len = 0;
    bool overflow = false;
    uint32_t next_energy = millis() + 500;

    for (;;) {
        /* 1. Serve gate/shadow requests from the compute core. The receive
         *    blocks for up to 5 ms so the arbiter does not spin at 100%
         *    CPU while still being responsive to the console. */
        GateReq req;
        if (xQueueReceive(g_req_q, &req, pdMS_TO_TICKS(5)) == pdTRUE) {
            GateRsp rsp;
            if (req.kind == GATE_KIND_GATE) {
                rsp.value = rlm_gate_eval(req.pooled);
                xQueueSend(g_gate_q, &rsp, portMAX_DELAY);
            } else {
                rsp.value = rlm_shadow_eval(req.pooled);
                xQueueSend(g_shadow_q, &rsp, portMAX_DELAY);
            }
        }

        /* 2. Battery heartbeat. */
        uint32_t now = millis();
        if ((int32_t)(now - next_energy) >= 0) {
            energy_tick();
            next_energy = now + 400;
        }

        /* 3. Async host I/O. */
        while (Serial.available()) {
            char c = (char)Serial.read();
            if (c == '\r') continue;
            if (c == '\n') {
                if (overflow) {
                    json_lock();
                    Serial.println("{\"err\":\"line too long\"}");
                    json_unlock();
                } else if (len) {
                    line[len] = 0;
                    handle_line(line);
                }
                len = 0;
                overflow = false;
            } else if (len < sizeof(line) - 1) {
                line[len++] = c;
            } else {
                /* Over-long line: discard it rather than executing a
                 * truncated command. */
                overflow = true;
            }
        }
    }
}

/* ---------------- boot / dispatcher ------------------------------------ */
void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}
    g_serial_mtx = xSemaphoreCreateMutex();

    g_req_q    = xQueueCreate(4, sizeof(GateReq));
    g_shadow_q = xQueueCreate(2, sizeof(GateRsp));
    g_gate_q   = xQueueCreate(2, sizeof(GateRsp));
    g_cmd_q    = xQueueCreate(2, sizeof(CmdJob));

    rlm_engine_init();
    g_compute_task = xTaskGetCurrentTaskHandle();  /* core 1 (APP_CPU)    */

    xTaskCreatePinnedToCore(arbiter_loop, "aagm_arbiter", 8192, nullptr,
                            3, &g_arbiter_task, 0 /* PRO_CPU, beside Wi-Fi */);

    Serial.printf(">RLM-AAGM fw=3 board=%s vocab=%u dim=%u max_rec=%u "
                  "max_tokens=%u tau_lo=%.3f budget=%u chat=ready\n",
                  ARDUINO_VARIANT, RLM_VOCAB_ROWS, RLM_DIM, RLM_MAX_RECURSION,
                  RLM_MAX_TOKENS, (double)RLM_TAU_LO, g_budget);
}

void loop() {                                      /* core 1 job dispatcher */
    CmdJob job;
    while (xQueueReceive(g_cmd_q, &job, 0) == pdTRUE) {
        /* TSTF is an experimental, opt-in path. It must be set before the
         * run starts, and it is per-run so that the reference configuration
         * is never affected by a previous command. */
        g_ws.tstf_enabled = g_tstf;
        switch (job.type) {
            case CMD_INFER: do_infer(job.text); break;
            case CMD_CHAT:  do_chat(job.text);  break;
            case CMD_BENCH: do_bench((int)job.value); break;
            case CMD_STAT:  do_stat(); break;
            default: break;
        }
    }
    vTaskDelay(pdMS_TO_TICKS(5));
}
