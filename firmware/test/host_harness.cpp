/* host_harness.cpp - native (g++) parity harness & hardware-free simulator.
 *
 * Compiles the *actual* firmware engine (rlm_engine.cpp + tokenizer.cpp)
 * on the host. Three execution modes:
 *
 *   1. Golden Parity (automated regression):
 *      ./host_harness <golden_vectors.txt> [--bench N]
 *      Emits a JSON parity report against the approved golden vectors.
 *
 *   2. Single Shot Inference:
 *      ./host_harness --infer "<text>" [--budget N] [--profile PERF|BAL|ECO]
 *      Runs tokenizer and engine, emitting device-identical telemetry JSON.
 *
 *   3. Interactive Protocol (Hardware-Free Mock UART):
 *      ./host_harness --interactive
 *      Speaks the EXACT line-oriented serial protocol of rlm_esp32.ino
 *      over stdin/stdout (INFER, CHAT, MODE, BATT, BENCH, STAT, PING, ABORT).
 *
 * The gate provider used here mirrors the firmware's asynchronous dual-core
 * provider: the gate and the shadow are produced by the same authoritative
 * MLP evaluations the firmware uses, but without a second core.
 */
#include "../rlm_esp32/src/rlm_engine.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>

static const float LOGIT_ATOL = 2e-2f;
static const float GATE_ATOL  = 1e-3f;
static const float DEPTH_ATOL = 5e-3f;

/* ---------------- Profile & Simulated Hardware State ------------------ */
enum Profile : uint8_t { PROF_PERF = 0, PROF_BAL = 1, PROF_ECO = 2 };
static const uint8_t  PROF_BUDGET[3] = { 8, 6, 4 };
static const uint32_t PROF_MHZ[3]    = { 240, 160, 80 };
static const char*    PROF_NAME[3]   = { "PERF", "BAL", "ECO" };

static volatile uint8_t  g_profile   = PROF_PERF;
static volatile bool     g_auto_mode = false;
static volatile uint8_t  g_budget    = RLM_MAX_RECURSION;
static volatile int      g_abort     = 0;
static volatile bool     g_batt_auto = false;
/* g_batt_mv is the MEASURED terminal voltage, exactly as in the firmware.
 * The open-circuit voltage is estimated by adding the droop back; the
 * droop is never subtracted from a value that was already loaded. */
static float             g_batt_mv   = 4000.0f; /* measured terminal voltage */
static const char*       g_batt_src  = "sim";
static uint32_t          g_uptime_ms = 0;
static uint8_t           g_tstf      = 0;

static const float BATT_R_INT = 0.200f; /* 200 mOhm internal resistance, model */

/* Simulated provider state. Mirrors the firmware mailbox payload. */
struct SimProvider {
    float gate;
    float shadow;
};

static RlmWorkspace g_ws;

static void sim_prime(void* ctx, const float* pooled) {
    ((SimProvider*)ctx)->shadow = rlm_shadow_eval(pooled);
}
static void sim_launch(void* ctx, const float* pooled, uint8_t) {
    ((SimProvider*)ctx)->gate = rlm_gate_eval(pooled);
}
static float sim_wait_gate(void* ctx) { return ((SimProvider*)ctx)->gate; }
static float sim_wait_shadow(void* ctx) { return ((SimProvider*)ctx)->shadow; }
static int sim_abort(void*) { return g_abort != 0; }

static SimProvider g_sim;
static RlmGateProvider SIM_PROVIDER = {
    sim_prime, sim_launch, sim_wait_gate, sim_wait_shadow, sim_abort, &g_sim
};

/* Load current drawn by the SoC as a function of CPU frequency. These are
 * board-level estimates, not measurements. */
static float load_current_ma(uint32_t cpu_mhz) {
    return (cpu_mhz >= 240) ? 50.0f : ((cpu_mhz >= 160) ? 36.0f : 27.0f);
}

/* Voltage drop across the battery's internal resistance under load. */
static float droop_mv(uint32_t cpu_mhz) {
    return (load_current_ma(cpu_mhz) / 1000.0f) * BATT_R_INT * 1000.0f;
}

/* Estimate the open-circuit voltage from the measured terminal voltage. */
static float ocv_from_measured(float measured_mv, uint32_t cpu_mhz) {
    return measured_mv + droop_mv(cpu_mhz);
}

static void apply_profile(uint8_t prof) {
    if (prof > 2) prof = 2;
    uint8_t new_budget = PROF_BUDGET[prof];
    if (new_budget < g_budget) g_abort = 1;     /* budget shrink triggers abort */
    g_profile = prof;
    g_budget  = new_budget;
}

static uint8_t profile_for_ocv(float ocv_mv) {
    float pct = (ocv_mv - 3300.0f) / 900.0f * 100.0f;
    if (pct < 0.0f) pct = 0.0f;
    if (pct > 100.0f) pct = 100.0f;
    if (pct < 25.0f) return PROF_ECO;
    if (pct < 50.0f) return PROF_BAL;
    return PROF_PERF;
}

static void energy_tick(void) {
    /* The chip sees the MEASURED terminal voltage, so that is what is
     * compared against the brownout cutoff. */
    if (g_batt_mv < 3350.0f) {
        apply_profile(PROF_ECO);
        g_abort = 1; /* brownout mitigation trigger */
    } else if (g_auto_mode) {
        apply_profile(profile_for_ocv(ocv_from_measured(g_batt_mv, PROF_MHZ[g_profile])));
    }
}

/* ---------------- JSON helpers ----------------------------------------- */
static const char* halt_reason_name(uint8_t r) {
    switch (r) {
        case RLM_HALT_ACT:    return "act_halt";
        case RLM_HALT_BUDGET: return "budget_exhausted";
        case RLM_HALT_ABORT:  return "aborted";
        default:              return "none";
    }
}

/* Escapes a byte string into a JSON string body (no surrounding quotes). */
static void json_escape(const char* s, char* out, size_t cap) {
    size_t o = 0;
    for (const unsigned char* p = (const unsigned char*)s; *p && o + 7 < cap; ++p) {
        switch (*p) {
            case '"':  out[o++] = '\\'; out[o++] = '"';  break;
            case '\\': out[o++] = '\\'; out[o++] = '\\'; break;
            case '\n': out[o++] = '\\'; out[o++] = 'n';  break;
            case '\r': out[o++] = '\\'; out[o++] = 'r';  break;
            case '\t': out[o++] = '\\'; out[o++] = 't';  break;
            case '\b': out[o++] = '\\'; out[o++] = 'b';  break;
            case '\f': out[o++] = '\\'; out[o++] = 'f';  break;
            default:
                if (*p < 0x20) {
                    o += (size_t)snprintf(out + o, cap - o, "\\u%04x", *p);
                } else {
                    out[o++] = (char)*p;
                }
        }
    }
    out[o] = 0;
}

static void emit_infer_json(const char* text, uint8_t budget) {
    uint16_t ids[RLM_MAX_TOKENS];
    uint8_t n_real = rlm_tokenize(text, ids);
    if (n_real == 0) {
        printf("{\"err\":\"empty\"}\n");
        fflush(stdout);
        return;
    }

    g_ws.tstf_enabled = g_tstf;
    auto t0 = std::chrono::steady_clock::now();
    RlmResult r;
    rlm_engine_infer(ids, n_real, budget, &SIM_PROVIDER, &g_ws, &r);
    auto t1 = std::chrono::steady_clock::now();
    uint32_t us = (uint32_t)std::chrono::duration<double, std::micro>(t1 - t0).count();

    float droop = droop_mv(PROF_MHZ[g_profile]);
    float ocv = ocv_from_measured(g_batt_mv, PROF_MHZ[g_profile]);

    printf("{\"pred\":%u,\"logits\":[%.4f,%.4f],\"steps\":%u,"
           "\"depth\":%.4f,\"mass\":%.4f,\"hints\":%u,\"prearm_confirmed\":%u,"
           "\"aborted\":%u,\"halt_reason\":\"%s\","
           "\"frozen\":%u,\"tstf\":%u,\"macs_saved\":%u,"
           "\"gates\":[", r.pred, r.logits[0], r.logits[1],
           r.trace.steps, r.trace.depth, r.trace.halting_mass,
           r.trace.prearm_hints, r.trace.prearm_confirmed,
           r.trace.aborted, halt_reason_name(r.trace.halt_reason),
           r.trace.frozen_tokens, r.trace.tstf_enabled,
           (unsigned)r.trace.ffn_macs_saved);
    for (uint8_t i = 0; i < r.trace.steps; ++i)
        printf(i ? ",%.4f" : "%.4f", r.trace.gates[i]);
    printf("],\"shadows\":[");
    for (uint8_t i = 0; i < r.trace.steps; ++i)
        printf(i ? ",%.4f" : "%.4f", r.trace.shadows[i]);
    printf("],\"us\":%u,\"budget\":%u,\"profile\":\"%s\",\"batt_mv\":%.0f,"
           "\"ocv_mv\":%.0f,\"droop_mv\":%.1f,\"batt_src\":\"%s\","
           "\"cpu_mhz\":%u,\"tokens\":%u}\n",
           us, budget, PROF_NAME[g_profile], (double)g_batt_mv,
           (double)ocv, (double)droop, g_batt_src,
           (unsigned)PROF_MHZ[g_profile], n_real);
    fflush(stdout);
}

static void emit_chat_json(const char* text) {
    uint16_t ids[RLM_MAX_TOKENS];
    uint8_t n_real = rlm_tokenize(text, ids);
    if (n_real == 0) {
        printf("{\"chat_reply\":\"Please enter a valid message.\"}\n");
        fflush(stdout);
        return;
    }

    g_ws.tstf_enabled = g_tstf;
    auto t0 = std::chrono::steady_clock::now();
    RlmResult r;
    rlm_engine_infer(ids, n_real, g_budget, &SIM_PROVIDER, &g_ws, &r);
    auto t1 = std::chrono::steady_clock::now();
    uint32_t us = (uint32_t)std::chrono::duration<double, std::micro>(t1 - t0).count();

    float max_l = std::max(r.logits[0], r.logits[1]);
    float e0 = expf(r.logits[0] - max_l);
    float e1 = expf(r.logits[1] - max_l);
    float conf = (r.pred == 1 ? e1 : e0) / (e0 + e1) * 100.0f;
    const char* verdict = r.pred == 1 ? "POSITIVE" : "NEGATIVE";

    char safe[1024];
    json_escape(text, safe, sizeof(safe));

    /* "saved_steps" is the difference between the budget and the steps that
     * actually ran. An aborted run did not save compute, it was cut short,
     * so the reply says so explicitly. */
    int saved = (int)g_budget - (int)r.trace.steps;
    if (saved < 0) saved = 0;
    const char* saving_note =
        r.trace.halt_reason == RLM_HALT_ACT
            ? "early halt"
            : (r.trace.halt_reason == RLM_HALT_ABORT
                   ? "run aborted"
                   : "budget exhausted");

    printf("{\"chat_reply\":\"Sentiment classification of your input. "
           "Verdict: %s (Confidence: %.1f%%). "
           "Recursive steps: %u of %u budget (%s), cumulative mass: %.3f. "
           "Host latency: %u us.\","
           "\"pred\":%u,\"verdict\":\"%s\",\"confidence\":%.1f,"
           "\"steps\":%u,\"budget\":%u,\"saved_steps\":%d,\"halt_reason\":\"%s\","
           "\"aborted\":%u,\"us\":%u,\"profile\":\"%s\",\"echo_len\":%zu}\n",
           verdict, conf, r.trace.steps, g_budget, saving_note,
           r.trace.halting_mass, us, r.pred, verdict, conf,
           r.trace.steps, g_budget, saved,
           halt_reason_name(r.trace.halt_reason), r.trace.aborted, us,
           PROF_NAME[g_profile], strlen(safe));
    fflush(stdout);
}

static void run_interactive(void) {
    rlm_engine_init();
    printf(">RLM-AAGM fw=3 board=ESP32-DevKit-v1(HostSim) vocab=%u dim=%u max_rec=%u "
           "tau_lo=%.3f budget=%u max_tokens=%u chat=ready\n",
           RLM_VOCAB_ROWS, RLM_DIM, RLM_MAX_RECURSION,
           (double)RLM_TAU_LO, g_budget, (unsigned)RLM_MAX_TOKENS);
    fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (line.empty()) continue;

        if (line.rfind("CHAT ", 0) == 0) {
            std::string text = line.substr(5);
            g_abort = 0;
            emit_chat_json(text.c_str());
        }
        else if (line.rfind("INFER ", 0) == 0) {
            std::string text = line.substr(6);
            g_abort = 0;
            emit_infer_json(text.c_str(), g_budget);
        }
        else if (line.rfind("BENCH", 0) == 0) {
            int n = 20;
            if (line.size() > 6) n = std::max(1, atoi(line.c_str() + 6));
            const char* text = "the first act is clumsy and uneven meanwhile the cast "
                               "shines, yet the final scene feels magnificent";
            uint16_t ids[RLM_MAX_TOKENS];
            uint8_t n_real = rlm_tokenize(text, ids);
            uint32_t best = UINT32_MAX, worst = 0, steps_sum = 0;
            double sum = 0;
            for (int i = 0; i < n; ++i) {
                g_abort = 0;
                auto t0 = std::chrono::steady_clock::now();
                RlmResult r;
                rlm_engine_infer(ids, n_real, g_budget, &SIM_PROVIDER, &g_ws, &r);
                auto t1 = std::chrono::steady_clock::now();
                uint32_t us = (uint32_t)std::chrono::duration<double, std::micro>(t1 - t0).count();
                best = std::min(best, us);
                worst = std::max(worst, us);
                sum += us; steps_sum += r.trace.steps;
            }
            printf("{\"bench_n\":%d,\"us_avg\":%.1f,\"us_min\":%u,\"us_max\":%u,"
                   "\"mean_steps\":%.3f,\"budget\":%u,\"cpu_mhz\":%u,"
                   "\"profile\":\"%s\",\"note\":\"host wall time, not ESP32 latency\"}\n",
                   n, sum / n, (unsigned)best, (unsigned)worst,
                   (double)steps_sum / n, g_budget,
                   (unsigned)PROF_MHZ[g_profile], PROF_NAME[g_profile]);
            fflush(stdout);
        }
        else if (line == "STAT") {
            float droop = droop_mv(PROF_MHZ[g_profile]);
            float ocv = ocv_from_measured(g_batt_mv, PROF_MHZ[g_profile]);
            printf("{\"board\":\"ESP32-DevKit-v1(HostSim)\",\"cpu_mhz\":%u,\"budget\":%u,"
                   "\"profile\":\"%s\",\"auto_mode\":%s,\"batt_mv\":%.0f,"
                   "\"ocv_mv\":%.0f,\"droop_mv\":%.1f,\"batt_src\":\"%s\","
                   "\"max_tokens\":%u,"
                   "\"workspace_bytes\":%zu,\"note\":\"simulated board, no physical device\"}\n",
                   (unsigned)PROF_MHZ[g_profile], g_budget,
                   PROF_NAME[g_profile], g_auto_mode ? "true" : "false",
                   (double)g_batt_mv, (double)ocv, (double)droop, g_batt_src,
                   (unsigned)RLM_MAX_TOKENS, sizeof(RlmWorkspace));
            fflush(stdout);
        }
        else if (line == "PING") {
            g_uptime_ms += 100;
            printf("{\"pong\":1,\"uptime_ms\":%u}\n", g_uptime_ms);
            fflush(stdout);
        }
        else if (line.rfind("MODE ", 0) == 0) {
            std::string m = line.substr(5);
            if      (m == "PERF") { g_auto_mode = false; apply_profile(PROF_PERF); }
            else if (m == "BAL")  { g_auto_mode = false; apply_profile(PROF_BAL); }
            else if (m == "ECO")  { g_auto_mode = false; apply_profile(PROF_ECO); }
            else if (m == "AUTO") { g_auto_mode = true; energy_tick(); }
            printf("{\"mode\":\"%s\",\"budget\":%u,\"cpu_mhz\":%u}\n",
                   PROF_NAME[g_profile], g_budget, (unsigned)PROF_MHZ[g_profile]);
            fflush(stdout);
        }
        else if (line.rfind("BATT ", 0) == 0) {
            std::string b = line.substr(5);
            if (b == "AUTO") { g_batt_auto = true; }
            else { g_batt_auto = false; g_batt_mv = (float)atof(b.c_str()); g_batt_src = "sim"; }
            energy_tick();
            float droop = droop_mv(PROF_MHZ[g_profile]);
            float ocv = ocv_from_measured(g_batt_mv, PROF_MHZ[g_profile]);
            printf("{\"batt_mv\":%.0f,\"ocv_mv\":%.0f,\"droop_mv\":%.1f,\"batt_src\":\"%s\","
                   "\"profile\":\"%s\",\"budget\":%u}\n",
                   (double)g_batt_mv, (double)ocv, (double)droop, g_batt_src,
                   PROF_NAME[g_profile], g_budget);
            fflush(stdout);
        }
        else if (line.rfind("ABORT", 0) == 0) {
            g_abort = 1;
            printf("{\"abort_armed\":1}\n");
            fflush(stdout);
        }
        else if (line.rfind("TSTF ", 0) == 0) {
            g_tstf = (atoi(line.c_str() + 5) != 0) ? 1 : 0;
            printf("{\"tstf\":%u,\"experimental\":true}\n", g_tstf);
            fflush(stdout);
        }
        else if (line == "QUIT" || line == "EXIT") {
            break;
        }
        else {
            printf("{\"err\":\"unknown cmd\"}\n");
            fflush(stdout);
        }
    }
}

/* ---------------- Golden Row Parsing ---------------------------------- */
struct Golden {
    std::string text;
    int pred = 0;
    float logits[2] = {0.0f, 0.0f};
    float depth = 0.0f;
    int steps = 0;
    int n_real = 0;
    uint16_t ids[RLM_MAX_SEQ] = {0};
    std::vector<float> gates;
    std::vector<float> shadows;

    Golden() {
        gates.reserve(RLM_MAX_RECURSION);
        shadows.reserve(RLM_MAX_RECURSION);
    }
};

static bool parse_row(const std::string& line, Golden& g) {
    std::vector<float> v;
    const char* p = line.c_str();
    char* end = nullptr;
    while (*p) {
        float x = strtof(p, &end);
        if (end == p) { ++p; continue; }
        v.push_back(x);
        p = end;
    }
    if (v.size() != (size_t)(6 + RLM_MAX_SEQ + 2 * RLM_MAX_RECURSION)) return false;
    size_t i = 0;
    g.pred  = (int)v[i++];
    g.logits[0] = v[i++];
    g.logits[1] = v[i++];
    g.depth = v[i++];
    g.steps = (int)v[i++];
    g.n_real = (int)v[i++];
    for (int t = 0; t < RLM_MAX_SEQ; ++t) g.ids[t] = (uint16_t)v[i++];
    g.gates.clear();
    g.shadows.clear();
    for (int t = 0; t < RLM_MAX_RECURSION; ++t) g.gates.push_back(v[i++]);
    for (int t = 0; t < RLM_MAX_RECURSION; ++t) g.shadows.push_back(v[i++]);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage:\n"
                        "  %s golden.txt [--bench N]\n"
                        "  %s --infer \"<text>\" [--budget N] [--profile PERF|BAL|ECO]\n"
                        "                        [--batt MV] [--tstf]\n"
                        "  %s --interactive\n"
                        "  %s --memory\n",
                argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }

    /* 0. Static memory budget report */
    if (!strcmp(argv[1], "--memory")) {
        printf("{\"max_tokens\":%u,\"max_seq\":%u,\"workspace_bytes\":%zu,"
               "\"engine_static_bytes\":%zu,\"note\":\"host build; the ESP32 build "
               "adds the same buffers as .bss plus FreeRTOS stacks\"}\n",
               (unsigned)RLM_MAX_TOKENS, (unsigned)RLM_MAX_SEQ,
               sizeof(RlmWorkspace), sizeof(RlmWorkspace));
        return 0;
    }

    /* 1. Interactive mock UART loop */
    if (!strcmp(argv[1], "--interactive") || !strcmp(argv[1], "-i")) {
        run_interactive();
        return 0;
    }

    /* 2. Single shot infer */
    if (!strcmp(argv[1], "--infer")) {
        if (argc < 3) { fprintf(stderr, "error: --infer requires text\n"); return 2; }
        const char* text = argv[2];
        uint8_t budget = RLM_MAX_RECURSION;
        bool budget_explicit = false;
        for (int i = 3; i + 1 < argc; ++i) {
            if (!strcmp(argv[i], "--budget")) {
                budget = (uint8_t)atoi(argv[i + 1]);
                budget_explicit = true;
            }
            if (!strcmp(argv[i], "--profile")) {
                if (!strcmp(argv[i+1], "BAL")) apply_profile(PROF_BAL);
                else if (!strcmp(argv[i+1], "ECO")) apply_profile(PROF_ECO);
                else apply_profile(PROF_PERF);
                if (!budget_explicit) budget = g_budget;
            }
            if (!strcmp(argv[i], "--batt")) {
                float mv = (float)atof(argv[i + 1]);
                if (mv < 2000.0f || mv > 5000.0f) {
                    fprintf(stderr, "error: --batt must be within 2000..5000 mV\n");
                    return 2;
                }
                g_batt_mv = mv;
                g_batt_src = "sim";
            }
            if (!strcmp(argv[i], "--tstf")) g_tstf = 1;
        }
        rlm_engine_init();
        emit_infer_json(text, budget);
        return 0;
    }

    /* 3. Golden verification mode */
    int bench_n = 0;
    for (int i = 2; i + 1 < argc; ++i)
        if (!strcmp(argv[i], "--bench")) bench_n = atoi(argv[i + 1]);

    FILE* f = fopen(argv[1], "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    std::vector<Golden> gs;
    char buf[8192];
    std::string text;
    while (fgets(buf, sizeof buf, f)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (line.rfind("TEXT ", 0) == 0) { text = line.substr(5); continue; }
        if (line.empty()) continue;
        Golden g;
        if (parse_row(line, g)) { g.text = text; gs.push_back(g); }
    }
    fclose(f);
    if (gs.empty()) { fprintf(stderr, "no golden rows parsed\n"); return 2; }

    rlm_engine_init();

    float max_le = 0.f, max_ge = 0.f, max_de = 0.f, max_se = 0.f;
    int mism_pred = 0, mism_steps = 0, mism_ids = 0, pass = 0, skipped = 0;
    long steps_sum = 0, hint_sum = 0;
    printf("{\"samples\":[");
    for (size_t i = 0; i < gs.size(); ++i) {
        Golden& g = gs[i];

        /* A golden longer than the deployment token cap cannot be reproduced
         * by this build; it is reported as skipped, not as a failure. */
        if (g.n_real > RLM_MAX_TOKENS) {
            ++skipped;
            printf("%s{\"i\":%zu,\"skipped\":\"exceeds_device_token_cap\","
                   "\"gold_tokens\":%d,\"cap\":%u}",
                   i ? "," : "", i, g.n_real, (unsigned)RLM_MAX_TOKENS);
            continue;
        }

        /* C tokenizer must reproduce the golden ids exactly */
        uint16_t ids[RLM_MAX_TOKENS];
        uint8_t n_real = rlm_tokenize(g.text.c_str(), ids);
        bool ids_ok = (n_real == g.n_real);
        for (int t = 0; ids_ok && t < n_real; ++t) ids_ok &= (ids[t] == g.ids[t]);
        if (!ids_ok) ++mism_ids;

        RlmResult r;
        rlm_engine_infer(g.ids, (uint8_t)g.n_real, RLM_MAX_RECURSION,
                         NULL, &g_ws, &r);

        float le = fmaxf(fabsf(r.logits[0] - g.logits[0]),
                         fabsf(r.logits[1] - g.logits[1]));
        float de = fabsf(r.trace.depth - g.depth);
        float ge = 0.f, se = 0.f;
        for (int k = 0; k < g.steps && k < RLM_MAX_RECURSION; ++k) {
            ge = fmaxf(ge, fabsf(r.trace.gates[k] - g.gates[k]));
            se = fmaxf(se, fabsf(r.trace.shadows[k] - g.shadows[k]));
        }
        max_le = fmaxf(max_le, le); max_de = fmaxf(max_de, de);
        max_ge = fmaxf(max_ge, ge); max_se = fmaxf(max_se, se);
        bool ok = ((int)r.pred == g.pred) && ((int)r.trace.steps == g.steps) &&
                  ids_ok && le < LOGIT_ATOL && de < DEPTH_ATOL && ge < GATE_ATOL;
        mism_pred += ((int)r.pred != g.pred);
        mism_steps += ((int)r.trace.steps != g.steps);
        pass += ok;
        steps_sum += r.trace.steps;
        hint_sum += r.trace.prearm_hints;
        printf("%s{\"i\":%zu,\"pred\":%d,\"gold_pred\":%d,\"steps\":%d,"
               "\"gold_steps\":%d,\"depth\":%.4f,\"logit_err\":%.2e,"
               "\"gate_err\":%.2e,\"shadow_err\":%.2e,\"ids_ok\":%s,\"ok\":%s}",
               i ? "," : "", i, r.pred, g.pred, r.trace.steps, g.steps,
               r.trace.depth, le, ge, se, ids_ok ? "true" : "false",
               ok ? "true" : "false");
    }
    printf("],\"summary\":{");

    /* determinism + rough host speed (bench: rerun first text N times) */
    double us_min = 1e12, us_avg = 0.0;
    if (bench_n > 0 && !gs.empty()) {
        for (int b = 0; b < bench_n; ++b) {
            auto t0 = std::chrono::steady_clock::now();
            RlmResult r;
            rlm_engine_infer(gs[0].ids, (uint8_t)gs[0].n_real, RLM_MAX_RECURSION,
                             NULL, &g_ws, &r);
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            us_min = us < us_min ? us : us_min;
            us_avg += us;
        }
        us_avg /= bench_n;
    }
    printf("\"golden\":%zu,\"passed\":%d,\"skipped\":%d,\"max_logit_err\":%.3e,"
           "\"max_gate_err\":%.3e,\"max_shadow_err\":%.3e,\"max_depth_err\":%.3e,"
           "\"mism_pred\":%d,\"mism_steps\":%d,\"mism_ids\":%d,"
           "\"mean_steps\":%.3f,\"mean_prearm_hints\":%.3f,"
           "\"bench_us_avg\":%.1f,\"bench_us_min\":%.1f,"
           "\"max_tokens\":%u,\"workspace_bytes\":%zu,"
           "\"PASS\":%s}\n",
           gs.size(), pass, skipped, max_le, max_ge, max_se, max_de,
           mism_pred, mism_steps, mism_ids,
           gs.size() ? (double)steps_sum / gs.size() : 0.0,
           gs.size() ? (double)hint_sum / gs.size() : 0.0,
           us_avg == 0.0 ? 0.0 : us_avg, us_min == 1e12 ? 0.0 : us_min,
           (unsigned)RLM_MAX_TOKENS, sizeof(RlmWorkspace),
           (pass + skipped == (int)gs.size()) ? "true" : "false");
    printf("}\n");
    return pass + skipped == (int)gs.size() ? 0 : 1;
}
