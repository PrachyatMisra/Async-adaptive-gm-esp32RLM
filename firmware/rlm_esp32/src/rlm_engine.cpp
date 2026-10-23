#include "rlm_engine.h"
#include "rlm_weights.h"

#include <math.h>
#include <string.h>

/* The device cap must not exceed the model context the weights were trained
 * with, otherwise positions past the trained table would be undefined. */
static_assert(RLM_MAX_TOKENS <= RLM_MAX_SEQ,
              "RLM_MAX_TOKENS must not exceed the trained context RLM_MAX_SEQ");

/* Positional encoding. Read-only once initialised, therefore shared by all
 * concurrent inferences. */
static float g_pos[RLM_MAX_TOKENS][RLM_DIM];

/* Magic-static initialisation: the table is built exactly once, even if
 * several threads race into rlm_engine_init(). */
static const bool g_pos_ready = []() {
    for (uint16_t pos = 0; pos < RLM_MAX_TOKENS; ++pos) {
        for (uint16_t d = 0; d < RLM_DIM; d += 2) {
            float angle = (float)pos * expf(-(float)d * logf(10000.0f) / RLM_DIM);
            g_pos[pos][d] = sinf(angle);
            g_pos[pos][d + 1] = cosf(angle);
        }
    }
    return true;
}();

/* MACs in one FFN block for one token: FFN1 (DIM x FFN) + FFN2 (FFN x DIM). */
#define RLM_FFN_MACS_PER_TOKEN (2u * RLM_DIM * RLM_FFN)

/* TSTF convergence threshold: mean |h_k - h_{k-1}| per channel below which a
 * token counts as temporally converged and its FFN is skipped next step. */
#define RLM_TSTF_DELTA_EPS 0.08f

static float sigmoidf_local(float x) {
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    float e = expf(x);
    return e / (1.0f + e);
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

static void linear_q(const float* x, uint16_t in, const int8_t* w,
                     const float* scales, const float* bias, uint16_t out,
                     float* y) {
    for (uint16_t r = 0; r < out; ++r) {
        float sum = bias ? bias[r] : 0.0f;
        const int8_t* row = w + (size_t)r * in;
        for (uint16_t c = 0; c < in; ++c)
            sum += (float)row[c] * scales[r] * x[c];
        y[r] = sum;
    }
}

static void layer_norm(const float* x, const float* weight, const float* bias,
                       float* y) {
    float mean = 0.0f;
    for (uint16_t i = 0; i < RLM_DIM; ++i) mean += x[i];
    mean /= (float)RLM_DIM;
    float var = 0.0f;
    for (uint16_t i = 0; i < RLM_DIM; ++i) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)RLM_DIM;
    float inv = 1.0f / sqrtf(var + RLM_LN_EPS);
    for (uint16_t i = 0; i < RLM_DIM; ++i)
        y[i] = (x[i] - mean) * inv * weight[i] + bias[i];
}

static void pool_tokens(float (*h)[RLM_DIM], uint8_t n_real,
                        float* pooled) {
    memset(pooled, 0, sizeof(float) * RLM_DIM);
    for (uint8_t t = 0; t < n_real; ++t)
        for (uint16_t d = 0; d < RLM_DIM; ++d) pooled[d] += h[t][d];
    float inv = 1.0f / (float)(n_real ? n_real : 1);
    for (uint16_t d = 0; d < RLM_DIM; ++d) pooled[d] *= inv;
}

/* One shared recursive block applied to every token.
 * Returns the number of tokens whose FFN was actually skipped (TSTF). */
static uint16_t recursive_prefix(RlmWorkspace* ws,
                                 float (*h_in)[RLM_DIM],
                                 uint8_t n_real,
                                 float (*h_out)[RLM_DIM],
                                 float* pooled,
                                 const uint8_t* frozen_mask,
                                 uint8_t tstf_enabled) {
    for (uint8_t t = 0; t < n_real; ++t)
        layer_norm(h_in[t], p_ln1_w, p_ln1_b, ws->norm[t]);

    for (uint8_t t = 0; t < n_real; ++t)
        linear_q(ws->norm[t], RLM_DIM, &w_attn_in[0][0], s_attn_in,
                 p_attn_in_bias, 3 * RLM_DIM, ws->qkv[t]);

    const float scale = 1.0f / sqrtf((float)RLM_HEAD_DIM);
    for (uint8_t i = 0; i < n_real; ++i)
        for (uint16_t d = 0; d < RLM_DIM; ++d) ws->attn[i][d] = 0.0f;

    /* Heads are independent and write to disjoint slices of attn, so scoring,
     * softmax and accumulation may be done one head at a time with a single
     * score matrix. The arithmetic per head is unchanged. */
    for (uint8_t head = 0; head < RLM_HEADS; ++head) {
        uint16_t offset = (uint16_t)head * RLM_HEAD_DIM;
        for (uint8_t i = 0; i < n_real; ++i) {
            float max_score = -INFINITY;
            for (uint8_t j = 0; j < n_real; ++j) {
                float score = 0.0f;
                for (uint16_t d = 0; d < RLM_HEAD_DIM; ++d)
                    score += ws->qkv[i][offset + d] *
                             ws->qkv[j][RLM_DIM + offset + d];
                ws->scores[i][j] = score * scale;
                if (ws->scores[i][j] > max_score)
                    max_score = ws->scores[i][j];
            }
            float denom = 0.0f;
            for (uint8_t j = 0; j < n_real; ++j) {
                ws->scores[i][j] = expf(ws->scores[i][j] - max_score);
                denom += ws->scores[i][j];
            }
            float inv_denom = 1.0f / denom;
            for (uint8_t j = 0; j < n_real; ++j)
                ws->scores[i][j] *= inv_denom;
        }
        for (uint8_t i = 0; i < n_real; ++i)
            for (uint8_t j = 0; j < n_real; ++j)
                for (uint16_t d = 0; d < RLM_HEAD_DIM; ++d)
                    ws->attn[i][offset + d] +=
                        ws->scores[i][j] * ws->qkv[j][2 * RLM_DIM + offset + d];
    }

    uint16_t bypassed = 0;
    for (uint8_t i = 0; i < n_real; ++i) {
        linear_q(ws->attn[i], RLM_DIM, &w_attn_out[0][0], s_attn_out,
                 p_attn_out_bias, RLM_DIM, ws->norm[i]);
        for (uint16_t d = 0; d < RLM_DIM; ++d)
            ws->norm[i][d] += h_in[i][d];
        /* attn now holds the post-attention residual; norm is about to be
         * overwritten by the second LayerNorm. */
        for (uint16_t d = 0; d < RLM_DIM; ++d)
            ws->attn[i][d] = ws->norm[i][d];

        /* Co-Design Feature: Temporal Token Saliency Freezing (TSTF).
         * Opt-in and experimental: freezing changes the computed state, so it
         * is never enabled for the reference configuration. */
        if (tstf_enabled && frozen_mask && frozen_mask[i]) {
            for (uint16_t d = 0; d < RLM_DIM; ++d) h_out[i][d] = ws->attn[i][d];
            bypassed++;
        } else {
            layer_norm(ws->norm[i], p_ln2_w, p_ln2_b, ws->norm[i]);
            linear_q(ws->norm[i], RLM_DIM, &w_ffn1[0][0], s_ffn1,
                     p_ffn1_bias, RLM_FFN, ws->ffn[i]);
            for (uint16_t d = 0; d < RLM_FFN; ++d) ws->ffn[i][d] = gelu(ws->ffn[i][d]);
            linear_q(ws->ffn[i], RLM_FFN, &w_ffn2[0][0], s_ffn2,
                     p_ffn2_bias, RLM_DIM, h_out[i]);
            for (uint16_t d = 0; d < RLM_DIM; ++d) h_out[i][d] += ws->attn[i][d];
        }
    }
    pool_tokens(h_out, n_real, pooled);
    return bypassed;
}

static void refine_tokens(float (*h)[RLM_DIM], uint8_t n_real) {
    for (uint8_t t = 0; t < n_real; ++t) {
        float refined[RLM_DIM];
        linear_q(h[t], RLM_DIM, &w_refine[0][0], s_refine,
                 p_refine_bias, RLM_DIM, refined);
        for (uint16_t d = 0; d < RLM_DIM; ++d)
            h[t][d] += RLM_REFINE_GAIN * refined[d];
    }
}

void rlm_engine_init(void) {
    (void)g_pos_ready;
}

float rlm_gate_eval(const float* pooled) {
    float hidden[RLM_GATE_HID];
    linear_q(pooled, RLM_DIM, &w_gate1[0][0], s_gate1,
             p_gate1_bias, RLM_GATE_HID, hidden);
    for (uint16_t i = 0; i < RLM_GATE_HID; ++i) hidden[i] = tanhf(hidden[i]);
    float output = 0.0f;
    linear_q(hidden, RLM_GATE_HID, &w_gate2[0][0], s_gate2,
             p_gate2_bias, 1, &output);
    return sigmoidf_local(output);
}

float rlm_shadow_eval(const float* pooled) {
    float output = 0.0f;
    linear_q(pooled, RLM_DIM, &w_shadow[0][0], s_shadow,
             p_shadow_bias, 1, &output);
    return sigmoidf_local(output);
}

void rlm_evaluate_head(const float* pooled, float* logits) {
    float hidden[RLM_GATE_HID];
    linear_q(pooled, RLM_DIM, &w_cls1[0][0], s_cls1,
             p_cls1_bias, RLM_GATE_HID, hidden);
    for (uint16_t i = 0; i < RLM_GATE_HID; ++i) hidden[i] = gelu(hidden[i]);
    linear_q(hidden, RLM_GATE_HID, &w_cls2[0][0], s_cls2,
             p_cls2_bias, RLM_CLASSES, logits);
}

/* ---- built-in synchronous provider (gates == NULL) -------------------- */
/* Its two-float state lives in the workspace, so it is reentrant. */
static void local_prime(void* ctx, const float* pooled) {
    RlmWorkspace* ws = (RlmWorkspace*)ctx;
    ws->local_shadow = rlm_shadow_eval(pooled);
}
static void local_launch(void* ctx, const float* pooled, uint8_t) {
    RlmWorkspace* ws = (RlmWorkspace*)ctx;
    ws->local_gate = rlm_gate_eval(pooled);
}
static float local_wait_gate(void* ctx) {
    return ((RlmWorkspace*)ctx)->local_gate;
}
static float local_wait_shadow(void* ctx) {
    return ((RlmWorkspace*)ctx)->local_shadow;
}
static int local_abort(void*) { return 0; }

void rlm_engine_infer(const uint16_t* ids, uint8_t n_real, uint8_t budget,
                      const RlmGateProvider* provider, RlmWorkspace* ws,
                      RlmResult* out) {
    rlm_engine_init();
    if (!ws || !out) return;
    RlmGateProvider local;
    if (!provider) {
        local.prime_shadow = local_prime;
        local.launch_gate   = local_launch;
        local.wait_gate     = local_wait_gate;
        local.wait_shadow   = local_wait_shadow;
        local.should_abort  = local_abort;
        local.ctx           = ws;
        provider = &local;
    }
    if (n_real == 0) n_real = 1;
    if (n_real > RLM_MAX_TOKENS) n_real = RLM_MAX_TOKENS;
    if (budget == 0 || budget > RLM_MAX_RECURSION) budget = RLM_MAX_RECURSION;
    memset(out, 0, sizeof(*out));
    out->trace.halt_reason = RLM_HALT_NONE;
    out->trace.tstf_enabled = ws->tstf_enabled ? 1 : 0;

    for (uint8_t t = 0; t < n_real; ++t) {
        const uint16_t id = ids[t] < RLM_VOCAB_ROWS ? ids[t] : 0;
        for (uint16_t d = 0; d < RLM_DIM; ++d)
            ws->h[0][t][d] = (float)w_emb[id][d] * s_emb[id] + g_pos[t][d];
    }
    pool_tokens(ws->h[0], n_real, ws->pooled);
    provider->prime_shadow(provider->ctx, ws->pooled);

    float depth = 0.0f;
    float mass = 0.0f;
    uint8_t current = 0;
    memset(ws->frozen_mask, 0, sizeof(ws->frozen_mask));
    uint8_t hint_pending = 0;   /* shadow hint issued on the previous step */

    for (uint8_t k = 0; k < budget; ++k) {
        if (provider->should_abort(provider->ctx)) {
            out->trace.aborted = 1;
            out->trace.halt_reason = RLM_HALT_ABORT;
            break;
        }
        float shadow = provider->wait_shadow(provider->ctx);

        uint16_t bypassed = recursive_prefix(ws, ws->h[current], n_real,
                                             ws->h[1 - current], ws->pooled,
                                             ws->frozen_mask, ws->tstf_enabled);
        if (bypassed) {
            out->trace.frozen_tokens =
                (uint8_t)(out->trace.frozen_tokens + bypassed);
            out->trace.ffn_macs_saved +=
                (uint32_t)bypassed * RLM_FFN_MACS_PER_TOKEN;
        }
        provider->launch_gate(provider->ctx, ws->pooled, k);
        refine_tokens(ws->h[1 - current], n_real);
        float gate = provider->wait_gate(provider->ctx);
        if (gate < 0.0f) gate = 0.0f;
        if (gate > 1.0f) gate = 1.0f;
        out->trace.gates[k] = gate;
        out->trace.shadows[k] = shadow;
        out->trace.steps++;
        depth += gate;
        mass += 1.0f - gate;

        if (shadow < RLM_TAU_LO) {
            out->trace.prearm_hints++;
            hint_pending = 1;
        } else {
            hint_pending = 0;
        }

        /* TSTF: convergence is measured on the *unmixed* state change between
         * consecutive recursive outputs, so it does not depend on the gate. */
        if (ws->tstf_enabled) {
            for (uint8_t t = 0; t < n_real; ++t) {
                float d_sum = 0.0f;
                for (uint16_t d = 0; d < RLM_DIM; ++d) {
                    d_sum += fabsf(ws->h[1 - current][t][d] -
                                   ws->h[current][t][d]);
                }
                ws->frozen_mask[t] =
                    ((d_sum / (float)RLM_DIM) < RLM_TSTF_DELTA_EPS) ? 1 : 0;
            }
        }

        for (uint8_t t = 0; t < n_real; ++t) {
            for (uint16_t d = 0; d < RLM_DIM; ++d) {
                ws->h[1 - current][t][d] = gate * ws->h[1 - current][t][d] +
                                           (1.0f - gate) * ws->h[current][t][d];
            }
        }

        current = (uint8_t)(1 - current);
        int halt = (k + 1 >= RLM_MIN_RECURSION) &&
                   (mass >= 1.0f - RLM_HALT_EPS);
        if (halt) {
            out->trace.halt_reason = RLM_HALT_ACT;
            if (hint_pending) out->trace.prearm_confirmed = 1;
            break;
        }
        provider->prime_shadow(provider->ctx, ws->pooled);
    }

    if (out->trace.halt_reason == RLM_HALT_NONE) {
        /* Budget exhausted without reaching the ACT threshold. */
        out->trace.halt_reason = RLM_HALT_BUDGET;
    }

    /* Final Classification Head */
    for (uint8_t t = 0; t < n_real; ++t)
        layer_norm(ws->h[current][t], p_lnout_w, p_lnout_b, ws->norm[t]);
    pool_tokens(ws->norm, n_real, ws->pooled);
    rlm_evaluate_head(ws->pooled, out->logits);

    out->pred = out->logits[1] > out->logits[0] ? 1 : 0;
    out->trace.depth = depth;
    out->trace.halting_mass = mass;
}
