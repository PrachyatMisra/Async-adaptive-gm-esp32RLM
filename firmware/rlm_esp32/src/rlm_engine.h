/* rlm_engine.h - Recursive Language Model (RLM) inference engine for ESP32.
 *
 * Portable C++ (no Arduino/STL/heap dependencies) implementing the exact
 * arithmetic of tools/aagm.py:AagmRLM:
 *   - weight-shared recursive block (LN -> MHA(4h) -> residual -> LN ->
 *     FFN(GELU) -> residual -> state refinement)
 *   - learned per-step gate g_k (mixing + ACT halting mass sum(1-g))
 *   - shadow halt-forecaster s_k (one-step-ahead retire pre-arm)
 *   - energy-context recursion budget with cooperative abort checkpoints
 *   - TSTF (Temporal Token Saliency Freezing), opt-in, experimental
 *
 * Numeric conventions: symmetric per-channel int8 weights dequantized
 * on the fly (accumulate int*float, apply row scale once), float32
 * activations, LayerNorm eps 1e-5, exact (erf) GELU, sigmoid/tanh via libm.
 *
 * Reentrancy: all large scratch lives in a caller-owned RlmWorkspace, so
 * independent inferences may run concurrently. rlm_gate_eval / rlm_shadow_eval
 * / rlm_evaluate_head use stack-local scratch and are safe to call from
 * another core while an inference is in flight.
 */
#ifndef RLM_ENGINE_H
#define RLM_ENGINE_H

#include <stdint.h>
#include "rlm_config.h"

/* Device input cap.
 *
 * RLM_MAX_SEQ (rlm_config.h) is the *model* context length the weights were
 * trained with. RLM_MAX_TOKENS is the *deployment* cap: the scratch buffers
 * are sized by it so the engine fits in classic-ESP32 DRAM (~160 KiB instead
 * of ~577 KiB). Inputs longer than the cap are truncated to their first
 * RLM_MAX_TOKENS tokens. Positional encodings for positions 0..31 are
 * identical to the 96-position table the model was trained with, so results
 * for inputs within the cap are bit-identical to a full-context build.
 */
#ifndef RLM_MAX_TOKENS
#define RLM_MAX_TOKENS 32
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Why the recursion stopped. RLM_HALT_ACT is the only "successful" reason. */
typedef enum {
    RLM_HALT_NONE   = 0,  /* still running (never observed in a final trace) */
    RLM_HALT_ACT    = 1,  /* cumulative ACT mass reached 1 - RLM_HALT_EPS      */
    RLM_HALT_BUDGET = 2,  /* recursion budget exhausted before the ACT threshold */
    RLM_HALT_ABORT  = 3   /* cooperative abort requested by the policy layer    */
} RlmHaltReason;

typedef struct {
    uint8_t  steps;                            /* recursion steps executed        */
    uint8_t  prearm_hints;                     /* shadow retire hints fired       */
    uint8_t  prearm_confirmed;                 /* hints followed by a real ACT halt */
    uint8_t  aborted;                          /* 1 when the run was aborted      */
    uint8_t  halt_reason;                      /* RlmHaltReason                   */
    float    depth;                            /* soft depth: sum of gates        */
    float    halting_mass;                     /* ACT mass at termination         */
    float    gates[RLM_MAX_RECURSION];         /* g_k actually used               */
    float    shadows[RLM_MAX_RECURSION];       /* s_k forecasts                   */

    /* TSTF (opt-in, experimental). Counters describe work actually skipped. */
    uint8_t  tstf_enabled;
    uint8_t  frozen_tokens;                    /* tokens whose FFN was bypassed   */
    uint32_t ffn_macs_saved;                   /* FFN MACs actually not executed  */
} RlmTrace;

typedef struct {
    uint8_t  pred;
    float    logits[RLM_CLASSES];
    RlmTrace trace;
} RlmResult;

/* Per-inference scratch. Sized by RLM_MAX_TOKENS; ~161 KiB on a 32-token
 * configuration. The caller owns it, so two engines can run concurrently. */
typedef struct RlmWorkspace {
    float h[2][RLM_MAX_TOKENS][RLM_DIM];
    float qkv[RLM_MAX_TOKENS][3 * RLM_DIM];
    float norm[RLM_MAX_TOKENS][RLM_DIM];
    float ffn[RLM_MAX_TOKENS][RLM_FFN];
    float scores[RLM_MAX_TOKENS][RLM_MAX_TOKENS];
    float attn[RLM_MAX_TOKENS][RLM_DIM];
    float pooled[RLM_DIM];
    uint8_t frozen_mask[RLM_MAX_TOKENS];
    /* State of the built-in synchronous gate provider (used when the caller
     * passes gates == NULL). Kept here so the provider stays reentrant. */
    float local_gate;
    float local_shadow;
    uint8_t tstf_enabled;   /* per-run opt-in; no global state, no races */
} RlmWorkspace;

typedef struct RlmGateProvider {
    void  (*prime_shadow)(void* ctx, const float* pooled_embed);
    void  (*launch_gate)(void* ctx, const float* pooled, uint8_t k);
    float (*wait_gate)(void* ctx);       /* g_k of the launched step        */
    float (*wait_shadow)(void* ctx);     /* s_k, precomputed one step ahead */
    int   (*should_abort)(void* ctx);
    void* ctx;
} RlmGateProvider;

/* Raw gate evaluators (public for dual-core off-core execution).
 * Reentrant: they keep their scratch on the stack. */
float rlm_gate_eval(const float* pooled);      /* authoritative gate MLP    */
float rlm_shadow_eval(const float* pooled);    /* shadow halt forecaster    */
void  rlm_evaluate_head(const float* pooled, float* logits); /* Output MLP head */

/* One-time init (idempotent, thread-safe). */
void rlm_engine_init(void);

/* Inference API. `ws` must be non-NULL and exclusively owned by the caller
 * for the duration of the call. Passing gates == NULL selects the built-in
 * synchronous provider, which evaluates the gate and shadow in-line from the
 * workspace state. */
void rlm_engine_infer(const uint16_t* ids, uint8_t n_real, uint8_t budget,
                      const RlmGateProvider* gates, RlmWorkspace* ws,
                      RlmResult* out);

/* FNV-1a hash tokenizer. Writes at most RLM_MAX_TOKENS ids; returns the
 * number of real (non-padding) tokens. */
uint8_t rlm_tokenize(const char* text, uint16_t* ids);

#ifdef __cplusplus
}
#endif

#endif /* RLM_ENGINE_H */
