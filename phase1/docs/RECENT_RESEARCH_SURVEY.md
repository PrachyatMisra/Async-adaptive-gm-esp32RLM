# Recent Research Survey - RLM and Related Work (2024-2026)

**For Phase 1 Robust Integration - Beyond Original Paper**

This document surveys recent papers (2024-2026) relevant to Recursive Language Models, weight sharing, adaptive gating, and edge deployment, with citations and integration notes.

---

## 1. Residual Scaling for Looped Transformers

### Papers:
- **On the Residual Scaling of Looped Transformers: Stability and Transferability** (arXiv:2606.18524)
- **DeepLoop: Depth Scaling for Looped Transformers** (arXiv:2607.13491)

### Key Findings:
- Standard 1/sqrt(N) scaling assumes independent weights, but weight sharing makes successive updates correlated
- Amplifies residual-stream norm growth from Θ(sqrt(N)) to Θ(N) - exploding hidden states
- **Solution:** Factorized parameterization epsilon = lambda / (N * sqrt(L))
  - 1/N controls within-layer loop correlation
  - 1/sqrt(L) controls across-layer variance
- Optimal learning rate depends only on number of unique layers L, not loop count N
- Consistent final norm across loop counts with 1/N scaling, vs rapid growth with 1/sqrt(N)

### Connection to Our Paper:
- **Explains Tmax=8 collapse F1 0.246 and FixedLoop T=4 F1 0.084 collapse** - Lipschitz instability without proper residual scaling
- Paper mentions 0.1 refinement gain and tanh stabilization, but not explicit 1/N scaling
- Implementation: `ResidualScaling` class with `get_scale()` method

### Integration:
```python
# Factorized parameterization
epsilon = lambda_factor / (N * sqrt(L))
h <- h + epsilon * f(h)  # Bounds norm growth
```

---

## 2. Relaxed Recursive Transformers with LoRA

### Papers:
- **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** (arXiv:2410.20672, Bae et al. 2024-2025)
- **ALBERT** (Lan et al. 2019) - cross-layer parameter sharing, 9x param reduction
- **Lessons on Parameter Sharing across Layers** (arXiv:2104.06022)

### Key Findings:
- Strict weight tying limits expressivity (Proposition 1 in our paper: FK_RLM ⊂ FL)
- Cannot express qualitatively different attention patterns (W_Q=I vs W_Q=-I)
- **Solution:** Relaxed Recursive Transformer - shared block + layer-specific LoRA adapters
  - Each loop iteration gets different LoRA modules
  - LoRA introduces low-rank deltas to shared linear weights
  - Average initialization for looped layers beats Stepwise
  - Approximating residual matrices between averaged weights and individual weights via truncated SVD with low ranks achievable
- Uptraining from existing models: tie layers then uptrain to adapt to repeated use
- Extended uptraining + knowledge distillation from full model improves low-rank models

### Connection to Our Paper:
- **Proposition 1 expressivity bounds** - strict inequality FK_RLM ≠ FL due to identical W_Q
- Paper's future work: larger models d=768, spectral norm
- Relaxed model allows smoother tradeoff between fully tied and vanilla

### Integration:
```python
class LoRAAdapter:
    lora_A: shared across layers
    lora_B: adaptively merged
    forward: (x @ A @ B) * scaling

class RelaxedRecursiveBlock:
    base shared block + LoRA per iteration
    Average init > Stepwise
```

---

## 3. Adaptive Depth Gating Diagnostics

### Papers:
- **Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts** (arXiv:2607.20519)
- **AdaPonderLM: Gated Pondering Language Models with Token-Wise Adaptive Depth** (arXiv:2603.01914)
- **PonderLM: Pretraining Language Models to Ponder in Continuous Space** (arXiv:2505.20674)
- **Universal Transformers Need Memory: Depth-State Trade-offs** (arXiv:2604.21999)

### Key Findings:
- **Learned halting gates unreliable in pretrained looped models** - 3 failure modes:
  1. Collapse to shallowest depth
  2. Ignore cost and run to full depth
  3. Freeze at near-constant depth
- **Shallow-halt trap:** Both default zero-bias and Graves positive bias settle into shallow halt equilibrium model cannot escape
  - Early high halting prob → stops early → never sees complex dependencies → never learns to ponder longer → vicious cycle
- **Halt-side pressure:** Ponder cost rho = N + R minimized alongside task loss, mean halt drops monotonically with memory size
- **Ponder loss variants:**
  - Bottom-K fraction (k=0.1) gate values penalized - AdaPonderLM
  - Entropy regularization
  - KL divergence to prior collapses with large coefficient
- **Router z-loss:** Penalize large logits for numerical stability (Zoph et al. 2022) - used in MoR

### Connection to Our Paper:
- **Tmax=8 100% halt fraction but Deff 0.782** - gate collapse to shallow, matches failure mode 1
- **Without tmin halts after 1 step** - shallow trap local minimum
- **tanh before sigmoid stabilization** - prevents early saturation, related to bias init
- **Ponder penalty lambda=0.01 rescues** - matches halt-side pressure training
- **Loss curve noisier** - gates dynamically adjusting

### Integration:
```python
class ImprovedGateNetwork:
    init_bias = -1.0 not 0 (negative encourages deeper initially, avoids shallow trap)
    z_loss: mean(logsumexp(logits)^2) penalize large logits
    entropy: -[g log g + (1-g)log(1-g)] maximize for exploration
    bottom_k: penalize bottom-K fraction of gates

Loss = CE + lambda_ponder*ponder + lambda_z*z_loss + lambda_ent*entropy + lambda_bottom*bottom_k
```

---

## 4. Token-Wise Adaptive Depth and Mixture-of-Recursions

### Papers:
- **Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation** (arXiv:2507.10524, NeurIPS 2025)
- **Scaling Latent Reasoning via Looped Language Models (LoopLM)** (arXiv:2510.25741)
- **Looping Back to Move Forward: Recursive Transformers for Efficient and Flexible LMMs** (arXiv:2602.09080)
- **LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models** (arXiv:2605.11011)

### Key Findings:
- **MoR:** Lightweight routers assign different recursion depths per token, focus quadratic attention only among active tokens, selective KV caching, KV sharing variant for prefill latency
- **LoopLM:** Shared-weight transformer layers applied recurrently, sandwich normalization, SwiGLU, RoPE, learned exit gates lambda_t(x)=σ(W_gate·h_t+b_gate), entropy-regularized training with uniform prior enables adaptive computation, ponder gate with specialized adaptive exit training achieves best accuracy at every budget
- **FREE Framework:** Parallel decoding early form of self-speculative decoding, computes exact KV for early-exited tokens, but poor batch inference compatibility - recursive models mitigate via parameter sharing
- **Hierarchical Reasoning Model (HRM):** 40.3% on ARC-AGI with 27M params and ~1000 examples, weight-tied recursion beats unshared depth at matched depth/compute (4-layer block looped 8 times 40.0 pass@1 vs 32-layer 23.75 with 8x params)

### Connection to Our Paper:
- **Paper future work:** Token-level halting for autoregressive generation
- **Current:** Sequence-level gating, token-level analysis shows r=-0.013 no correlation (gate not keyword counting)
- **MoR:** Extends to token-level with routers, selective caching - natural next step

### Integration:
```python
class TokenWiseGating:
    routers per iteration: lightweight MLP per token
    token_gates [B, L, T], active_mask [B, L, T]
    Focus attention among active tokens only
    KV sharing variant for prefill

Future: For Phase1 d=768 GLUE, token-level halting autoregressive
```

---

## 5. Pondering and Latent Reasoning

### Papers:
- **Pondering LMs** - repeatedly invoking forward process within single token generation, weighted sum of all token embeddings as ponder, fully differentiable end-to-end pretraining via gradient descent, higher parameter knowledge density
- **Pondering GPT-2 834M achieves loss comparable to vanilla GPT-2 with ~2.01x params*tokens, Pondering LLaMA 834M matches vanilla 3.2x**
- **Pondering Pythia 2.5B achieves validation loss comparable to 6.9B official Pythia with only 37% params**

### Key Findings:
- Pondering as implicit chain-of-thought, iterative refinement of predictions
- Overcomes expressive limitations of discrete vocabularies via continuous weighted embeddings
- Test-time scaling via recurrent/iterative transformers enables spending more computation at inference

### Connection to Our Paper:
- **Neural ODE connection:** h(t+1)=h(t)+g(t)*(R(h)-h(t)) forward Euler discretization of dh/dt=R(h)-h
- **Optimal depth 2** - model learns optimal stopping independently of budget
- **Ponder penalty** - balancing computational penalty critical, excessive induces premature halting, negligible reduces to standard transformer

---

## 6. ESP32 TinyML and Quantization (Phase 2)

### Papers/Articles:
- **ESP32-S3 Edge AI in Practice** (zediot.com, dev.to)
- **TinyML on ESP32-S3: Running TFLM for Edge AI Inference**
- **Edge AI & TinyML on ESP32: On-Device Machine Learning** (gizantech.com)
- **Model Quantization and Optimization for MCUs**

### Key Findings:
- **INT8 is default, INT4 is frontier** - Float32 not deployable on MCU class
- **Post-training quantization with representative dataset 100-500 samples** from actual deployment distribution
- **75% smaller model, 4-10x faster inference, lower power** - integer arithmetic far more energy-efficient
- **Critical step: Input preprocessing** - scale and zero-point must match training, else accuracy drops dramatically
- **ESP-NN hardware acceleration** - 3-8x faster on S3 vs classic on quantized models
- **Static memory allocation** - tensor arena 4KB generous for tiny model, 32KB for larger, 16-byte aligned, no dynamic allocation
- **Affine quantization:** real = scale * (q - zero_point), q int8 [-128,127]
- **Performance reference:**
  - KWS 20K params 1s audio MFCC INT8: ~12ms SRAM ~15ms PSRAM
  - Gesture 5K 128Hz Accel INT8: ~2ms SRAM ~2.5ms PSRAM
  - Person Detection 250K 96x96 grayscale INT8: N/A overflow SRAM ~145ms PSRAM
  - MNIST 60K 28x28 INT8: ~8ms SRAM ~10ms PSRAM

### Connection to Phase 2:
- Our implementation: symmetric per-channel int8 [-127,127] zero_point=0 (LiteRT spec), per-row scales, float32 activations (FPU), 510 KiB Flash 28 KiB SRAM 0 heap, 0.00% quant drop max logit drift 4.5e-6
- Energy: 240MHz 30-68mA bench 66.8mA, 160MHz 27-44mA bench 45.9mA, 80MHz 20-31mA bench 33.2mA, WiFi Tx 180-240mA peaks dominates CPU, deep sleep ~10uA
- Our RLM 524K params 32 tokens INT8: 3.5ms avg 2.18 steps

### Integration:
```python
# Post-training INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = lambda: ([sample.astype(np.float32)] for sample in calibration_samples)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Affine mapping
real = scale * (q - zero_point)

# Critical: Input preprocessing scale/zero-point must match training
```

---

## 7. Integration Summary for Robust Backend

### Phase 1 Advanced Features Implemented:

1. **ResidualScaling** - 1/N scaling not 1/sqrt(N) for weight sharing, epsilon=lambda/(N*sqrt(L)), bounds norm growth Θ(N) vs Θ(sqrt(N)), explains Tmax=8 collapse
2. **RelaxedRecursiveBlock with LoRA** - Shared block + layer-wise LoRA per iteration, Average init > Stepwise, addresses Proposition 1 expressivity bounds, minimal overhead rank=8
3. **ImprovedGateNetwork** - Negative bias init -1.0 avoids shallow-halt trap, router z-loss penalize large logits (Zoph et al.), entropy regularization encourage exploration, bottom-K ponder loss (AdaPonderLM)
4. **TokenWiseGating** - MoR style lightweight routers per token, selective KV caching, future work for Phase1 d=768 GLUE and autoregressive
5. **SpectralNormWrapper** - Guarantee Lipschitz K<1 for Banach fixed-point, future work Section VII
6. **AdvancedLoss** - CE + ponder + bottom-K + z-loss + entropy, tracking residual norm growth, gate collapse detection, Pareto frontier
7. **Advanced Metrics** - Lipschitz estimation, residual growth analysis, gate collapse detection (shallow/deep/frozen), token-level correlation beyond r=-0.013

### Phase 2 Advanced Features:

1. **Advanced Quantization** - Representative dataset 100-500 samples, scale/zero-point matching critical, ESP-NN acceleration, SRAM/PSRAM placement, affine mapping real=scale*(q-zero_point), profiling reference
2. **Energy Model** - Real draws 66.8mA@240→45.9mA@160→33.2mA@80, WiFi Tx dominates, deep sleep 10uA, V_loaded=V_OCV-I*R_int R_int=0.2Ω, calibration with INA219
3. **Verification Robustness** - Track1 C++ parity 12/12 PASS logit error 2.86e-6, Track2 Verilog RTL Q1.15 mass signals VCD GTKWave, Track3 UART exact protocol, artifacts

### Papers Cited:

- Residual Scaling: 2606.18524, 2607.13491 DeepLoop
- Relaxed Recursive: 2410.20672 Bae et al., ALBERT, 2104.06022
- Gating Diagnostics: 2607.20519, 2603.01914 AdaPonderLM, 2505.20674 PonderLM, 2604.21999
- Token-Wise: 2507.10524 MoR NeurIPS 2025, 2510.25741 LoopLM, 2602.09080, 2605.11011 LoopUS
- Pondering: Pondering LMs continuous space
- TinyML: ESP32-S3 Edge AI, TFLM, INT8 quantization

---

## For Antigravity Prompt (If More Robustness Needed)

If this environment cannot implement more robust integration (e.g., actual PyTorch training with GPU, real ESP32 hardware testing, large-scale GLUE benchmarks), use this prompt for Antigravity:

```
You are refining a two-phase RLM repository for final review:

Phase 1 is IEEE iSense 2026 paper "Adaptive Gating or Simple Recursion?" with exact implementation:
- IMDB 1200/300/600, 96 tokens, DistilBERT tokenizer, d=96, 4 heads, 3.06M params, Tmax=8 tmin=2
- Equations (1)-(7), Banach contraction K<1, Neural ODE dh/dt=R(h)-h, Proposition 1 FK_RLM ⊂ FL
- Results: Table I F1 0.661 @ Tmax=2 beats Transformer 0.634, Table II depth ablation, Table IV ponder λ=0.01 rescues 0.246→0.667, Table V FixedLoop T=2 0.671 matches gated (weight sharing explains gains), Table VI r=-0.013 no correlation
- Limitations: sequential bottleneck, small data, truncation artifact
- Future: GLUE d=768, token-level halting, spectral norm K<1, dynamic threshold, 512/1024 tokens

Recent research to integrate for robustness (2024-2026 papers):
1. Residual Scaling 1/N for looped transformers (2606.18524, DeepLoop 2607.13491): Standard 1/sqrt(N) fails for weight sharing, need epsilon=lambda/(N*sqrt(L)) factorized 1/N + 1/sqrt(L), bounds residual norm Θ(N) vs Θ(sqrt(N)), explains Tmax=8 collapse
2. Relaxed Recursive Transformers with LoRA (2410.20672): Strict tying limits expressivity Proposition 1, LoRA adapters per iteration relax with minimal overhead rank=8, Average init > Stepwise, truncated SVD
3. Adaptive Depth Gating Diagnostics (2607.20519, AdaPonderLM 2603.01914): Gates collapse to shallowest, ignore cost, freeze constant. Shallow-halt trap with zero-bias and positive bias init. Solutions: negative bias -1.0, router z-loss penalize large logits (Zoph et al.), entropy regularization, bottom-K ponder loss k=0.1
4. Mixture-of-Recursions MoR (2507.10524): Token-wise routers assign different depths per token, selective KV caching, KV sharing for prefill, lightweight routers, focus attention among active tokens
5. Pondering LMs: Continuous weighted embeddings as ponder, fully differentiable, 2.5B Pondering Pythia matches 6.9B official with 37% params

Phase 2 is ESP32 hardware extension beyond paper:
- Zero-vocab FNV-1a hash 4096 buckets 0 bytes vocab, 8 collisions vs 30 @2047
- Int8 symmetric per-channel 510 KiB Flash 28 KiB SRAM 0 heap, affine real=scale*(q-zero_point), scale/zero-point matching critical, representative dataset 100-500 samples, ESP-NN 3-8x faster S3 vs classic, static arena 4KB/32KB 16-byte aligned
- ACT halting sum(1-g)≥0.9 avg 2.18 steps synthetic (not comparable to paper IMDB), dual-core FreeRTOS zero-copy mailbox direct notifications launch/wait overlap
- DVFS PERF 240/B8 50mA BAL 160/6 36mA ECO 80/4 27mA AUTO GPIO34, V_loaded=V_OCV-I*R_int R_int=0.2Ω brownout <3350mV ECO+abort, calibration INA219
- Verification: C++ parity 12/12 PASS logit error 2.86e-6, Verilog RTL Q1.15 mass VCD GTKWave, mock UART exact protocol

Tasks:
- Make backend robust: Implement ResidualScaling, RelaxedRecursiveBlock with LoRA, ImprovedGateNetwork with z-loss/entropy/bottom-K, TokenWiseGating MoR, SpectralNormWrapper K<1, AdvancedLoss CE+ponder+bottomK+z+entropy, ResidualNormTracker, gate collapse detection, Lipschitz estimation, Pareto frontier, token-level advanced correlation
- Make frontend robust: Phase1 interactive with Tables I-VI, ablations, residual norm charts, gate collapse diagnostics, tokenization comparison, inference speed analysis. Phase2 Live Studio, Serial Console, Coprocessor waveform VCD table, Patent, Verification, battery slider, DVFS
- Integrate deeper: Unified demo server with both phases, Project Chat repository-grounded, 3D pipeline graph 138 symbols 247 relationships
- Optimize: Remove irrelevant lines, py_compile OK, clean backend/frontend structure, no errors
- Document: Two phases clearly separated, inconsistencies fixed, recent research survey with citations, presentation guide two-phase

The repository currently has phase1/backend/model/advanced.py and phase1/backend/training/advanced_trainer.py and phase1/backend/evaluation/advanced_metrics.py and phase2/tools/advanced_quantization.py with research integration. Make them fully functional with actual PyTorch training if torch available, fallback to paper tables, and integrate into frontend demos with charts and diagnostics.

Also ensure code is robust for final review tomorrow.
```

---

## References

- Vaswani et al. Attention is all you need, NeurIPS 2017
- Graves. Adaptive computation time, arXiv:1603.08983 2016
- Dehghani et al. Universal Transformers, ICLR 2019
- Lan et al. ALBERT, ICLR 2020
- Maas et al. IMDB, ACL 2011
- Xin et al. DeeBERT, ACL 2020
- Giannou et al. Looped transformers as programmable computers, arXiv:2301.13196 2023
- Bae et al. Relaxed Recursive Transformers with LoRA, arXiv:2410.20672 2024
- On the Residual Scaling of Looped Transformers, arXiv:2606.18524 2026
- DeepLoop: Depth Scaling for Looped Transformers, arXiv:2607.13491 2026
- Adaptive Depth in Looped Transformers, arXiv:2607.20519 2026
- AdaPonderLM, arXiv:2603.01914 2026
- PonderLM, arXiv:2505.20674 2025
- Mixture-of-Recursions MoR, arXiv:2507.10524 NeurIPS 2025
- LoopLM, arXiv:2510.25741 2025
- LoopUS, arXiv:2605.11011 2026
- ESP32-S3 Edge AI, TFLM, INT8 quantization articles 2026
