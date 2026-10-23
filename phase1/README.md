# Phase 1: Recursive Language Model (RLM) - IEEE iSense 2026 Paper Implementation

**Paper Title:** Adaptive Gating or Simple Recursion? Isolating Weight-Sharing Gains in Data-Scarce Language Modeling
**Authors:** Prachyat Misra, Prof. Parvathy AK - VIT Chennai
**Venue:** IEEE iSense 2026 (Accepted)
**Status:** Paper-approved for final review presentation

---

## Overview

Phase 1 implements the **exact RLM architecture from the accepted paper**, isolating weight-sharing gains from adaptive gating in data-scarce language modeling.

**Core Research Question:** Why do conventional models employ all layers and heavy computations even when the answer is straightforward?

**Proposed Solution:** Replace L distinct transformer layers with a single shared block applied recurrently, governed by a learned scalar gate that halts recursion when representation converges.

---

## Key Findings (From Paper)

### Main Results - Table I

| Model | Acc | Prec | Rec | F1 | Params |
|-------|-----|------|-----|----|--------|
| Transformer (2L) | 0.648 | 0.647 | 0.622 | **0.634** | 3.16M |
| BiLSTM (2L) | 0.620 | 0.600 | 0.673 | **0.635** | 3.05M |
| **RLM (Tmax=2)** | 0.543 | 0.520 | 0.908 | **0.661** | 3.06M |
| RLM (Tmax=8) | 0.562 | 0.782 | 0.146 | **0.246** | 3.06M |

**Finding:** RLM peaks at Tmax=2 with F1=0.661, outperforming baselines under controlled 3.06M budget. Tmax=8 collapses to F1=0.246 due to gate saturation.

### Recursion Depth Ablation - Table II

| Tmax | Acc | F1 | Deff | Halt% | Time |
|------|-----|----|------|-------|------|
| 1 | 0.623 | 0.636 | 0.918 | 0% | 3.2s |
| **2** | 0.543 | **0.661** | 1.276 | 0% | 5.2s |
| 4 | 0.632 | 0.592 | 0.957 | 100% | 6.7s |
| 8 | 0.562 | 0.246 | 0.782 | 100% | 6.3s |

Contrary to expectation, performance peaks at Tmax=2, not monotonically with budget.

### Ponder Penalty - Table IV (Phase 2 Enhancement)

| Tmax | λ | Acc | F1 | Deff | g_bar |
|------|---|-----|----|------|-------|
| 8 | 0.00 | 0.562 | 0.246 | 0.781 | 0.391 |
| **8** | **0.01** | 0.558 | **0.667** | **0.054** | 0.027 |
| 8 | 0.05 | 0.532 | 0.146 | 0.023 | 0.011 |
| 2 | 0.00 | 0.543 | 0.661 | 1.276 | 0.638 |

**Rescue:** λ=0.01 successfully rescues Tmax=8: F1 0.246→0.667 while Deff drops to 0.054.

### Fixed-Loop Isolation - Table V

| Model | Acc | F1 | Prec | Rec |
|-------|-----|----|------|-----|
| FixedLoop (T=1) | 0.623 | 0.641 | 0.601 | 0.687 |
| **FixedLoop (T=2)** | 0.547 | **0.671** | 0.521 | 0.946 |
| FixedLoop (T=4) | 0.525 | 0.084 | 0.765 | 0.044 |
| RLM (Tmax=2, gated) | 0.543 | 0.661 | 0.520 | 0.908 |

**Crucial:** FixedLoop T=2 F1=0.671 matches gated RLM F1=0.661. Proves **weight sharing alone explains gains**, not adaptive gating at this scale.

### Token-Level Gate Correlation - Table VI

| Group | Mean Gate | n |
|-------|-----------|---|
| High sentiment density (ρ>0.05) | 0.659 | 10 |
| Low sentiment density (ρ≤0.02) | 0.639 | 444 |
| Correct predictions | 0.639 | 326 |

**Pearson r=-0.013 p=0.754** - No correlation between sentiment word density and gate values.

---

## Architecture - Paper Section III

### Theoretical Framework

**Standard Transformer vs Weight-Shared Recurrence:**

```
h~(t+1) = R(h(t); θ)                    (1) - shared block
h(t+1) = g(t)*h~(t+1) + (1-g(t))*h(t)   (2) - soft mixing
g(t) = σ(W2·tanh(W1·MeanPool(h~(t+1))+b1)+b2)  (3) - gate
Deff(x) = Σ g(t)                        (4) - effective depth
```

**Fixed-Point Convergence (Theorem 1 Banach Contraction):**

States where gate halts (g(h*)=0) or representation converges (R(h*)=h*) are stable attractors. If block is Lipschitz K<1, iterates converge to unique fixed point.

**Neural ODE Connection:**

```
h(t+1) = h(t) + g(t)*(R(h(t))-h(t))    (6)
dh/dt = R(h;θ) - h                      (7) - forward Euler discretization
```

Fixed point R(h*)=h* corresponds to equilibrium dh/dt=0.

**Expressivity Bounds (Proposition 1):**

FK_RLM ⊂ FL for L=K. Weight-sharing limits expressivity (cannot have layer-specialized attention patterns like W_Q=I and W_Q=-I in different layers) but provides stabilizing inductive bias.

---

## Dataset - Paper Section IV.B

- **Source:** Balanced subset of IMDB movie reviews [Maas et al. 2011]
- **Split:** 1200 train / 300 val / 600 test = 2100 total
- **Preprocessing:** Truncated/padded to 96 tokens using DistilBERT tokenizer
- **Controlled:** Seed 42, CPU training, Python 3.10, PyTorch 2.x, HuggingFace

**Truncation Artifact Hypothesis:** 96-token truncation cuts reviews where most retain positive tone, negative sentiment often appears later - explains positive-recall bias.

---

## Training - Paper Section IV.C

- **Hidden dim:** d=96
- **Transformer baseline:** 2 distinct encoder layers, 4 heads, FFN 4×, 3.16M params
- **RLM:** Shares 1 recursive block, 3.06M params, Tmax=8, tmin=2
- **BiLSTM:** 2 bidirectional layers, 3.05M params
- **Optimizer:** AdamW lr=5×10⁻⁴ for RLM, wd=10⁻⁴
- **Epochs:** 4, Seed 42, CPU
- **Tricks:**
  - Without tmin, model halts after single step (local minimum)
  - Adding tanh before sigmoid prevents gates saturating to near-zero in early epochs
  - Loss curve noisier than transformer - gates dynamically adjusting

---

## Tokenization Analysis

### Paper Choice: DistilBERT Tokenizer

**Advantages:**
- Subword handling: OOV words split into meaningful pieces
- Pretrained alignment: leverages DistilBERT pretraining
- Standard benchmark: comparable with literature
- Context preservation: 96 tokens captures most IMDB reviews

**Disadvantages:**
- Vocab storage: ~100KB+ table
- Not suitable for 520KB SRAM microcontrollers
- Requires HuggingFace ecosystem

**Paper Rationale:** Controlled academic evaluation with standard tokenizer.

### Phase 2 Extension: FNV-1a Hash Tokenizer

**For ESP32 deployment (Phase 2):**
- Zero vocab storage: 0 bytes dictionary
- Constant memory, streaming-friendly
- Deterministic C and Python identical implementations
- Collision: ~8 pairs with 4096 buckets vs 30 with 2047

**Tradeoff:** Phase 1 targets research understanding, Phase 2 targets edge deployment.

---

## Inference Speed Analysis

### Paper Observations

| Tmax | Time | Deff | Halt% | F1 |
|------|------|------|-------|----|
| 1 | 3.2s | 0.918 | 0% | 0.636 |
| 2 | 5.2s | 1.276 | 0% | 0.661 |
| 8 | 6.3s | 0.782 | 100% | 0.246 |

- Training times reflect sequential bottleneck: recurrent application of R cannot be parallelized
- Model maintains accuracy while significantly reducing computational time (paper claim)
- Optimal depth 2 maintains acceptable costs while preserving context

### Theoretical Speedup

- **Weight sharing:** 8× parameter reduction vs 8-layer distinct transformer
- **Adaptive halting:** Early halt at Tmax=2 saves 75% steps vs fixed T=8
- **Bottleneck:** Sequential recursion prevents parallelization - future work: token-level halting, spectral normalization

---

## Repository Structure

```
phase1/
├── backend/
│   ├── model/
│   │   ├── config.py          # RLMConfig matching paper Section IV.C
│   │   ├── tokenizer.py       # DistilBERT wrapper + fallback
│   │   ├── rlm.py             # RecursiveLanguageModel exact paper equations
│   │   └── baselines.py       # Transformer 2L and BiLSTM baselines
│   ├── training/
│   │   ├── dataset.py         # IMDB 1200/300/600 subset
│   │   └── trainer.py         # AdamW + ponder penalty + AMP
│   ├── evaluation/
│   │   ├── metrics.py         # Acc, Prec, Rec, F1, Bootstrap CI
│   │   └── ablation.py        # Tmax, ponder, fixed-loop, token-level
│   └── pipeline/
│       └── pipeline.py        # End-to-end pipeline
├── frontend/
│   ├── app.py                 # Interactive demo
│   └── templates/             # Visualization
├── data/                      # Dataset cache
├── experiments/
│   └── results/               # JSON results
└── README.md                  # This file
```

---

## Quickstart

### Installation

```bash
pip install torch transformers datasets
# or minimal
pip install -r requirements.txt
```

### Run Paper Replication (Quick - No Training)

```python
from phase1.backend.pipeline.pipeline import run_paper_replication

results = run_paper_replication(quick=True)
print(results["paper_tables"]["table_i_baseline"])
```

### Full Training (Requires Torch)

```python
from phase1.backend.model.config import RLMConfig
from phase1.backend.pipeline.pipeline import RLMPipeline

config = RLMConfig()
pipeline = RLMPipeline(config)
results = pipeline.run_full_pipeline(seed=42, source="hf")
pipeline.save_results()
```

### Interactive Demo

```bash
python phase1/frontend/app.py
# Open http://localhost:8000/phase1
```

### CLI Demo

```bash
python phase1/demo.py --text "the movie was brilliant and amazing"
```

---

## Limitations & Future Work (Paper Section VII)

**Limitations:**
- Sequential bottleneck: recurrent application cannot be parallelized
- Small training set (1200) prevents gate from learning non-trivial halting policies
- 96-token truncation artifact

**Future Work:**
- Evaluate on GLUE benchmarks with d=768
- Token-level halting for autoregressive generation
- Spectral normalization to formally guarantee contraction
- Optimize threshold dynamically rather than fixed hyperparameter
- Longer sequences (512/1024 tokens) for comprehensive gating demonstration

---

## Relation to Phase 2

**Phase 1:** Academic RLM with DistilBERT tokenizer, 3.06M params, F1 0.661 peak at Tmax=2, analysis of weight-sharing vs gating.

**Phase 2:** Hardware extension for ESP32:
- Zero-vocab FNV-1a hash tokenizer (0 bytes vocab)
- Int8 symmetric per-channel quantization (510 KiB Flash, 28 KiB SRAM)
- ACT-style halting: sum(1-g) >= 0.9, mean 2.18 steps (synthetic corpus)
- Dual-core FreeRTOS: compute on APP_CPU, arbiter on PRO_CPU
- Battery droop compensation, DVFS, cooperative abort
- Verilog coprocessor, golden parity harness

**Important:** Phase 2 results (92.33% accuracy, 72.8% energy reduction) are on synthetic lexicon-bootstrapped corpus, **not comparable** to Phase 1 IMDB results. Paper is Phase 1 only.

---

## References

[1] Vaswani et al. Attention is all you need, NeurIPS 2017
[2] Graves. Adaptive computation time, arXiv:1603.08983 2016
[3] Dehghani et al. Universal Transformers, ICLR 2019
[4] Lan et al. ALBERT, ICLR 2020
[5] Maas et al. Learning word vectors for sentiment analysis, ACL 2011
[6] Xin et al. DeeBERT, ACL 2020
[9] Giannou et al. Looped transformers as programmable computers, arXiv:2301.13196

---

## Presentation Checklist for Final Review

- [ ] Dataset: 1200/300/600 IMDB, 96 tokens, DistilBERT tokenizer, seed 42
- [ ] Architecture: d=96, 4 heads, FFN 384, shared block 3.06M vs Transformer 3.16M
- [ ] Equations: (1) recurrence, (2) soft mixing, (3) gate with tanh, (4) Deff
- [ ] Results: Table I F1 0.661 peak at Tmax=2, Table II depth ablation, Table IV ponder rescue 0.246→0.667, Table V fixed-loop 0.671 matches gated, Table VI r=-0.013
- [ ] Theory: Banach contraction, expressivity bounds, Neural ODE
- [ ] Limitations: sequential bottleneck, small data, truncation artifact
- [ ] Demo: Interactive inference showing gate values, Deff, halting

**Key Talking Point:** "The primary finding is gating mechanism learned optimal depth 2 and maintained it independently of allocated budget. Weight sharing explains gains over distinct-weight baselines. Ponder penalty mitigates collapse at high depths."
