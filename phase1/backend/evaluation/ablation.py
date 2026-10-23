"""
Phase 1 Ablation Studies - Paper Section V
Implements all 3 enhancements from paper
"""
from typing import Dict, List

def ponder_penalty_ablation():
    """
    Paper Table IV - Ponder penalty rescues Tmax=8
    Loss = L_task + lambda * sum g(t)
    """
    return {
        "description": "Ponder penalty regularizes gating, forcing early halting to prevent collapse",
        "equation": "L_total = L_ce + lambda * (1/Tmax) * sum_{t=1}^{Tmax} g(t)",
        "results": [
            {"tmax": 8, "lambda": 0.00, "f1": 0.246, "deff": 0.781, "g_bar": 0.391, "status": "Collapsed"},
            {"tmax": 8, "lambda": 0.01, "f1": 0.667, "deff": 0.054, "g_bar": 0.027, "status": "Rescued - best"},
            {"tmax": 8, "lambda": 0.05, "f1": 0.146, "deff": 0.023, "g_bar": 0.011, "status": "Over-regularized"},
            {"tmax": 2, "lambda": 0.00, "f1": 0.661, "deff": 1.276, "g_bar": 0.638, "status": "Healthy gate"},
            {"tmax": 2, "lambda": 0.01, "f1": 0.275, "deff": 0.035, "g_bar": 0.018, "status": "Degraded - penalty hurts"},
        ],
        "key_finding": "At Tmax=8, lambda=0.01 rescues F1 0.246->0.667 while Deff drops to 0.054",
        "tradeoff": "Excessive penalty induces premature halting, negligible penalty reduces to standard transformer",
        "calibration": "Depth reduction curve aligns with confidence scores"
    }

def fixed_loop_isolation():
    """
    Paper Table V - Fixed-Loop Isolation Test
    Identical shared weights but no adaptive gate
    """
    return {
        "description": "Isolate gate's contribution vs weight-sharing",
        "method": "FixedLoop Transformer: same shared block, no gate, fixed T steps",
        "results": {
            "fixed_t1": {"acc": 0.623, "f1": 0.641, "note": "Single pass"},
            "fixed_t2": {"acc": 0.547, "f1": 0.671, "note": "Matches gated RLM - weight sharing is driver"},
            "fixed_t4": {"acc": 0.525, "f1": 0.084, "note": "Collapses aggressively - Lipschitz instability"},
            "gated_tmax2": {"acc": 0.543, "f1": 0.661, "note": "Gated version"},
            "gated_tmax8_lambda": {"acc": 0.558, "f1": 0.667, "note": "With ponder penalty rescue"},
        },
        "conclusion": "At 1200 training samples, performance benefit comes from weight-shared recursion, not adaptive gating",
        "open_question": "Why FixedLoop T=4 collapses to F1=0.084 even faster than gated RLM Tmax=8? Hypothesis: Lipschitz instability without gate-mediated residual pathways",
        "parameter_constrained": "Model is parameter-constrained rather than adaptation-constrained at this scale"
    }

def token_level_gate_analysis():
    """
    Paper Table VI - Token-level gate correlation
    """
    return {
        "method": "Curated lexicon, compute sentiment density rho_s = (#sentiment words)/len",
        "correlation": "Pearson r=-0.013 p=0.754 - no correlation",
        "groups": [
            {"group": "High sentiment density rho>0.05", "mean_gate": 0.659, "n": 10},
            {"group": "Low sentiment density rho<=0.02", "mean_gate": 0.639, "n": 444},
            {"group": "Correct predictions", "mean_gate": 0.639, "n": 326},
            {"group": "Incorrect predictions", "mean_gate": 0.637, "n": 274},
            {"group": "Class negative", "mean_gate": 0.632, "n": 306},
            {"group": "Class positive", "mean_gate": 0.644, "n": 294},
        ],
        "interpretation": "Gate does not simply correlate with lexical sentiment density",
        "outliers": "Complex sentences where model correctly avoids early halting - expected",
        "implication": "Gate may encode more abstract properties than keyword counting"
    }

def recursion_depth_ablation():
    """
    Paper Table II - Tmax ablation
    """
    return {
        "description": "Contrary to expectation that performance improves with recursion budget",
        "results": [
            {"tmax": 1, "acc": 0.623, "f1": 0.636, "deff": 0.918, "halt": "0%", "time": "3.2s"},
            {"tmax": 2, "acc": 0.543, "f1": 0.661, "deff": 1.276, "halt": "0%", "time": "5.2s", "peak": True},
            {"tmax": 4, "acc": 0.632, "f1": 0.592, "deff": 0.957, "halt": "100%", "time": "6.7s"},
            {"tmax": 6, "acc": 0.612, "f1": 0.639, "deff": 1.092, "halt": "100%", "time": "8.6s"},
            {"tmax": 8, "acc": 0.562, "f1": 0.246, "deff": 0.782, "halt": "100%", "time": "6.3s", "collapse": True},
        ],
        "finding": "Peak at Tmax=2, decreases at Tmax=4, collapses at Tmax=8",
        "gate_collapse": "At Tmax=8, 100% halt fraction but Deff only 0.782 - gate collapse",
        "optimal_stopping": "Model learns optimal depth 2 and maintains it independently of budget"
    }

def get_all_ablations():
    return {
        "ponder": ponder_penalty_ablation(),
        "fixed_loop": fixed_loop_isolation(),
        "token_level": token_level_gate_analysis(),
        "depth": recursion_depth_ablation()
    }
