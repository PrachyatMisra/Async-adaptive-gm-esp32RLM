"""
Phase 1 Metrics - Paper Tables I-VI
"""
from typing import List, Dict, Tuple
import math
import random

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Accuracy, Precision, Recall, F1"""
    tp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 0)

    acc = (tp + tn) / len(y_true) if y_true else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

def bootstrap_ci(y_true: List[int], y_pred: List[int], n_bootstrap: int = 1000, seed: int = 42) -> Dict:
    """Bootstrap 95% CI for F1 - Paper Table III"""
    rng = random.Random(seed)
    n = len(y_true)
    f1_scores = []

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n-1) for _ in range(n)]
        sample_true = [y_true[i] for i in indices]
        sample_pred = [y_pred[i] for i in indices]
        metrics = compute_metrics(sample_true, sample_pred)
        f1_scores.append(metrics["f1"])

    f1_scores.sort()
    mean = sum(f1_scores) / len(f1_scores)
    std = math.sqrt(sum((x - mean) ** 2 for x in f1_scores) / len(f1_scores))

    # 95% CI
    low_idx = int(0.025 * len(f1_scores))
    high_idx = int(0.975 * len(f1_scores))

    return {
        "mean": round(mean, 3),
        "std": round(std, 3),
        "ci_low": round(f1_scores[low_idx], 3),
        "ci_high": round(f1_scores[high_idx], 3),
        "n_bootstrap": n_bootstrap
    }

def get_paper_tables():
    """Exact tables from paper for verification"""
    return {
        "table_i_baseline": {
            "title": "BASELINE RESULTS (IMDB, 96 TOKENS, SEED=42)",
            "columns": ["Model", "Acc", "Prec", "Rec", "F1"],
            "rows": [
                ["Transformer (2L)", 0.648, 0.647, 0.622, 0.634],
                ["BiLSTM (2L)", 0.620, 0.600, 0.673, 0.635],
                ["RLM (Tmax=2)", 0.543, 0.520, 0.908, 0.661],
                ["RLM (Tmax=8)", 0.562, 0.782, 0.146, 0.246],
            ],
            "finding": "RLM peaks at Tmax=2 with F1=0.661, beating baselines. Tmax=8 collapses."
        },
        "table_ii_ablation": {
            "title": "RLM RECURSION-DEPTH ABLATION",
            "columns": ["Tmax", "Acc", "F1", "Deff", "Halt%", "Time"],
            "rows": [
                [1, 0.623, 0.636, 0.918, "0%", "3.2s"],
                [2, 0.543, 0.661, 1.276, "0%", "5.2s"],
                [4, 0.632, 0.592, 0.957, "100%", "6.7s"],
                [6, 0.612, 0.639, 1.092, "100%", "8.6s"],
                [8, 0.562, 0.246, 0.782, "100%", "6.3s"],
            ],
            "finding": "Performance peaks at Tmax=2, decreases at Tmax=4, collapses at Tmax=8"
        },
        "table_iii_bootstrap": {
            "title": "BOOTSTRAP 95% CONFIDENCE INTERVALS (F1)",
            "columns": ["Model", "F1", "Mean", "Std", "CI Low", "CI High"],
            "rows": [
                ["Transformer", 0.634, 0.024, 0.586, 0.678, ""],
                ["BiLSTM", 0.635, 0.022, 0.591, 0.680, ""],
                ["RLM (Tmax=8)", 0.246, 0.030, 0.187, 0.304, ""],
            ],
            "finding": "Transformer and BiLSTM overlap, RLM Tmax=8 significantly worse"
        },
        "table_iv_ponder": {
            "title": "PONDER PENALTY ABLATION: lambda VS PERFORMANCE",
            "columns": ["Tmax", "lambda", "Acc", "F1", "Deff", "g_bar"],
            "rows": [
                [8, 0.00, 0.562, 0.246, 0.781, 0.391],
                [8, 0.01, 0.558, 0.667, 0.054, 0.027],
                [8, 0.05, 0.532, 0.146, 0.023, 0.011],
                [8, 0.10, 0.530, 0.145, 0.020, 0.010],
                [2, 0.00, 0.543, 0.661, 1.276, 0.638],
                [2, 0.01, 0.560, 0.275, 0.035, 0.018],
                [2, 0.05, 0.553, 0.247, 0.016, 0.008],
            ],
            "finding": "lambda=0.01 rescues Tmax=8: F1 0.246->0.667, Deff drops to 0.054"
        },
        "table_v_fixedloop": {
            "title": "FIXED-LOOP VS GATED RLM",
            "columns": ["Model", "Acc", "F1", "Prec", "Rec"],
            "rows": [
                ["FixedLoop (T=1)", 0.623, 0.641, 0.601, 0.687],
                ["FixedLoop (T=2)", 0.547, 0.671, 0.521, 0.946],
                ["FixedLoop (T=4)", 0.525, 0.084, 0.765, 0.044],
                ["RLM (Tmax=2, gated)", 0.543, 0.661, 0.520, 0.908],
                ["RLM (Tmax=8, lambda=0.01)", 0.558, 0.667, 0.529, 0.901],
            ],
            "finding": "FixedLoop T=2 F1=0.671 matches gated RLM F1=0.661 - weight sharing explains gains"
        },
        "table_vi_token": {
            "title": "MEAN GATE VALUE BY INPUT COMPOSITION (Tmax=2)",
            "columns": ["Group", "Mean Gate g_bar", "n"],
            "rows": [
                ["High sentiment density (rho>0.05)", 0.659, 10],
                ["Low sentiment density (rho<=0.02)", 0.639, 444],
                ["Correct predictions", 0.639, 326],
                ["Incorrect predictions", 0.637, 274],
                ["Class: negative", 0.632, 306],
                ["Class: positive", 0.644, 294],
            ],
            "finding": "Pearson r=-0.013 p=0.754 no correlation between sentiment density and gate"
        }
    }

def token_level_correlation_analysis():
    """Paper Section V.B.3 - token-level gate correlation"""
    return {
        "method": "Curated lexicon to compute sentiment density rho_s vs gate g",
        "result": "r=-0.013 p=0.754",
        "interpretation": "Gate does not correlate with lexical sentiment density",
        "outliers": "Complex sentences where model correctly avoids early halting",
        "implication": "Gate learns abstract halting policy, not simple keyword counting"
    }
