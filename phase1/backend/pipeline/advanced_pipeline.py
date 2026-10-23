"""
Phase 1 Advanced Pipeline - Full robust integration with recent research
Combines paper replication + residual scaling + LoRA + gating diagnostics + tokenwise + spectral norm
"""
from typing import Dict, List
import json
import time
from pathlib import Path

from ..model.config import RLMConfig
from ..model.advanced import get_advanced_research_summary
from ..training.advanced_trainer import get_advanced_training_config
from ..evaluation.advanced_metrics import get_paper_tables_with_advanced
from ..training.dataset import load_imdb_subset, get_dataset_stats
from ..evaluation.metrics import get_paper_tables
from ..evaluation.ablation import get_all_ablations
from ..model.tokenizer import analyze_tokenization_advantages

class AdvancedRLMPipeline:
    """
    Advanced pipeline integrating all research (2024-2026) for robust backend
    """

    def __init__(self, config: RLMConfig = None):
        self.config = config or RLMConfig()
        self.results = {}

    def run_research_stage(self):
        print("[advanced] Stage: Recent research survey (2024-2026)")
        summary = get_advanced_research_summary()
        self.results["advanced_research"] = summary
        return summary

    def run_advanced_training_stage(self):
        print("[advanced] Stage: Advanced training config")
        config = get_advanced_training_config()
        self.results["advanced_training"] = config
        return config

    def run_advanced_metrics_stage(self):
        print("[advanced] Stage: Advanced metrics with diagnostics")
        tables = get_paper_tables_with_advanced()
        self.results["advanced_tables"] = tables
        return tables

    def run_tokenization_stage(self):
        print("[advanced] Stage: Tokenization deep analysis")
        analysis = analyze_tokenization_advantages()
        self.results["tokenization"] = analysis
        return analysis

    def run_full_advanced_pipeline(self, seed: int = 42, source: str = "synthetic"):
        start = time.time()
        print("="*70)
        print("Phase 1 Advanced RLM Pipeline - Paper + Recent Research (2024-2026)")
        print("="*70)

        # Dataset
        print(f"[advanced] Dataset loading seed={seed} source={source}")
        corpus = load_imdb_subset(seed=seed, n_train=self.config.train_samples,
                                  n_val=self.config.val_samples,
                                  n_test=self.config.test_samples, source=source)
        stats = get_dataset_stats(corpus)
        self.results["dataset"] = stats

        # Research
        self.run_research_stage()
        self.run_advanced_training_stage()
        self.run_advanced_metrics_stage()
        self.run_tokenization_stage()

        # Ablations
        print("[advanced] Stage: Ablations")
        ablations = get_all_ablations()
        self.results["ablations"] = ablations

        # Paper tables
        print("[advanced] Stage: Paper tables")
        tables = get_paper_tables()
        self.results["paper_tables"] = tables

        elapsed = time.time() - start
        self.results["pipeline"] = {
            "elapsed_seconds": round(elapsed, 2),
            "mode": "advanced - paper + recent research 2024-2026",
            "config": {
                "hidden_dim": self.config.hidden_dim,
                "max_seq_len": self.config.max_seq_len,
                "max_recursion": self.config.max_recursion,
                "min_recursion": self.config.min_recursion,
                "seed": seed
            },
            "research_integration": [
                "Residual scaling 1/N (2606.18524, DeepLoop 2607.13491) - explains Tmax=8 collapse",
                "Relaxed Recursive with LoRA (2410.20672) - addresses Proposition 1 expressivity",
                "Gating diagnostics (2607.20519, AdaPonderLM 2603.01914) - shallow-halt trap, z-loss, entropy, bottom-K",
                "Token-wise MoR (2507.10524) - lightweight routers, selective KV caching",
                "Spectral norm K<1 - Banach fixed-point guarantee (Future Work VII)"
            ],
            "robust_backend": [
                "ResidualScaling epsilon=lambda/(N*sqrt(L)) factorized",
                "RelaxedRecursiveBlock LoRA rank=8 Average init",
                "ImprovedGateNetwork bias -1.0 z-loss entropy bottom-K",
                "TokenWiseGating MoR selective KV",
                "SpectralNormWrapper Lipschitz K<1",
                "AdvancedLoss CE+ponder+bottomK+z+entropy",
                "ResidualNormTracker gate collapse detection",
                "Lipschitz estimation Pareto frontier"
            ]
        }

        print("="*70)
        print(f"[advanced] Done in {elapsed:.2f}s - Robust backend ready")
        print("="*70)
        return self.results

    def save_results(self, path: str = "phase1/experiments/results/advanced_pipeline.json"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"[advanced] Results saved to {p}")
        return p

def run_advanced_pipeline(quick: bool = True):
    pipeline = AdvancedRLMPipeline()
    results = pipeline.run_full_advanced_pipeline(quick=quick)
    return results

if __name__ == "__main__":
    run_advanced_pipeline()
