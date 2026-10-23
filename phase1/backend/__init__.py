"""
Phase 1 Backend - Robust integration of paper + recent research
"""
from .model.config import RLMConfig, BaselineConfig, BiLSTMConfig, TrainingConfig
from .model.tokenizer import DistilBERTTokenizerWrapper, analyze_tokenization_advantages
from .model.rlm import RecursiveLanguageModel, FixedLoopTransformer, get_model_stats
from .model.baselines import TransformerBaseline, BiLSTMBaseline, get_baseline_results

try:
    from .model.advanced import (
        ResidualScaling, LoRAAdapter, RelaxedRecursiveBlock,
        ImprovedGateNetwork, TokenWiseGating, AdvancedRLM,
        get_advanced_research_summary
    )
    from .training.advanced_trainer import AdvancedLoss, ResidualNormTracker, get_advanced_training_config
    from .evaluation.advanced_metrics import (
        estimate_lipschitz_constant, analyze_residual_growth,
        detect_gate_collapse, compute_pareto_frontier
    )
    ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"[phase1 backend] Advanced import failed: {e}")
    ADVANCED_AVAILABLE = False

from .training.dataset import load_imdb_subset, build_synthetic_imdb, get_dataset_stats
from .training.trainer import IMDBDataset, get_training_pipeline
from .evaluation.metrics import compute_metrics, bootstrap_ci, get_paper_tables
from .evaluation.ablation import get_all_ablations
from .pipeline.pipeline import RLMPipeline, run_paper_replication

__all__ = [
    "RLMConfig", "BaselineConfig", "BiLSTMConfig", "TrainingConfig",
    "DistilBERTTokenizerWrapper", "analyze_tokenization_advantages",
    "RecursiveLanguageModel", "FixedLoopTransformer", "get_model_stats",
    "TransformerBaseline", "BiLSTMBaseline", "get_baseline_results",
    "load_imdb_subset", "build_synthetic_imdb", "get_dataset_stats",
    "IMDBDataset", "get_training_pipeline",
    "compute_metrics", "bootstrap_ci", "get_paper_tables",
    "get_all_ablations",
    "RLMPipeline", "run_paper_replication",
    "ADVANCED_AVAILABLE"
]

if ADVANCED_AVAILABLE:
    __all__.extend([
        "ResidualScaling", "LoRAAdapter", "RelaxedRecursiveBlock",
        "ImprovedGateNetwork", "TokenWiseGating", "AdvancedRLM",
        "get_advanced_research_summary",
        "AdvancedLoss", "ResidualNormTracker", "get_advanced_training_config",
        "estimate_lipschitz_constant", "analyze_residual_growth",
        "detect_gate_collapse", "compute_pareto_frontier"
    ])
