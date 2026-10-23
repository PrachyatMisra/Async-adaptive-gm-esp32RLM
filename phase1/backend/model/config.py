"""
Phase 1 - RLM Configuration
Based on IEEE iSense 2026 paper: Adaptive Gating or Simple Recursion?
Implements exact hyperparameters from Section IV.C
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class RLMConfig:
    """Exact config from paper Section IV.C"""
    # Tokenization
    max_seq_len: int = 96
    tokenizer_name: str = "distilbert-base-uncased"

    # Architecture - matches paper 3.06M params
    hidden_dim: int = 96
    num_heads: int = 4
    ffn_expansion: int = 4  # 4x expansion -> 384
    num_classes: int = 2
    vocab_size: int = 30522  # DistilBERT vocab

    # Recursion - paper Tmax=8, tmin=2
    max_recursion: int = 8
    min_recursion: int = 2
    halt_threshold: float = 0.5  # g(t) < tau halts after tmin

    # Gate network dimensions - W1 in R^{d x d/2}, W2 in R^{d/2 x 1}
    @property
    def gate_hidden_dim(self) -> int:
        return self.hidden_dim // 2

    @property
    def ffn_dim(self) -> int:
        return self.hidden_dim * self.ffn_expansion

    # Training - paper Section IV.C
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 4
    batch_size: int = 32
    seed: int = 42
    dropout: float = 0.1

    # Ponder penalty - Section V.B Phase 2 Enhancements
    ponder_lambda: float = 0.0  # 0.0 baseline, 0.01 rescues Tmax=8

    # Dataset - paper Section IV.B
    train_samples: int = 1200
    val_samples: int = 300
    test_samples: int = 600

@dataclass
class BaselineConfig(RLMConfig):
    """2-layer Transformer baseline - 3.16M params"""
    num_layers: int = 2

@dataclass
class BiLSTMConfig(RLMConfig):
    """BiLSTM baseline - 3.05M params"""
    lstm_layers: int = 2
    lstm_hidden: int = 96

@dataclass
class TrainingConfig:
    """Training pipeline config"""
    device: str = "cpu"  # paper: CPU training
    use_amp: bool = True
    grad_clip: float = 1.0
    eval_every: int = 1
    save_dir: str = "phase1/experiments/results"
