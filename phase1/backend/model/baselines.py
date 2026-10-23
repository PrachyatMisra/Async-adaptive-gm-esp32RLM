"""
Phase 1 Baselines - Transformer and BiLSTM
Paper Section IV.C and Table I
"""
from typing import Dict

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import BaselineConfig, BiLSTMConfig

if TORCH_AVAILABLE:
    class TransformerBaseline(nn.Module):
        """2-layer distinct transformer - 3.16M params, F1 0.634"""
        def __init__(self, config: BaselineConfig):
            super().__init__()
            self.config = config
            d = config.hidden_dim

            self.embedding = nn.Embedding(config.vocab_size, d, padding_idx=0)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=config.num_heads,
                    dim_feedforward=config.ffn_dim,
                    dropout=config.dropout,
                    batch_first=True
                ) for _ in range(config.num_layers)
            ])
            self.classifier = nn.Sequential(
                nn.Linear(d, config.gate_hidden_dim),
                nn.GELU(),
                nn.Linear(config.gate_hidden_dim, config.num_classes)
            )
            self.norm = nn.LayerNorm(d)

        def forward(self, input_ids, attention_mask=None):
            h = self.embedding(input_ids)
            # Create padding mask for transformer
            if attention_mask is not None:
                padding_mask = (attention_mask == 0)
            else:
                padding_mask = None

            for layer in self.layers:
                h = layer(h, src_key_padding_mask=padding_mask)

            # Mean pooling
            if attention_mask is not None:
                mask_exp = attention_mask.unsqueeze(-1).float()
                pooled = (h * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
            else:
                pooled = h.mean(1)

            logits = self.classifier(self.norm(pooled))
            return logits, {"steps": self.config.num_layers, "depth": float(self.config.num_layers)}

    class BiLSTMBaseline(nn.Module):
        """2-layer BiLSTM - 3.05M params, F1 0.635"""
        def __init__(self, config: BiLSTMConfig):
            super().__init__()
            self.config = config
            d = config.hidden_dim

            self.embedding = nn.Embedding(config.vocab_size, d, padding_idx=0)
            self.lstm = nn.LSTM(
                input_size=d,
                hidden_size=config.lstm_hidden,
                num_layers=config.lstm_layers,
                bidirectional=True,
                batch_first=True,
                dropout=config.dropout if config.lstm_layers > 1 else 0
            )
            self.classifier = nn.Sequential(
                nn.Linear(config.lstm_hidden * 2, config.gate_hidden_dim),
                nn.GELU(),
                nn.Linear(config.gate_hidden_dim, config.num_classes)
            )

        def forward(self, input_ids, attention_mask=None):
            h = self.embedding(input_ids)
            lstm_out, _ = self.lstm(h)

            if attention_mask is not None:
                mask_exp = attention_mask.unsqueeze(-1).float()
                pooled = (lstm_out * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
            else:
                pooled = lstm_out.mean(1)

            logits = self.classifier(pooled)
            return logits, {"steps": self.config.lstm_layers, "depth": float(self.config.lstm_layers)}

else:
    class TransformerBaseline:
        def __init__(self, config): self.config = config
    class BiLSTMBaseline:
        def __init__(self, config): self.config = config

def get_baseline_results():
    """Paper Table I results"""
    return {
        "transformer_2l": {"acc": 0.648, "prec": 0.647, "rec": 0.622, "f1": 0.634, "params": "3.16M"},
        "bilstm_2l": {"acc": 0.620, "prec": 0.600, "rec": 0.673, "f1": 0.635, "params": "3.05M"},
        "rlm_tmax2": {"acc": 0.543, "prec": 0.520, "rec": 0.908, "f1": 0.661, "params": "3.06M"},
        "rlm_tmax8": {"acc": 0.562, "prec": 0.782, "rec": 0.146, "f1": 0.246, "params": "3.06M", "note": "Gate collapse"},
    }
