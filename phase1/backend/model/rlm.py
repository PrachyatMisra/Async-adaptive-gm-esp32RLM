"""
Phase 1 - Recursive Language Model (RLM)
Exact implementation from IEEE iSense 2026 paper Section III

Equations:
(1) h~(t+1) = R(h(t); theta)  - shared recursive block
(2) h(t+1) = g(t) * h~(t+1) + (1-g(t)) * h(t)  - soft mixing
(3) g(t) = sigma(W2 * tanh(W1 * MeanPool(h~(t+1)) + b1) + b2)
(4) Deff(x) = sum g(t)  - effective depth
(6) h(t+1) = h(t) + g(t)*(R(h(t))-h(t))  - ODE view

Key paper findings implemented:
- Tmax=2 peaks at F1=0.661, Tmax=8 collapses to 0.246
- Ponder penalty lambda*sum g(t) rescues Tmax=8 to F1=0.667 at lambda=0.01
- Fixed-Loop (no gate) at T=2 achieves F1=0.671 matching gated RLM
- Token-level correlation r=-0.013 p=0.754 (no sentiment density correlation)
"""
from typing import Dict, List, Tuple, Optional
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from .config import RLMConfig

if TORCH_AVAILABLE:
    class RecursiveBlock(nn.Module):
        """
        Weight-shared recursive block R(·;θ)
        Single transformer layer: LayerNorm -> MHA -> Residual -> LayerNorm -> FFN -> Residual -> Refinement
        """
        def __init__(self, config: RLMConfig):
            super().__init__()
            self.config = config
            d = config.hidden_dim

            # Pre-attention norm
            self.ln1 = nn.LayerNorm(d)
            # Multi-head attention (4 heads, Dk=24, D=96)
            self.attn = nn.MultiheadAttention(d, config.num_heads, dropout=config.dropout, batch_first=True)
            # Post-attention norm
            self.ln2 = nn.LayerNorm(d)
            # FFN: 384-wide GELU (4x expansion)
            self.ffn = nn.Sequential(
                nn.Linear(d, config.ffn_dim),
                nn.GELU(),
                nn.Linear(config.ffn_dim, d),
                nn.Dropout(config.dropout)
            )
            # State refinement (paper mentions 0.1 gain)
            self.refinement = nn.Linear(d, d)

            # Gate network: W1 in R^{d x d/2}, W2 in R^{d/2 x 1} with tanh
            self.gate_network = nn.Sequential(
                nn.Linear(d, config.gate_hidden_dim),
                nn.Tanh(),  # Critical: paper notes tanh before sigmoid stabilizes training
                nn.Linear(config.gate_hidden_dim, 1)
            )

        def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None):
            """
            h: [B, L, D]
            mask: [B, L] attention mask
            Returns: h_new [B, L, D], gate [B], pooled [B, D]
            """
            B, L, D = h.shape

            # LayerNorm + Attention with residual
            h_norm = self.ln1(h)
            key_padding_mask = None
            if mask is not None:
                key_padding_mask = (mask == 0)  # True for padded positions

            attn_out, _ = self.attn(h_norm, h_norm, h_norm, key_padding_mask=key_padding_mask)
            h = h + attn_out

            # LayerNorm + FFN with residual
            h = h + self.ffn(self.ln2(h))

            # Pooled representation for gating - masked mean pooling (paper correction)
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()  # [B, L, 1]
                n_real = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
                pooled = (h * mask_expanded).sum(dim=1) / n_real  # [B, D]
            else:
                pooled = h.mean(dim=1)  # [B, D]

            # Gate: sigma(W2 * tanh(W1 * pooled + b1) + b2)
            gate = torch.sigmoid(self.gate_network(pooled)).squeeze(-1)  # [B]

            # State refinement: h + 0.1 * Linear(h)
            h_refined = h + 0.1 * self.refinement(h)

            return h_refined, gate, pooled

    class RecursiveLanguageModel(nn.Module):
        """
        RLM from paper Section III
        Replaces L distinct layers with single shared block applied recurrently
        """
        def __init__(self, config: RLMConfig):
            super().__init__()
            self.config = config
            d = config.hidden_dim

            # Embedding + Positional Encoding
            self.embedding = nn.Embedding(config.vocab_size, d, padding_idx=0)
            self.pos_encoding = self._build_sinusoidal_pe(config.max_seq_len, d)

            # Single shared recursive block
            self.block = RecursiveBlock(config)

            # Output classifier
            self.output_norm = nn.LayerNorm(d)
            self.classifier = nn.Sequential(
                nn.Linear(d, config.gate_hidden_dim),
                nn.GELU(),
                nn.Linear(config.gate_hidden_dim, config.num_classes)
            )

            self.dropout = nn.Dropout(config.dropout)

        @staticmethod
        def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0)  # [1, L, D]

        def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
            """
            input_ids: [B, L]
            attention_mask: [B, L]
            Returns: logits [B, num_classes], trace dict
            """
            B, L = input_ids.shape
            device = input_ids.device

            # Initial representation h(0) = Embed(x) + PE(x)
            h = self.embedding(input_ids)  # [B, L, D]
            pe = self.pos_encoding[:, :L, :].to(device)
            h = h + pe
            h = self.dropout(h)

            if attention_mask is None:
                attention_mask = (input_ids != 0).long()

            gates = []
            depths = torch.zeros(B, device=device)
            halted = torch.zeros(B, dtype=torch.bool, device=device)
            steps_run = torch.zeros(B, dtype=torch.long, device=device)

            # Recurrent application
            for t in range(self.config.max_recursion):
                # Only run for non-halted sequences
                running = ~halted
                if not running.any() and t >= self.config.min_recursion:
                    break

                steps_run += running.long()

                h_new, g_t, pooled = self.block(h, attention_mask)

                # Soft mixing: h(t+1) = g(t) * h~(t+1) + (1-g(t)) * h(t)
                g_expanded = g_t.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                h_next = g_expanded * h_new + (1.0 - g_expanded) * h

                # Only update running sequences
                h = torch.where(running.unsqueeze(-1).unsqueeze(-1), h_next, h)
                depths += running.float() * g_t
                gates.append(g_t.detach())

                # Halting condition: g(t) < tau after min depth
                if t + 1 >= self.config.min_recursion:
                    newly_halted = (g_t < self.config.halt_threshold) & running
                    halted = halted | newly_halted

            # Final classification - pooled output
            h_norm = self.output_norm(h)
            if attention_mask is not None:
                mask_exp = attention_mask.unsqueeze(-1).float()
                n_real = mask_exp.sum(dim=1).clamp(min=1)
                pooled_out = (h_norm * mask_exp).sum(dim=1) / n_real
            else:
                pooled_out = h_norm.mean(dim=1)

            logits = self.classifier(pooled_out)

            trace = {
                "depth": depths,  # Deff = sum g(t)
                "gates": torch.stack(gates) if gates else torch.zeros(0),
                "steps": len(gates),
                "steps_run": steps_run,
                "halted": halted,
                "ponder": depths.mean() / self.config.max_recursion
            }

            return logits, trace

        def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, trace: Dict, lambda_ponder: float = 0.0):
            ce_loss = F.cross_entropy(logits, labels)
            if lambda_ponder > 0:
                ponder_loss = lambda_ponder * trace["ponder"]
                return ce_loss + ponder_loss, {"ce": ce_loss.item(), "ponder": ponder_loss.item()}
            return ce_loss, {"ce": ce_loss.item()}

    class FixedLoopTransformer(nn.Module):
        """
        Fixed-Loop baseline - identical shared weights but no adaptive gate
        Paper Table V: FixedLoop(T=2) achieves F1=0.671 matching gated RLM
        Proves weight-sharing alone explains gains at this scale
        """
        def __init__(self, config: RLMConfig, fixed_steps: int = 2):
            super().__init__()
            self.config = config
            self.fixed_steps = fixed_steps
            d = config.hidden_dim

            self.embedding = nn.Embedding(config.vocab_size, d, padding_idx=0)
            self.pos_encoding = RecursiveLanguageModel._build_sinusoidal_pe(config.max_seq_len, d)
            self.block = RecursiveBlock(config)
            self.output_norm = nn.LayerNorm(d)
            self.classifier = nn.Sequential(
                nn.Linear(d, config.gate_hidden_dim),
                nn.GELU(),
                nn.Linear(config.gate_hidden_dim, config.num_classes)
            )
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
            B, L = input_ids.shape
            device = input_ids.device

            h = self.embedding(input_ids)
            pe = self.pos_encoding[:, :L, :].to(device)
            h = h + pe
            h = self.dropout(h)

            if attention_mask is None:
                attention_mask = (input_ids != 0).long()

            # Fixed T steps, no gating
            for t in range(self.fixed_steps):
                h_new, _, _ = self.block(h, attention_mask)
                h = h_new  # No soft mixing, just sequential

            h_norm = self.output_norm(h)
            if attention_mask is not None:
                mask_exp = attention_mask.unsqueeze(-1).float()
                n_real = mask_exp.sum(dim=1).clamp(min=1)
                pooled_out = (h_norm * mask_exp).sum(dim=1) / n_real
            else:
                pooled_out = h_norm.mean(dim=1)

            logits = self.classifier(pooled_out)
            trace = {
                "depth": torch.full((B,), float(self.fixed_steps), device=device),
                "steps": self.fixed_steps,
                "gates": torch.ones(self.fixed_steps, B, device=device)
            }
            return logits, trace

else:
    # Fallback mock classes for environments without torch
    class RecursiveBlock:
        def __init__(self, config): self.config = config
    class RecursiveLanguageModel:
        def __init__(self, config): self.config = config
    class FixedLoopTransformer:
        def __init__(self, config, fixed_steps=2):
            self.config = config
            self.fixed_steps = fixed_steps

def get_model_stats():
    """Returns model statistics matching paper"""
    return {
        "rlm": {
            "params": "3.06M",
            "hidden_dim": 96,
            "heads": 4,
            "ffn": 384,
            "max_recursion": 8,
            "min_recursion": 2,
            "paper_table": "Table I: RLM Tmax=2 Acc 0.543 Prec 0.520 Rec 0.908 F1 0.661"
        },
        "transformer_baseline": {
            "params": "3.16M",
            "layers": 2,
            "hidden_dim": 96,
            "heads": 4,
            "paper_table": "Table I: Transformer 2L Acc 0.648 Prec 0.647 Rec 0.622 F1 0.634"
        },
        "bilstm_baseline": {
            "params": "3.05M",
            "layers": 2,
            "bidirectional": True,
            "paper_table": "Table I: BiLSTM 2L Acc 0.620 Prec 0.600 Rec 0.673 F1 0.635"
        },
        "fixed_loop": {
            "paper_table": "Table V: FixedLoop T=2 Acc 0.547 F1 0.671 Prec 0.521 Rec 0.946",
            "finding": "Weight sharing alone explains gains, not adaptive gating at this scale"
        }
    }
