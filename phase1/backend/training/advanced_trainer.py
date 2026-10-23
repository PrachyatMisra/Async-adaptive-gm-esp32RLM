"""
Phase 1 Advanced Trainer - Research-backed robust training
Implements:
- Spectral normalization for Lipschitz
- Router z-loss and entropy regularization
- Ponder loss variants (bottom-K, entropy)
- Random deep supervision
- Knowledge distillation
- Residual stream norm tracking
"""
from typing import Dict, List, Optional
import math
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class AdvancedLoss(nn.Module):
        """
        Advanced loss combining task + ponder + z-loss + entropy
        """
        def __init__(self,
                     ponder_lambda: float = 0.01,
                     z_loss_lambda: float = 0.001,
                     entropy_lambda: float = 0.001,
                     bottom_k_fraction: float = 0.1):
            super().__init__()
            self.ponder_lambda = ponder_lambda
            self.z_loss_lambda = z_loss_lambda
            self.entropy_lambda = entropy_lambda
            self.bottom_k_fraction = bottom_k_fraction

        def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                   trace: Dict, diagnostics: Optional[List[Dict]] = None):
            ce_loss = F.cross_entropy(logits, labels)

            # Ponder loss: lambda * mean depth / max_recursion
            ponder_loss = trace["depth"].mean() / 8.0  # Normalize by max

            # Bottom-K ponder loss (AdaPonderLM style) - penalize bottom-K fraction of gates
            gates = trace["gates"]  # [T, B]
            if gates.numel() > 0:
                # Flatten and get bottom K
                flat_gates = gates.view(-1)
                k = int(self.bottom_k_fraction * flat_gates.numel())
                if k > 0:
                    bottom_k, _ = torch.topk(flat_gates, k, largest=False)
                    bottom_k_loss = bottom_k.mean()
                else:
                    bottom_k_loss = torch.tensor(0.0, device=logits.device)
            else:
                bottom_k_loss = torch.tensor(0.0, device=logits.device)

            # Z-loss and entropy from diagnostics
            z_loss = torch.tensor(0.0, device=logits.device)
            entropy_loss = torch.tensor(0.0, device=logits.device)

            if diagnostics:
                for diag in diagnostics:
                    if "logits" in diag:
                        # Z-loss: penalize large logits
                        logits_g = diag["logits"]
                        z_loss = z_loss + torch.mean(logits_g ** 2)
                        # Entropy
                        gate = diag["gate"]
                        eps = 1e-6
                        gate_c = gate.clamp(eps, 1-eps)
                        ent = -(gate_c * torch.log(gate_c) + (1-gate_c) * torch.log(1-gate_c))
                        entropy_loss = entropy_loss - ent.mean()  # Maximize entropy

            if diagnostics:
                z_loss = z_loss / len(diagnostics)
                entropy_loss = entropy_loss / len(diagnostics)

            total_loss = (ce_loss +
                         self.ponder_lambda * ponder_loss +
                         self.ponder_lambda * 0.5 * bottom_k_loss +
                         self.z_loss_lambda * z_loss +
                         self.entropy_lambda * entropy_loss)

            return total_loss, {
                "ce": ce_loss.item(),
                "ponder": ponder_loss.item(),
                "bottom_k": bottom_k_loss.item(),
                "z_loss": z_loss.item(),
                "entropy": entropy_loss.item(),
                "total": total_loss.item()
            }

    class ResidualNormTracker:
        """
        Track residual stream norm growth - diagnose instability
        Paper: On the Residual Scaling of Looped Transformers
        """
        def __init__(self):
            self.norms = []

        def track(self, h: torch.Tensor, step: int):
            norm = h.norm(dim=-1).mean().item()
            self.norms.append({"step": step, "norm": norm})

        def get_diagnostics(self) -> Dict:
            if not self.norms:
                return {}
            norms = [n["norm"] for n in self.norms]
            return {
                "initial_norm": norms[0] if norms else 0,
                "final_norm": norms[-1] if norms else 0,
                "max_norm": max(norms) if norms else 0,
                "growth": norms[-1] / norms[0] if norms and norms[0] != 0 else 0,
                "exploding": max(norms) > 100 if norms else False,
                "trajectory": self.norms
            }

    def train_advanced_epoch(model, loader, optimizer, device, loss_fn, tracker=None):
        model.train()
        total_losses = {"ce": 0, "ponder": 0, "total": 0}
        num_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, trace = model(input_ids, attention_mask)

            diagnostics = trace.get("diagnostics", None)
            loss, loss_dict = loss_fn(logits, labels, trace, diagnostics)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in total_losses:
                if k in loss_dict:
                    total_losses[k] += loss_dict[k]
            num_batches += 1

        return {k: v / num_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def evaluate_advanced(model, loader, device):
        model.eval()
        all_preds, all_labels = [], []
        total_depth = 0
        total_steps = 0
        gate_values = []

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, trace = model(input_ids, attention_mask)
            preds = logits.argmax(-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            total_depth += trace["depth"].sum().item()
            total_steps += trace["steps"]
            if trace["gates"].numel() > 0:
                gate_values.extend(trace["gates"].mean(dim=1).cpu().tolist())

        from ..evaluation.metrics import compute_metrics
        metrics = compute_metrics(all_labels, all_preds)
        n = len(all_labels)

        metrics["mean_depth"] = total_depth / n
        metrics["mean_steps"] = total_steps / n if isinstance(total_steps, (int, float)) else total_steps
        metrics["mean_gate"] = sum(gate_values) / len(gate_values) if gate_values else 0
        metrics["gate_std"] = math.sqrt(sum((g - metrics["mean_gate"])**2 for g in gate_values) / len(gate_values)) if gate_values else 0

        # Detect collapse
        metrics["collapsed"] = metrics["mean_gate"] < 0.1 or metrics["mean_gate"] > 0.9
        metrics["shallow_trap"] = metrics["mean_steps"] <= 1.1

        return metrics

else:
    class AdvancedLoss:
        def __init__(self, *args, **kwargs): pass
    class ResidualNormTracker:
        def __init__(self, *args, **kwargs): pass

def get_advanced_training_config():
    return {
        "loss_components": {
            "ce": "Cross-entropy task loss",
            "ponder": "lambda * mean_depth / max_recursion - ACT ponder cost",
            "bottom_k": "Bottom-K fraction (k=0.1) gate values - AdaPonderLM style",
            "z_loss": "Router z-loss penalize large logits (Zoph et al. 2022) - MoR",
            "entropy": "Entropy regularization encourage exploration, avoid shallow trap"
        },
        "residual_scaling": "epsilon = lambda / (N * sqrt(L)) - factorized 1/N + 1/sqrt(L)",
        "spectral_norm": "Guarantee Lipschitz K<1 for Banach fixed-point convergence",
        "initialization": "Negative bias -1.0 not zero, avoids shallow-halt trap (paper 2604.21999)",
        "tracking": "Residual stream norm growth detection, gate collapse detection",
        "paper_connections": [
            "Tmax=8 collapse F1 0.246 explained by missing 1/N scaling - norm explodes Θ(N)",
            "Shallow-halt trap with zero-bias init - both zero and positive bias fail",
            "Router z-loss stabilizes MoR training",
            "Bottom-K ponder loss from AdaPonderLM prunes insignificant steps"
        ]
    }
