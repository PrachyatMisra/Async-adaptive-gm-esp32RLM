"""
Phase 1 Trainer - with ponder penalty
Paper Section IV.C and V.B
"""
from typing import Dict, List, Optional
import time
import random

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..model.config import RLMConfig, TrainingConfig
from ..model.tokenizer import DistilBERTTokenizerWrapper

if TORCH_AVAILABLE:
    class IMDBDataset(Dataset):
        def __init__(self, texts: List[str], labels: List[int], tokenizer: DistilBERTTokenizerWrapper):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            enc = self.tokenizer.encode(self.texts[idx])
            return {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)
            }

    def train_epoch(model, loader, optimizer, device, ponder_lambda: float = 0.0):
        model.train()
        total_loss = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, trace = model(input_ids, attention_mask)
            loss, loss_dict = model.compute_loss(logits, labels, trace, ponder_lambda)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(model, loader, device):
        model.eval()
        all_preds, all_labels = [], []
        total_depth = 0
        total_steps = 0
        halt_count = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, trace = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            total_depth += trace["depth"].sum().item()
            total_steps += trace["steps_run"].sum().item() if isinstance(trace["steps_run"], torch.Tensor) else trace["steps"]
            if "halted" in trace:
                halt_count += trace["halted"].sum().item()

        # Metrics
        from ..evaluation.metrics import compute_metrics
        metrics = compute_metrics(all_labels, all_preds)
        n = len(all_labels)
        metrics["mean_depth"] = total_depth / n
        metrics["mean_steps"] = total_steps / n
        metrics["halt_fraction"] = halt_count / n if n else 0

        return metrics

else:
    class IMDBDataset:
        def __init__(self, *args, **kwargs): pass
    def train_epoch(*args, **kwargs): return 0.0
    def evaluate(*args, **kwargs): return {}

def get_training_pipeline():
    """Returns paper training pipeline description"""
    return {
        "optimizer": "AdamW lr=5e-4 wd=1e-4 (RLM)",
        "epochs": 4,
        "seed": 42,
        "device": "CPU (paper: controlled local environment)",
        "amp": "Automatic Mixed Precision for memory efficiency",
        "ponder_penalty": "L = L_task + lambda * sum g(t), lambda=0.01 rescues Tmax=8",
        "tmin_trick": "Without tmin, model halts after 1 step - local minimum",
        "tanh_stabilization": "Adding tanh before sigmoid prevents early saturation",
        "paper_observation": "RLM loss curve noisier than transformer - gates dynamically adjusting"
    }
