#!/usr/bin/env python3
"""
Phase 1 Training Script - Paper Replication
Implements full training pipeline if torch available, else uses paper tables
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1.backend.model.config import RLMConfig
from phase1.backend.pipeline.pipeline import RLMPipeline, run_paper_replication

def main():
    parser = argparse.ArgumentParser(description="Phase 1 RLM Training - Paper Replication")
    parser.add_argument("--quick", action="store_true", help="Use paper tables, no training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["auto", "hf", "synthetic"], default="auto")
    parser.add_argument("--tmax", type=int, default=8)
    parser.add_argument("--ponder", type=float, default=0.0, help="Ponder lambda")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    if args.quick:
        print("[train] Quick mode - paper tables")
        results = run_paper_replication(quick=True)
        print(f"[train] Tables loaded: {list(results['paper_tables'].keys())}")
        return

    config = RLMConfig(
        max_recursion=args.tmax,
        ponder_lambda=args.ponder,
        epochs=args.epochs,
        seed=args.seed
    )

    try:
        import torch
        print(f"[train] Torch available {torch.__version__}, running full pipeline")
        pipeline = RLMPipeline(config)
        results = pipeline.run_full_pipeline(seed=args.seed, source=args.dataset)
        out_path = pipeline.save_results()
        print(f"[train] Results saved to {out_path}")

        # Print key tables
        print("\n=== Paper Replication Summary ===")
        for key, table in results.get("paper_tables", {}).items():
            print(f"\n{key}: {table['title']}")
            for row in table["rows"][:3]:
                print(f"  {row}")

    except ImportError as e:
        print(f"[train] Torch not available ({e}), falling back to paper tables")
        results = run_paper_replication(quick=True)
        print("[train] Use --quick for paper tables, or pip install torch transformers datasets")

if __name__ == "__main__":
    main()
