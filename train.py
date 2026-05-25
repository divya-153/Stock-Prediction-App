"""
train.py — CLI training script.

Usage:
    python train.py --ticker RELIANCE.NS
    python train.py --ticker RELIANCE.NS --no-optuna --epochs 10
    python train.py --ticker AAPL --trials 30 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure src/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_ohlcv
from src.trainer import train
from src.utils import MODELS_DIR, get_logger, set_global_seed

log = get_logger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NSE/Global stock OHLC predictor.")
    p.add_argument("--ticker",     required=True,       help="yfinance ticker, e.g. RELIANCE.NS")
    p.add_argument("--no-optuna",  action="store_true", help="Skip Optuna tuning (use defaults)")
    p.add_argument("--trials",     type=int, default=20, help="Optuna trials per model (default 20)")
    p.add_argument("--epochs",     type=int, default=20, help="Deep model max epochs (default 20)")
    p.add_argument("--emb-dim",    type=int, default=64, help="Embedding dimension (default 64)")
    p.add_argument("--no-wf",      action="store_true", help="Skip walk-forward evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed()

    log.info("=== Training pipeline START === ticker=%s", args.ticker)

    # 1. Load data
    log.info("Loading OHLCV data for %s…", args.ticker)
    df_raw = load_ohlcv(args.ticker)

    # 2. Run training
    result = train(
        ticker          = args.ticker,
        df_raw          = df_raw,
        use_optuna      = not args.no_optuna,
        optuna_trials   = args.trials,
        deep_epochs     = args.epochs,
        embedding_dim   = args.emb_dim,
        run_walk_forward= not args.no_wf,
    )

    # 3. Print summary
    log.info("=== Training pipeline END ===")
    print("\n── Walk-Forward Metrics ──")
    for fold in result["fold_metrics"]:
        print(
            f"  Fold {fold['fold_id']} (test {fold['test_year']}) | "
            + " | ".join(f"{k.replace('target_','')}: {v:.4f}" for k, v in fold["mae"].items())
        )

    print("\n── Best Deep HPs ──")
    print(json.dumps(result["best_deep_hp"], indent=2))

    print("\n── Best LightGBM HPs ──")
    for tgt, hp in result["best_lgbm_hp"].items():
        print(f"  [{tgt}]  {json.dumps(hp)}")

    print(f"\nModels saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
