"""
utils.py — Shared utilities: logging, seeds, constants, path helpers.

UPGRADED: Multi-window sequence lengths (30 / 60 / 120 days),
          risk model path helper, wicks model path helper.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Project layout
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
# Version is the feature count — forces retrain when FEATURE_COLS changes
_FEATURE_VERSION = "v67"   # update this whenever FEATURE_COLS changes
MODELS_DIR = PROJECT_ROOT / f"models_{_FEATURE_VERSION}"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SEED        : int = 42
FETCH_YEARS : int = 15

# Multi-window sequence lengths (short / mid / long term)
SEQ_SHORT : int = 30
SEQ_MID   : int = 60
SEQ_LONG  : int = 120
SEQUENCE_LEN = SEQ_SHORT     # kept for legacy callers (inference uses longest)

# Minimum rows needed before any sample can be built
MIN_HISTORY : int = SEQ_LONG + 1

WALK_FORWARD_FOLDS = [
    {"train_end": 2018, "test_year": 2019},
    {"train_end": 2019, "test_year": 2020},
    {"train_end": 2020, "test_year": 2021},
    {"train_end": 2021, "test_year": 2022},
    {"train_end": 2022, "test_year": 2023},
    {"train_end": 2023, "test_year": 2024},
    {"train_end": 2024, "test_year": 2025},
    # Event-window fold: exposes Optuna to the Jan–Apr 2026 spike regime
    {"train_end": 2025, "test_year": 2026},
]

# ─────────────────────────────────────────────────────────────────────────────
# Loss / reconstruction weights  (centralised so trainer + predictor agree)
# ─────────────────────────────────────────────────────────────────────────────

# target_close sample weight multiplier — 2× penalises close errors during training.
# The primary condition is: predicted close within ₹10 of actual close.
CLOSE_LOSS_WEIGHT: float = 2.0

# Risk-based range expansion factors applied at inference time.
# HIGH/EXTREME risk → widen the predicted High-Low range to avoid under-estimation
# on spike days.  LOW/MODERATE → no expansion.
RISK_RANGE_EXPANSION: dict[str, float] = {
    "LOW"     : 1.00,
    "MODERATE": 1.00,
    "HIGH"    : 1.25,
    "EXTREME" : 1.35,
}

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h   = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────
def _safe(ticker: str) -> str:
    return ticker.replace(".", "_").replace("^", "").replace("/", "_")

def model_path(ticker: str, suffix: str) -> Path:
    return MODELS_DIR / f"{_safe(ticker)}_{suffix}"

def scaler_path(ticker: str, name: str) -> Path:
    return model_path(ticker, f"scaler_{name}.joblib")

def lgbm_path(ticker: str, target: str) -> Path:
    return model_path(ticker, f"lgbm_{target}.joblib")

def risk_model_path(ticker: str) -> Path:
    return model_path(ticker, "lgbm_risk.joblib")

def deep_model_path(ticker: str) -> Path:
    return model_path(ticker, "deep_model.keras")

def embedder_path(ticker: str) -> Path:
    return model_path(ticker, "embedder.keras")
def spike_lgbm_path(ticker: str) -> Path:
    """Specialist LightGBM trained only on spike-day rows (range_expansion_ratio > 1.5)."""
    return model_path(ticker, "lgbm_spike_range.joblib")