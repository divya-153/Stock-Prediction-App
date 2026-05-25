"""
sequence_builder.py — Multi-window sequence builder (30 / 60 / 120 days).

Zero leakage: for each sample i, the window ending at row (i + SEQ_LONG - 1)
feeds into all three branch inputs; the target is row (i + SEQ_LONG).

Why three windows?
  SEQ_SHORT (30d) → LSTM captures recent momentum / microstructure
  SEQ_MID   (60d) → LSTM captures monthly seasonality
  SEQ_LONG (120d) → Transformer captures regime shifts / macro trends

At inference, we build each window from the latest available rows.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.feature_engineering import FEATURE_COLS, TARGET_COLS
from src.utils import SEQ_LONG, SEQ_MID, SEQ_SHORT, get_logger

log = get_logger(__name__)

# Alias so callers can import a single "max window" constant
MAX_SEQ = SEQ_LONG


# ─────────────────────────────────────────────────────────────────────────────
# Training-time builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    feature_scaler: RobustScaler | None = None,
    target_scalers: dict[str, RobustScaler] | None = None,
    fit_scalers: bool = True,
) -> Tuple[
    dict[str, np.ndarray],     # X_seq: {"short": (N,30,F), "mid": (N,60,F), "long": (N,120,F)}
    np.ndarray,                # X_flat (N, F)  — last step of long window, scaled
    dict[str, np.ndarray],     # y_dict
    RobustScaler,              # feature_scaler
    dict[str, RobustScaler],   # target_scalers
    pd.DatetimeIndex,          # dates aligned with each sample
]:
    n_rows = len(df)
    if n_rows < MAX_SEQ + 1:
        raise ValueError(
            f"Only {n_rows} rows — need at least {MAX_SEQ + 1} for multi-window sequences."
        )

    feature_arr = df[FEATURE_COLS].values.astype(np.float32)

    # ── Fit / apply feature scaler ────────────────────────────────────────────
    if feature_scaler is None:
        feature_scaler = RobustScaler()
    if fit_scalers:
        fa = feature_scaler.fit_transform(feature_arr)
    else:
        fa = feature_scaler.transform(feature_arr)

    # ── Fit / apply target scalers ────────────────────────────────────────────
    if target_scalers is None:
        target_scalers = {c: RobustScaler() for c in TARGET_COLS}
    y_raw    = {c: df[c].values.astype(np.float32).reshape(-1, 1) for c in TARGET_COLS}
    y_scaled = {}
    for c in TARGET_COLS:
        if fit_scalers:
            y_scaled[c] = target_scalers[c].fit_transform(y_raw[c]).ravel()
        else:
            y_scaled[c] = target_scalers[c].transform(y_raw[c]).ravel()

    # ── Build three windows aligned on the same sample set ────────────────────
    n_samples  = n_rows - MAX_SEQ
    n_features = fa.shape[1]

    X_short = np.zeros((n_samples, SEQ_SHORT, n_features), dtype=np.float32)
    X_mid   = np.zeros((n_samples, SEQ_MID,   n_features), dtype=np.float32)
    X_long  = np.zeros((n_samples, SEQ_LONG,  n_features), dtype=np.float32)
    X_flat  = np.zeros((n_samples, n_features),            dtype=np.float32)

    for i in range(n_samples):
        end = i + MAX_SEQ          # exclusive end; row at `end` is the target
        X_long[i]  = fa[end - SEQ_LONG : end]
        X_mid[i]   = fa[end - SEQ_MID  : end]
        X_short[i] = fa[end - SEQ_SHORT: end]
        X_flat[i]  = fa[end - 1]   # latest row before the target

    X_seq = {"short": X_short, "mid": X_mid, "long": X_long}

    # Targets: row at index MAX_SEQ onwards
    y_dict = {c: y_scaled[c][MAX_SEQ:] for c in TARGET_COLS}
    dates  = df.index[MAX_SEQ:]

    log.info(
        "build_sequences: short%s mid%s long%s flat%s | %d samples",
        X_short.shape, X_mid.shape, X_long.shape, X_flat.shape, n_samples,
    )
    return X_seq, X_flat, y_dict, feature_scaler, target_scalers, dates


# ─────────────────────────────────────────────────────────────────────────────
# Inference-time builder (one sample from latest data)
# ─────────────────────────────────────────────────────────────────────────────

def build_inference_sequence(
    df: pd.DataFrame,
    feature_scaler: RobustScaler,
) -> Tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Build one multi-window sample from the last MAX_SEQ rows of *df*.

    Returns
    -------
    X_seq : {"short": (1,30,F), "mid": (1,60,F), "long": (1,120,F)}
    X_flat: (1, F)
    """
    if len(df) < MAX_SEQ:
        raise ValueError(f"Need ≥{MAX_SEQ} rows for inference, got {len(df)}.")

    fa = feature_scaler.transform(
        df[FEATURE_COLS].values[-MAX_SEQ:].astype(np.float32)
    )

    return (
        {
            "short": fa[np.newaxis, -SEQ_SHORT:, :],
            "mid"  : fa[np.newaxis, -SEQ_MID:,   :],
            "long" : fa[np.newaxis, :,            :],
        },
        fa[-1:, :],   # X_flat
    )
