"""
feature_engineering.py — Feature engineering with spike-robustness enhancements.

LEAKAGE RULE: Every feature uses only information available BEFORE the target row.

NEW in this version (spike-error diagnosis fixes):
  - Removed raw volatility feature (was noisy + redundant with range features)
  - Added ATR (Average True Range) — a proper, stable volatility proxy
  - Added range_expansion_ratio: how much yesterday's range deviated from its 10d avg
    (key signal for detecting incoming high-volatility days)
  - Added body_ratio: (|Close - Open|) / Range — measures candle body compression
  - Added upper_shadow / lower_shadow: wick ratios (regime shift signal)
  - Added volume_spike: whether yesterday's volume was anomalous
  - Added range_z_score: z-score of yesterday's range vs rolling distribution
    (most direct "are we in a spike regime?" feature available without future data)
  - Log-transformed range target: reduces regression skew on extreme days
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Feature columns
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS: List[str] = [
    # Lag features
    "open_lag1", "high_lag1", "low_lag1", "close_lag1",
    "close_lag2", "close_lag3",
    # Gap features
    "gap_1", "avg_gap_5", "avg_gap_10",
    # Range features (core)
    "range_1", "avg_range_5", "avg_range_10",
    # Range regime features (NEW — spike detection)
    "range_expansion_ratio",   # range_1 / avg_range_10
    "range_z_score_10",        # (range_1 - avg_range_10) / std_range_10
    "atr_14",                  # Average True Range over 14 days (leakage-safe)
    
    # Range momentum features
    "range_momentum",
    "range_acceleration",
    "range_breakout",
    # Multi-window ATR (NEW — short/long vol comparison)
    "atr_5",               # fast ATR — reacts within a week to new vol regime
    "atr_28",              # slow ATR — long-term vol baseline
    "atr_ratio_5_28",      # atr_5/atr_28 > 1.5 → short-term vol elevated vs baseline
    # Realized volatility estimators (NEW — statistically efficient regime detectors)
    "parkinson_vol",       # log(H/L)² / (4·ln2) rolling mean — spike-sensitive
    "garman_klass_vol",    # full OHLC estimator — most efficient single-day vol
    "hv_ratio_5_20",       # 5d HV / 20d HV — "is recent vol elevated vs normal?"
    "calm_streak",         # days since H-L range was above its 20d median (calm-before-storm)
    # Volatility mean-reversion features (NEW)
    "consec_high_range",   # consecutive days with range > 20d median (vol cluster signal)
    "consec_low_range",    # consecutive days with range < 70% of 20d median (breakout precursor)
    
    # Candle structure (NEW — regime signals)
    "body_ratio_1",            # |Close-Open| / Range yesterday
    "upper_shadow_1",          # (High - max(Open,Close)) / Range
    "lower_shadow_1",          # (min(Open,Close) - Low) / Range
    # Return features
    "return_1", "return_3", "return_5",
    "pct_return_1", "pct_return_5",
    # Rolling statistics
    "rolling_mean_5", "rolling_mean_10",
    "rolling_std_5", "rolling_std_10",
    # Volume features
    "volume_1", "volume_change", "avg_volume_5",
    "volume_spike",            # NEW: volume_1 / avg_volume_5 (anomaly ratio)
    # Close-position features
    "close_pos_1", "avg_close_pos_5",
    # Calendar features
    "day_of_week", "week_of_year", "month", "day_of_year",
    # Macro-economic features (new)
    "brent_ret1", "wti_ret1", "oil_avg_ret1",
    "usdinr_ret1",
    "vix_level1", "vix_ret1",
    "sp500_ret1",
    "macro_shock",
    "macro_shock_extreme", # NEW: binary flag — combined macro shock > 2σ
     # Sentiment & event features (new)
    "sentiment_score",
    "sentiment_magnitude",
    "sentiment_volume",
    "event_ceasefire",
    "event_sanctions",
    "event_opec",
    "event_central_bank",
    "event_fiscal",
    "event_shock",
]

TARGET_COLS: List[str] = ["target_open", "target_log_range", "target_closepos", "target_close"]

# Legacy name kept for compatibility with predictor inverse-transform
_TARGET_RANGE_COL = "target_log_range"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features and targets from raw OHLCV data.
    Drops NaN rows at the end. Returns clean DataFrame.
    """
    df = df_raw.copy()

    _add_lag_features(df)
    _add_gap_features(df)
    _add_range_features(df)
    _add_range_regime_features(df)
    _add_realized_vol_features(df)          # NEW — Parkinson, Garman-Klass, HV ratio
    _add_candle_structure_features(df)
    _add_return_features(df)
    _add_rolling_stats(df)
    _add_volume_features(df)
    _add_close_position_features(df)
    _add_calendar_features(df)
    # Macro features (no leakage — all values are shifted by 1 day inside the module)
    from src.macro_features import build_macro_features, MACRO_FEATURE_COLS
    macro_df = build_macro_features(df.index)
    for col in MACRO_FEATURE_COLS:
        df[col] = macro_df[col].values
 
    # Sentiment + event features (no leakage — shifted by 1 day inside the module)
    from src.sentiment_features import build_sentiment_features, SENTIMENT_FEATURE_COLS
    sentiment_df = build_sentiment_features(df.index)
    for col in SENTIMENT_FEATURE_COLS:
        df[col] = sentiment_df[col].values
 
    _add_targets(df)

    before = len(df)

    # remove infinity values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # clip extreme values (prevents float32 overflow)
    df[FEATURE_COLS] = df[FEATURE_COLS].clip(-1e6, 1e6)

    # drop NaN rows
    df.dropna(subset=FEATURE_COLS + TARGET_COLS, inplace=True)

    log.info(
        "build_features: %d raw rows → %d clean rows (dropped %d).",
        before, len(df), before - len(df),
    )
    return df

"""
PATCH for feature_engineering.py
=================================

Add the function below IMMEDIATELY AFTER the existing build_features() function
(around line 537 in your file, right after the closing `return df` line).

Do NOT replace anything. Just insert this block after build_features().
The existing build_features(), calendar_features_for_date(), and all
private helpers (_add_lag_features, etc.) remain completely unchanged.
"""


def build_features_for_inference(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering variant for recursive multi-step inference.

    Unlike build_features(), this function:
      - Does NOT compute TARGET_COLS (Open/High/Low/Close targets don't
        exist for synthetic rows appended during multi-step forecasting)
      - Does NOT drop rows where any feature is NaN (that would discard
        the last synthetic row, which has boundary NaN from rolling windows)
      - Forward-fills up to 2 boundary NaN values at the tail only, to
        handle rolling warm-up on newly-appended rows
      - Clips and replaces inf identically to build_features()

    Called by predictor._predict_one_step() at each recursive step.
    The real training path still uses build_features() with strict dropna.
    """
    df = df_raw.copy()

    _add_lag_features(df)
    _add_gap_features(df)
    _add_range_features(df)
    _add_range_regime_features(df)
    _add_realized_vol_features(df)          # NEW — Parkinson, Garman-Klass, HV ratio
    _add_candle_structure_features(df)
    _add_return_features(df)
    _add_rolling_stats(df)
    _add_volume_features(df)
    _add_close_position_features(df)
    _add_calendar_features(df)

    from src.macro_features import build_macro_features, MACRO_FEATURE_COLS
    macro_df = build_macro_features(df.index)
    for col in MACRO_FEATURE_COLS:
        df[col] = macro_df[col].values

    from src.sentiment_features import build_sentiment_features, SENTIMENT_FEATURE_COLS
    sentiment_df = build_sentiment_features(df.index)
    for col in SENTIMENT_FEATURE_COLS:
        df[col] = sentiment_df[col].values

    # Clean up — same as build_features() but without requiring TARGET_COLS
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].clip(-1e6, 1e6)

    # Forward-fill boundary NaN at the tail (max 2 rows only).
    # This handles rolling warm-up gaps on newly-appended synthetic rows
    # without corrupting real historical data further up the frame.
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill(limit=2)

    # Drop rows where ALL feature cols are NaN (true empty rows only).
    # This is intentionally lenient — a row with some NaN is kept so
    # that the last synthetic row survives into the inference pipeline.
    df.dropna(subset=FEATURE_COLS, how="all", inplace=True)

    return df

def calendar_features_for_date(target_date: pd.Timestamp) -> dict:
    return {
        "day_of_week"  : int(target_date.dayofweek),
        "week_of_year" : int(target_date.isocalendar().week),
        "month"        : int(target_date.month),
        "day_of_year"  : int(target_date.day_of_year),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_lag_features(df: pd.DataFrame) -> None:
    df["open_lag1"]  = df["Open"].shift(1)
    df["high_lag1"]  = df["High"].shift(1)
    df["low_lag1"]   = df["Low"].shift(1)
    df["close_lag1"] = df["Close"].shift(1)
    df["close_lag2"] = df["Close"].shift(2)
    df["close_lag3"] = df["Close"].shift(3)


def _add_gap_features(df: pd.DataFrame) -> None:
    df["gap_1"]      = df["open_lag1"] - df["close_lag2"]
    daily_gap        = df["Open"].shift(1) - df["Close"].shift(2)
    df["avg_gap_5"]  = daily_gap.rolling(5).mean()
    df["avg_gap_10"] = daily_gap.rolling(10).mean()


def _add_range_features(df: pd.DataFrame) -> None:
    daily_range       = (df["High"] - df["Low"]).shift(1)
    df["range_1"]     = daily_range
    df["avg_range_5"] = daily_range.rolling(5).mean()
    df["avg_range_10"]= daily_range.rolling(10).mean()


def _add_range_regime_features(df: pd.DataFrame) -> None:
    """
    FIX #1: Proper regime / spike-detection features.

    range_expansion_ratio: how large is yesterday's range vs its 10d average?
      > 1.5 → regime expansion likely → model should widen predicted range
      This is the single most predictive leading indicator of a high-error day.

    range_z_score_10: how many SDs above/below normal is yesterday's range?
      Gives LightGBM a clear threshold split for "unusual day".

    atr_14: Average True Range (Wilder). True range accounts for gaps:
      TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
      All values shifted so only past data is used.
    """
    daily_range = (df["High"] - df["Low"]).shift(1)
    avg10       = daily_range.rolling(10).mean()
    std10       = daily_range.rolling(10).std().replace(0, np.nan)

    df["range_expansion_ratio"] = (daily_range / avg10.replace(0, np.nan)).fillna(1.0)
    df["range_z_score_10"]      = ((daily_range - avg10) / std10).fillna(0.0)

    # True range (all shifted by 1 — pure past data)
    H  = df["High"].shift(1)
    L  = df["Low"].shift(1)
    PC = df["Close"].shift(2)   # previous-previous close (one step earlier than H/L)
    tr = pd.concat([H - L, (H - PC).abs(), (L - PC).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    
    # Range momentum features
    df["range_momentum"]    = daily_range - avg10
    df["range_acceleration"]= daily_range.diff()
    df["range_breakout"]    = (daily_range / avg10).fillna(1.0)

    # FIX/NEW — Multi-window ATR (short/slow comparison gives vol-regime ratio)
    df["atr_5"]  = tr.rolling(5).mean()
    df["atr_28"] = tr.rolling(28).mean()
    df["atr_ratio_5_28"] = (df["atr_5"] / df["atr_28"].replace(0, np.nan)).fillna(1.0)

    # NEW — Volatility mean-reversion / breakout precursor features
    # All computed on lag-1 data (daily_range already shifted) — zero leakage
    med20 = daily_range.rolling(20).median()
    above_med = (daily_range > med20.replace(0, np.nan)).astype(int)
    below_med = (daily_range < (med20.replace(0, np.nan) * 0.70)).astype(int)

    # Consecutive days above/below median — capped at 5 to bound the feature
    df["consec_high_range"] = (
        above_med.groupby((above_med == 0).cumsum()).cumcount().clip(0, 5)
    )
    df["consec_low_range"] = (
        below_med.groupby((below_med == 0).cumsum()).cumcount().clip(0, 5)
    )

    # calm_streak: how many consecutive days was range BELOW 20d median?
    # High values (≥3) precede breakouts — the "calm before storm" signal.
    is_calm = (daily_range < med20.replace(0, np.nan)).astype(int)
    df["calm_streak"] = (
        is_calm.groupby((is_calm == 0).cumsum()).cumcount().clip(0, 10)
    )


def _add_realized_vol_features(df: pd.DataFrame) -> None:
    """
    NEW — Realized volatility estimators (all lag-1, zero leakage).

    Parkinson estimator:  uses only High/Low — 5.55× more efficient than
    close-only HV. Near-zero on calm days, spikes sharply on volatile days.

    Garman-Klass estimator: uses full OHLC — the most efficient single-day
    estimator. Captures both gap and intra-day range information.

    hv_ratio_5_20: 5-day historical vol / 20-day HV.
      > 1.5 = recent vol regime elevated vs baseline.
      < 0.7 = unusually quiet vs recent history (potential breakout precursor).

    All three are computed on lag-1 OHLCV so no forward information is used.
    """
    H  = df["High"].shift(1)
    L  = df["Low"].shift(1)
    O  = df["Open"].shift(1)
    C  = df["Close"].shift(1)
    PC = df["Close"].shift(2)   # needed for log-return series

    hl_ratio = (H / L.replace(0, np.nan)).clip(1e-6, None)
    co_ratio = (C / O.replace(0, np.nan)).clip(1e-6, None)

    # Parkinson: rolling 10-day mean of the daily estimator
    park_daily = (np.log(hl_ratio) ** 2) / (4.0 * np.log(2))
    df["parkinson_vol"] = np.sqrt(park_daily.rolling(10, min_periods=5).mean()).fillna(0.0)

    # Garman-Klass: rolling 10-day mean of the daily estimator
    gk_daily = (
        0.5 * (np.log(hl_ratio) ** 2)
        - (2.0 * np.log(2) - 1.0) * (np.log(co_ratio) ** 2)
    ).clip(0.0)   # numerical safety — should be ≥ 0 by construction
    df["garman_klass_vol"] = np.sqrt(gk_daily.rolling(10, min_periods=5).mean()).fillna(0.0)

    # HV ratio: short-term (5d) vs medium-term (20d) historical vol
    log_ret = np.log((C / PC.replace(0, np.nan)).clip(1e-6, None))
    hv5  = log_ret.rolling(5,  min_periods=3).std()
    hv20 = log_ret.rolling(20, min_periods=10).std().replace(0, np.nan)
    df["hv_ratio_5_20"] = (hv5 / hv20).fillna(1.0).clip(0.0, 5.0)


def _add_candle_structure_features(df: pd.DataFrame) -> None:
    """
    FIX #2: Candle microstructure. These encode whether yesterday was a
    trend day (large body), doji (small body), or spike (long wicks).
    They are excellent LightGBM split candidates for regime detection.
    All computed on yesterday (shift(1)) — zero leakage.
    """
    H  = df["High"].shift(1)
    L  = df["Low"].shift(1)
    O  = df["Open"].shift(1)
    C  = df["Close"].shift(1)
    rng = (H - L).replace(0, np.nan)

    df["body_ratio_1"]   = ((C - O).abs() / rng).fillna(0.5)
    df["upper_shadow_1"] = ((H - pd.concat([O, C], axis=1).max(axis=1)) / rng).fillna(0.0)
    df["lower_shadow_1"] = ((pd.concat([O, C], axis=1).min(axis=1) - L) / rng).fillna(0.0)


def _add_return_features(df: pd.DataFrame) -> None:
    close = df["Close"]
    df["return_1"]     = np.log(close.shift(1) / close.shift(2))
    df["return_3"]     = np.log(close.shift(1) / close.shift(4))
    df["return_5"]     = np.log(close.shift(1) / close.shift(6))
    df["pct_return_1"] = (close.shift(1) - close.shift(2)) / close.shift(2)
    df["pct_return_5"] = (close.shift(1) - close.shift(6)) / close.shift(6)


def _add_rolling_stats(df: pd.DataFrame) -> None:
    cs = df["Close"].shift(1)
    df["rolling_mean_5"]  = cs.rolling(5).mean()
    df["rolling_mean_10"] = cs.rolling(10).mean()
    df["rolling_std_5"]   = cs.rolling(5).std()
    df["rolling_std_10"]  = cs.rolling(10).std()


def _add_volume_features(df: pd.DataFrame) -> None:
    vs = df["Volume"].shift(1)
    avg_vol5           = vs.rolling(5).mean().replace(0, np.nan)
    df["volume_1"]     = vs
    df["volume_change"]= vs / df["Volume"].shift(2) - 1.0
    df["avg_volume_5"] = avg_vol5
    # FIX #3: Volume spike ratio — a clean anomaly signal
    df["volume_spike"] = (vs / avg_vol5).fillna(1.0)


def _add_close_position_features(df: pd.DataFrame) -> None:
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    cp  = (df["Close"] - df["Low"]) / rng
    cs  = cp.shift(1)
    df["close_pos_1"]    = cs
    df["avg_close_pos_5"]= cs.rolling(5).mean()


def _add_calendar_features(df: pd.DataFrame) -> None:
    idx = df.index
    df["day_of_week"]  = idx.dayofweek.astype(np.int16)
    df["week_of_year"] = idx.isocalendar().week.values.astype(np.int16)
    df["month"]        = idx.month.astype(np.int16)
    df["day_of_year"]  = idx.day_of_year.astype(np.int16)


def _add_targets(df: pd.DataFrame) -> None:
    """
    FIX #4: Log-transform the range target.

    WHY: target_range is heavily right-skewed — extreme spike days have ranges
    5–10× the median. RobustScaler partially handles this, but LightGBM still
    optimises MAE on the raw scale, meaning spike days dominate gradients OR
    get under-weighted by quantile splits.

    Solution: predict log(range) instead of range.
    At inference we exp() it back. This compresses the tail enormously and
    makes LightGBM's splits uniform across quiet and volatile regimes.

    target_open and target_closepos are unaffected (open is in price space,
    closepos is already bounded [0,1]).
    """
    df["target_open"] = df["Open"]

    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    # log(range) — add small epsilon for numerical safety
    df["target_log_range"] = np.log(rng + 1e-6)

    df["target_closepos"] = ((df["Close"] - df["Low"]) / rng).clip(0.0, 1.0)

    # Direct close target — anchors the close-first prediction hierarchy.
    # Predicting close directly removes the instability of reconstructing it
    # from closepos × range, which amplifies errors on high-volatility days.
    df["target_close"] = df["Close"]