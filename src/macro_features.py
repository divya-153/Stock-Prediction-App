"""
macro_features.py — Macro-economic feature engineering.

Adds daily returns and volatility for:
  - Brent Crude Oil  (BZ=F)
  - WTI Crude Oil    (CL=F)
  - USD/INR FX rate  (INR=X)
  - India VIX        (^INDIAVIX)
  - S&P 500          (^GSPC)

Design rules (zero leakage, robustness):
  - All macro features are shifted by 1 day (only yesterday's data is known
    when predicting today's OHLC → strictly no leakage)
  - Missing tickers / download failures are handled gracefully:
      missing columns → filled with 0.0 (neutral / no signal)
      so your model continues to work even if a feed is unavailable
  - Features are near-zero on normal days; spike sharply on shock days
    → LightGBM will learn to widen OHLC predictions only when macro shocks occur
  - All features are dimensionless (log-returns, z-scores, ratios)
    so they play well with RobustScaler
  - A single public function `build_macro_features(df_index)` is the only
    external surface; it returns a DataFrame aligned to the provided index.

Macro columns added to FEATURE_COLS (in feature_engineering.py):
  brent_ret1        — yesterday's log-return of Brent Crude
  wti_ret1          — yesterday's log-return of WTI Crude
  oil_avg_ret1      — average of brent_ret1 and wti_ret1 (robust to feed gaps)
  usdinr_ret1       — yesterday's log-return of USD/INR (INR depreciation = positive)
  vix_level1        — yesterday's India VIX level (normalised to z-score vs 252d mean)
  vix_ret1          — yesterday's log-return of India VIX (spike signal)
  sp500_ret1        — yesterday's log-return of S&P 500 (global risk proxy)
  macro_shock       — composite shock score: |oil_avg_ret1|*0.3 + |usdinr_ret1|*0.3
                      + vix_z*0.2 + |sp500_ret1|*0.2 (0 on calm days, >1 on shock days)
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import DATA_DIR, FETCH_YEARS, get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Macro tickers (yfinance symbols)
# ─────────────────────────────────────────────────────────────────────────────
_MACRO_TICKERS: dict[str, str] = {
    "brent"  : "BZ=F",       # Brent Crude Oil futures
    "wti"    : "CL=F",       # WTI Crude Oil futures
    "usdinr" : "INR=X",      # USD/INR exchange rate
    "vix"    : "^INDIAVIX",  # India VIX
    "sp500"  : "^GSPC",      # S&P 500
}

# Columns this module adds to the feature set
MACRO_FEATURE_COLS: list[str] = [
    "brent_ret1",
    "wti_ret1",
    "oil_avg_ret1",
    "usdinr_ret1",
    "vix_level1",
    "vix_ret1",
    "sp500_ret1",
    "macro_shock",
    "macro_shock_extreme",   # NEW: binary flag — composite shock > 2σ (multiple signals firing simultaneously)
]

# Cache file for the merged macro DataFrame
_MACRO_CACHE = DATA_DIR / "macro_data.parquet"
_MACRO_CACHE_DAYS = 1   # refresh daily (same policy as OHLCV cache)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_macro_features(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Return a DataFrame of macro features aligned to *target_index*.

    Parameters
    ----------
    target_index : DatetimeIndex from your main df (e.g. from build_features())
                   Must be timezone-naive, sorted ascending.

    Returns
    -------
    pd.DataFrame with columns = MACRO_FEATURE_COLS, index = target_index.
    All missing values are filled with 0.0 (neutral, no macro signal).

    Usage (in feature_engineering.py::build_features):
        macro_df = build_macro_features(df.index)
        df = df.join(macro_df, how="left")
    """
    raw = _load_macro_raw()

    out = pd.DataFrame(0.0, index=target_index, columns=MACRO_FEATURE_COLS)

    if raw.empty:
        log.warning("Macro data unavailable — all macro features set to 0.0.")
        return out

    # Compute features on the raw daily macro frame
    feat = _compute_macro_features(raw)

    # Align: reindex to target_index, forward-fill up to 5 days (weekends/holidays)
    # then fill remaining NaN with 0.0
    feat_aligned = (
        feat
        .reindex(target_index.union(feat.index))   # expand index
        .sort_index()
        .ffill(limit=5)                             # bridge weekends / holidays
        .reindex(target_index)                      # back to target
        .fillna(0.0)                                # any remaining gaps → neutral
    )

    out.update(feat_aligned)
    log.info(
        "Macro features built: %d rows, %d columns, non-zero rows: %d",
        len(out), len(MACRO_FEATURE_COLS),
        int((out != 0.0).any(axis=1).sum()),
    )
    return out


def get_macro_snapshot(as_of_date: date) -> dict[str, float]:
    """
    Return macro feature values for a single date (used at inference time).
    Returns dict with 0.0 fallback for all features if data is unavailable.
    """
    raw = _load_macro_raw()
    fallback = {c: 0.0 for c in MACRO_FEATURE_COLS}

    if raw.empty:
        return fallback

    feat = _compute_macro_features(raw)

    # Use the last available row on or before as_of_date
    ts = pd.Timestamp(as_of_date)
    avail = feat[feat.index <= ts]
    if avail.empty:
        return fallback

    row = avail.iloc[-1]
    return {c: float(row.get(c, 0.0)) for c in MACRO_FEATURE_COLS}


# ─────────────────────────────────────────────────────────────────────────────
# Raw data loader (cache-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _load_macro_raw(use_cache: bool = True) -> pd.DataFrame:
    """
    Download (or load from cache) close prices for all macro tickers.
    Returns DataFrame with columns = keys of _MACRO_TICKERS, DatetimeIndex.
    Returns empty DataFrame on complete failure.
    """
    if use_cache and _MACRO_CACHE.exists():
        age = (date.today() - date.fromtimestamp(_MACRO_CACHE.stat().st_mtime)).days
        if age < _MACRO_CACHE_DAYS:
            try:
                df = pd.read_parquet(_MACRO_CACHE)
                log.info("Macro data loaded from cache (%d rows).", len(df))
                return df
            except Exception:
                pass  # corrupt cache → re-download

    return _download_macro(use_cache)


def _download_macro(save_cache: bool = True) -> pd.DataFrame:
    """
    Download close prices for all macro tickers via yfinance.
    Tickers that fail are silently dropped (column stays absent).
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — macro features disabled.")
        return pd.DataFrame()

    today     = date.today()
    end_date  = today - timedelta(days=1)
    start_date = date(end_date.year - FETCH_YEARS, end_date.month, end_date.day)

    frames: dict[str, pd.Series] = {}
    for name, symbol in _MACRO_TICKERS.items():
        try:
            raw = yf.download(
                symbol,
                start=str(start_date),
                end=str(end_date + timedelta(days=1)),
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                log.warning("Macro ticker %s (%s): empty download.", name, symbol)
                continue

            # Handle MultiIndex columns (yfinance sometimes returns them)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            close = raw["Close"].copy()
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close.sort_index(inplace=True)
            close.ffill(inplace=True)
            close.dropna(inplace=True)

            frames[name] = close
            log.info("Macro %s (%s): %d rows.", name, symbol, len(close))

        except Exception as exc:
            log.warning("Macro ticker %s (%s) failed: %s", name, symbol, exc)

    if not frames:
        return pd.DataFrame()

    # Merge all series into one DataFrame on their union of dates
    df = pd.DataFrame(frames)
    df.index.name = "Date"
    df.sort_index(inplace=True)
    df.ffill(inplace=True)   # forward-fill gaps (holidays, early closes)

    if save_cache:
        try:
            df.to_parquet(_MACRO_CACHE)
            log.info("Macro data cached to %s (%d rows).", _MACRO_CACHE.name, len(df))
        except Exception as exc:
            log.warning("Could not cache macro data: %s", exc)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation (all shifted by 1 day → no leakage)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_macro_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute macro features from raw close prices.
    ALL outputs are shifted by 1 day so only past data is used.

    Returns DataFrame with columns = MACRO_FEATURE_COLS.
    """
    out = pd.DataFrame(index=raw.index)

    # ── Oil returns ────────────────────────────────────────────────────────────
    brent_ret = _log_return(raw, "brent")   # shape: (N,)
    wti_ret   = _log_return(raw, "wti")

    out["brent_ret1"]   = brent_ret.shift(1)
    out["wti_ret1"]     = wti_ret.shift(1)

    # Oil average: use whichever feeds are present (robust to single-feed gaps)
    oil_stack = pd.concat(
        [c for c in [out["brent_ret1"], out["wti_ret1"]]
         if c.notna().any()],
        axis=1,
    )
    out["oil_avg_ret1"] = oil_stack.mean(axis=1) if not oil_stack.empty else 0.0

    # ── USD/INR return ─────────────────────────────────────────────────────────
    # Positive return = INR weakens vs USD = typically bearish for Indian equities
    out["usdinr_ret1"] = _log_return(raw, "usdinr").shift(1)

    # ── India VIX ─────────────────────────────────────────────────────────────
    if "vix" in raw.columns:
        vix       = raw["vix"].replace(0, np.nan).ffill()
        vix_s1    = vix.shift(1)

        # z-score of VIX level vs trailing 252-day window (normalises absolute level)
        vix_mean  = vix_s1.rolling(252, min_periods=60).mean()
        vix_std   = vix_s1.rolling(252, min_periods=60).std().replace(0, np.nan)
        out["vix_level1"] = ((vix_s1 - vix_mean) / vix_std).fillna(0.0)

        # VIX daily log-return (captures spike days regardless of absolute level)
        out["vix_ret1"] = np.log(vix / vix.shift(1)).shift(1).fillna(0.0)
    else:
        out["vix_level1"] = 0.0
        out["vix_ret1"]   = 0.0

    # ── S&P 500 return ─────────────────────────────────────────────────────────
    out["sp500_ret1"] = _log_return(raw, "sp500").shift(1)

    # ── Composite macro shock score ────────────────────────────────────────────
    # Weighted sum of absolute shocks; near-zero on calm days, >1 on extreme days.
    # Weights: oil 30% + FX 30% + VIX 20% + S&P500 20%
    # Each component is normalised to its rolling 252d std before weighting,
    # so the score is comparable across different macro regimes.
    def _norm_abs(series: pd.Series, window: int = 252) -> pd.Series:
        """Absolute value normalised by rolling std. 0.0 if std unavailable."""
        std = series.rolling(window, min_periods=30).std().replace(0, np.nan)
        return (series.abs() / std).fillna(0.0)

    shock = (
        0.30 * _norm_abs(out["oil_avg_ret1"])
      + 0.30 * _norm_abs(out["usdinr_ret1"])
      + 0.20 * out["vix_level1"].abs()          # already z-scored
      + 0.20 * _norm_abs(out["sp500_ret1"])
    )
    out["macro_shock"] = shock.clip(0.0, 5.0)   # cap at 5σ to avoid outlier dominance

    # NEW: binary flag for extreme combined macro shock (>2σ = oil+FX+VIX all firing)
    # Gives LightGBM a clean binary split for the highest-risk macro days.
    out["macro_shock_extreme"] = (shock > 2.0).astype(float)

    # ── Final cleanup ──────────────────────────────────────────────────────────
    # Replace inf / very large values, then fill remaining NaN with 0.0
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.clip(-10.0, 10.0, inplace=True)
    out.fillna(0.0, inplace=True)

    return out[MACRO_FEATURE_COLS]


def _log_return(df: pd.DataFrame, col: str) -> pd.Series:
    """Safe log-return for a column; returns 0.0 series if column absent."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    prices = df[col].replace(0, np.nan).ffill()
    return np.log(prices / prices.shift(1)).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Cache invalidation utility
# ─────────────────────────────────────────────────────────────────────────────

def invalidate_macro_cache() -> None:
    """Force a fresh download next time (useful after network issues)."""
    if _MACRO_CACHE.exists():
        _MACRO_CACHE.unlink()
    log.info("Macro data cache invalidated.")