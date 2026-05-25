"""
data_loader.py — Download and clean OHLCV data using yfinance.

Rules:
  - 15 years of daily data, auto-updated to latest date
  - Sorted ascending by date
  - Missing values handled safely
  - No leakage: this module only returns raw price data
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.utils import DATA_DIR, FETCH_YEARS, get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv(
    ticker: str,
    use_cache: bool = True,
    cache_days: int = 1,
) -> pd.DataFrame:
    """
    Download (or load from cache) FETCH_YEARS of daily OHLCV data for *ticker*.

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume]
    DatetimeIndex sorted ascending, timezone-naive.

    Raises
    ------
    ValueError  – if the downloaded frame is empty or too short.
    """
    cache_file = DATA_DIR / f"{ticker.replace('/', '_').replace('^','')}_ohlcv.parquet"

    # ── Try cache ──────────────────────────────────────────────────────────────
    if use_cache and cache_file.exists():
        age = (date.today() - date.fromtimestamp(cache_file.stat().st_mtime)).days
        if age < cache_days:
            log.info("Loading %s from cache (%s).", ticker, cache_file.name)
            df = pd.read_parquet(cache_file)
            return _validate(df, ticker)

    # ── Download ───────────────────────────────────────────────────────────────
    today = date.today()
    end_date = today - timedelta(days=1)
    start_date = date(end_date.year - FETCH_YEARS, end_date.month, end_date.day)

    log.info("Downloading %s: %s → %s", ticker, start_date, end_date)
    raw: pd.DataFrame = yf.download(
        ticker,
        start=str(start_date),
        end=str(end_date + timedelta(days=1)),   # end is exclusive in yfinance
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"yfinance returned empty data for '{ticker}'.")

    df = _clean(raw)
    df = _validate(df, ticker)
    
    # ── Persist ────────────────────────────────────────────────────────────────
    csv_file = DATA_DIR / f"{ticker.replace('/', '_').replace('^','')}_ohlcv.csv"
    df.to_csv(csv_file)

    df.to_parquet(cache_file)
    log.info("Saved %d rows to %s.", len(df), cache_file.name)
    return df

   


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns, handle MultiIndex, remove NaN rows."""
    # yfinance sometimes returns a MultiIndex column (ticker, field)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only OHLCV
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns after download: {missing}")

    df = raw[needed].copy()

    # Ensure DatetimeIndex, timezone-naive
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    # Sort ascending
    df.sort_index(inplace=True)

    # Drop rows with all-NaN OHLC
    df.dropna(subset=["Open", "High", "Low", "Close"], how="all", inplace=True)

    # Forward-fill isolated NaN cells (e.g. volume gaps)
    df.ffill(inplace=True)

    # Drop any remaining NaN rows
    df.dropna(inplace=True)

    # Basic sanity: High >= Low
    bad = df["High"] < df["Low"]
    if bad.any():
        log.warning("Dropping %d rows where High < Low.", bad.sum())
        df = df[~bad]

    return df


def _validate(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    min_rows = 252 * 3   # at least 3 years
    if len(df) < min_rows:
        raise ValueError(
            f"Only {len(df)} rows for '{ticker}' — need at least {min_rows}."
        )
    log.info("%s: %d rows, %s → %s", ticker, len(df), df.index[0].date(), df.index[-1].date())
    return df
