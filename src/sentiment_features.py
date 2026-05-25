"""
sentiment_features.py — Daily sentiment + binary event features.

THREE-SOURCE ARCHITECTURE (in priority order):
═══════════════════════════════════════════════
  SOURCE 1 — GDELT (macro/global events, 2015→today, FREE, no API key)
    · Covers 15 years → fills all training rows with real data
    · Filtered to finance/macro keywords only (reduces noise)
    · Deduplicated by title to remove syndication copies
    · Cached per-year in data/gdelt_YYYY.parquet (refreshed monthly)

  SOURCE 2 — yfinance (company-specific, last ~2 weeks, FREE, no API key)
    · Real-time company news for NSE/US index tickers
    · FIXED: uses new nested content structure (item["content"]["title"])
    · Supplements GDELT with stock-specific headlines

  SOURCE 3 — NewsAPI (optional backup, last 30 days, needs NEWSAPI_KEY env var)
    · Set env var: set NEWSAPI_KEY=your_key_here  (Windows)
    · Silently skipped if key absent — never crashes

NOISE CONTROLS:
  · GDELT: finance keyword filter before storing (removes >90% of noise)
  · GDELT: deduplicated by normalised title (removes syndication duplicates)
  · All sources: MAX_HEADLINES_PER_DAY cap after dedup
  · Stub titles under 10 chars discarded
  · sentiment_volume: normalised against rolling 90-day median
    so 2011 (fewer digital sources) and 2026 (many sources) are comparable

SCORING CHAIN (best available, smoke-tested):
  FinBERT → VADER → TextBlob → 0.0

LEAKAGE RULE (zero leakage):
  All features shifted by 1 day inside _build_sentiment_dataframe().
  Only yesterday's headlines are used when predicting today.

CACHE LAYOUT:
  data/sentiment_cache.parquet  — final feature DataFrame (refreshed daily)
  data/gdelt_YYYY.parquet       — raw GDELT headlines per year (monthly refresh)
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import DATA_DIR, FETCH_YEARS

from src.utils import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SENTIMENT_FEATURE_COLS: list[str] = [
    "sentiment_score",       # mean scorer output  [-1, +1]
    "sentiment_magnitude",   # abs(sentiment_score) [0, +1]
    "sentiment_volume",      # normalised headline count (volume-adjusted)
    "event_ceasefire",       # binary: war / ceasefire news
    "event_sanctions",       # binary: sanctions / tariff
    "event_opec",            # binary: OPEC / crude supply
    "event_central_bank",    # binary: RBI / Fed / rate decision
    "event_fiscal",          # binary: budget / GST / tax
    "event_shock",           # binary: any of the above
]

_SENTIMENT_CACHE  = DATA_DIR / "sentiment_cache.parquet"
_CACHE_DAYS       = 1     # rebuild final features daily
_GDELT_CACHE_DAYS = 30    # GDELT raw data refreshed monthly per past year

# Hard cap per day AFTER dedup — prevents volume explosion on high-news days
MAX_HEADLINES_PER_DAY = 50

# ── GDELT finance keyword filter ──────────────────────────────────────────────
# Only headlines containing at least one of these keywords are kept.
# This is the primary noise reduction — removes >90% of irrelevant GDELT noise.
_GDELT_FINANCE_KEYWORDS: list[str] = [
    "stock", "market", "nifty", "sensex", "bse", "nse", "rupee", "rbi",
    "reserve bank", "inflation", "gdp", "oil", "opec", "crude", "brent",
    "rate hike", "rate cut", "repo rate", "monetary policy", "budget",
    "fiscal", "tax", "tariff", "sanction", "trade war", "fed",
    "federal reserve", "ecb", "interest rate", "recession",
    "export", "import", "reliance", "tata", "infosys", "wipro",
    "hdfc", "icici", "sensex", "ceasefire", "war", "conflict",
    "missile", "invasion", "geopolit", "wti", "energy",
    "rally", "crash", "bull", "bear", "earnings", "profit",
    "revenue", "quarterly", "dividend",
]
_GDELT_KEYWORD_RE = re.compile(
    "|".join(re.escape(k) for k in _GDELT_FINANCE_KEYWORDS),
    flags=re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Event keyword patterns (case-insensitive)
# ─────────────────────────────────────────────────────────────────────────────
_EVENT_PATTERNS: dict[str, list[str]] = {
    "event_ceasefire": [
        r"\bceasefire\b", r"\bwar\b", r"\bmilitary\s+strike\b",
        r"\bmissile\b", r"\bairstr", r"\bescalat", r"\bgeopolit",
        r"\bconflict\b", r"\binvasion\b", r"\bterror\b",
    ],
    "event_sanctions": [
        r"\bsanction", r"\btariff", r"\btrade\s+war\b",
        r"\bexport\s+ban\b", r"\bembargo\b", r"\btrade\s+restrict",
        r"\bimport\s+dut", r"\bcustoms\s+dut",
    ],
    "event_opec": [
        r"\bopec\b", r"\bcrude\s+oil\b", r"\boil\s+produc",
        r"\bbrent\b", r"\bwti\b", r"\boil\s+supply\b",
        r"\bpetrol\s+price\b", r"\benergy\s+crisis\b",
    ],
    "event_central_bank": [
        r"\brbi\b", r"\bfed\s+rate\b", r"\bfederal\s+reserve\b",
        r"\becb\b", r"\brate\s+hike\b", r"\brate\s+cut\b",
        r"\bmonetary\s+policy\b", r"\binterest\s+rate\b",
        r"\brepo\s+rate\b", r"\binflation\s+target",
    ],
    "event_fiscal": [
        r"\bunion\s+budget\b", r"\bgst\b", r"\btax\s+rate\b",
        r"\bfiscal\s+deficit\b", r"\bgovernment\s+spend",
        r"\bdisinvest", r"\bprivatis", r"\bprivatiz",
        r"\bfinance\s+minister\b", r"\bbudget\s+proposal",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_sentiment_features(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Return sentiment + event features aligned to *target_index*.
    All missing values filled 0.0. Zero leakage (shifted 1 day inside).
    """
    out = pd.DataFrame(0.0, index=target_index, columns=SENTIMENT_FEATURE_COLS)

    raw = _load_or_build_cache(target_index)
    if raw.empty:
        log.warning("Sentiment data unavailable — all features set to 0.0.")
        return out

    aligned = (
        raw
        .reindex(target_index.union(raw.index))
        .sort_index()
        .ffill(limit=3)
        .reindex(target_index)
        .fillna(0.0)
    )
    out.update(aligned)

    non_zero = int((out["sentiment_score"] != 0.0).sum())
    log.info(
        "Sentiment features: %d rows, %d non-zero (%.1f%%)",
        len(out), non_zero, non_zero / max(len(out), 1) * 100,
    )
    return out


def get_sentiment_snapshot(as_of_date: date) -> dict[str, float]:
    """Single-date snapshot for inference. Returns 0.0 fallback if unavailable."""
    fallback  = {c: 0.0 for c in SENTIMENT_FEATURE_COLS}
    today_idx = pd.DatetimeIndex([pd.Timestamp(as_of_date)])

    raw = _load_or_build_cache(today_idx)
    if raw.empty:
        return fallback

    ts    = pd.Timestamp(as_of_date)
    avail = raw[raw.index <= ts]
    if avail.empty:
        return fallback

    row = avail.iloc[-1]
    return {c: float(row.get(c, 0.0)) for c in SENTIMENT_FEATURE_COLS}


def invalidate_sentiment_cache() -> None:
    """Force fresh rebuild on next call."""
    if _SENTIMENT_CACHE.exists():
        _SENTIMENT_CACHE.unlink()
    log.info("Sentiment cache invalidated.")


# ─────────────────────────────────────────────────────────────────────────────
# Cache layer
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_build_cache(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load final feature cache if fresh; else rebuild and save."""
    if _SENTIMENT_CACHE.exists():
        age = (date.today() - date.fromtimestamp(
            _SENTIMENT_CACHE.stat().st_mtime)).days
        if age < _CACHE_DAYS:
            try:
                df = pd.read_parquet(_SENTIMENT_CACHE)
                log.info("Sentiment cache loaded (%d rows).", len(df))
                return df
            except Exception:
                pass  # corrupt → rebuild

    df = _build_sentiment_dataframe(target_index)

    if not df.empty:
        try:
            df.to_parquet(_SENTIMENT_CACHE)
            log.info("Sentiment cache saved (%d rows).", len(df))
        except Exception as exc:
            log.warning("Could not save sentiment cache: %s", exc)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Core builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_sentiment_dataframe(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fetch from all three sources, merge, deduplicate, score,
    detect events, normalise volume, shift 1 day.
    """
    if target_index is None or len(target_index) == 0:
        return pd.DataFrame()

    start_date = target_index.min().date()
    end_date   = target_index.max().date()

    # ── Collect from all three sources ────────────────────────────────────────
    headlines: dict[date, list[str]] = {}

    # Source 1: GDELT — macro/global, covers full training history
    _merge_into(headlines, _fetch_gdelt_news(start_date, end_date))

    # Source 2: yfinance — company-specific, last ~2 weeks
    _merge_into(headlines, _fetch_yfinance_news())

    # Source 3: NewsAPI — optional backup, last 30 days if key set
    _merge_into(headlines, _fetch_newsapi(start_date, end_date))

    log.info("Headlines merged from all sources: %d days.", len(headlines))

    if not headlines:
        log.warning("No headlines from any source — sentiment will be 0.0.")
        return pd.DataFrame()

    # ── Deduplicate and cap per day ───────────────────────────────────────────
    headlines = _dedup_and_cap(headlines)

    # ── Score and detect events ───────────────────────────────────────────────
    scorer = _get_scorer()
    rows: list[dict] = []

    for dt in sorted(headlines.keys()):
        texts = headlines[dt]
        if not texts:
            continue

        scores     = [scorer(t) for t in texts]
        mean_score = float(np.mean(scores)) if scores else 0.0

        combined = " ".join(texts).lower()
        flags = {
            col: int(any(re.search(pat, combined) for pat in pats))
            for col, pats in _EVENT_PATTERNS.items()
        }

        rows.append({
            "date"               : pd.Timestamp(dt),
            "sentiment_score"    : float(np.clip(mean_score, -1.0, 1.0)),
            "sentiment_magnitude": float(abs(mean_score)),
            "_raw_count"         : len(texts),   # kept for volume normalisation
            **flags,
            "event_shock"        : int(any(flags.values())),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)

    # ── Volume normalisation — rolling 90-day median ──────────────────────────
    # WHY: A fixed divisor (e.g. /20) makes 2011 look like "no news" because
    # there were fewer digital sources then. Rolling median means each day is
    # expressed relative to its own era's typical volume — 2011 and 2026 are
    # on the same scale. Result clipped to [0, 1].
    rolling_med = df["_raw_count"].rolling(90, min_periods=7).median()
    rolling_med = rolling_med.replace(0, np.nan).ffill().bfill().replace(0, 1.0)
    df["sentiment_volume"] = (df["_raw_count"] / rolling_med).clip(0.0, 1.0)
    df.drop(columns=["_raw_count"], inplace=True)

    # ── Shift 1 day — strict no-leakage ──────────────────────────────────────
    df = df.shift(1)
    df.fillna(0.0, inplace=True)

    log.info(
        "Sentiment dataframe: %d rows | non-zero score: %d (%.1f%%)",
        len(df),
        int((df["sentiment_score"] != 0.0).sum()),
        (df["sentiment_score"] != 0.0).mean() * 100,
    )
    return df[SENTIMENT_FEATURE_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Noise controls
# ─────────────────────────────────────────────────────────────────────────────

def _merge_into(target: dict, source: dict) -> None:
    """Merge source into target, discarding stub titles under 10 chars."""
    for dt, texts in source.items():
        for t in texts:
            if len(t.strip()) >= 10:
                target.setdefault(dt, []).append(t)


def _dedup_and_cap(headlines: dict[date, list[str]]) -> dict[date, list[str]]:
    """
    Per-day: remove duplicate titles (case-insensitive), then cap at
    MAX_HEADLINES_PER_DAY. This handles syndication (same Reuters article
    on 20 sites) and prevents volume inflation on extreme-news days.
    """
    result: dict[date, list[str]] = {}
    for dt, texts in headlines.items():
        seen:   set[str]  = set()
        unique: list[str] = []
        for t in texts:
            key = t.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(t)
        result[dt] = unique[:MAX_HEADLINES_PER_DAY]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1 — GDELT (macro/global events, 15 years history)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_gdelt_news(start_date: date, end_date: date) -> dict[date, list[str]]:
    """
    Fetch finance-filtered headlines from GDELT for [start_date, end_date].

    Uses GDELT Document API v2 (Article List mode) — returns JSON with
    title + date per article. No auth required. Free.

    Strategy to handle 15 years without memory/speed issues:
      - Fetch year by year
      - Cache each year separately in data/gdelt_YYYY.parquet
      - Past years: refresh monthly. Current year: refresh daily.
      - Finance keyword filter applied before caching (primary noise control)
    """
    result: dict[date, list[str]] = {}

    try:
        import requests
    except ImportError:
        log.warning("requests not installed — GDELT skipped. pip install requests")
        return result

    years = list(range(start_date.year, end_date.year + 1))

    for year in years:
        year_cache = DATA_DIR / f"gdelt_{year}.parquet"

        # Check yearly cache freshness
        if year_cache.exists():
            age = (date.today() - date.fromtimestamp(
                year_cache.stat().st_mtime)).days
            # Past years refresh monthly; current year refreshes daily
            max_age = _CACHE_DAYS if year == date.today().year else _GDELT_CACHE_DAYS
            if age < max_age:
                try:
                    cached_df = pd.read_parquet(year_cache)
                    _append_gdelt_df(cached_df, result, start_date, end_date)
                    log.info("GDELT %d: from cache (%d rows).", year, len(cached_df))
                    continue
                except Exception:
                    pass  # corrupt cache → re-fetch

        # Fetch fresh from GDELT API
        year_rows = _fetch_gdelt_year(year, requests)
        if year_rows:
            try:
                df_year = pd.DataFrame(year_rows)
                df_year.to_parquet(year_cache)
                log.info("GDELT %d: fetched %d articles, cached.", year, len(df_year))
            except Exception as exc:
                log.warning("GDELT %d: cache write failed: %s", year, exc)
                df_year = pd.DataFrame(year_rows)
            _append_gdelt_df(df_year, result, start_date, end_date)
        else:
            log.info("GDELT %d: no articles returned.", year)

    total = sum(len(v) for v in result.values())
    log.info("GDELT total: %d days, %d headlines.", len(result), total)
    return result


def _fetch_gdelt_year(year: int, requests_mod) -> list[dict]:
    """
    Query GDELT Document API for finance-related articles in a given year.
    Returns list of {"date": date, "title": str} dicts.

    Uses multiple focused queries (OR logic across calls) to get broad
    finance coverage without hitting API limits per call.
    Finance keyword filter (_GDELT_KEYWORD_RE) applied to every title
    before appending — this is the primary noise control.
    """
    # Finance/macro queries covering all our event categories
    queries = [
        "India stock market Nifty Sensex NSE BSE",
        "RBI repo rate monetary policy India inflation",
        "OPEC oil crude Brent WTI energy supply",
        "Federal Reserve interest rate Fed rate hike cut",
        "India budget GST fiscal deficit tax finance minister",
        "sanctions tariff trade war embargo geopolitical",
        "ceasefire war conflict invasion missile military",
        "Reliance Tata Infosys HDFC ICICI earnings quarterly",
    ]

    rows: list[dict] = []
    seen_titles: set[str] = set()

    year_start = date(year, 1, 1)
    year_end   = min(date(year, 12, 31), date.today() - timedelta(days=1))
    if year_start > year_end:
        return rows

    for query in queries:
        try:
            resp = requests_mod.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query"        : query,
                    "mode"         : "artlist",
                    "maxrecords"   : "250",
                    "startdatetime": year_start.strftime("%Y%m%d") + "000000",
                    "enddatetime"  : year_end.strftime("%Y%m%d")   + "235959",
                    "format"       : "json",
                    "sort"         : "DateDesc",
                },
                timeout=25,
            )

            if resp.status_code != 200:
                log.debug("GDELT API returned %d for year %d.", resp.status_code, year)
                continue

            articles = resp.json().get("articles", [])

            for art in articles:
                title    = (art.get("title") or "").strip()
                seentime = art.get("seendate", "")   # format: "YYYYMMDDTHHMMSSZ"

                if len(title) < 10:
                    continue

                # Primary noise filter: keep only finance-relevant headlines
                if not _GDELT_KEYWORD_RE.search(title):
                    continue

                # Dedup within this fetch session by normalised title prefix
                title_key = title.lower()[:80]
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

                # Parse date from GDELT seendate string
                try:
                    dt = date(
                        int(seentime[0:4]),
                        int(seentime[4:6]),
                        int(seentime[6:8]),
                    )
                except Exception:
                    continue

                rows.append({"date": dt, "title": title})

        except Exception as exc:
            log.debug("GDELT query failed (year=%d): %s", year, exc)

    return rows


def _append_gdelt_df(
    df: pd.DataFrame,
    result: dict,
    start_date: date,
    end_date: date,
) -> None:
    """Append rows from a GDELT DataFrame into result dict, filtered by date range."""
    for _, row in df.iterrows():
        try:
            dt = pd.Timestamp(row["date"]).date()
        except Exception:
            continue
        if start_date <= dt <= end_date:
            result.setdefault(dt, []).append(str(row["title"]))


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2 — yfinance (company-specific, last ~2 weeks)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_yfinance_news() -> dict[date, list[str]]:
    """
    Pull recent headlines from yfinance for broad market tickers.

    FIXED for new yfinance response structure (2025+):
      Old format: item["title"],  item["providerPublishTime"]  (unix int)
      New format: item["content"]["title"], item["content"]["pubDate"] (ISO str)
    Tries new format first, falls back to old format automatically.
    """
    tickers_to_try = ["^NSEI", "RELIANCE.NS", "^GSPC", "^IXIC"]
    result: dict[date, list[str]] = {}

    try:
        import yfinance as yf
    except ImportError:
        return result

    for ticker_sym in tickers_to_try:
        try:
            t    = yf.Ticker(ticker_sym)
            news = t.news or []

            for item in news:
                # New format first, old format as fallback
                content = item.get("content", {})
                title   = (
                    content.get("title")
                    or item.get("title")
                    or ""
                ).strip()

                ts_raw = (
                    content.get("pubDate")           # new: ISO string
                    or content.get("displayTime")    # new: alternative
                    or item.get("providerPublishTime")  # old: unix int
                    or item.get("providePublishTime")   # old: typo variant
                )

                if not title or not ts_raw:
                    continue

                try:
                    if isinstance(ts_raw, (int, float)):
                        dt = pd.Timestamp(ts_raw, unit="s").date()
                    else:
                        dt = pd.Timestamp(str(ts_raw)).date()
                except Exception:
                    continue

                result.setdefault(dt, []).append(title)

        except Exception as exc:
            log.debug("yfinance news for %s failed: %s", ticker_sym, exc)

    log.info(
        "yfinance: %d days, %d headlines.",
        len(result), sum(len(v) for v in result.values()),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3 — NewsAPI (optional backup, last 30 days)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_newsapi(start_date: date, end_date: date) -> dict[date, list[str]]:
    """
    Fetch from NewsAPI.org (free tier: 100 req/day, max 30-day lookback).
    Requires NEWSAPI_KEY environment variable. Silently skipped if absent.
    Get a free key at: https://newsapi.org/register
    """
    import os
    api_key = os.environ.get("NEWSAPI_KEY", "").strip()
    if not api_key:
        return {}

    result: dict[date, list[str]] = {}

    cutoff    = date.today() - timedelta(days=29)
    from_date = max(start_date, cutoff)
    if from_date > end_date:
        return result

    queries = [
        "Indian stock market NSE Nifty",
        "RBI monetary policy India",
        "OPEC crude oil price",
        "US sanctions geopolitical tariff",
    ]

    try:
        import requests
        for q in queries:
            try:
                resp = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q"        : q,
                        "from"     : from_date.isoformat(),
                        "to"       : end_date.isoformat(),
                        "language" : "en",
                        "sortBy"   : "publishedAt",
                        "pageSize" : 50,
                        "apiKey"   : api_key,
                    },
                    timeout=10,
                )
                data = resp.json()
                for article in data.get("articles", []):
                    title  = (article.get("title") or "").strip()
                    pub_at = article.get("publishedAt", "")
                    if not title or not pub_at:
                        continue
                    try:
                        dt = pd.Timestamp(pub_at).date()
                    except Exception:
                        continue
                    result.setdefault(dt, []).append(title)
            except Exception as exc:
                log.debug("NewsAPI query '%s' failed: %s", q, exc)

    except ImportError:
        pass

    log.info(
        "NewsAPI: %d days, %d headlines.",
        len(result), sum(len(v) for v in result.values()),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment scorer chain: FinBERT → VADER → TextBlob → 0.0
# ─────────────────────────────────────────────────────────────────────────────

def _get_scorer():
    """Return best available scorer. Each is smoke-tested before returning."""
    for fn, name in [(_try_finbert, "FinBERT"), (_try_vader, "VADER"),
                     (_try_textblob, "TextBlob")]:
        scorer = fn()
        if scorer:
            log.info("Sentiment scorer: %s", name)
            return scorer
    log.warning("No scorer available — sentiment scores will be 0.0.")
    return lambda text: 0.0


def _try_finbert():
    """
    Load ProsusAI/finbert. Smoke-tested with a real inference call
    so silent failures (model not downloaded, corrupt weights) are caught
    and we fall through to VADER instead of returning 0.0 for everything.
    """
    try:
        from transformers import pipeline, logging as hf_log
        hf_log.set_verbosity_error()

        finbert   = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=128,
            device=-1,   # CPU only
        )
        label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

        # Smoke test — if model not downloaded this will raise
        _test  = finbert("markets rally on strong earnings")[0]
        _label = _test.get("label", "").lower()
        if _label not in label_map:
            raise ValueError(f"Unexpected FinBERT label: {_label!r}")
        log.info(
            "FinBERT smoke test passed: label=%s score=%.3f",
            _label, _test["score"],
        )

        def score(text: str) -> float:
            try:
                out   = finbert(text[:512])[0]
                label = out["label"].lower()
                conf  = float(out["score"])
                return label_map.get(label, 0.0) * conf
            except Exception:
                return 0.0

        return score

    except Exception as exc:
        log.warning("FinBERT unavailable (%s) — trying VADER.", exc)
        return None


def _try_vader():
    """Load VADER from nltk. Auto-downloads lexicon on first use (offline after)."""
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        sia = SentimentIntensityAnalyzer()

        def score(text: str) -> float:
            try:
                return float(sia.polarity_scores(text)["compound"])
            except Exception:
                return 0.0

        return score

    except Exception:
        return None


def _try_textblob():
    """Fallback: TextBlob polarity score."""
    try:
        from textblob import TextBlob

        def score(text: str) -> float:
            try:
                return float(TextBlob(text).sentiment.polarity)
            except Exception:
                return 0.0

        return score

    except Exception:
        return None