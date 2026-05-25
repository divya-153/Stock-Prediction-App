"""
predictor.py — Inference pipeline with recursive multi-step forecasting.

Changes vs previous version:
  - Multi-window sequences (30/60/120) fed into embedder
  - Fused vector normalised with fused_scaler before LightGBM
  - Risk score from dedicated LightGBM classifier (spec §6)
  - Wick reconstruction (spec §4):
      High = max(Open, Close) + upper_wick_adj
      Low  = min(Open, Close) - lower_wick_adj
      where wick adjustments are derived from the predicted ATR
  - log-range inverse: range = exp(pred_log_range) - epsilon
  - [NEW] Recursive multi-step forecasting for any future date:
      predict_multi_step() iterates day-by-day, appending predicted
      OHLCV to the synthetic history so lag/rolling features update
      correctly at each step. predict() routes automatically.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_ohlcv
from src.feature_engineering import (
    FEATURE_COLS,
    TARGET_COLS,
    build_features,
    build_features_for_inference,   # NEW — inference-safe variant
    calendar_features_for_date,
)
from src.market_calendar import is_market_open
from src.sequence_builder import MAX_SEQ, build_inference_sequence
from src.trainer import load_artifacts
from src.utils import get_logger, RISK_RANGE_EXPANSION

log = get_logger(__name__)

CALENDAR_COLS = ["day_of_week", "week_of_year", "month", "day_of_year"]

# Risk label thresholds (spec §6)
_RISK_THRESHOLDS = [
    (0.25, "LOW"),
    (0.55, "MODERATE"),
    (0.80, "HIGH"),
    (1.01, "EXTREME"),
]


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Trading day enumeration helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_trading_days(start_date: date, end_date: date) -> list[date]:
    """
    Return all trading days in (start_date, end_date] — exclusive of
    start_date, inclusive of end_date — skipping weekends and NSE holidays.

    Used by the recursive engine to enumerate the steps it must traverse.
    """
    days: list[date] = []
    current = start_date + timedelta(days=1)
    while current <= end_date:
        is_open, _ = is_market_open(current)
        if is_open:
            days.append(current)
        current += timedelta(days=1)
    return days


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Volume estimation for synthetic rows
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_volume(df_synthetic: pd.DataFrame) -> float:
    """
    Return a volume estimate for the next synthetic row.
    Uses the 5-day rolling average of the Volume column.
    Volume is not modelled; a stable average prevents cascade errors
    in volume-based features (volume_spike, avg_volume_5, etc.).
    """
    vol = df_synthetic["Volume"].dropna()
    if len(vol) == 0:
        return 1_000_000.0
    return float(vol.tail(5).mean())


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Single-step inference on a growing synthetic history
# ─────────────────────────────────────────────────────────────────────────────

def _predict_one_step(
    step_date: date,
    df_synthetic: pd.DataFrame,
    artifacts: dict,
) -> tuple[dict, pd.Series]:
    """
    Run the full inference pipeline for step_date using df_synthetic as
    the history (which may contain previously-predicted synthetic rows).

    Returns
    -------
    pred_dict : dict with Open, High, Low, Close, risk_score, risk_label, …
    new_row   : pd.Series (OHLCV) to append to df_synthetic for the next step
    """
    embedder         = artifacts["embedder"]
    lgbm_models      = artifacts["lgbm_models"]
    risk_model       = artifacts["risk_model"]
    feat_scaler      = artifacts["feat_scaler"]
    fused_scaler     = artifacts["fused_scaler"]
    tgt_scalers      = artifacts["tgt_scalers"]
    spike_lgbm_model = artifacts.get("spike_lgbm_model")   # optional — may be None

    future_ts = pd.Timestamp(step_date)

    # 1. Build features on the full growing history (inference-safe: no dropna on tail)
    df_feat = build_features_for_inference(df_synthetic)

    # Slice: only rows strictly before step_date (no leakage)
    df_ctx = df_feat[df_feat.index < future_ts]

    if len(df_ctx) < MAX_SEQ:
        raise ValueError(
            f"Multi-step: need ≥{MAX_SEQ} rows before {step_date}, "
            f"have {len(df_ctx)} after feature build. "
            "Ensure sufficient real history exists before the forecast horizon."
        )

    # 2. Multi-window sequences from the last MAX_SEQ rows of context
    X_seq, X_flat = build_inference_sequence(df_ctx.iloc[-MAX_SEQ:], feat_scaler)

    # 3. Calendar override — use step_date's exact calendar values
    cal         = calendar_features_for_date(future_ts)
    cal_indices = [FEATURE_COLS.index(c) for c in CALENDAR_COLS]
    X_flat_adj  = X_flat.copy()
    for name, idx in zip(CALENDAR_COLS, cal_indices):
        X_flat_adj[0, idx] = float(cal[name])

    # 4. Macro override — last known snapshot up to step_date
    from src.macro_features import get_macro_snapshot, MACRO_FEATURE_COLS
    macro_snap = get_macro_snapshot(step_date)
    for mac_col, mac_val in macro_snap.items():
        if mac_col in FEATURE_COLS:
            X_flat_adj[0, FEATURE_COLS.index(mac_col)] = float(mac_val)

    # 5. Sentiment override — last known snapshot up to step_date
    from src.sentiment_features import get_sentiment_snapshot, SENTIMENT_FEATURE_COLS
    sentiment_snap = get_sentiment_snapshot(step_date)
    for sent_col, sent_val in sentiment_snap.items():
        if sent_col in FEATURE_COLS:
            X_flat_adj[0, FEATURE_COLS.index(sent_col)] = float(sent_val)

    # 6. Deep embedding + LightGBM
    embedding    = embedder.predict(X_seq, verbose=0) * 0.5
    X_fused_raw  = np.hstack([embedding, X_flat_adj])
    X_fused_norm = fused_scaler.transform(X_fused_raw)
    pred_scaled  = {t: float(m.predict(X_fused_norm)[0]) for t, m in lgbm_models.items()}

    # 7. Risk scoring
    risk_score, risk_label = _score_risk(risk_model, X_fused_norm, df_ctx)

    # 8. OHLC reconstruction (close-first hierarchy)
    prev_close = float(df_ctx["Close"].iloc[-1])

    if "target_close" in tgt_scalers:
        close_pred = _inv(tgt_scalers, "target_close", pred_scaled["target_close"])
    else:
        log.warning("target_close scaler not found — using closepos fallback.")
        log_range_raw_fb = _inv(tgt_scalers, "target_log_range", pred_scaled["target_log_range"])
        _range_fb        = max(float(np.exp(log_range_raw_fb)) - 1e-6, 0.0)
        closepos_fb      = float(np.clip(
            _inv(tgt_scalers, "target_closepos", pred_scaled["target_closepos"]), 0.0, 1.0
        ))
        close_pred = prev_close - _range_fb / 2.0 + _range_fb * closepos_fb

    # Bug 4 FIX: risk-conditional open blending.
    # OLD: fixed 0.6/0.4 split pulled open toward prev_close on ALL days,
    #      systematically under-predicting gap openings on spike/news days.
    # NEW: on HIGH/EXTREME risk days trust the model almost fully.
    open_model = _inv(tgt_scalers, "target_open", pred_scaled["target_open"])
    _open_blend = {
        "LOW"     : (0.55, 0.45),
        "MODERATE": (0.65, 0.35),
        "HIGH"    : (0.80, 0.20),
        "EXTREME" : (0.95, 0.05),
    }.get(risk_label, (0.60, 0.40))
    open_pred = _open_blend[0] * open_model + _open_blend[1] * prev_close

    # Improvement 9: sensible minimum range floor (prevents degenerate near-zero predictions)
    _min_range = float((df_ctx["High"] - df_ctx["Low"]).tail(10).mean()) * 0.10
    log_range_raw = _inv(tgt_scalers, "target_log_range", pred_scaled["target_log_range"])
    range_model   = max(float(np.exp(log_range_raw)) - 1e-6, _min_range)

    recent_range = (df_ctx["High"] - df_ctx["Low"]).tail(15)
    avg_range    = float(recent_range.mean())
    std_range    = float(recent_range.std())

    range_pred = 0.6 * range_model + 0.4 * avg_range
    if std_range > 0:
        range_pred += 0.15 * std_range

    expansion_factor = RISK_RANGE_EXPANSION.get(risk_label, 1.0)
    range_pred = range_pred * expansion_factor

    # Improvement 8: spike specialist model blending on HIGH/EXTREME risk days.
    # The specialist was trained exclusively on spike-day rows and captures the
    # tail distribution that the general model under-weights.
    if spike_lgbm_model is not None and risk_label in ("HIGH", "EXTREME"):
        spike_log_range_raw = _inv(
            tgt_scalers, "target_log_range",
            float(spike_lgbm_model.predict(X_fused_norm)[0])
        )
        spike_range = max(float(np.exp(spike_log_range_raw)) - 1e-6, _min_range)
        blend_w = 0.45 if risk_label == "HIGH" else 0.60   # give specialist more weight on EXTREME
        range_pred = (1.0 - blend_w) * range_pred + blend_w * spike_range
        log.info("Spike specialist blend (%s): general=%.2f spike=%.2f → blended=%.2f",
                 risk_label, range_pred / (1 - blend_w + 1e-9), spike_range, range_pred)

    event_shock_val = int(sentiment_snap.get("event_shock", 0))
    if event_shock_val == 1 and expansion_factor < 1.20:
        range_pred = range_pred * 1.20

    # Bug 2 FIX: risk-aware upper clip instead of hard 90th-percentile cap.
    # OLD: capped at quantile(0.90) — spike days (which ARE the 90th–100th percentile)
    #      were systematically clipped to the point of being meaningless.
    # NEW: cap scales with risk so HIGH/EXTREME days can exceed recent history.
    _upper_cap_mult = {
        "LOW"     : 0.90,
        "MODERATE": 1.10,
        "HIGH"    : 1.50,
        "EXTREME" : 3.00,
    }.get(risk_label, 1.0)
    range_pred = float(np.clip(
        range_pred,
        recent_range.quantile(0.10),
        recent_range.quantile(0.90) * _upper_cap_mult,
    ))

    # Improvement 4: risk floor — if risk is HIGH/EXTREME but model predicts
    # a small range (model uncertainty), enforce a minimum based on ATR.
    if risk_label in ("HIGH", "EXTREME"):
        last_atr = float(df_ctx.get("atr_14", pd.Series([avg_range])).dropna().iloc[-1])
        atr_floor = last_atr * {"HIGH": 1.20, "EXTREME": 1.50}.get(risk_label, 1.0)
        if range_pred < atr_floor:
            log.info("Risk floor applied (%s): range_pred %.2f → %.2f (ATR floor)",
                     risk_label, range_pred, atr_floor)
            range_pred = atr_floor

    pct_change = (close_pred - prev_close) / prev_close * 100.0

    open_pred, high_pred, low_pred, close_pred = _reconstruct_ohlc_close_first(
        open_pred, close_pred, range_pred, df_ctx
    )

    # 9. Build result dict
    pred_dict: dict[str, Any] = {
        "market_closed"       : False,
        "mode"                : "future",
        "date"                : step_date,
        "Open"                : round(open_pred,  2),
        "High"                : round(high_pred,  2),
        "Low"                 : round(low_pred,   2),
        "Close"               : round(close_pred, 2),
        "prev_close"          : round(prev_close, 2),
        "pct_change"          : round(pct_change, 4),
        "range"               : round(range_pred, 2),
        "risk_score"          : round(risk_score, 3),
        "risk_label"          : risk_label,
        "sentiment_score"     : round(float(sentiment_snap.get("sentiment_score", 0.0)), 3),
        "sentiment_magnitude" : round(float(sentiment_snap.get("sentiment_magnitude", 0.0)), 3),
        "event_shock"         : event_shock_val,
        "active_events"       : _active_event_labels(sentiment_snap),
        "event_shock_reason"  : _event_shock_reason(sentiment_snap),
        "step_from_last_real" : None,   # filled by caller
    }

    # 10. Build the synthetic OHLCV row to append for the next step
    new_row = pd.Series({
        "Open"  : open_pred,
        "High"  : high_pred,
        "Low"   : low_pred,
        "Close" : close_pred,
        "Volume": _estimate_volume(df_synthetic),
    }, name=future_ts)

    return pred_dict, new_row


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Recursive multi-step forecasting engine
# ─────────────────────────────────────────────────────────────────────────────

def predict_multi_step(ticker: str, target_date: date) -> dict[str, Any]:
    """
    Recursive multi-step forecasting for any future target_date.

    Algorithm:
      1. Enumerate all trading days from last_real_date+1 to target_date.
      2. For each step, predict using the current synthetic history.
      3. Append predicted OHLCV row to the history.
      4. Re-run feature engineering on the updated history for the next step.
      5. Return the prediction for target_date.

    This ensures:
      - Each intermediate step sees its predecessors' predicted values
        through lag, rolling, range, and candle-structure features.
      - Calendar features (day_of_week, month, etc.) are exact for each step.
      - Macro/sentiment features use the last-known values (correct for future).
      - No repeated/static predictions across steps.

    For dates <= last real data date, delegates to predict() (backtest path).
    """
    # Market open check
    is_open, reason = is_market_open(target_date)
    if not is_open:
        return {"market_closed": True, "reason": reason, "date": target_date}

    # Load real data and artifacts once (shared across all steps)
    artifacts      = load_artifacts(ticker)
    df_raw         = load_ohlcv(ticker, use_cache=True)
    last_real_date = df_raw.index[-1].date()

    # Enumerate the trading days we need to traverse
    trading_days = _get_trading_days(last_real_date, target_date)

    if len(trading_days) == 0:
        # target_date is on or before the last real date — use backtest path
        return predict(ticker, target_date)

    log.info(
        "[%s] Multi-step forecast: %d trading step(s) from %s to %s",
        ticker, len(trading_days), last_real_date, target_date,
    )

    # Start with the real OHLCV history; grows with each predicted step
    df_synthetic = df_raw.copy()
    last_pred: dict[str, Any] = {}

    for step_num, step_date in enumerate(trading_days, start=1):
        log.info(
            "[%s] Step %d/%d — predicting %s",
            ticker, step_num, len(trading_days), step_date,
        )
        pred_dict, new_row = _predict_one_step(step_date, df_synthetic, artifacts)
        pred_dict["step_from_last_real"] = step_num
        last_pred = pred_dict

        # Append the synthetic OHLCV row so subsequent steps see it as history
        new_df = pd.DataFrame(
            [[new_row["Open"], new_row["High"], new_row["Low"],
              new_row["Close"], new_row["Volume"]]],
            index=[pd.Timestamp(step_date)],
            columns=["Open", "High", "Low", "Close", "Volume"],
        )
        df_synthetic = pd.concat([df_synthetic, new_df])
        df_synthetic.index = pd.to_datetime(df_synthetic.index)
        # Guard against duplicate index entries (shouldn't happen, but safe)
        df_synthetic = df_synthetic[~df_synthetic.index.duplicated(keep="last")]

    return last_pred


# ─────────────────────────────────────────────────────────────────────────────
# Public API  (UPDATED — routes multi-step calls through new engine)
# ─────────────────────────────────────────────────────────────────────────────

def predict(ticker: str, target_date: date) -> dict[str, Any]:
    """
    Unified prediction. Returns one of three shapes:
      market_closed → {"market_closed": True, "reason": ..., "date": ...}
      future/1-step → {"market_closed": False, "mode": "future", OHLC, risk, ...}
      future/multi  → same shape, produced by predict_multi_step()
      backtest      → future + actual OHLC + error metrics

    Routing:
      - If target_date is more than 1 trading day beyond the last real data,
        predict_multi_step() is called automatically.
      - Single-step future (tomorrow) and all backtest dates use the
        original single-step inference path below.
    """
    is_open, reason = is_market_open(target_date)
    if not is_open:
        return {"market_closed": True, "reason": reason, "date": target_date}

    mode = "future" if target_date >= date.today() else "backtest"

    # For future predictions only: check whether multi-step routing is needed
    if mode == "future":
        df_raw_check   = load_ohlcv(ticker, use_cache=True)
        last_real_date = df_raw_check.index[-1].date()
        trading_days   = _get_trading_days(last_real_date, target_date)

        if len(trading_days) > 1:
            # More than one step ahead — delegate to recursive engine
            return predict_multi_step(ticker, target_date)
        # len == 1 or 0: fall through to single-step logic below

    # ── Original single-step inference (preserved exactly) ────────────────────
    artifacts        = load_artifacts(ticker)
    embedder         = artifacts["embedder"]
    lgbm_models      = artifacts["lgbm_models"]
    risk_model       = artifacts["risk_model"]
    feat_scaler      = artifacts["feat_scaler"]
    fused_scaler     = artifacts["fused_scaler"]
    tgt_scalers      = artifacts["tgt_scalers"]
    spike_lgbm_model = artifacts.get("spike_lgbm_model")   # optional

    df_raw    = load_ohlcv(ticker, use_cache=True)
    df        = build_features(df_raw)
    future_ts = pd.Timestamp(target_date)

    df_past = df[df.index < future_ts]
    if len(df_past) < MAX_SEQ:
        raise ValueError(
            f"Need ≥{MAX_SEQ} rows before {target_date}, have {len(df_past)}."
        )

    # Multi-window sequences
    X_seq, X_flat = build_inference_sequence(df_past.iloc[-MAX_SEQ:], feat_scaler)

    # Override calendar features with target date's values
    cal         = calendar_features_for_date(future_ts)
    cal_indices = [FEATURE_COLS.index(c) for c in CALENDAR_COLS]
    X_flat_adj  = X_flat.copy()
    for name, idx in zip(CALENDAR_COLS, cal_indices):
        X_flat_adj[0, idx] = float(cal[name])

    # Override macro features with the latest available macro snapshot
    from src.macro_features import get_macro_snapshot, MACRO_FEATURE_COLS
    macro_snap = get_macro_snapshot(target_date)
    for mac_col, mac_val in macro_snap.items():
        if mac_col in FEATURE_COLS:
            mac_idx = FEATURE_COLS.index(mac_col)
            X_flat_adj[0, mac_idx] = float(mac_val)

    # Override sentiment features with the latest available sentiment snapshot
    from src.sentiment_features import get_sentiment_snapshot, SENTIMENT_FEATURE_COLS
    sentiment_snap = get_sentiment_snapshot(target_date)
    for sent_col, sent_val in sentiment_snap.items():
        if sent_col in FEATURE_COLS:
            sent_idx = FEATURE_COLS.index(sent_col)
            X_flat_adj[0, sent_idx] = float(sent_val)

    # Deep embedding
    embedding = embedder.predict(X_seq, verbose=0) * 0.5     # (1, emb_dim)
    X_fused_raw  = np.hstack([embedding, X_flat_adj])        # (1, emb_dim + n_feat)
    X_fused_norm = fused_scaler.transform(X_fused_raw)       # normalised

    # Predict OHLC components
    pred_scaled = {t: float(m.predict(X_fused_norm)[0]) for t, m in lgbm_models.items()}

    # Risk score first — needed to expand range before reconstruction
    risk_score, risk_label = _score_risk(risk_model, X_fused_norm, df_past)

    # Close-first hierarchy
    prev_close = float(df_past["Close"].iloc[-1])
    if "target_close" in tgt_scalers:
        close_pred = _inv(tgt_scalers, "target_close", pred_scaled["target_close"])
    else:
        log.warning("target_close scaler not found — falling back to closepos reconstruction.")
        log_range_raw_fb = _inv(tgt_scalers, "target_log_range", pred_scaled["target_log_range"])
        _range_fb        = max(float(np.exp(log_range_raw_fb)) - 1e-6, 0.0)
        closepos_fb      = float(np.clip(
            _inv(tgt_scalers, "target_closepos", pred_scaled["target_closepos"]), 0.0, 1.0
        ))
        close_pred = prev_close - _range_fb / 2.0 + _range_fb * closepos_fb

    # Bug 4 FIX: risk-conditional open blending (same logic as _predict_one_step).
    open_model = _inv(tgt_scalers, "target_open", pred_scaled["target_open"])
    _open_blend = {
        "LOW"     : (0.55, 0.45),
        "MODERATE": (0.65, 0.35),
        "HIGH"    : (0.80, 0.20),
        "EXTREME" : (0.95, 0.05),
    }.get(risk_label, (0.60, 0.40))
    open_pred = _open_blend[0] * open_model + _open_blend[1] * prev_close

    # Improvement 9: sensible minimum range floor
    _min_range = float((df_past["High"] - df_past["Low"]).tail(10).mean()) * 0.10
    log_range_raw = _inv(tgt_scalers, "target_log_range", pred_scaled["target_log_range"])
    range_model   = max(float(np.exp(log_range_raw)) - 1e-6, _min_range)

    recent_range = (df_past["High"] - df_past["Low"]).tail(15)
    avg_range    = float(recent_range.mean())
    std_range    = float(recent_range.std())

    range_pred = 0.6 * range_model + 0.4 * avg_range
    if std_range > 0:
        range_pred += 0.15 * std_range

    # Risk-aware range expansion
    expansion_factor = RISK_RANGE_EXPANSION.get(risk_label, 1.0)
    range_pred = range_pred * expansion_factor
    if expansion_factor > 1.0:
        log.info(
            "Risk-aware expansion: %s × %.2f → range_pred=%.4f",
            risk_label, expansion_factor, range_pred,
        )

    # Improvement 8: spike specialist blending on HIGH/EXTREME days
    if spike_lgbm_model is not None and risk_label in ("HIGH", "EXTREME"):
        spike_log_range_raw = _inv(
            tgt_scalers, "target_log_range",
            float(spike_lgbm_model.predict(X_fused_norm)[0])
        )
        spike_range = max(float(np.exp(spike_log_range_raw)) - 1e-6, _min_range)
        blend_w = 0.45 if risk_label == "HIGH" else 0.60
        range_pred = (1.0 - blend_w) * range_pred + blend_w * spike_range
        log.info("Spike specialist blend (%s): blended range=%.2f", risk_label, range_pred)

    # Additional event_shock expansion (minimum 20% on active shock days)
    event_shock_val = int(sentiment_snap.get("event_shock", 0))
    if event_shock_val == 1 and expansion_factor < 1.20:
        range_pred = range_pred * 1.20
        log.info("event_shock=1 — range expanded by 20%%.")

    # Bug 2 FIX: risk-aware upper clip (replaces hard 90th-percentile cap).
    _upper_cap_mult = {
        "LOW"     : 0.90,
        "MODERATE": 1.10,
        "HIGH"    : 1.50,
        "EXTREME" : 3.00,
    }.get(risk_label, 1.0)
    range_pred = float(np.clip(
        range_pred,
        recent_range.quantile(0.10),
        recent_range.quantile(0.90) * _upper_cap_mult,
    ))

    # Improvement 4: ATR-based risk floor — prevents model returning tiny range on HIGH/EXTREME
    if risk_label in ("HIGH", "EXTREME"):
        last_atr = float(df_past.get("atr_14", pd.Series([avg_range])).dropna().iloc[-1])
        atr_floor = last_atr * {"HIGH": 1.20, "EXTREME": 1.50}.get(risk_label, 1.0)
        if range_pred < atr_floor:
            log.info("Risk floor (%s): range_pred %.2f → %.2f", risk_label, range_pred, atr_floor)
            range_pred = atr_floor

    pct_change = (close_pred - prev_close) / prev_close * 100.0

    # Close-first OHLC reconstruction (spec §4)
    open_pred, high_pred, low_pred, close_pred = _reconstruct_ohlc_close_first(
        open_pred, close_pred, range_pred, df_past
    )

    result: dict[str, Any] = {
        "market_closed"       : False,
        "mode"                : mode,
        "date"                : target_date,
        "Open"                : round(open_pred,  2),
        "High"                : round(high_pred,  2),
        "Low"                 : round(low_pred,   2),
        "Close"               : round(close_pred, 2),
        "prev_close"          : round(prev_close, 2),
        "pct_change"          : round(pct_change, 4),
        "range"               : round(range_pred, 2),
        "risk_score"          : round(risk_score, 3),
        "risk_label"          : risk_label,
        "sentiment_score"     : round(float(sentiment_snap.get("sentiment_score", 0.0)), 3),
        "sentiment_magnitude" : round(float(sentiment_snap.get("sentiment_magnitude", 0.0)), 3),
        "event_shock"         : int(sentiment_snap.get("event_shock", 0)),
        "active_events"       : _active_event_labels(sentiment_snap),
        "event_shock_reason"  : _event_shock_reason(sentiment_snap),
    }

    if mode == "backtest":
        result = _attach_actuals(result, df_raw, future_ts)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Wick-aware OHLC reconstruction  (spec §4)
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_ohlc(
    open_pred: float,
    range_pred: float,
    closepos_pred: float,
    df_past: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """
    Spec §4:
      Close = Low + Range × close_position
      High  = max(Open, Close) + upper_wick_adj
      Low   = min(Open, Close) - lower_wick_adj

    Wick adjustments are estimated from the most recent ATR:
      upper_wick_adj = avg_upper_shadow_ratio × range_pred
      lower_wick_adj = avg_lower_shadow_ratio × range_pred

    This produces realistic candles instead of symmetric ones.
    """
    hist  = df_past.iloc[-20:]
    rng_h = (hist["High"] - hist["Low"]).replace(0, np.nan)
    body_hi = hist[["Open", "Close"]].max(axis=1)
    body_lo = hist[["Open", "Close"]].min(axis=1)
    upper_ratios = ((hist["High"] - body_hi) / rng_h).clip(0, 0.5)
    lower_ratios = ((body_lo - hist["Low"])  / rng_h).clip(0, 0.5)

    upper_wick_adj = float(upper_ratios.median()) * range_pred
    lower_wick_adj = float(lower_ratios.median()) * range_pred

    low_base   = open_pred - range_pred / 2.0
    high_base  = open_pred + range_pred / 2.0
    close_pred = low_base + range_pred * closepos_pred

    body_top = max(open_pred, close_pred)
    body_bot = min(open_pred, close_pred)
    high_pred = body_top + upper_wick_adj
    low_pred  = body_bot - lower_wick_adj

    high_pred = max(high_pred, open_pred, close_pred)
    low_pred  = min(low_pred,  open_pred, close_pred)

    return open_pred, high_pred, low_pred, close_pred


# ─────────────────────────────────────────────────────────────────────────────
# Close-first OHLC reconstruction  (spec §4, updated)
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_ohlc_close_first(
    open_pred: float,
    close_pred: float,
    range_pred: float,
    df_past: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """
    Close-first OHLC reconstruction (spec §4, updated).

    Inputs
    ------
    open_pred  : predicted open (blended with prev_close in caller)
    close_pred : predicted close (from dedicated target_close LightGBM head)
    range_pred : predicted High-Low range (risk-expanded by caller)
    df_past    : historical OHLCV used for wick ratio estimation

    Logic
    -----
    1. Body boundaries are derived from the two directly-predicted values
       (open and close) — no indirect reconstruction from closepos.
    2. Upper and lower wick adjustments are estimated from the median wick
       ratios of the last 20 historical candles, scaled by range_pred.
    3. Hard constraints ensure OHLC sanity:
         High ≥ max(Open, Close),  Low ≤ min(Open, Close)
    """
    hist     = df_past.iloc[-20:]
    rng_h    = (hist["High"] - hist["Low"]).replace(0, np.nan)
    body_hi  = hist[["Open", "Close"]].max(axis=1)
    body_lo  = hist[["Open", "Close"]].min(axis=1)
    upper_ratios = ((hist["High"] - body_hi) / rng_h).clip(0, 0.5)
    lower_ratios = ((body_lo - hist["Low"])  / rng_h).clip(0, 0.5)

    upper_wick_adj = float(upper_ratios.median()) * range_pred
    lower_wick_adj = float(lower_ratios.median()) * range_pred

    body_top  = max(open_pred, close_pred)
    body_bot  = min(open_pred, close_pred)
    high_pred = body_top + upper_wick_adj
    low_pred  = body_bot - lower_wick_adj

    high_pred = max(high_pred, open_pred, close_pred)
    low_pred  = min(low_pred,  open_pred, close_pred)

    return open_pred, high_pred, low_pred, close_pred


# ─────────────────────────────────────────────────────────────────────────────
# Risk scoring  (spec §6)
# ─────────────────────────────────────────────────────────────────────────────

def _score_risk(
    risk_model, X_fused_norm: np.ndarray, df_past: pd.DataFrame
) -> tuple[float, str]:
    """
    Use the dedicated LightGBM risk classifier to output a probability score.
    Falls back to the feature-based heuristic if the model is missing.
    """
    if risk_model is not None:
        try:
            if hasattr(risk_model, "predict_proba"):
                score = float(risk_model.predict_proba(X_fused_norm)[0, 1])
            else:
                score = float(np.clip(risk_model.predict(X_fused_norm)[0], 0.0, 1.0))
        except Exception as exc:
            log.warning("Risk model predict failed (%s), using heuristic.", exc)
            score = _heuristic_risk(df_past)
    else:
        score = _heuristic_risk(df_past)

    label = next(lbl for thresh, lbl in _RISK_THRESHOLDS if score < thresh)
    return score, label


def _heuristic_risk(df_past: pd.DataFrame) -> float:
    """
    Fallback heuristic risk score when no trained risk model is available.
    """
    last = df_past.iloc[-1]

    expansion = float(last.get("range_expansion_ratio", 1.0))
    vol_spike  = float(last.get("volume_spike",          1.0))
    z          = abs(float(last.get("range_z_score_10",  0.0)))

    oil_ret = float(last.get("oil_avg_ret1", 0.0))
    fx_ret  = float(last.get("usdinr_ret1",  0.0))

    price_component = (
        0.30 * min((expansion - 1.0) / 2.0, 1.0)
      + 0.20 * min((vol_spike  - 1.0) / 3.0, 1.0)
      + 0.15 * min(z / 3.0, 1.0)
    )
    oil_component = 0.20 * min(max(oil_ret / 0.05, 0.0), 1.0)
    fx_component  = 0.15 * min(max(fx_ret  / 0.01, 0.0), 1.0)

    score = price_component + oil_component + fx_component
    return float(np.clip(score, 0.0, 1.0))


def _active_event_labels(sentiment_snap: dict) -> list[str]:
    """
    Return human-readable labels for every event flag that fired today.
    Used for UI display only.
    """
    label_map = {
        "event_ceasefire"   : "Ceasefire/War",
        "event_sanctions"   : "Sanctions/Tariff",
        "event_opec"        : "OPEC/Oil",
        "event_central_bank": "Central Bank",
        "event_fiscal"      : "Budget/Fiscal",
    }
    return [
        label
        for col, label in label_map.items()
        if int(sentiment_snap.get(col, 0)) == 1
    ]


def _event_shock_reason(sentiment_snap: dict) -> str:
    """
    Build a specific, human-readable explanation of what triggered the event shock.
    Returns a detailed string describing each active event type, or an empty
    string if no shock is active.
    """
    if int(sentiment_snap.get("event_shock", 0)) != 1:
        return ""

    reason_map = {
        "event_ceasefire"   : "Ceasefire / War / Geopolitical conflict detected",
        "event_sanctions"   : "Sanctions / Tariff / Trade-war action detected",
        "event_opec"        : "OPEC decision / Oil supply shock detected",
        "event_central_bank": "Central bank rate action / monetary policy shock detected",
        "event_fiscal"      : "Budget / Fiscal stimulus / Government spending shock detected",
    }
    active = [
        reason_map[col]
        for col in reason_map
        if int(sentiment_snap.get(col, 0)) == 1
    ]

    if not active:
        # event_shock=1 but no individual flag fired — composite/macro shock
        return "Composite macro / geopolitical shock detected (multiple signals above threshold)"

    return " · ".join(active)


# ─────────────────────────────────────────────────────────────────────────────
# Backtest helper
# ─────────────────────────────────────────────────────────────────────────────

def _attach_actuals(result: dict, df_raw: pd.DataFrame, ts: pd.Timestamp) -> dict:
    rows = df_raw[df_raw.index == ts]
    if rows.empty:
        result.update({
            "actual_Open": None, "actual_High": None,
            "actual_Low": None,  "actual_Close": None,
            "mae": None, "rmse": None, "data_available": False,
        })
        return result

    r  = rows.iloc[0]
    aO, aH, aL, aC = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])
    predicted = [result["Open"], result["High"], result["Low"], result["Close"]]
    actuals   = [aO, aH, aL, aC]
    # R² for a single-date prediction: compute on the 4 OHLC values.
    # This reflects how well the predicted OHLC structure matches actuals.
    # Values near 1.0 = near-perfect; near 0 = predicts mean; negative = worse than mean.
    r2 = float(r2_score(actuals, predicted))
    result.update({
        "actual_Open"  : round(aO, 2),
        "actual_High"  : round(aH, 2),
        "actual_Low"   : round(aL, 2),
        "actual_Close" : round(aC, 2),
        "mae"          : round(mean_absolute_error(actuals, predicted), 4),
        "rmse"         : round(np.sqrt(mean_squared_error(actuals, predicted)), 4),
        "r2"           : round(r2, 4),
        "err_open"     : round(abs(result["Open"]  - aO), 2),
        "err_high"     : round(abs(result["High"]  - aH), 2),
        "err_low"      : round(abs(result["Low"]   - aL), 2),
        "err_close"    : round(abs(result["Close"] - aC), 2),
        "data_available": True,
    })
    return result


def _inv(tgt_scalers: dict, key: str, value: float) -> float:
    return float(tgt_scalers[key].inverse_transform([[value]])[0, 0])