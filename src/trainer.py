"""
trainer.py — Full training pipeline (upgraded).

Steps:
  1. Feature engineering
  2. Multi-window sequence building (30 / 60 / 120 days)
  3. Walk-forward validation (evaluation only — no leakage)
  4. Optuna tuning: deep model
  5. Train deep model on full data → extract embeddings
  6. Fuse embeddings + scaled tabular features
  7. Optuna tuning + train 3 LightGBM OHLC models
  8. Optuna tuning + train separate LightGBM risk model (spec §6)
  9. Save all artifacts

Spec constraints enforced:
  - No random splits (walk-forward only)
  - Scalers fitted only on train fold data (no leakage)
  - Transformer mandatory (inside build_hybrid_model)
  - Risk model is a separate LightGBM (§6)

FIX (v2): Robust Keras save/load with multi-format fallback:
  Tries  .keras  →  .h5  →  SavedModel directory
  so training never silently loses the embedder on older TF builds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler

from src.deep_model import (
    build_hybrid_model,
    extract_embeddings,
    train_deep_model,
)
from src.feature_engineering import FEATURE_COLS, TARGET_COLS, build_features
from src.optuna_tuner import (
    DEEP_HP_DEFAULTS,
    LGB_HP_DEFAULTS,
    RISK_LGB_DEFAULTS,
    tune_deep_model,
    tune_lgbm,
    tune_risk_model,
)
from src.sequence_builder import MAX_SEQ, build_sequences
from src.utils import (
    SEED,
    CLOSE_LOSS_WEIGHT,
    deep_model_path,
    embedder_path,
    get_logger,
    lgbm_path,
    risk_model_path,
    scaler_path,
    spike_lgbm_path,
)
from src.walk_forward import generate_folds

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Feature columns used in the fused vector (tabular part)
# ─────────────────────────────────────────────────────────────────────────────
_RISK_FEAT_COLS = [
    "atr_14", "volume_spike", "range_expansion_ratio",
    "range_z_score_10", "range_1", "avg_range_10",
]


# ─────────────────────────────────────────────────────────────────────────────
# Keras save / load helpers  (multi-format fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _keras_save(model, base_path: Path) -> Path:
    """
    Save a Keras model trying formats in order until one succeeds:
      1. <base_path>          (.keras  — Keras 3 / TF ≥ 2.12)
      2. <base_path>.h5       (legacy HDF5 — works on any TF 2.x)
      3. <base_path>_savedmodel  (TF SavedModel directory)

    Returns the path that was actually written.
    Raises RuntimeError if all three attempts fail.
    """
    attempts = [
        (str(base_path),                       "native .keras"),
        (str(base_path.with_suffix(".h5")),    ".h5 fallback"),
        (str(base_path.parent / (base_path.stem + "_savedmodel")), "SavedModel dir"),
    ]
    last_exc = None
    for path_str, label in attempts:
        try:
            model.save(path_str)
            log.info("Keras model saved (%s): %s", label, path_str)
            return Path(path_str)
        except Exception as exc:
            log.warning("Keras save attempt '%s' failed: %s", label, exc)
            last_exc = exc

    raise RuntimeError(
        f"All Keras save attempts failed for {base_path}. Last error: {last_exc}\n"
        "Check that TensorFlow is properly installed: pip install tensorflow>=2.14"
    ) from last_exc


def _keras_load(base_path: Path):
    """
    Load a Keras model trying the same fallback chain used by _keras_save.
    Raises FileNotFoundError if none of the expected paths exist.
    """
    import tensorflow as tf

    candidates = [
        base_path,
        base_path.with_suffix(".h5"),
        base_path.parent / (base_path.stem + "_savedmodel"),
    ]
    for p in candidates:
        if p.exists():
            log.info("Loading Keras model from: %s", p)
            return tf.keras.models.load_model(str(p))

    raise FileNotFoundError(
        f"No Keras model found at any of: {[str(c) for c in candidates]}.\n"
        "Run training first:  python train.py --ticker TICKER"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def train(
    ticker: str,
    df_raw: pd.DataFrame,
    use_optuna: bool = True,
    optuna_trials: int = 20,
    deep_epochs: int = 20,
    embedding_dim: int = 64,
    run_walk_forward: bool = True,
) -> dict[str, Any]:
    from src.utils import set_global_seed
    set_global_seed(SEED)

    log.info("[%s] Step 1 — Feature engineering", ticker)
    df = build_features(df_raw)

    log.info("[%s] Step 2 — Multi-window sequence building", ticker)
    X_seq, X_flat, y_dict, feat_scaler, tgt_scalers, dates = build_sequences(
        df, fit_scalers=True
    )

    log.info("[%s] Step 3 — Walk-forward validation", ticker)
    fold_metrics: list[dict] = []
    if run_walk_forward:
        fold_metrics = _walk_forward_eval(df, feat_scaler, tgt_scalers)

    log.info("[%s] Step 4 — Tune deep model", ticker)
    best_deep_hp = DEEP_HP_DEFAULTS.copy()
    if use_optuna:
        try:
            # Tune on target_close — anchors the hierarchy (close → open → high/low).
            # Close is the primary accuracy target; the deep embedding should
            # specialise toward close prediction.
            best_deep_hp = tune_deep_model(
                X_seq,
                {"output_close": y_dict["target_close"],
                 "output_range": y_dict["target_log_range"]},
                n_trials=optuna_trials,
            )
        except Exception as exc:
            log.warning("Deep tuning failed (%s) — using defaults.", exc)

    log.info("[%s] Step 5 — Train deep model on target_close (close-first hierarchy)", ticker)
    full_model, embedder = build_hybrid_model(
        n_features    = X_seq["short"].shape[2],
        embedding_dim = embedding_dim,
        lstm_units    = best_deep_hp.get("lstm_units",    DEEP_HP_DEFAULTS["lstm_units"]),
        lstm_layers   = best_deep_hp.get("lstm_layers",   DEEP_HP_DEFAULTS["lstm_layers"]),
        num_heads     = best_deep_hp.get("num_heads",     DEEP_HP_DEFAULTS["num_heads"]),
        dropout       = best_deep_hp.get("dropout",       DEEP_HP_DEFAULTS["dropout"]),
        learning_rate = best_deep_hp.get("learning_rate", DEEP_HP_DEFAULTS["learning_rate"]),
    )
    # FIX + Improvement 3: multi-task training on close AND log_range.
    # This forces the embedding to encode volatility structure as well as price
    # level — richer signal for LightGBM on spike days.
    train_deep_model(
        full_model, X_seq,
        {"output_close": y_dict["target_close"],
         "output_range": y_dict["target_log_range"]},
        epochs=deep_epochs,
    )

    log.info("[%s] Step 6 — Extract embeddings + fuse features", ticker)
    embeddings = extract_embeddings(embedder, X_seq) * 0.5
    X_fused = np.hstack([embeddings, X_flat])

    # Normalise fused vector (spec §2)
    fused_scaler = RobustScaler()
    X_fused_norm = fused_scaler.fit_transform(X_fused)

    log.info("[%s] Step 7 — Train LightGBM OHLC models", ticker)
    lgbm_models  = {}
    best_lgbm_hp = {}

    # Improvement 2: Continuous spike-day sample weights.
    # OLD: binary event_shock flag only — missed all price-driven spikes with no news.
    # NEW: continuous weight proportional to how volatile each day was, so
    #      LightGBM gradient updates scale with actual market impact.
    #
    # Weight formula (additive, not multiplicative — avoids runaway weights):
    #   base  = clip(range_expansion_ratio, 1.0, 4.0)   ← 1× quiet, up to 4× spike
    #   boost += (CLOSE_LOSS_WEIGHT-1) × event_shock     ← extra for news-driven days
    #   final = base_weight / mean(base_weight)          ← normalise: mean weight = 1
    from src.feature_engineering import FEATURE_COLS as _FCOLS
    from src.sequence_builder import MAX_SEQ as _MAX_SEQ

    # Build range-expansion array aligned with X_flat (starts at MAX_SEQ offset)
    if "range_expansion_ratio" in _FCOLS:
        re_col_idx = _FCOLS.index("range_expansion_ratio")
        range_exp_arr = X_flat[:, re_col_idx]
        # Clip: 1.0 floor (normal days weight 1×), 4.0 ceiling (extreme spike days weight 4×)
        range_weights = np.clip(range_exp_arr, 1.0, 4.0).astype(np.float32)
    else:
        range_weights = np.ones(len(X_flat), dtype=np.float32)

    # Add event_shock boost on top (additive)
    if "event_shock" in _FCOLS:
        shock_col_idx = _FCOLS.index("event_shock")
        shock_flags   = X_flat[:, shock_col_idx].clip(0, 1)
        range_weights = range_weights + (CLOSE_LOSS_WEIGHT - 1.0) * shock_flags

    # Normalise so mean weight = 1.0 (keeps effective learning rate stable)
    range_weights /= range_weights.mean()
    base_sample_weights = range_weights

    spike_pct = float((base_sample_weights > 1.5).mean() * 100)
    log.info(
        "Sample weights: %.1f%% rows weighted >1.5×  (range-expansion + shock-day boost)",
        spike_pct,
    )

    for target in TARGET_COLS:
        y_scaled = y_dict[target]

        # target_close: double weight on ALL samples (primary accuracy target)
        # plus the shock-day boost already in base_sample_weights.
        if target == "target_close":
            sample_weights = base_sample_weights * CLOSE_LOSS_WEIGHT
        else:
            sample_weights = base_sample_weights

        lgb_hp = LGB_HP_DEFAULTS.copy()
        if use_optuna:
            try:
                lgb_hp = tune_lgbm(
                    X_fused_norm, y_scaled, target_name=target, n_trials=optuna_trials
                )
            except Exception as exc:
                log.warning("LGB tuning failed for %s (%s).", target, exc)
        best_lgbm_hp[target] = lgb_hp
        lgbm_models[target]  = _fit_lgbm_regressor(
            X_fused_norm, y_scaled, lgb_hp, sample_weight=sample_weights
        )
        log.info("  LightGBM [%s] trained.", target)

    log.info("[%s] Step 8 — Train risk model (LightGBM §6)", ticker)
    risk_model, risk_hp = _train_risk_model(
        df, X_fused_norm, use_optuna, optuna_trials
    )

    # Improvement 8: Spike specialist LightGBM for log_range on spike days only.
    # Trained on rows where range_expansion_ratio > 1.5 (yesterday was a high-vol day).
    # At inference, blended with the general model when risk = HIGH / EXTREME.
    # Uses same X_fused_norm so no extra feature engineering is required.
    spike_lgbm_model = None
    if "range_expansion_ratio" in _FCOLS:
        re_col_idx   = _FCOLS.index("range_expansion_ratio")
        spike_mask   = X_flat[:, re_col_idx] > 1.5
        n_spike      = int(spike_mask.sum())
        log.info("[%s] Spike specialist: %d spike-day rows (%.1f%% of total)",
                 ticker, n_spike, n_spike / len(X_flat) * 100)
        if n_spike >= 60:   # need enough samples to avoid overfitting
            X_spike  = X_fused_norm[spike_mask]
            y_spike  = y_dict["target_log_range"][spike_mask]
            sw_spike = base_sample_weights[spike_mask]
            spike_lgb_hp = LGB_HP_DEFAULTS.copy()
            if use_optuna:
                try:
                    spike_lgb_hp = tune_lgbm(
                        X_spike, y_spike,
                        target_name="spike_log_range",
                        n_trials=max(10, optuna_trials // 2),
                    )
                except Exception as exc:
                    log.warning("Spike LGB tuning failed (%s) — using defaults.", exc)
            spike_lgbm_model = _fit_lgbm_regressor(
                X_spike, y_spike, spike_lgb_hp, sample_weight=sw_spike
            )
            log.info("[%s] Spike specialist LightGBM trained.", ticker)
        else:
            log.warning("[%s] Too few spike rows (%d < 60) — spike specialist skipped.", ticker, n_spike)

    log.info("[%s] Step 9 — Save all artifacts", ticker)
    _save_artifacts(
        ticker, full_model, embedder, lgbm_models, risk_model,
        feat_scaler, tgt_scalers, fused_scaler, spike_lgbm_model,
    )

    log.info("[%s] Training complete.", ticker)
    return {
        "fold_metrics"      : fold_metrics,
        "best_deep_hp"      : best_deep_hp,
        "best_lgbm_hp"      : best_lgbm_hp,
        "risk_hp"           : risk_hp,
        "embedding_dim"     : embedding_dim,
        "n_features"        : X_seq["short"].shape[2],
        "spike_model_trained": spike_lgbm_model is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Risk model training  (spec §6  — extended labels + macro/sentiment features)
# ─────────────────────────────────────────────────────────────────────────────

def _train_risk_model(
    df: pd.DataFrame,
    X_fused_norm: np.ndarray,
    use_optuna: bool,
    optuna_trials: int,
):
    """
    Build and train the risk LightGBM classifier.

    LABEL LOGIC (extended):
    A day is labelled "high risk" (y=1) if ANY of the following is true:
      [A] range_expansion_ratio > 1.5   — price range expanded sharply
      [B] volume_spike > 2.0            — abnormal trading volume
      [C] oil_avg_ret1 > +0.05          — oil surged > 5% yesterday  ← NEW
      [D] usdinr_ret1  > +0.01          — INR weakened > 1% yesterday ← NEW

    Rules C and D use macro features already in FEATURE_COLS (shifted by 1 day,
    so there is zero leakage — they represent yesterday's macro moves).

    FEATURE SET:
    The full X_fused_norm vector (embedding + all tabular features, already
    normalised) is used so the classifier can exploit macro + sentiment signals
    as well as price signals.  This is strictly broader than the previous
    price-only 6-column subset.

    OUTPUT:
    risk_model.predict_proba(X)[:, 1] → P(high_risk) in [0, 1]
    This feeds directly into risk_score in predictor.py.
    """
    risk_df = df.iloc[MAX_SEQ:].copy()

    # ── BUG 1 FIX — Condition A: use TODAY's ACTUAL realized range as label ──
    # OLD (wrong): used range_expansion_ratio which is a lag-1 FEATURE — the
    #   classifier trivially memorised the feature→label mapping and learned
    #   nothing about tomorrow's volatility.
    # NEW (correct): label = "did today's actual range exceed 1.5× the 10d avg?"
    #   At TRAINING time all historical actuals are known → zero leakage.
    #   At INFERENCE time the classifier uses only lag-1 features to predict
    #   this label for the unknown future day → also zero leakage.
    today_range    = risk_df["High"] - risk_df["Low"]         # today's actual range
    hist_avg_range = (
        (df["High"] - df["Low"])
        .shift(1)
        .rolling(10, min_periods=5)
        .mean()
        .iloc[MAX_SEQ:]
        .replace(0, np.nan)
    )
    hist_std_range = (
        (df["High"] - df["Low"])
        .shift(1)
        .rolling(10, min_periods=5)
        .std()
        .iloc[MAX_SEQ:]
        .replace(0, np.nan)
    )
    # High risk: today's range > 1.5× the lag-1 rolling average  OR  > 1.5σ above it
    range_ratio  = (today_range.values / hist_avg_range.values)
    range_zscore = ((today_range.values - hist_avg_range.values) / hist_std_range.values)
    cond_price   = (range_ratio > 1.5) | (range_zscore > 1.5)

    # Condition B: volume spike on TODAY (uses today's actual Volume — valid as label)
    vol_today = risk_df["Volume"]
    avg_vol5  = risk_df["Volume"].shift(1).rolling(5, min_periods=3).mean().replace(0, np.nan)
    cond_vol  = ((vol_today / avg_vol5) > 2.5)

    # ── Condition C: oil surge > +5% (uses oil_avg_ret1, shifted inside macro) ─
    # oil_avg_ret1 is already lag-1 (yesterday's return) — zero leakage.
    if "oil_avg_ret1" in risk_df.columns:
        cond_oil = risk_df["oil_avg_ret1"] > 0.05          # +5% daily log-return
    else:
        log.warning("oil_avg_ret1 not found in df — oil spike condition disabled.")
        cond_oil = pd.Series(False, index=risk_df.index)

    # ── Condition D: INR weakening > +1% vs USD ──────────────────────────────
    # Positive usdinr_ret1 means INR weakened (more rupees per dollar).
    if "usdinr_ret1" in risk_df.columns:
        cond_fx = risk_df["usdinr_ret1"] > 0.01            # +1% daily log-return
    else:
        log.warning("usdinr_ret1 not found in df — FX spike condition disabled.")
        cond_fx = pd.Series(False, index=risk_df.index)

    # ── Combined label ────────────────────────────────────────────────────────
    y_risk = (cond_price | cond_vol | cond_oil | cond_fx).astype(int).values
    # Replace NaN positions (rolling warm-up) with 0
    y_risk = np.nan_to_num(y_risk, nan=0).astype(int)

    pos_rate = y_risk.mean() * 100
    log.info(
        "Risk labels (FIXED): %d positive / %d total (%.1f%%)  "
        "[price/zscore=%d | vol=%d | oil=%d | fx=%d]",
        int(y_risk.sum()), len(y_risk), pos_rate,
        int(np.nan_to_num(cond_price, nan=0).sum()),
        int(np.nan_to_num(cond_vol,   nan=0).sum()),
        int(cond_oil.sum()), int(cond_fx.sum()),
    )

    if y_risk.sum() < 10:
        log.warning(
            "Very few positive risk labels (%d) — classifier may underperform.",
            int(y_risk.sum()),
        )

    # ── Tune + train risk classifier ─────────────────────────────────────────
    # X_fused_norm already contains embedding + all FEATURE_COLS (macro + sentiment
    # included), so no extra feature engineering is needed here.
    risk_hp = RISK_LGB_DEFAULTS.copy()
    if use_optuna:
        try:
            risk_hp = tune_risk_model(
                X_fused_norm, y_risk, n_trials=optuna_trials
            )
        except Exception as exc:
            log.warning("Risk model tuning failed (%s) — using defaults.", exc)

    try:
        import lightgbm as lgb
        p = {**RISK_LGB_DEFAULTS, **risk_hp, "random_state": SEED, "verbose": -1}
        risk_model = lgb.LGBMClassifier(**p)
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        log.warning("LightGBM unavailable — falling back to LogisticRegression for risk model.")
        risk_model = LogisticRegression(max_iter=500, class_weight="balanced")

    risk_model.fit(X_fused_norm, y_risk)
    log.info(
        "Risk model trained | positive-class rate=%.1f%% | features=%d",
        pos_rate, X_fused_norm.shape[1],
    )
    return risk_model, risk_hp


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward evaluation (leakage-free)
# ─────────────────────────────────────────────────────────────────────────────

def _walk_forward_eval(
    df: pd.DataFrame,
    feat_scaler: RobustScaler,
    tgt_scalers: dict,
) -> list[dict]:
    from src.sequence_builder import build_sequences

    folds   = generate_folds(df)
    metrics = []

    for fold in folds:
        train_df = df.iloc[fold.train_idx]
        test_df  = df.iloc[np.concatenate([fold.train_idx, fold.test_idx])]

        if len(train_df) < MAX_SEQ + 1:
            log.warning("Fold %d: insufficient train rows — skip.", fold.fold_id)
            continue

        try:
            X_tr_seq, X_tr_flat, y_tr, fs, ts, _ = build_sequences(
                train_df, fit_scalers=True
            )
            X_te_seq, X_te_flat, y_te, _, _, _ = build_sequences(
                test_df, feature_scaler=fs, target_scalers=ts, fit_scalers=False
            )
        except ValueError as e:
            log.warning("Fold %d build failed: %s", fold.fold_id, e)
            continue

        n_feat = X_tr_seq["short"].shape[2]
        # Bug 3 FIX: train on actual targets (not zeros) so embeddings carry real signal.
        # Use multi-task dict so the walk-forward embedding also encodes volatility.
        try:
            fm, emb = build_hybrid_model(n_features=n_feat, embedding_dim=32)
            train_deep_model(
                fm, X_tr_seq,
                {"output_close": y_tr["target_close"],
                 "output_range": y_tr["target_log_range"]},
                epochs=5, validation_split=0.0, patience=3,
            )
            emb_tr = extract_embeddings(emb, X_tr_seq)
            emb_te = extract_embeddings(emb, X_te_seq)
        except Exception as exc:
            log.warning("Fold %d deep model failed (%s) — using zero embeddings.", fold.fold_id, exc)
            emb_tr = np.zeros((len(X_tr_seq["short"]), 32), np.float32)
            emb_te = np.zeros((len(X_te_seq["short"]), 32), np.float32)

        n_train_samples = len(X_tr_seq["short"])
        emb_te_only = emb_te[n_train_samples:]
        Xf_te_only  = X_te_flat[n_train_samples:]

        # Bug 5 FIX: apply fold-local fused scaler (train only on train fold).
        # OLD: raw unscaled fused vector went into LightGBM — inconsistent with
        #      final pipeline which applies RobustScaler on the fused vector.
        fold_fused_scaler = RobustScaler()
        X_tr_f = fold_fused_scaler.fit_transform(np.hstack([emb_tr,      X_tr_flat]))
        X_te_f = fold_fused_scaler.transform(    np.hstack([emb_te_only, Xf_te_only]))

        # Improvement 10: track spike-day MAE separately for diagnostic visibility.
        # Spike rows = test days where range_expansion_ratio > 1.5 (volatile day follows).
        from src.feature_engineering import FEATURE_COLS as _WF_FCOLS
        if "range_expansion_ratio" in _WF_FCOLS:
            re_idx    = _WF_FCOLS.index("range_expansion_ratio")
            spike_mask_te = Xf_te_only[:, re_idx] > 1.5
        else:
            spike_mask_te = np.zeros(len(Xf_te_only), dtype=bool)
        has_spike = spike_mask_te.any()

        fold_mae = {}
        for tgt in TARGET_COLS:
            m  = _fit_lgbm_regressor(X_tr_f, y_tr[tgt], LGB_HP_DEFAULTS, sample_weight=None)
            pr = m.predict(X_te_f)
            y_te_orig = ts[tgt].inverse_transform(y_te[tgt][n_train_samples:].reshape(-1, 1)).ravel()
            pr_orig   = ts[tgt].inverse_transform(pr.reshape(-1, 1)).ravel()
            fold_mae[tgt] = float(mean_absolute_error(y_te_orig, pr_orig))
            # Spike-day MAE (only log on log_range to avoid spam)
            if has_spike and tgt == "target_log_range":
                spike_mae = float(mean_absolute_error(
                    y_te_orig[spike_mask_te], pr_orig[spike_mask_te]
                ))
                fold_mae[f"{tgt}_spike"] = spike_mae
                fold_mae[f"{tgt}_normal"] = float(mean_absolute_error(
                    y_te_orig[~spike_mask_te], pr_orig[~spike_mask_te]
                )) if (~spike_mask_te).any() else float("nan")

        entry = {"fold_id": fold.fold_id, "train_end": fold.train_end,
                 "test_year": fold.test_year, "mae": fold_mae}
        metrics.append(entry)
        log.info(
            "Fold %d (test=%d): all=%.4f | spike=%.4f | normal=%.4f",
            fold.fold_id, fold.test_year,
            fold_mae.get("target_log_range", float("nan")),
            fold_mae.get("target_log_range_spike", float("nan")),
            fold_mae.get("target_log_range_normal", float("nan")),
        )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fit_lgbm_regressor(
    X: np.ndarray,
    y: np.ndarray,
    hp: dict,
    sample_weight: np.ndarray | None = None,
):
    """
    Fit a LightGBM Huber regressor with optional per-sample weights.

    sample_weight is used to penalise close errors (CLOSE_LOSS_WEIGHT×) and
    event-shock-day errors more heavily during training, without changing the
    evaluation metric or inference logic.
    """
    try:
        import lightgbm as lgb
        p = {**LGB_HP_DEFAULTS, **hp, "random_state": SEED, "verbose": -1}

        p.pop("objective", None)
        p.pop("alpha", None)

        model = lgb.LGBMRegressor(
            objective="huber",
            alpha=0.7,
            **p
        )

    except ImportError:
        from sklearn.linear_model import Ridge
        model = Ridge()
        sample_weight = None   # Ridge .fit() does not use sample_weight

    if sample_weight is not None:
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load  (robust multi-format)
# ─────────────────────────────────────────────────────────────────────────────

def _save_artifacts(
    ticker, full_model, embedder, lgbm_models, risk_model,
    feat_scaler, tgt_scalers, fused_scaler, spike_lgbm_model=None,
) -> None:
    # ── Keras models (full model + embedder) ──────────────────────────────────
    _keras_save(full_model, deep_model_path(ticker))
    _keras_save(embedder,   embedder_path(ticker))

    # ── Scikit-learn / LightGBM artifacts (joblib) ────────────────────────────
    for tgt, m in lgbm_models.items():
        joblib.dump(m, lgbm_path(ticker, tgt))
    joblib.dump(risk_model,   risk_model_path(ticker))
    joblib.dump(feat_scaler,  scaler_path(ticker, "features"))
    joblib.dump(fused_scaler, scaler_path(ticker, "fused"))
    for tgt, sc in tgt_scalers.items():
        joblib.dump(sc, scaler_path(ticker, tgt))

    # ── Spike specialist model (optional — only saved if trained) ─────────────
    if spike_lgbm_model is not None:
        joblib.dump(spike_lgbm_model, spike_lgbm_path(ticker))
        log.info("Spike specialist LightGBM saved: %s", spike_lgbm_path(ticker).name)

    log.info("All artifacts saved for %s.", ticker)


def load_artifacts(ticker: str) -> dict:
    """
    Load all saved artifacts.

    Tries .keras → .h5 → SavedModel directory for Keras models
    (mirrors the fallback used by _save_artifacts).
    Raises FileNotFoundError with a clear message if not trained.
    """
    embedder     = _keras_load(embedder_path(ticker))
    feat_scaler  = joblib.load(scaler_path(ticker, "features"))
    fused_scaler = joblib.load(scaler_path(ticker, "fused"))
    tgt_scalers  = {t: joblib.load(scaler_path(ticker, t)) for t in TARGET_COLS}
    lgbm_models  = {t: joblib.load(lgbm_path(ticker, t))  for t in TARGET_COLS}

    rp = risk_model_path(ticker)
    risk_model = joblib.load(rp) if rp.exists() else None

    # Spike specialist model — optional (present only if enough spike-day history existed)
    sp = spike_lgbm_path(ticker)
    spike_lgbm_model = joblib.load(sp) if sp.exists() else None
    if spike_lgbm_model is not None:
        log.info("Spike specialist LightGBM loaded for %s.", ticker)

    log.info("Artifacts loaded for %s.", ticker)
    return {
        "embedder"         : embedder,
        "lgbm_models"      : lgbm_models,
        "risk_model"       : risk_model,
        "feat_scaler"      : feat_scaler,
        "fused_scaler"     : fused_scaler,
        "tgt_scalers"      : tgt_scalers,
        "spike_lgbm_model" : spike_lgbm_model,
    }