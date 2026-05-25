"""
optuna_tuner.py — Optuna tuning for deep model and LightGBM.

Spec compliance:
  - 20 trials max
  - Huber loss for all LightGBM models
  - min_child_samples tuned (prevents spike-day overfitting)
  - Scalers fitted only on training data (no leakage)
  - Risk model tuned separately
"""

from __future__ import annotations

import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.utils import SEED, get_logger

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEEP_HP_DEFAULTS = dict(
    lstm_units=64, lstm_layers=1, num_heads=4,
    dropout=0.2, learning_rate=1e-3,
)

LGB_HP_DEFAULTS = dict(
    max_depth=6, num_leaves=50,
    learning_rate=0.05, n_estimators=500,
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20,
    objective="huber",
    alpha=0.9,
)

RISK_LGB_DEFAULTS = dict(
    max_depth=4, num_leaves=31,
    learning_rate=0.05, n_estimators=200,
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=30,
    objective="binary",    # risk is a 0/1 classification target
    class_weight="balanced",  # NEW: compensates for minority high-risk days
)



# ─────────────────────────────────────────────────────────────────────────────
# Deep model tuning
# ─────────────────────────────────────────────────────────────────────────────

def tune_deep_model(
    X_seq_train: dict[str, np.ndarray],
    y_train: "np.ndarray | dict",
    n_trials: int = 20,
    timeout: int = 300,
    val_fraction: float = 0.15,
) -> dict:
    """
    Tune deep model hyperparameters via Optuna.

    y_train accepts:
      - np.ndarray  (N,)  — treated as close target (backward compatible)
      - dict {"output_close": arr, "output_range": arr} — multi-task (preferred)

    Optimisation metric: validation MAE on the close output only.
    """
    try:
        from src.deep_model import build_hybrid_model, train_deep_model
    except ImportError:
        log.warning("TF unavailable — using deep model defaults.")
        return DEEP_HP_DEFAULTS.copy()

    # Normalise y to close array + optional range array
    if isinstance(y_train, dict):
        y_close = y_train.get("output_close", next(iter(y_train.values())))
        y_range = y_train.get("output_range", np.zeros_like(y_close))
    else:
        y_close = y_train
        y_range = np.zeros_like(y_close)

    n_val  = max(60, int(len(y_close) * val_fraction))
    X_tr   = {k: v[:-n_val] for k, v in X_seq_train.items()}
    X_vl   = {k: v[-n_val:] for k, v in X_seq_train.items()}
    y_tr   = {"output_close": y_close[:-n_val], "output_range": y_range[:-n_val]}
    y_vl_c = y_close[-n_val:]   # evaluate only on close for simplicity
    n_feat = X_seq_train["short"].shape[2]

    def objective(trial: optuna.Trial) -> float:
        hp = dict(
            lstm_units    = trial.suggest_int("lstm_units", 32, 128, step=32),
            lstm_layers   = trial.suggest_int("lstm_layers", 1, 2),
            num_heads     = trial.suggest_categorical("num_heads", [2, 4, 8]),
            dropout       = trial.suggest_float("dropout", 0.1, 0.4),
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        )
        fm, _ = build_hybrid_model(n_features=n_feat, **hp)
        train_deep_model(fm, X_tr, y_tr, epochs=8, batch_size=64,
                         validation_split=0.0, patience=3)
        # Multi-task model returns list of arrays — take close output (index 0)
        raw_preds = fm.predict(X_vl, verbose=0)
        close_preds = (raw_preds[0].ravel() if isinstance(raw_preds, list)
                       else raw_preds.ravel())
        return float(np.mean(np.abs(close_preds - y_vl_c)))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    log.info("Deep best: %s  val_MAE=%.6f", study.best_params, study.best_value)
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM OHLC model tuning (Huber)
# ─────────────────────────────────────────────────────────────────────────────

def tune_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_name: str,
    n_trials: int = 20,
    timeout: int = 120,
    val_fraction: float = 0.15,
) -> dict:
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("LGB unavailable for %s.", target_name)
        return LGB_HP_DEFAULTS.copy()

    n_val = max(30, int(len(y_train) * val_fraction))
    X_tr  = X_train[:-n_val];  X_vl = X_train[-n_val:]
    y_tr  = y_train[:-n_val];  y_vl = y_train[-n_val:]

    def objective(trial: optuna.Trial) -> float:
        p = dict(
            max_depth        = trial.suggest_int("max_depth", 3, 10),
            num_leaves       = trial.suggest_int("num_leaves", 20, 100),
            learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            n_estimators     = trial.suggest_int("n_estimators", 100, 800),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples= trial.suggest_int("min_child_samples", 10, 50),
            alpha            = trial.suggest_float("alpha", 0.7, 0.95),
            objective        = "huber",
            random_state     = SEED,
            verbose          = -1,
        )
        m = lgb.LGBMRegressor(**p)
        m.fit(X_tr, y_tr)
        return float(np.mean(np.abs(m.predict(X_vl) - y_vl)))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    log.info("LGB [%s] best: %s  val_MAE=%.6f", target_name, study.best_params, study.best_value)
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# Risk model tuning (binary classification)
# ─────────────────────────────────────────────────────────────────────────────

def tune_risk_model(
    X_train: np.ndarray,
    y_binary: np.ndarray,
    n_trials: int = 20,
    timeout: int = 60,
    val_fraction: float = 0.15,
) -> dict:
    """
    Tune LightGBM binary classifier for risk labelling.

    Uses ROC-AUC as the optimisation metric (more informative than log-loss
    when the positive class — high-risk days — is a minority, which is typical
    after extending labels with macro conditions).
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return RISK_LGB_DEFAULTS.copy()

    n_val = max(30, int(len(y_binary) * val_fraction))
    X_tr  = X_train[:-n_val];  X_vl = X_train[-n_val:]
    y_tr  = y_binary[:-n_val]; y_vl = y_binary[-n_val:]

    # Guard: if validation fold has only one class, AUC is undefined
    if len(np.unique(y_vl)) < 2:
        log.warning("Risk model: validation fold has a single class — skipping Optuna, using defaults.")
        return RISK_LGB_DEFAULTS.copy()

    def objective(trial: optuna.Trial) -> float:
        p = dict(
            max_depth        = trial.suggest_int("max_depth", 2, 8),
            num_leaves       = trial.suggest_int("num_leaves", 10, 60),
            learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            n_estimators     = trial.suggest_int("n_estimators", 50, 300),
            min_child_samples= trial.suggest_int("min_child_samples", 20, 60),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            objective        = "binary",
            class_weight     = "balanced",   # NEW: handles minority class
            random_state     = SEED,
            verbose          = -1,
        )
        m = lgb.LGBMClassifier(**p)
        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_vl)[:, 1]
        # Maximise AUC → minimise negative AUC
        return -float(roc_auc_score(y_vl, probs))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_auc = -study.best_value
    log.info(
        "Risk model Optuna done | best params: %s | val_AUC=%.4f",
        study.best_params, best_auc,
    )
    return study.best_params