"""
deep_model.py — Multi-window Hybrid LSTM + Transformer feature extractor.

Architecture per the spec:
  Three input branches (short=30, mid=60, long=120):
    Each branch:
      LSTM  → captures temporal momentum at that window scale
      Transformer Encoder → captures regime shifts / long-range dependencies
    Each branch output: concat(LSTM_out, Transformer_out) → branch_emb

  Fusion:
    concat(branch_short_emb, branch_mid_emb, branch_long_emb)
    → Dense(embedding_dim, relu)
    → final embedding vector

  Role: feature extractor only.
        A Dense(1) head is added during training and discarded after.
        LightGBM receives the embedding vector, never raw sequences.

Transformer rules (spec §5):
  - Multi-head attention (2, 4, or 8 heads, tunable)
  - Accepts sequence embedding from LSTM output (residual path)
  - Learns regime shifts via self-attention over the full window
  - Output is sequence-level (GlobalAveragePooling1D), not token-level
  - Does NOT directly predict price
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.utils import SEQ_LONG, SEQ_MID, SEQ_SHORT, SEED, get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer encoder block
# ─────────────────────────────────────────────────────────────────────────────

def _transformer_block(
    inputs: tf.Tensor,
    num_heads: int,
    ff_dim: int,
    dropout: float,
    name_prefix: str = "tr",
) -> tf.Tensor:
    """
    Single Transformer encoder block.
    Input shape: (batch, seq_len, d_model)
    Output shape: same

    key_dim is per-head dimension.  We use d_model // num_heads so the
    total attention capacity scales with d_model regardless of head count.
    """
    d_model  = inputs.shape[-1]
    key_dim  = max(1, d_model // num_heads)

    # Multi-head self-attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name=f"{name_prefix}_mha",
    )(inputs, inputs)
    attn = layers.Dropout(dropout, name=f"{name_prefix}_attn_drop")(attn)
    x    = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(inputs + attn)

    # Position-wise FFN
    ff = layers.Dense(ff_dim, activation="relu", name=f"{name_prefix}_ff1")(x)
    ff = layers.Dropout(dropout, name=f"{name_prefix}_ff_drop")(ff)
    ff = layers.Dense(d_model,  name=f"{name_prefix}_ff2")(ff)
    ff = layers.Dropout(dropout, name=f"{name_prefix}_ff2_drop")(ff)
    x  = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(x + ff)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Single-window branch: LSTM → Transformer → branch embedding
# ─────────────────────────────────────────────────────────────────────────────

def _build_branch(
    seq_input: keras.Input,
    lstm_units: int,
    lstm_layers: int,
    num_heads: int,
    tr_ff_dim: int,
    dropout: float,
    branch_name: str,
) -> tf.Tensor:
    """
    Build one LSTM+Transformer branch for a single time-window input.

    Pipeline:
      seq_input  (batch, seq_len, n_feat)
        → LSTM stack  →  lstm_out  (batch, lstm_units)
        → Transformer (on the same seq_input, projected)  →  tr_out  (batch, proj_dim)
        → Concatenate([lstm_out, tr_out])
        → Dense(branch_emb_dim, relu)

    The Transformer receives the original sequence (not LSTM output) so it
    can attend over the full window without LSTM's sequential bottleneck.
    LSTM and Transformer are thus complementary:
      LSTM  → sequential / recurrent momentum
      Transformer → global self-attention / regime awareness
    """
    n_feat   = seq_input.shape[-1]
    # Transformer projection dimension: multiple of num_heads, ≥ 16
    proj_dim = max(num_heads * 4, (n_feat // num_heads + 1) * num_heads)

    # ── LSTM branch ──────────────────────────────────────────────────────────
    x_lstm = seq_input
    for i in range(lstm_layers):
        ret_seq = (i < lstm_layers - 1)
        x_lstm  = layers.LSTM(
            lstm_units, return_sequences=ret_seq,
            name=f"{branch_name}_lstm_{i}"
        )(x_lstm)
        x_lstm  = layers.Dropout(dropout, name=f"{branch_name}_lstm_drop_{i}")(x_lstm)
    # shape: (batch, lstm_units)

    # ── Transformer branch ────────────────────────────────────────────────────
    x_tr = layers.Dense(proj_dim, name=f"{branch_name}_tr_proj")(seq_input)
    x_tr = _transformer_block(
        x_tr, num_heads=num_heads, ff_dim=tr_ff_dim,
        dropout=dropout, name_prefix=f"{branch_name}_tr",
    )
    # Sequence-level pooling (spec §5: NOT token-level prediction)
    x_tr = layers.GlobalAveragePooling1D(name=f"{branch_name}_tr_gap")(x_tr)
    # shape: (batch, proj_dim)

    # ── Merge within branch ───────────────────────────────────────────────────
    merged = layers.Concatenate(name=f"{branch_name}_concat")([x_lstm, x_tr])
    # Normalise before passing to the fusion stage
    merged = layers.LayerNormalization(epsilon=1e-6, name=f"{branch_name}_ln")(merged)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Full model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_hybrid_model(
    n_features: int,
    embedding_dim: int = 64,
    lstm_units: int = 64,
    lstm_layers: int = 1,
    num_heads: int = 4,
    transformer_ff_dim: int = 128,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tuple[keras.Model, keras.Model]:
    """
    Build the three-window hybrid LSTM+Transformer model.

    Inputs: three named inputs — "short", "mid", "long"
    Output: embedding vector of size *embedding_dim*

    Returns
    -------
    full_model : compiled with MAE loss (used during training)
    embedder   : same graph up to the embedding layer (used for feature extraction)
    """
    tf.random.set_seed(SEED)

    inp_short = keras.Input(shape=(SEQ_SHORT, n_features), name="short")
    inp_mid   = keras.Input(shape=(SEQ_MID,   n_features), name="mid")
    inp_long  = keras.Input(shape=(SEQ_LONG,  n_features), name="long")

    emb_short = _build_branch(inp_short, lstm_units, lstm_layers, num_heads,
                               transformer_ff_dim, dropout, "short")
    emb_mid   = _build_branch(inp_mid,   lstm_units, lstm_layers, num_heads,
                               transformer_ff_dim, dropout, "mid")
    emb_long  = _build_branch(inp_long,  lstm_units, lstm_layers, num_heads,
                               transformer_ff_dim, dropout, "long")

    # ── Fuse three branches ───────────────────────────────────────────────────
    fused = layers.Concatenate(name="fusion")([emb_short, emb_mid, emb_long])
    embedding = layers.Dense(embedding_dim, activation="relu", name="embedding")(fused)
    embedding = layers.Dropout(dropout, name="embed_drop")(embedding)
    # Final normalisation so embeddings are on a consistent scale for LightGBM
    embedding = layers.LayerNormalization(epsilon=1e-6, name="embed_ln")(embedding)

    # ── Multi-task training heads (both discarded after training — only embedding kept) ──
    # FIX: training on both close AND log_range simultaneously forces the embedding
    # to encode volatility structure, not just price level. This directly improves
    # spike-day range predictions by giving LightGBM richer embeddings.
    output_close = layers.Dense(1, name="output_close")(embedding)
    output_range = layers.Dense(1, name="output_range")(embedding)

    inputs     = [inp_short, inp_mid, inp_long]
    full_model = keras.Model(
        inputs=inputs,
        outputs=[output_close, output_range],
        name="hybrid_full_multitask",
    )
    full_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"output_close": "mae", "output_range": "mae"},
        # Range weighted 1.5× — embedding must encode volatility structure for spike days
        loss_weights={"output_close": 1.0, "output_range": 1.5},
    )
    embedder = keras.Model(inputs=inputs, outputs=embedding, name="embedder")

    total_params = full_model.count_params()
    log.info(
        "Built 3-window hybrid (multi-task): LSTM(%d×%d) + Transformer(heads=%d) × 3 windows "
        "→ emb_dim=%d  |  params=%d",
        lstm_layers, lstm_units, num_heads, embedding_dim, total_params,
    )
    return full_model, embedder


# ─────────────────────────────────────────────────────────────────────────────
# Training helper
# ─────────────────────────────────────────────────────────────────────────────

def train_deep_model(
    model: keras.Model,
    X_seq: dict[str, np.ndarray],
    y: "np.ndarray | dict",
    epochs: int = 20,
    batch_size: int = 64,
    validation_split: float = 0.1,
    patience: int = 5,
) -> keras.callbacks.History:
    """
    Train *model* on multi-window inputs.

    Parameters
    ----------
    model   : compiled full_model from build_hybrid_model()
    X_seq   : {"short": arr, "mid": arr, "long": arr}
    y       : (N,) array  OR  dict {"output_close": arr, "output_range": arr}
              Multi-task dict is preferred — it forces the embedding to encode
              both price level and volatility structure simultaneously.
              If a plain ndarray is passed it is treated as the close target and
              a zero-filled range target is added automatically (backward compat).
    """
    # FIX Bug 3: normalise y to dict so multi-task model always receives both targets.
    # Legacy callers passing a plain ndarray continue to work unchanged.
    if isinstance(y, np.ndarray):
        y_input: dict = {
            "output_close": y,
            "output_range": np.zeros_like(y),
        }
    else:
        y_input = y

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience,
        restore_best_weights=True, verbose=0,
    )
    lr_sched = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3,
        min_lr=1e-6, verbose=0,
    )
    history = model.fit(
        X_seq, y_input,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[es, lr_sched],
        verbose=0,
    )
    log.info(
        "Deep model trained: %d epochs  train_loss=%.6f",
        len(history.epoch), history.history["loss"][-1],
    )
    return history


def extract_embeddings(embedder: keras.Model, X_seq: dict[str, np.ndarray]) -> np.ndarray:
    """
    Run multi-window X_seq through embedder.
    Returns (N, emb_dim).
    """
    return embedder.predict(X_seq, verbose=0)