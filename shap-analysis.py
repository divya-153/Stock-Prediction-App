import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.trainer import load_artifacts
from src.data_loader import load_ohlcv
from src.feature_engineering import build_features, FEATURE_COLS
from src.sequence_builder import build_inference_sequence


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TICKER = "RELIANCE.NS"
TARGET_MODEL = "target_log_range"


# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────

artifacts = load_artifacts(TICKER)

model = artifacts["lgbm_models"][TARGET_MODEL]
fused_scaler = artifacts["fused_scaler"]
feat_scaler = artifacts["feat_scaler"]
embedder = artifacts["embedder"]


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

df_raw = load_ohlcv(TICKER)
df = build_features(df_raw)


# ─────────────────────────────────────────────
# BUILD SHAP DATASET
# ─────────────────────────────────────────────

X_rows = []

for i in range(len(df) - 300, len(df) - 120):

    df_window = df.iloc[:i]

    if len(df_window) < 120:
        continue

    X_seq, X_flat = build_inference_sequence(df_window, feat_scaler)

    emb = embedder.predict(X_seq, verbose=0)

    # flatten embedding (IMPORTANT)
    emb_flat = emb.reshape(emb.shape[0], -1)

    X_fused = np.hstack([emb_flat, X_flat])
    X_fused = fused_scaler.transform(X_fused)

    X_rows.append(X_fused[0])   # single sample per window


X = np.array(X_rows)


# ─────────────────────────────────────────────
# FEATURE NAMES (CRITICAL FIX)
# ─────────────────────────────────────────────

embedding_dim = embedder.output_shape[-1]

feature_names = (
    [f"embed_{i}" for i in range(embedding_dim)] +
    FEATURE_COLS
)

X_df = pd.DataFrame(X, columns=feature_names)


# ─────────────────────────────────────────────
# SHAP EXPLAINER
# ─────────────────────────────────────────────

print("Building SHAP explainer...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_df)


# ─────────────────────────────────────────────
# GLOBAL IMPORTANCE
# ─────────────────────────────────────────────

print("Computing global importance...")

mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

df_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": mean_abs_shap
}).sort_values("importance", ascending=False)


print("\n🔥 TOP 20 FEATURES DRIVING RANGE:\n")
print(df_importance.head(20))


# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────

plt.figure()
shap.summary_plot(shap_values, X_df, show=False)
plt.title("Global SHAP Summary - Range Model")
plt.show()


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

df_importance.to_csv("shap_global_importance.csv", index=False)

print("\n✅ Saved: shap_global_importance.csv")