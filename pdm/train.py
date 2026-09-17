"""
train.py
--------
Trains the models described in the synopsis, plus one extra comparison model:
  1. A PyTorch autoencoder (deep learning) that learns "healthy" engine
     behaviour and flags anomalies via reconstruction error.
  2. A Random Forest (machine learning) that predicts Remaining Useful
     Life (RUL) from sensor readings -> used for failure prediction /
     risk classification.
  3. An XGBoost regressor -- a second ML model trained on the exact same
     data/split as the Random Forest, so the two can be compared head to
     head on the same held-out engines instead of reporting just one
     algorithm's numbers.

Dataset: NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001 subset).

Evaluation methodology (important):
  - Data is split by ENGINE UNIT (not by row). Splitting by row would leak
    information, since consecutive cycles of the same engine are highly
    correlated -- a naive random row split lets the model "see" an engine
    it's being tested on, just at a neighbouring cycle.
  - 80 engines are used for training, 20 are held out entirely for
    validation. Metrics reported below come only from those 20 unseen
    engines.
  - After validation metrics are captured, the FINAL deployed models are
    refit on all 100 engines (standard practice: validate on a held-out
    split, then retrain on everything before shipping).
  - A second, independent evaluation against NASA's official test set
    (data/test_FD001.txt + RUL_FD001.txt) is done separately in
    evaluate.py, using the models saved here.

Run:  python train.py
Produces artifacts in ./models/ that the FastAPI app loads at runtime.
"""

import json
import os
import tempfile
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from dl_model import Autoencoder
from numpy_autoencoder import NumpyAutoencoder

DATA_DIR = "data"
MODEL_DIR = "models"
RUL_CAP = 125            # standard RUL clipping for FD001 (Saxena et al.)
ANOMALY_WINDOW = 0.3     # first 30% of an engine's life is treated as "healthy" (for autoencoder training)
ANOMALY_HEALTHY_PERCENTILE = 99.5  # stricter cutoff to reduce false positives on normal degradation
ANOMALY_LABEL_RUL = 20   # a cycle is labeled "true anomaly" if true RUL <= this (matches API's "Critical" risk level)
VAL_FRACTION = 0.2       # fraction of ENGINES (not rows) held out for validation
SEED = 42

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# Sensors with ~zero variance in FD001 (uninformative, dropped) — verified
# empirically on the training set.
DROP_COLS = ["op3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]

FEATURE_COLS = [c for c in COLS if c not in DROP_COLS and c not in ("unit", "cycle")]


def load_train():
    df = pd.read_csv(f"{DATA_DIR}/train_FD001.txt", sep=r"\s+", header=None, names=COLS)
    df = df.drop(columns=DROP_COLS)
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=RUL_CAP)
    df["life_frac"] = df["cycle"] / max_cycle
    df["true_anomaly"] = (df["RUL"] <= ANOMALY_LABEL_RUL).astype(int)
    return df


def train_autoencoder(X_healthy: np.ndarray, n_features: int, seed: int = SEED):
    bottleneck = max(3, n_features // 3)
    torch.manual_seed(seed)
    model = Autoencoder(n_features=n_features, bottleneck=bottleneck)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_healthy, dtype=torch.float32)
    n_epochs, batch_size, n_samples = 150, 64, X_t.shape[0]

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            batch = X_t[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
    model.eval()
    return model, bottleneck


def evaluate_split(name, X, df_split, autoencoder, anomaly_threshold, rf, xgb):
    """Compute honest metrics for one held-out split (no rows seen during training)."""
    y_true_rul = df_split["RUL"].values

    def _reg_metrics(model):
        y_pred = np.clip(model.predict(X), 0, RUL_CAP)
        mae = mean_absolute_error(y_true_rul, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true_rul, y_pred)))
        return mae, rmse

    rf_mae, rf_rmse = _reg_metrics(rf)
    xgb_mae, xgb_rmse = _reg_metrics(xgb)

    recon = autoencoder.predict(X)
    recon_err = np.mean((X - recon) ** 2, axis=1)
    y_pred_anomaly = (recon_err > anomaly_threshold).astype(int)
    y_true_anomaly = df_split["true_anomaly"].values

    acc = accuracy_score(y_true_anomaly, y_pred_anomaly)
    prec = precision_score(y_true_anomaly, y_pred_anomaly, zero_division=0)
    rec = recall_score(y_true_anomaly, y_pred_anomaly, zero_division=0)
    f1 = f1_score(y_true_anomaly, y_pred_anomaly, zero_division=0)
    cm = confusion_matrix(y_true_anomaly, y_pred_anomaly).tolist()

    print(f"\n--- {name} (n={len(df_split)} rows, {df_split['unit'].nunique()} engines) ---")
    print(f"  RUL regression (Random Forest) -> MAE: {rf_mae:.2f} cycles | RMSE: {rf_rmse:.2f} cycles")
    print(f"  RUL regression (XGBoost)       -> MAE: {xgb_mae:.2f} cycles | RMSE: {xgb_rmse:.2f} cycles")
    print(f"  Anomaly detector -> Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {cm}")

    return {
        "n_rows": int(len(df_split)),
        "n_engines": int(df_split["unit"].nunique()),
        "rul_mae": float(rf_mae),          # kept for backwards compat (Random Forest)
        "rul_rmse": float(rf_rmse),
        "random_forest": {"rul_mae": float(rf_mae), "rul_rmse": float(rf_rmse)},
        "xgboost": {"rul_mae": float(xgb_mae), "rul_rmse": float(xgb_rmse)},
        "anomaly_accuracy": float(acc),
        "anomaly_precision": float(prec),
        "anomaly_recall": float(rec),
        "anomaly_f1": float(f1),
        "confusion_matrix": cm,
    }


def main():
    df = load_train()
    all_units = df["unit"].unique()
    print(f"Loaded {len(df)} rows from {len(all_units)} engines")
    print(f"Base failure-rate in data: {df['true_anomaly'].mean():.1%} of cycles are within "
          f"{ANOMALY_LABEL_RUL} cycles of failure (true anomaly)")

    # ---- 1. Unit-level train/validation split (prevents leakage) ----
    rng = np.random.RandomState(SEED)
    shuffled = rng.permutation(all_units)
    n_val = max(1, int(len(all_units) * VAL_FRACTION))
    val_units = set(shuffled[:n_val])
    train_units = set(shuffled[n_val:])
    print(f"Train engines: {len(train_units)} | Held-out validation engines: {len(val_units)}")

    df_train = df[df["unit"].isin(train_units)].reset_index(drop=True)
    df_val = df[df["unit"].isin(val_units)].reset_index(drop=True)

    # ---- 2. Fit preprocessing + models on TRAIN ENGINES ONLY ----
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_train[FEATURE_COLS])
    X_val = scaler.transform(df_val[FEATURE_COLS])

    healthy_mask = df_train["life_frac"] <= ANOMALY_WINDOW
    X_healthy = X_train[healthy_mask.values]
    print(f"\nTraining PyTorch autoencoder on {len(X_healthy)} healthy samples (train engines only)")
    n_features = X_train.shape[1]
    model, bottleneck = train_autoencoder(X_healthy, n_features)

    weights = model.export_weights()
    val_weights_path = os.path.join(tempfile.gettempdir(), "pdm_val_autoencoder_weights.npz")
    np.savez(val_weights_path, **weights)
    autoencoder = NumpyAutoencoder(val_weights_path)

    recon_healthy = autoencoder.predict(X_healthy)
    err_healthy = np.mean((X_healthy - recon_healthy) ** 2, axis=1)
    anomaly_threshold = float(np.percentile(err_healthy, ANOMALY_HEALTHY_PERCENTILE))

    rf = RandomForestRegressor(n_estimators=60, max_depth=10, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, df_train["RUL"].values)

    xgb = XGBRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
    )
    xgb.fit(X_train, df_train["RUL"].values)

    # ---- 3. HONEST evaluation on the 20 held-out engines the model never saw ----
    val_metrics = evaluate_split("Held-out validation engines", X_val, df_val, autoencoder, anomaly_threshold, rf, xgb)

    # ---- 4. Refit final models on ALL 100 engines for deployment ----
    print("\nRefitting final models on all engines for deployment...")
    scaler_final = MinMaxScaler()
    X_all = scaler_final.fit_transform(df[FEATURE_COLS])
    healthy_mask_all = df["life_frac"] <= ANOMALY_WINDOW
    X_healthy_all = X_all[healthy_mask_all.values]
    model_final, _ = train_autoencoder(X_healthy_all, n_features)
    weights_final = model_final.export_weights()
    np.savez(f"{MODEL_DIR}/autoencoder_weights.npz", **weights_final)
    autoencoder_final = NumpyAutoencoder(f"{MODEL_DIR}/autoencoder_weights.npz")

    recon_healthy_all = autoencoder_final.predict(X_healthy_all)
    err_healthy_all = np.mean((X_healthy_all - recon_healthy_all) ** 2, axis=1)
    anomaly_threshold_final = float(np.percentile(err_healthy_all, ANOMALY_HEALTHY_PERCENTILE))

    rf_final = RandomForestRegressor(n_estimators=60, max_depth=10, random_state=SEED, n_jobs=-1)
    rf_final.fit(X_all, df["RUL"].values)

    xgb_final = XGBRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
    )
    xgb_final.fit(X_all, df["RUL"].values)

    joblib.dump(scaler_final, f"{MODEL_DIR}/scaler.joblib")
    joblib.dump(rf_final, f"{MODEL_DIR}/rf_rul.joblib")
    joblib.dump(xgb_final, f"{MODEL_DIR}/xgb_rul.joblib")

    with torch.no_grad():
        torch_recon = model_final(torch.tensor(X_healthy_all[:50], dtype=torch.float32)).numpy()
    numpy_recon = autoencoder_final.predict(X_healthy_all[:50])
    max_diff = np.max(np.abs(torch_recon - numpy_recon))
    print(f"Max diff between torch and numpy-replica outputs (sanity check): {max_diff:.2e}")

    meta = {
        "feature_cols": FEATURE_COLS,
        "anomaly_threshold": anomaly_threshold_final,
        "rul_cap": RUL_CAP,
        "anomaly_label_rul": ANOMALY_LABEL_RUL,
        "bottleneck_dim": bottleneck,
        "evaluation": {
            "methodology": (
                f"Metrics below come from {len(val_units)} engines held out entirely "
                f"during training (unit-level split, not row-level, to avoid leakage "
                f"between neighbouring cycles of the same engine). Random Forest and "
                f"XGBoost were trained on the identical split for a fair comparison."
            ),
            "held_out_validation": val_metrics,
        },
        "val_mae": val_metrics["rul_mae"],
    }
    with open(f"{MODEL_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved artifacts + evaluation metrics to", MODEL_DIR)

    demo_units = sorted(df["unit"].unique().tolist())
    demo_df = df[df["unit"].isin(demo_units)][["unit", "cycle", "RUL"] + FEATURE_COLS]
    demo_df.to_csv(f"{MODEL_DIR}/demo_engines.csv", index=False)
    print(f"Saved demo dataset with all {len(demo_units)} engines")

    print(
        "\nNOTE: run `python evaluate.py` next for an independent evaluation against "
        "NASA's official held-out test set (data/test_FD001.txt + RUL_FD001.txt)."
    )


if __name__ == "__main__":
    main()
