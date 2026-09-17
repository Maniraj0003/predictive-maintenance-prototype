import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from numpy_autoencoder import NumpyAutoencoder

DATA_DIR = "data"
MODEL_DIR = "models"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load_test_with_full_rul(meta):
    df = pd.read_csv(f"{DATA_DIR}/test_FD001.txt", sep=r"\s+", header=None, names=COLS)
    rul_final = pd.read_csv(f"{DATA_DIR}/RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL_final"])
    rul_final["unit"] = rul_final.index + 1  # RUL_FD001.txt is ordered by unit 1..100, one row each

    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.merge(rul_final, on="unit")
    last_cycle_per_unit = df.groupby("unit")["cycle"].transform("max")

    # Back-calculate true RUL at every cycle, capped exactly like training
    df["RUL"] = (df["RUL_final"] + (last_cycle_per_unit - df["cycle"])).clip(upper=meta["rul_cap"])
    df["true_anomaly"] = (df["RUL"] <= meta["anomaly_label_rul"]).astype(int)
    return df


def main():
    with open(f"{MODEL_DIR}/meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    df = load_test_with_full_rul(meta)
    print(f"Loaded official test set: {len(df)} rows from {df['unit'].nunique()} engines")
    print(f"Base failure-rate in test data: {df['true_anomaly'].mean():.1%} of cycles are within "
          f"{meta['anomaly_label_rul']} cycles of failure")

    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    autoencoder = NumpyAutoencoder(f"{MODEL_DIR}/autoencoder_weights.npz")
    rf = joblib.load(f"{MODEL_DIR}/rf_rul.joblib")
    xgb = joblib.load(f"{MODEL_DIR}/xgb_rul.joblib")

    X = scaler.transform(df[feature_cols])

    # --- RUL regression: Random Forest vs XGBoost, same data, same metrics ---
    y_true_rul = df["RUL"].values
    last_idx = df.groupby("unit")["cycle"].idxmax()

    def _reg_report(model):
        y_pred = np.clip(model.predict(X), 0, meta["rul_cap"])
        mae = mean_absolute_error(y_true_rul, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true_rul, y_pred)))
        # Official competition protocol also scores only the LAST observed
        # cycle per test engine ("if you stopped monitoring right now, how
        # good is your RUL estimate?").
        mae_last = mean_absolute_error(y_true_rul[last_idx], y_pred[last_idx])
        rmse_last = float(np.sqrt(mean_squared_error(y_true_rul[last_idx], y_pred[last_idx])))
        return {
            "rul_mae_all_cycles": float(mae),
            "rul_rmse_all_cycles": float(rmse),
            "rul_mae_last_cycle": float(mae_last),
            "rul_rmse_last_cycle": float(rmse_last),
        }

    rf_report = _reg_report(rf)
    xgb_report = _reg_report(xgb)

    # --- Anomaly classification ---
    recon = autoencoder.predict(X)
    recon_err = np.mean((X - recon) ** 2, axis=1)
    y_pred_anomaly = (recon_err > meta["anomaly_threshold"]).astype(int)
    y_true_anomaly = df["true_anomaly"].values

    acc = accuracy_score(y_true_anomaly, y_pred_anomaly)
    prec = precision_score(y_true_anomaly, y_pred_anomaly, zero_division=0)
    rec = recall_score(y_true_anomaly, y_pred_anomaly, zero_division=0)
    f1 = f1_score(y_true_anomaly, y_pred_anomaly, zero_division=0)
    cm = confusion_matrix(y_true_anomaly, y_pred_anomaly).tolist()

    print("\n=== Official NASA test set (FD001) — never used in training ===")
    print(f"RUL regression (Random Forest, all cycles)  -> MAE: {rf_report['rul_mae_all_cycles']:.2f} | RMSE: {rf_report['rul_rmse_all_cycles']:.2f}")
    print(f"RUL regression (XGBoost, all cycles)        -> MAE: {xgb_report['rul_mae_all_cycles']:.2f} | RMSE: {xgb_report['rul_rmse_all_cycles']:.2f}")
    print(f"RUL regression (Random Forest, last cycle)  -> MAE: {rf_report['rul_mae_last_cycle']:.2f} | RMSE: {rf_report['rul_rmse_last_cycle']:.2f}")
    print(f"RUL regression (XGBoost, last cycle)        -> MAE: {xgb_report['rul_mae_last_cycle']:.2f} | RMSE: {xgb_report['rul_rmse_last_cycle']:.2f}")
    print(f"Anomaly detector -> Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {cm}")

    report = {
        "n_rows": int(len(df)),
        "n_engines": int(df["unit"].nunique()),
        "random_forest": rf_report,
        "xgboost": xgb_report,
        # kept for backwards-compat with earlier report consumers (Random Forest)
        "rul_mae_all_cycles": rf_report["rul_mae_all_cycles"],
        "rul_rmse_all_cycles": rf_report["rul_rmse_all_cycles"],
        "rul_mae_last_cycle": rf_report["rul_mae_last_cycle"],
        "rul_rmse_last_cycle": rf_report["rul_rmse_last_cycle"],
        "anomaly_accuracy": float(acc),
        "anomaly_precision": float(prec),
        "anomaly_recall": float(rec),
        "anomaly_f1": float(f1),
        "confusion_matrix": cm,
        "note": "Evaluated on data/test_FD001.txt + RUL_FD001.txt, which the models never saw during training.",
    }
    with open(f"{MODEL_DIR}/test_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {MODEL_DIR}/test_evaluation_report.json")


if __name__ == "__main__":
    main()
