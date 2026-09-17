import json
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, BASE_DIR)  # so `numpy_autoencoder` is importable

from numpy_autoencoder import NumpyAutoencoder  # noqa: E402

app = FastAPI(title="Predictive Maintenance API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load artifacts once at cold start
# ---------------------------------------------------------------------------
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
# Trained with real PyTorch (see train.py / dl_model.py); replayed here with
# a dependency-free NumPy forward pass so the deployed API doesn't need the
# ~1GB torch runtime (see numpy_autoencoder.py for details).
autoencoder = NumpyAutoencoder(os.path.join(MODEL_DIR, "autoencoder_weights.npz"))
RUL_MODELS = {
    "random_forest": joblib.load(os.path.join(MODEL_DIR, "rf_rul.joblib")),
    "xgboost": joblib.load(os.path.join(MODEL_DIR, "xgb_rul.joblib")),
}
DEFAULT_RUL_MODEL = "xgboost"  # slightly lower MAE on both held-out validation and the official test set
with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    META = json.load(f)

TEST_EVAL_PATH = os.path.join(MODEL_DIR, "test_evaluation_report.json")
TEST_EVAL = None
if os.path.exists(TEST_EVAL_PATH):
    with open(TEST_EVAL_PATH) as f:
        TEST_EVAL = json.load(f)

FEATURE_COLS: List[str] = META["feature_cols"]
ANOMALY_THRESHOLD: float = META["anomaly_threshold"]
RUL_CAP: int = META["rul_cap"]

DEMO_DF = pd.read_csv(os.path.join(MODEL_DIR, "demo_engines.csv"))


class SensorReading(BaseModel):
    op1: float = Field(..., description="Operational setting 1")
    op2: float = Field(..., description="Operational setting 2")
    s2: float
    s3: float
    s4: float
    s7: float
    s8: float
    s9: float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s17: float
    s20: float
    s21: float


class PredictionResponse(BaseModel):
    predicted_rul: float
    reconstruction_error: float
    anomaly_threshold: float
    is_anomaly: bool
    risk_level: str
    model_used: str


def _risk_level(rul: float) -> str:
    if rul <= 20:
        return "Critical"
    if rul <= 50:
        return "Warning"
    return "Healthy"


def _resolve_rul_model(model: str):
    if model not in RUL_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Choose one of {list(RUL_MODELS)}")
    return RUL_MODELS[model]


def _predict_row(row: dict, model: str = DEFAULT_RUL_MODEL) -> PredictionResponse:
    rul_model = _resolve_rul_model(model)
    x = pd.DataFrame([row])[FEATURE_COLS]
    x_scaled = scaler.transform(x)

    recon = autoencoder.predict(x_scaled)
    recon_error = float(np.mean((x_scaled - recon) ** 2))
    is_anomaly = recon_error > ANOMALY_THRESHOLD

    pred_rul = float(np.clip(rul_model.predict(x_scaled)[0], 0, RUL_CAP))

    return PredictionResponse(
        predicted_rul=round(pred_rul, 1),
        reconstruction_error=round(recon_error, 5),
        anomaly_threshold=round(ANOMALY_THRESHOLD, 5),
        is_anomaly=bool(is_anomaly),
        risk_level=_risk_level(pred_rul),
        model_used=model,
    )


def _predict_batch(df_rows: pd.DataFrame, model: str = DEFAULT_RUL_MODEL) -> pd.DataFrame:
    """Vectorized version of _predict_row for a whole engine's lifecycle at
    once -- one scaler.transform, one autoencoder forward pass, one predict
    call, instead of re-running the models per row. This is what makes
    /api/demo/{unit_id} fast even for engines with 300+ cycles."""
    rul_model = _resolve_rul_model(model)
    x_scaled = scaler.transform(df_rows[FEATURE_COLS])

    recon = autoencoder.predict(x_scaled)
    recon_error = np.mean((x_scaled - recon) ** 2, axis=1)
    is_anomaly = recon_error > ANOMALY_THRESHOLD

    pred_rul = np.clip(rul_model.predict(x_scaled), 0, RUL_CAP)

    out = pd.DataFrame({
        "predicted_rul": np.round(pred_rul, 1),
        "reconstruction_error": np.round(recon_error, 5),
        "anomaly_threshold": round(ANOMALY_THRESHOLD, 5),
        "is_anomaly": is_anomaly,
        "risk_level": [_risk_level(r) for r in pred_rul],
        "model_used": model,
    })
    return out


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    return {
        "feature_cols": FEATURE_COLS,
        "anomaly_threshold": ANOMALY_THRESHOLD,
        "rul_cap": RUL_CAP,
        "val_mae_cycles": META.get("val_mae"),
        "available_rul_models": list(RUL_MODELS.keys()),
        "default_rul_model": DEFAULT_RUL_MODEL,
        "model": {
            "anomaly_detector": "PyTorch autoencoder (trained on healthy engine cycles), served via a NumPy re-implementation of the trained weights",
            "failure_predictor": "Random Forest Regressor and XGBoost Regressor (selectable) -> Remaining Useful Life (cycles)",
            "dataset": "NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001)",
        },
    }


@app.get("/api/evaluation")
def evaluation():
    """Honest model evaluation metrics.

    - held_out_validation: 20 engines held out from the 100 training
      engines, never seen during training (unit-level split, computed in
      train.py).
    - official_test_set: NASA's official FD001 test set + RUL labels,
      a completely separate dataset (computed in evaluate.py).
    """
    return {
        "held_out_validation": META.get("evaluation", {}).get("held_out_validation"),
        "methodology": META.get("evaluation", {}).get("methodology"),
        "official_test_set": TEST_EVAL,
    }


@app.get("/api/demo-units")
def demo_units():
    return {"units": sorted(DEMO_DF["unit"].unique().tolist())}


@app.get("/api/demo/{unit_id}")
def demo_unit(unit_id: int, model: str = DEFAULT_RUL_MODEL):
    sub = DEMO_DF[DEMO_DF["unit"] == unit_id].sort_values("cycle").reset_index(drop=True)
    if sub.empty:
        raise HTTPException(status_code=404, detail="Unknown demo unit")

    preds = _predict_batch(sub, model=model)
    history = pd.concat(
        [sub[["cycle"]].astype(int), sub[["RUL"]].rename(columns={"RUL": "true_rul"}), preds],
        axis=1,
    )
    return {"unit": unit_id, "model_used": model, "history": history.to_dict(orient="records")}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(reading: SensorReading, model: str = DEFAULT_RUL_MODEL):
    return _predict_row(reading.dict(), model=model)


# Local dev entrypoint: `python api/index.py`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
