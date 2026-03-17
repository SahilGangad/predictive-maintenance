import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib, os
from schemas import MachineInput, PredictionResponse
from fastapi.responses import FileResponse

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'model', 'artifacts')

model         = None
scaler        = None
label_encoder = None
feature_cols  = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, label_encoder, feature_cols
    model         = joblib.load(os.path.join(ARTIFACTS_DIR, 'model.pkl'))
    scaler        = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl'))
    feature_cols  = joblib.load(os.path.join(ARTIFACTS_DIR, 'feature_cols.pkl'))
    print("✅ Model loaded")
    yield

app = FastAPI(title="AI4I Predictive Maintenance API", lifespan=lifespan)

def get_risk_level(prob: float) -> str:
    if prob < 0.2:  return "LOW"
    if prob < 0.5:  return "MEDIUM"
    if prob < 0.75: return "HIGH"
    return "CRITICAL"

def build_features(data: MachineInput) -> np.ndarray:
    # ── Original features ─────────────────────────────────
    type_enc          = label_encoder.transform([data.machine_type])[0]
    air_temp          = data.air_temperature
    process_temp      = data.process_temperature
    rpm               = data.rotational_speed
    torque            = data.torque
    tool_wear         = data.tool_wear

    # ── Engineered features (must match notebook exactly) ─
    temp_delta              = process_temp - air_temp
    power_watts             = torque * (rpm * 2 * np.pi / 60)
    wear_torque_interaction = tool_wear * torque
    speed_torque_ratio      = rpm / (torque + 1e-6)
    wear_bin                = int(pd.cut(
                                [tool_wear],
                                bins=[0, 60, 120, 180, 300],
                                labels=[0, 1, 2, 3],
                                include_lowest=True
                              )[0] or 0)
    high_temp_flag          = int(process_temp > 312.0)   # Q90 from training data
    low_speed_flag          = int(rpm < 1380.0)           # Q10 from training data

    # ── Return in same order as FEATURE_COLS ──────────────
    return np.array([[
        type_enc, air_temp, process_temp, rpm,
        torque, tool_wear,
        temp_delta, power_watts, wear_torque_interaction,
        speed_torque_ratio, wear_bin, high_temp_flag, low_speed_flag
    ]])

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None,
            "features_expected": len(feature_cols) if feature_cols else 0}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: MachineInput):
    try:
        features        = build_features(data)
        features_scaled = scaler.transform(features)
        prediction      = model.predict(features_scaled)[0]
        probability     = model.predict_proba(features_scaled)[0][1]
        risk            = get_risk_level(float(probability))

        return PredictionResponse(
            failure_predicted=bool(prediction),
            failure_probability=round(float(probability), 4),
            risk_level=risk,
            message="⚠️ Maintenance required!" if prediction else "✅ Machine operating normally."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(data: list[MachineInput]):
    return [predict(d) for d in data]