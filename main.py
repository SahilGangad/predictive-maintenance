import time
import asyncio
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
import joblib, os
from schemas import MachineInput, PredictionResponse
from fastapi.responses import FileResponse

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'model', 'artifacts')

model         = None
scaler        = None
label_encoder = None
feature_cols  = None
process_pool  = None   # ProcessPoolExecutor for CPU-bound batch scoring

# Workers are forked (default start method on Linux) AFTER the model is
# loaded below, so each worker process inherits its own in-memory copy of
# model/scaler/label_encoder/feature_cols via copy-on-write — no re-loading
# or IPC needed per request.
WORKER_COUNT = max(1, (os.cpu_count() or 2) - 1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, label_encoder, feature_cols, process_pool
    model         = joblib.load(os.path.join(ARTIFACTS_DIR, 'model.pkl'))
    scaler        = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl'))
    feature_cols  = joblib.load(os.path.join(ARTIFACTS_DIR, 'feature_cols.pkl'))
    print("✅ Model loaded")

    # Pool is created *after* the model is loaded so forked workers already
    # have it in memory.
    process_pool = ProcessPoolExecutor(max_workers=WORKER_COUNT)
    print(f"✅ Process pool started with {WORKER_COUNT} workers")

    yield

    process_pool.shutdown(wait=True)

app = FastAPI(title="AI4I Predictive Maintenance API", lifespan=lifespan)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for now)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def _score_record(data: MachineInput) -> dict:
    """
    Pure, top-level scoring function with no FastAPI/HTTP concerns.
    Must stay top-level (not a closure/method) so it can be pickled and
    sent to worker processes by ProcessPoolExecutor.
    """
    features        = build_features(data)
    features_scaled = scaler.transform(features)
    prediction      = model.predict(features_scaled)[0]
    probability     = model.predict_proba(features_scaled)[0][1]
    risk            = get_risk_level(float(probability))

    return {
        "failure_predicted": bool(prediction),
        "failure_probability": round(float(probability), 4),
        "risk_level": risk,
        "message": "⚠️ Maintenance required!" if prediction else "✅ Machine operating normally."
    }

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
        return PredictionResponse(**_score_record(data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(data: list[MachineInput]):
    """
    Sequential batch scoring — single process, single core. Fine for small
    batches; kept as the baseline to compare against the parallel version.
    """
    try:
        return [PredictionResponse(**_score_record(d)) for d in data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch/parallel", response_model=list[PredictionResponse])
async def predict_batch_parallel(data: list[MachineInput]):
    """
    CPU-bound batch scoring, parallelized across processes.

    Model inference (Random Forest / XGBoost predict + predict_proba) is
    CPU-bound, not I/O-bound — so this uses a ProcessPoolExecutor (real OS
    processes, sidesteps the GIL) rather than asyncio/threads, which would
    only help if we were waiting on network/disk I/O, not crunching numbers.

    The endpoint itself is `async def` only so the event loop stays free
    to serve other requests while these workers run — the parallelism
    doing the actual work is the process pool, not the `async` keyword.
    """
    try:
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(process_pool, _score_record, d) for d in data]
        results = await asyncio.gather(*tasks)
        return [PredictionResponse(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/benchmark")
def benchmark(data: list[MachineInput]):
    """
    Scores the same batch sequentially vs. via the process pool and returns
    timing for both, so the speedup (or overhead, for small batches) is
    directly visible instead of asserted.
    """
    t0 = time.perf_counter()
    _ = [_score_record(d) for d in data]
    sequential_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    futures = [process_pool.submit(_score_record, d) for d in data]
    _ = [f.result() for f in futures]
    parallel_seconds = time.perf_counter() - t0

    return {
        "batch_size": len(data),
        "worker_count": WORKER_COUNT,
        "sequential_seconds": round(sequential_seconds, 4),
        "parallel_seconds": round(parallel_seconds, 4),
        "speedup_x": round(sequential_seconds / parallel_seconds, 2) if parallel_seconds > 0 else None,
        "note": "Process-pool overhead (spawning/serializing to workers) can "
                "make small batches slower in parallel than sequential — the "
                "crossover point is exactly what this endpoint lets you measure."
    }