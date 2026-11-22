import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from src.serving.schemas import PredictionResponse, TransactionRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraud_api")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/fraud_pipeline.joblib")
META_PATH = os.getenv("META_PATH", "artifacts/metadata.json")

state = {"model": None, "meta": {}}


def _sanitize_for_json(obj):
    """Convert non-finite floats (NaN, Infinity) to strings so 422 error responses serialize cleanly."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, BaseException):
        return str(obj)
    return obj


def _load_artifacts() -> None:
    """Safely load serialized model and metadata, leaving API in degraded mode on any failure."""
    state["model"] = None
    state["meta"] = {}

    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        logger.error(
            "Startup degraded: model or metadata missing | model_exists=%s | meta_exists=%s",
            os.path.exists(MODEL_PATH),
            os.path.exists(META_PATH),
        )
        return

    try:
        loaded_model = joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.exception(
            "Startup degraded: failed to deserialize model artifact at %s (%s: %s)",
            MODEL_PATH,
            type(exc).__name__,
            exc,
        )
        return

    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            loaded_meta = json.load(f)
        if not isinstance(loaded_meta, dict):
            raise TypeError(f"Expected metadata JSON object, got {type(loaded_meta).__name__}")
    except Exception as exc:
        logger.exception(
            "Startup degraded: failed to parse metadata JSON at %s (%s: %s)",
            META_PATH,
            type(exc).__name__,
            exc,
        )
        return

    state["model"] = loaded_model
    state["meta"] = loaded_meta
    logger.info(
        "Loaded model artifact successfully | path=%s | version=%s",
        MODEL_PATH,
        state["meta"].get("model_version", "unknown"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    try:
        yield
    finally:
        state["model"] = None
        state["meta"] = {}


app = FastAPI(title="Fraud Detection Real-Time Scoring API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": _sanitize_for_json(exc.errors())},
    )


def build_feature_row(payload: TransactionRequest) -> pd.DataFrame:
    data = payload.model_dump()
    hour_of_day = (data["Time"] / 3600.0) % 24.0
    data["hour_sin"] = float(np.sin(2.0 * np.pi * hour_of_day / 24.0))
    data["hour_cos"] = float(np.cos(2.0 * np.pi * hour_of_day / 24.0))
    cols = state["meta"].get(
        "feature_cols", ["Amount", "hour_sin", "hour_cos"] + [f"V{i}" for i in range(1, 29)]
    )
    return pd.DataFrame([{c: data[c] for c in cols}]).astype(float)


@app.get("/health")
def health():
    loaded = state["model"] is not None
    return {
        "status": "healthy" if loaded else "degraded",
        "model_loaded": loaded,
        "model_version": state["meta"].get("model_version", "unavailable"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionRequest):
    start = time.perf_counter()
    threshold = float(state["meta"].get("decision_threshold", 0.50))
    fallback_cutoff = float(state["meta"].get("fallback_amount_p995", 2500.0))

    try:
        if state["model"] is None:
            raise RuntimeError("Model artifact not loaded in application state")
        X_row = build_feature_row(payload)
        prob = float(state["model"].predict_proba(X_row)[0, 1])
        is_fraud = bool(prob >= threshold)
        mode = "ml_model"
    except Exception as exc:
        logger.exception(
            "ML inference failed for transaction_id=%s (%s: %s); engaging fallback_rule",
            payload.transaction_id,
            type(exc).__name__,
            exc,
        )
        safe_amount = payload.Amount if payload.Amount is not None else 0.0
        is_fraud = bool(safe_amount >= fallback_cutoff)
        prob = 0.95 if is_fraud else 0.01
        mode = "fallback_rule"

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PredictionResponse(
        transaction_id=payload.transaction_id,
        fraud_probability=round(prob, 6),
        is_fraud=is_fraud,
        threshold_used=threshold,
        scoring_mode=mode,
        latency_ms=round(elapsed_ms, 3),
    )