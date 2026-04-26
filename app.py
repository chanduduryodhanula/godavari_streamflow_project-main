from __future__ import annotations
 
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
 
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
 
# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("godavari")
 
# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path("artifacts/models/best_model.pkl")
 
FEATURE_ORDER = [
    "Rainfall_mm",
    "Temperature_C",
    "Rainfall_t-1",
    "Rainfall_t-2",
    "Rainfall_t-3",
    "Streamflow_t-1",
    "Streamflow_t-2",
    "Streamflow_t-3",
    "Month",
    "Day",
]
 
# Flood thresholds (m³/s) — adjust to match your basin calibration
THRESHOLD_ELEVATED = 150.0
THRESHOLD_FLOOD    = 300.0
 
# Reasonable physical bounds for input validation
BOUNDS: dict[str, tuple[float, float]] = {
    "Rainfall_mm":    (0.0,   500.0),
    "Temperature_C":  (-5.0,  55.0),
    "Streamflow_t-1": (0.0,   10_000.0),
    "Streamflow_t-2": (0.0,   10_000.0),
    "Streamflow_t-3": (0.0,   10_000.0),
}
 
# ── Global model bundle ───────────────────────────────────────────────────────
_BUNDLE: dict[str, Any] | None = None
_LOAD_TS: str | None = None
 
 
# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    global _BUNDLE, _LOAD_TS
    log.info("Loading model from %s …", MODEL_PATH)
    if not MODEL_PATH.exists():
        log.critical("Model file not found at %s — aborting startup.", MODEL_PATH)
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    _BUNDLE = joblib.load(MODEL_PATH)
    _LOAD_TS = datetime.now(timezone.utc).isoformat()
    log.info("Model loaded successfully. Bundle keys: %s", list(_BUNDLE.keys()))
    yield
    log.info("Shutting down Godavari API.")
 
 
# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Godavari Streamflow Prediction API",
    description=(
        "AI-powered hydrological forecasting for the Godavari River Basin, India. "
        "Accepts current and lagged hydrometeorological inputs and returns a "
        "predicted streamflow value (m³/s) using a trained Random Forest model."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
 
 
# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
 
    Rainfall_mm:    float = Field(..., alias="Rainfall_mm",    description="Current rainfall (mm)",          ge=0,    le=500)
    Temperature_C:  float = Field(..., alias="Temperature_C",  description="Current temperature (°C)",       ge=-5,   le=55)
    Rainfall_t_1:   float = Field(..., alias="Rainfall_t-1",   description="Rainfall 1 day ago (mm)",        ge=0,    le=500)
    Rainfall_t_2:   float = Field(..., alias="Rainfall_t-2",   description="Rainfall 2 days ago (mm)",       ge=0,    le=500)
    Rainfall_t_3:   float = Field(..., alias="Rainfall_t-3",   description="Rainfall 3 days ago (mm)",       ge=0,    le=500)
    Streamflow_t_1: float = Field(..., alias="Streamflow_t-1", description="Streamflow 1 day ago (m³/s)",    ge=0,    le=10000)
    Streamflow_t_2: float = Field(..., alias="Streamflow_t-2", description="Streamflow 2 days ago (m³/s)",   ge=0,    le=10000)
    Streamflow_t_3: float = Field(..., alias="Streamflow_t-3", description="Streamflow 3 days ago (m³/s)",   ge=0,    le=10000)
 
    @field_validator("Temperature_C")
    @classmethod
    def temperature_sanity(cls, v: float) -> float:
        if v > 50:
            log.warning("Unusually high temperature received: %.1f °C", v)
        return v
 
 
class PredictResponse(BaseModel):
    prediction_streamflow_m3s: float = Field(..., description="Predicted streamflow (m³/s)")
    flow_status:               str   = Field(..., description="Risk classification")
    alert_message:             str   = Field(..., description="Human-readable advisory")
    season:                    str   = Field(..., description="Hydrological season at prediction time")
    month:                     int   = Field(..., description="Month used for seasonal encoding")
    day:                       int   = Field(..., description="Day-of-month used for encoding")
    model_version:             str   = Field(..., description="Model bundle version tag")
    predicted_at:              str   = Field(..., description="UTC timestamp of prediction")
    inference_ms:              float = Field(..., description="Inference latency in milliseconds")
 
 
class HealthResponse(BaseModel):
    status:       str
    message:      str
    model_loaded: bool
    model_path:   str
    loaded_at:    str | None
    api_version:  str
 
 
class ModelInfoResponse(BaseModel):
    model_type:       str
    feature_count:    int
    encoded_columns:  list[str]
    model_version:    str
    loaded_at:        str | None
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def _month_to_season(month: int) -> str:
    if month in (6, 7, 8, 9):
        return "Monsoon"
    if month in (10, 11, 12, 1, 2):
        return "Winter"
    return "Summer"
 
 
def _classify_flow(value: float) -> tuple[str, str]:
    """Returns (status_label, advisory_message)."""
    if value < THRESHOLD_ELEVATED:
        return (
            "Normal Flow",
            "Flow is within the safe operating range. No immediate action required.",
        )
    if value < THRESHOLD_FLOOD:
        return (
            "Elevated Flow",
            "Flow is elevated. Monitor downstream gauges closely and prepare response teams.",
        )
    return (
        "Flood Risk",
        "High-flow event detected — activate the flood warning protocol immediately.",
    )
 
 
def _prepare_features(payload: PredictRequest, encoded_columns: list[str]) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    season = _month_to_season(now.month)
 
    row: dict[str, Any] = {
        "Rainfall_mm":    payload.Rainfall_mm,
        "Temperature_C":  payload.Temperature_C,
        "Rainfall_t-1":   payload.Rainfall_t_1,
        "Rainfall_t-2":   payload.Rainfall_t_2,
        "Rainfall_t-3":   payload.Rainfall_t_3,
        "Streamflow_t-1": payload.Streamflow_t_1,
        "Streamflow_t-2": payload.Streamflow_t_2,
        "Streamflow_t-3": payload.Streamflow_t_3,
        "Month":          now.month,
        "Day":            now.day,
        "Data_Type":      "Real",
        "Season":         season,
    }
 
    df = pd.DataFrame([row])
    encoded = pd.get_dummies(df, columns=["Data_Type", "Season"], drop_first=False)
    encoded = encoded.reindex(columns=encoded_columns, fill_value=0)
    return encoded, season, now.month, now.day
 
 
# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    return response
 
 
# ── Exception handlers ────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. Please try again later."},
    )
 
 
# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    """Quick liveness check — returns API and model status."""
    return HealthResponse(
        status="ok",
        message="Godavari Streamflow Prediction API is running.",
        model_loaded=_BUNDLE is not None,
        model_path=str(MODEL_PATH),
        loaded_at=_LOAD_TS,
        api_version=app.version,
    )
 
 
@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info() -> ModelInfoResponse:
    """Returns metadata about the currently loaded model bundle."""
    if _BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    model = _BUNDLE["model"]
    encoded_columns = _BUNDLE.get("encoded_columns", [])
    return ModelInfoResponse(
        model_type=type(model).__name__,
        feature_count=len(encoded_columns),
        encoded_columns=encoded_columns,
        model_version=_BUNDLE.get("version", "unknown"),
        loaded_at=_LOAD_TS,
    )
 
 
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Run streamflow prediction.
 
    Accepts current and 3-day lagged hydrometeorological inputs.
    Returns predicted streamflow (m³/s), risk classification, and advisory.
    """
    if _BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
 
    t0 = time.perf_counter()
 
    try:
        model          = _BUNDLE["model"]
        encoded_cols   = _BUNDLE["encoded_columns"]
    except KeyError as exc:
        log.error("Invalid model bundle — missing key: %s", exc)
        raise HTTPException(status_code=500, detail=f"Invalid model bundle format: {exc}") from exc
 
    try:
        X, season, month, day = _prepare_features(request, encoded_cols)
        raw_pred   = float(model.predict(X)[0])
        prediction = max(0.0, round(raw_pred, 3))   # clamp negatives from tree variance
    except Exception as exc:
        log.exception("Prediction pipeline failed.")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
 
    elapsed_ms = (time.perf_counter() - t0) * 1000
    flow_status, alert_msg = _classify_flow(prediction)
 
    log.info(
        "Prediction: %.3f m³/s | status=%s | rain=%.1f mm | temp=%.1f °C | latency=%.1f ms",
        prediction, flow_status, request.Rainfall_mm, request.Temperature_C, elapsed_ms,
    )
 
    return PredictResponse(
        prediction_streamflow_m3s=prediction,
        flow_status=flow_status,
        alert_message=alert_msg,
        season=season,
        month=month,
        day=day,
        model_version=_BUNDLE.get("version", "unknown"),
        predicted_at=datetime.now(timezone.utc).isoformat(),
        inference_ms=round(elapsed_ms, 2),
    )
 
 
@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(requests_list: list[PredictRequest]) -> list[PredictResponse]:
    """
    Run predictions for multiple input rows in a single request.
    Returns a list of PredictResponse objects in the same order.
    """
    if _BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if len(requests_list) > 100:
        raise HTTPException(status_code=400, detail="Batch size must not exceed 100.")
    return [predict(req) for req in requests_list]
 