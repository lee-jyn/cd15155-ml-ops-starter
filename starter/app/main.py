"""
Sentiment Prediction API.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
    GET  /metrics         — Prometheus metrics
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from utils import load_classifier

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("sentiment-api")


def log(level: str, message: str, **kwargs) -> None:
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level.upper(),
        "message": message,
        **kwargs,
    }
    logger.log(
        getattr(logging, level.upper(), logging.INFO), json.dumps(entry, default=str)
    )


PREDICTION_REQUESTS = Counter(
    "prediction_requests",
    "Total number of predictions served, labelled by sentiment",
    ["sentiment"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_ms",
    "Model inference latency in milliseconds",
    buckets=(5, 10, 25, 50, 75, 100, 250, 500, 1000, 2500, 5000),
)
PREDICTION_ERRORS = Counter(
    "prediction_errors",
    "Total number of failed prediction calls",
)

load_dotenv()

MAX_TEXT_CHARS = 2000

classifiers = {}


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: list[str]


class PredictionResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    log("INFO", "Loading model...")
    classifiers["sentiment"] = load_classifier()
    log("INFO", "Model loaded successfully")
    yield
    classifiers.clear()
    log("INFO", "Model unloaded")


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


def run_predictions(texts: list[str]) -> list[PredictionResult]:
    try:
        long_texts = [i for i, t in enumerate(texts) if len(t) > MAX_TEXT_CHARS]
        if long_texts:
            log(
                "WARNING",
                f"Input text exceeds {MAX_TEXT_CHARS} characters and will be truncated",
                indices=long_texts,
            )

        start = time.perf_counter()
        outputs = classifiers["sentiment"](texts, truncation=True, max_length=512)
        latency_ms = (time.perf_counter() - start) * 1000

        results = [
            PredictionResult(
                text=text,
                sentiment=output["label"],
                confidence=output["score"],
                latency_ms=latency_ms,
            )
            for text, output in zip(texts, outputs)
        ]

        PREDICTION_LATENCY.observe(latency_ms)
        for result in results:
            PREDICTION_REQUESTS.labels(sentiment=result.sentiment).inc()
            log(
                "INFO",
                "Prediction served",
                sentiment=result.sentiment,
                confidence=round(result.confidence, 4),
                latency_ms=round(result.latency_ms, 2),
            )
        return results
    except Exception as e:
        PREDICTION_ERRORS.inc()
        log("ERROR", "Prediction failed", error=str(e), batch_size=len(texts))
        raise


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def ensure_model_loaded() -> None:
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/health")
def health():
    ensure_model_loaded()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    ensure_model_loaded()
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be empty")
    return run_predictions([request.text])[0]


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest):
    ensure_model_loaded()
    if not request.texts:
        raise HTTPException(status_code=422, detail="Texts list must not be empty")
    if any(not text.strip() for text in request.texts):
        raise HTTPException(status_code=422, detail="Texts must not be empty")
    return run_predictions(request.texts)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )
