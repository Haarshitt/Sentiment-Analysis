"""FastAPI application for transformer-based sentiment classification."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI, Request, status

from app.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    HealthResponse,
    PredictRequest,
    SentimentPrediction,
)
from app.services import MODEL_NAME, SentimentService


class SentimentInference(Protocol):
    """Small interface used by routes and replaceable in tests."""

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def predict(self, text: str) -> SentimentPrediction: ...

    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]: ...


def create_app(service: SentimentInference | None = None) -> FastAPI:
    """Create the API app, optionally with an injected service for tests."""
    model_service = service or SentimentService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load the model once during startup."""
        app.state.model_service.load()
        yield

    app = FastAPI(
        title="Sentiment Analysis API",
        version="1.0.0",
        description=(
            "Classify text with Hugging Face's "
            "`distilbert-base-uncased-finetuned-sst-2-english` model."
        ),
        lifespan=lifespan,
    )
    app.state.model_service = model_service

    def get_service(request: Request) -> SentimentInference:
        return request.app.state.model_service

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Operational"],
        summary="Check API readiness",
        description="Returns service status and whether the model is initialized.",
    )
    def health(request: Request) -> HealthResponse:
        """Return the readiness status without running inference."""
        loaded_service = get_service(request)
        return HealthResponse(
            status="ok",
            model=MODEL_NAME,
            model_loaded=loaded_service.is_loaded,
        )

    @app.post(
        "/predict",
        response_model=SentimentPrediction,
        status_code=status.HTTP_200_OK,
        tags=["Inference"],
        summary="Predict sentiment for one text",
        description=(
            "Returns a lowercase sentiment label and confidence score. "
            "Input is truncated to the model's 512-token maximum."
        ),
    )
    def predict(payload: PredictRequest, request: Request) -> SentimentPrediction:
        """Run one sentiment prediction."""
        return get_service(request).predict(payload.text)

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        status_code=status.HTTP_200_OK,
        tags=["Inference"],
        summary="Predict sentiment for multiple texts",
        description=(
            "Classifies up to 32 texts in one request and preserves input order. "
            "Each text is truncated to the model's 512-token maximum."
        ),
    )
    def predict_batch(
        payload: BatchPredictRequest, request: Request
    ) -> BatchPredictionResponse:
        """Run a batch of sentiment predictions."""
        predictions = get_service(request).predict_batch(payload.texts)
        return BatchPredictionResponse(predictions=predictions)

    return app


app = create_app()
