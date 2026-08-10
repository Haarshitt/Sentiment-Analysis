"""Hugging Face inference service with process-level model caching."""

from __future__ import annotations

from threading import Lock
from typing import Any

from app.schemas import SentimentPrediction

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


class SentimentService:
    """Loads one Transformers pipeline and exposes prediction methods."""

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._load_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        """Return whether the Transformers pipeline is ready."""
        return self._pipeline is not None

    def load(self) -> None:
        """Initialize the model once for this application process."""
        if self._pipeline is not None:
            return

        with self._load_lock:
            if self._pipeline is not None:
                return

            # Import here so API unit tests can run without Transformers installed.
            from transformers import pipeline

            self._pipeline = pipeline(
                task="sentiment-analysis",
                model=MODEL_NAME,
                tokenizer=MODEL_NAME,
            )

    def predict(self, text: str) -> SentimentPrediction:
        """Classify one text string."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        """Classify text strings, preserving their input order."""
        self.load()
        assert self._pipeline is not None

        outputs = self._pipeline(texts, truncation=True, max_length=512)
        return [
            SentimentPrediction(
                label=str(output["label"]).lower(),
                confidence=float(output["score"]),
            )
            for output in outputs
        ]
