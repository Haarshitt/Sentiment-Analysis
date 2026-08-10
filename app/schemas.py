"""Pydantic request and response models for the API."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_TEXT_LENGTH = 5_000
MAX_BATCH_SIZE = 32

TextInput = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Text to classify. Whitespace-only input is rejected.",
    ),
]


class PredictRequest(BaseModel):
    """Payload for a single sentiment prediction."""

    text: TextInput

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"text": "The onboarding experience was fast and intuitive."}
            ]
        }
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject strings that only contain whitespace."""
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class BatchPredictRequest(BaseModel):
    """Payload for multiple sentiment predictions."""

    texts: list[TextInput] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"One to {MAX_BATCH_SIZE} texts to classify.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "texts": [
                        "This product exceeded my expectations.",
                        "The checkout flow was confusing and slow.",
                    ]
                }
            ]
        }
    )

    @field_validator("texts")
    @classmethod
    def texts_must_not_contain_blanks(cls, values: list[str]) -> list[str]:
        """Ensure every batch item contains visible text."""
        if any(not value.strip() for value in values):
            raise ValueError("texts must not contain blank entries")
        return values


class SentimentPrediction(BaseModel):
    """A normalized model prediction."""

    label: str = Field(
        description="Normalized lowercase sentiment label.",
        examples=["positive"],
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Model confidence score from 0 to 1.",
        examples=[0.9987],
    )


class BatchPredictionResponse(BaseModel):
    """Predictions returned in the same order as the request texts."""

    predictions: list[SentimentPrediction] = Field(
        description="One prediction per submitted text, in request order."
    )


class HealthResponse(BaseModel):
    """Service readiness status."""

    status: str = Field(examples=["ok"])
    model: str = Field(examples=["distilbert-base-uncased-finetuned-sst-2-english"])
    model_loaded: bool = Field(
        description="Whether the model pipeline has been initialized.",
        examples=[True],
    )
