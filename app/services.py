"""Memory-efficient Hugging Face tokenizer and ONNX Runtime inference service."""

from __future__ import annotations

import os
from threading import Lock
from typing import Any

import numpy as np

from app.schemas import SentimentPrediction

# Transformers is used only for tokenization. Set this before its lazy imports so
# it never attempts to initialize a PyTorch backend.
os.environ.setdefault("USE_TORCH", "0")

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
ONNX_REPOSITORY = "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
ONNX_MODEL_FILE = "onnx/model_quantized.onnx"
MAX_MODEL_TOKENS = 512
CLASS_LABELS = {0: "negative", 1: "positive"}


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute row-wise softmax without overflow from large logits."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("logits must have shape (batch_size, class_count)")

    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def postprocess_logits(logits: np.ndarray) -> list[SentimentPrediction]:
    """Convert DistilBERT logits into ordered, normalized predictions."""
    probabilities = stable_softmax(logits)
    if probabilities.shape[1] != len(CLASS_LABELS):
        raise ValueError(
            f"expected {len(CLASS_LABELS)} sentiment classes, got "
            f"{probabilities.shape[1]}"
        )

    class_ids = np.argmax(probabilities, axis=1)
    return [
        SentimentPrediction(
            label=CLASS_LABELS[int(class_id)],
            confidence=float(probabilities[index, class_id]),
        )
        for index, class_id in enumerate(class_ids)
    ]


class SentimentService:
    """Loads one quantized ONNX model and Hugging Face tokenizer per process."""

    def __init__(self) -> None:
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._load_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        """Return whether both tokenizer and ONNX inference session are ready."""
        return self._session is not None and self._tokenizer is not None

    def load(self) -> None:
        """Download and initialize the quantized CPU model once per process."""
        if self.is_loaded:
            return

        with self._load_lock:
            if self.is_loaded:
                return

            # These imports stay local so API unit tests do not need model packages.
            from huggingface_hub import hf_hub_download
            from onnxruntime import InferenceSession, SessionOptions
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(ONNX_REPOSITORY)
            model_path = hf_hub_download(
                repo_id=ONNX_REPOSITORY,
                filename=ONNX_MODEL_FILE,
            )
            session_options = SessionOptions()
            session_options.intra_op_num_threads = 1
            session = InferenceSession(
                model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )

            self._tokenizer = tokenizer
            self._session = session

    def predict(self, text: str) -> SentimentPrediction:
        """Classify one text string."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        """Classify text strings in input order using direct ONNX inference."""
        self.load()
        assert self._session is not None
        assert self._tokenizer is not None

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_MODEL_TOKENS,
            return_tensors="np",
        )
        input_names = [model_input.name for model_input in self._session.get_inputs()]
        missing_inputs = [name for name in input_names if name not in encoded]
        if missing_inputs:
            raise RuntimeError(
                "Tokenizer did not produce required ONNX inputs: "
                + ", ".join(missing_inputs)
            )

        onnx_inputs = {
            name: np.asarray(encoded[name], dtype=np.int64) for name in input_names
        }
        outputs = self._session.run(None, onnx_inputs)
        return postprocess_logits(np.asarray(outputs[0]))
