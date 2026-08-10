"""API tests using a fake inference service; no model download is required."""

import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import SentimentPrediction
from app.services import SentimentService, postprocess_logits, stable_softmax


class FakeSentimentService:
    """Deterministic stand-in for Hugging Face inference."""

    def __init__(self) -> None:
        self.is_loaded = False
        self.calls: list[list[str]] = []

    def load(self) -> None:
        self.is_loaded = True

    def predict(self, text: str) -> SentimentPrediction:
        self.calls.append([text])
        return SentimentPrediction(label="positive", confidence=0.91)

    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        self.calls.append(texts)
        return [
            SentimentPrediction(
                label="positive" if "good" in text.lower() else "negative",
                confidence=0.91,
            )
            for text in texts
        ]


def build_client() -> tuple[TestClient, FakeSentimentService]:
    """Return a test client whose app uses fake, local-only inference."""
    fake_service = FakeSentimentService()
    return TestClient(create_app(fake_service)), fake_service


def test_health_initializes_service_once() -> None:
    client, fake_service = build_client()

    with client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "model_loaded": True,
    }
    assert fake_service.is_loaded is True


def test_root_redirects_to_swagger_docs() -> None:
    client, _ = build_client()

    with client:
        response = client.get("/", follow_redirects=False)
        head_response = client.head("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
    assert head_response.status_code == 307
    assert head_response.headers["location"] == "/docs"


def test_predict_returns_normalized_prediction() -> None:
    client, fake_service = build_client()

    with client:
        response = client.post("/predict", json={"text": "A good result."})

    assert response.status_code == 200
    assert response.json() == {"label": "positive", "confidence": 0.91}
    assert fake_service.calls == [["A good result."]]


def test_batch_preserves_order() -> None:
    client, fake_service = build_client()
    texts = ["A good experience.", "This was awful."]

    with client:
        response = client.post("/predict/batch", json={"texts": texts})

    assert response.status_code == 200
    assert response.json() == {
        "predictions": [
            {"label": "positive", "confidence": 0.91},
            {"label": "negative", "confidence": 0.91},
        ]
    }
    assert fake_service.calls == [texts]


def test_predict_rejects_blank_text() -> None:
    client, _ = build_client()

    with client:
        response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422
    assert "text must not be blank" in response.text


def test_batch_rejects_more_than_limit() -> None:
    client, _ = build_client()

    with client:
        response = client.post("/predict/batch", json={"texts": ["ok"] * 33})

    assert response.status_code == 422


def test_openapi_includes_endpoint_summaries_and_schemas() -> None:
    client, _ = build_client()

    schema = client.get("/openapi.json").json()

    assert schema["paths"]["/predict"]["post"]["summary"] == "Predict sentiment for one text"
    assert "SentimentPrediction" in schema["components"]["schemas"]


def test_stable_softmax_handles_large_logits() -> None:
    probabilities = stable_softmax(np.array([[10_000.0, 10_001.0]]))

    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[0, 1] > probabilities[0, 0]


def test_postprocess_logits_maps_class_ids_and_preserves_order() -> None:
    predictions = postprocess_logits(
        np.array(
            [
                [8.0, -8.0],
                [-4.0, 4.0],
                [0.0, 1.0],
            ]
        )
    )

    assert [prediction.label for prediction in predictions] == [
        "negative",
        "positive",
        "positive",
    ]
    assert all(0.0 <= prediction.confidence <= 1.0 for prediction in predictions)


def test_service_runs_onnx_with_tokenizer_numpy_inputs() -> None:
    class ModelInput:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeTokenizer:
        def __call__(self, texts: list[str], **_: object) -> dict[str, np.ndarray]:
            assert texts == ["good", "bad"]
            return {
                "input_ids": np.array([[101, 1], [101, 2]], dtype=np.int32),
                "attention_mask": np.array([[1, 1], [1, 1]], dtype=np.int32),
            }

    class FakeSession:
        def __init__(self) -> None:
            self.received_inputs: dict[str, np.ndarray] | None = None

        def get_inputs(self) -> list[ModelInput]:
            return [ModelInput("input_ids"), ModelInput("attention_mask")]

        def run(
            self, _: None, inputs: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            self.received_inputs = inputs
            return [np.array([[-3.0, 3.0], [3.0, -3.0]])]

    service = SentimentService()
    fake_session = FakeSession()
    service._tokenizer = FakeTokenizer()
    service._session = fake_session

    predictions = service.predict_batch(["good", "bad"])

    assert [prediction.label for prediction in predictions] == ["positive", "negative"]
    assert fake_session.received_inputs is not None
    assert all(
        value.dtype == np.int64 for value in fake_session.received_inputs.values()
    )
