"""API tests using a fake inference service; no model download is required."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import SentimentPrediction


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
