# Sentiment Analysis API

A simple, production-minded REST API that classifies text sentiment with Hugging Face Transformers tokenization and a quantized ONNX export of the pretrained [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) model.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Haarshitt/Sentiment-Analysis)

## Features

- `POST /predict` classifies one text string.
- `POST /predict/batch` classifies up to 32 texts while preserving input order.
- `GET /health` reports API readiness.
- Response labels are normalized to lowercase (`positive` or `negative`).
- Pydantic validation rejects blank text, caps a text at 5,000 characters, and caps a batch at 32 items.
- A shared ONNX Runtime session and Hugging Face tokenizer load once at application startup, rather than once per request.
- Inference uses Xenova's quantized `onnx/model_quantized.onnx` artifact on the CPU; PyTorch is not installed.
- Interactive Swagger UI is available at `/docs`; endpoint descriptions, schemas, and example payloads are included automatically.

## Project structure

```text
.
├── app/
│   ├── main.py       # Routes, lifespan, and OpenAPI metadata
│   ├── schemas.py    # Request/response validation models
│   └── services.py   # Cached Hugging Face inference service
├── tests/
│   └── test_api.py   # API tests with fake inference
├── Dockerfile
└── requirements.txt
```

## Setup

Python 3.10–3.12 is recommended (the included Docker image uses Python 3.11).

```bash
git clone <your-repository-url>
cd sentiment-analysis-api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run locally

```bash
uvicorn app.main:app --reload
```

On first startup, the tokenizer and Xenova's quantized ONNX model download to the local Hugging Face cache. Then open:

- Base URL: `http://127.0.0.1:8000` (redirects to Swagger UI)
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`
- Health check: `http://127.0.0.1:8000/health`

## API examples

### Single prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"The interface is clean and easy to use."}'
```

Example response:

```json
{
  "label": "positive",
  "confidence": 0.9998
}
```

### Batch prediction

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["I love the fast delivery.","The app keeps crashing."]}'
```

Example response:

```json
{
  "predictions": [
    {"label": "positive", "confidence": 0.9989},
    {"label": "negative", "confidence": 0.9976}
  ]
}
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

## Test

```bash
pytest -q
```

Tests inject a deterministic fake service and unit-test ONNX logit post-processing, so they do not download or run model assets.

## Docker

```bash
docker build -t sentiment-analysis-api .
docker run --rm -p 8000:8000 sentiment-analysis-api
```

The first container start downloads the tokenizer and quantized ONNX model unless the image is extended to pre-cache them.

## Deploy on Render

Click the **Deploy to Render** button near the top of this README, sign in to
Render, review the Blueprint, and approve the deployment. Render reads
`render.yaml` from this repository and configures the build command, start
command, Python version, and health check automatically.

After the deployment finishes, append `/docs` to the generated
`https://<service-name>.onrender.com` URL to open the live Swagger UI.

### Why ONNX Runtime on Render free tier?

Render free web services have a 512 MB memory limit. The previous PyTorch-backed
`transformers.pipeline` approach loaded the PyTorch runtime in addition to model
weights and could exceed that limit. This implementation keeps Hugging Face
Transformers for `AutoTokenizer`, but removes PyTorch and runs Xenova's
quantized [`onnx/model_quantized.onnx`](https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/tree/main/onnx)
artifact (about 68 MB) through ONNX Runtime's CPU provider. This substantially
reduces the inference footprint for the 512 MB environment; actual memory use
still depends on process overhead and concurrent traffic.

## Architecture

1. FastAPI validates and documents request/response contracts from Pydantic models.
2. The app lifespan calls `SentimentService.load()` at startup.
3. `SentimentService` downloads `onnx/model_quantized.onnx` from the Xenova repository, then initializes one CPU-only ONNX Runtime session per process behind a lock.
4. `AutoTokenizer` creates NumPy `input_ids` and `attention_mask` tensors; the service passes only the inputs required by the ONNX model.
5. The service applies a numerically stable softmax to logits, maps class `0` to `negative` and class `1` to `positive`, and preserves batch order.

## Resume talking points

- Built a containerized FastAPI inference service around a quantized ONNX DistilBERT SST-2 classifier, avoiding PyTorch for constrained deployments.
- Designed validated single and batch REST endpoints with OpenAPI/Swagger documentation and predictable response schemas.
- Reduced runtime memory pressure on a 512 MB Render instance by using CPU ONNX Runtime with Hugging Face tokenization and a cached, thread-safe session.
- Added health monitoring, input-size safeguards, numerically stable logit post-processing, and mocked API tests that avoid model downloads in CI.
