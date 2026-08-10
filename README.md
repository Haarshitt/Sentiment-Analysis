# Sentiment Analysis API

A simple, production-minded REST API that classifies text sentiment with Hugging Face Transformers tokenization and a quantized ONNX export of the pretrained [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) model.



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


## Architecture

1. FastAPI validates and documents request/response contracts from Pydantic models.
2. The app lifespan calls `SentimentService.load()` at startup.
3. `SentimentService` downloads `onnx/model_quantized.onnx` from the Xenova repository, then initializes one CPU-only ONNX Runtime session per process behind a lock.
4. `AutoTokenizer` creates NumPy `input_ids` and `attention_mask` tensors; the service passes only the inputs required by the ONNX model.
5. The service applies a numerically stable softmax to logits, maps class `0` to `negative` and class `1` to `positive`, and preserves batch order.

