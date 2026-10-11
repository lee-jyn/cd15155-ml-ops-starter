"""
Integration tests for the Sentiment Analysis API.
"""

import pytest
from fastapi.testclient import TestClient

VALID_SENTIMENTS = {"positive", "negative", "neutral"}

HEADLINES = [
    "The company reported record profits and raised its dividend.",
    "The firm filed for bankruptcy after massive losses.",
    "Stocks closed flat on Friday amid low trading volume.",
]


def assert_valid_result(result: dict, text: str) -> None:
    assert set(result) == {"text", "sentiment", "confidence", "latency_ms"}
    assert result["text"] == text
    assert result["sentiment"] in VALID_SENTIMENTS
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["latency_ms"] >= 0.0


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("text", HEADLINES)
def test_predict_returns_valid_response(client: TestClient, text):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    assert_valid_result(response.json(), text)


def test_predict_empty_text_returns_422(client: TestClient):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_predict_batch(client: TestClient):
    response = client.post("/predict/batch", json={"texts": HEADLINES})
    assert response.status_code == 200

    results = response.json()
    assert isinstance(results, list)
    assert len(results) == len(HEADLINES)
    for result, text in zip(results, HEADLINES):
        assert_valid_result(result, text)


def test_predict_batch_empty_list_returns_422(client: TestClient):
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422
