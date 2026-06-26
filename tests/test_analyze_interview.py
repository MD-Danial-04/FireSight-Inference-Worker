from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)

SAMPLE_TRANSCRIPT = (
    "The device is a PMD, model Xiaomi M365. It uses a lithium-ion battery. "
    "The battery brand is Samsung. I was charging it in the living room when I heard a pop."
)

SAMPLE_QUESTIONS = [
    {"id": "device-type", "prompt": "What type of mobility device is it?", "hint": "e.g. PMD"},
    {"id": "device-model", "prompt": "What is the device's model?"},
    {"id": "battery-brand", "prompt": "What is the battery's brand?"},
    {"id": "charging-location", "prompt": "Where was the device when the incident occurred?"},
]

LLM_ANALYSIS_RESPONSE = """
```json
{
  "coverage": [
    {
      "id": "device-type",
      "status": "answered",
      "answer": "PMD",
      "evidence": "The device is a PMD",
      "confidence": 0.95
    },
    {
      "id": "device-model",
      "status": "answered",
      "answer": "Xiaomi M365",
      "evidence": "model Xiaomi M365",
      "confidence": 0.9
    },
    {
      "id": "battery-brand",
      "status": "answered",
      "answer": "Samsung",
      "evidence": "battery brand is Samsung",
      "confidence": 0.85
    },
    {
      "id": "charging-location",
      "status": "partial",
      "answer": "Living room",
      "evidence": "charging it in the living room",
      "confidence": 0.7
    }
  ],
  "follow_ups": [
    {
      "related_question_id": "charging-location",
      "prompt": "Was the device on charge or in use at the time of the incident?",
      "prompt_conduct": "[MS] Was the device on charge or in use at the time of the incident?",
      "reason": "Location mentioned but charging context needs clarification"
    }
  ]
}
```
"""


def test_analyze_interview_returns_fake_coverage():
    response = client.post(
        "/v1/analyze-interview",
        json={"transcript": SAMPLE_TRANSCRIPT, "questions": SAMPLE_QUESTIONS},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "fake"
    assert len(data["coverage"]) == len(SAMPLE_QUESTIONS)
    assert all(item["id"] in {q["id"] for q in SAMPLE_QUESTIONS} for item in data["coverage"])
    assert all(
        item["status"] in {"answered", "partial", "unanswered", "unclear"}
        for item in data["coverage"]
    )
    assert all("answer" in item for item in data["coverage"])
    assert isinstance(data["follow_ups"], list)


def test_analyze_interview_requires_transcript():
    response = client.post(
        "/v1/analyze-interview",
        json={"transcript": "", "questions": SAMPLE_QUESTIONS},
    )
    assert response.status_code == 422


def test_analyze_interview_llm_parses_json():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": LLM_ANALYSIS_RESPONSE}}],
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(settings, "use_fake_extraction", False),
        patch("app.analyze_interview.httpx.AsyncClient", return_value=mock_client),
    ):
        response = client.post(
            "/v1/analyze-interview",
            json={"transcript": SAMPLE_TRANSCRIPT, "questions": SAMPLE_QUESTIONS},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "ollama"
    assert data["coverage"][0]["status"] == "answered"
    assert data["coverage"][0]["id"] == "device-type"
    assert data["coverage"][0]["answer"] == "PMD"
    assert len(data["follow_ups"]) == 1
    assert data["follow_ups"][0]["related_question_id"] == "charging-location"
    assert data["follow_ups"][0]["prompt_conduct"]


def test_analyze_interview_fake_includes_prompt_conduct_for_non_english():
    response = client.post(
        "/v1/analyze-interview",
        json={
            "transcript": SAMPLE_TRANSCRIPT,
            "questions": SAMPLE_QUESTIONS,
            "interview_language": "ms",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert any(
        follow_up["prompt_conduct"].startswith("[MS]")
        for follow_up in data["follow_ups"]
    )


def test_analyze_interview_llm_invalid_json_returns_502():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "not valid json"}}],
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(settings, "use_fake_extraction", False),
        patch("app.analyze_interview.httpx.AsyncClient", return_value=mock_client),
    ):
        response = client.post(
            "/v1/analyze-interview",
            json={"transcript": SAMPLE_TRANSCRIPT, "questions": SAMPLE_QUESTIONS},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "LLM returned invalid JSON"
