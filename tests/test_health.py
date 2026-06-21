from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.config import settings
from app.extract import EXTRACTABLE_FIELDS, _parse_llm_json
from app.main import app

client = TestClient(app)

FAM_STOP_MESSAGE = (
    "LF812 stop for location at 7 Gul Ave. Case classified as False alarm malfunction "
    "of manual Call point, at zone 7. Upon investigation No smoke no fire. "
    "Case handed over to SGT3 alsyraf waT190350 from Nanyang NPC"
)

LLM_JSON_RESPONSE = """
```json
{
  "fields": {
    "applianceCallSign": "LF812",
    "locationOfFire": "7 Gul Ave",
    "fireInvolved": "False alarm malfunction",
    "methodOfExtinguishment": "",
    "damagesSustained": "",
    "probableCause": "False alarm malfunction of manual call point",
    "ignitionSource": "",
    "ignitionFuel": "",
    "eventsCircumstances": "No smoke no fire",
    "areaOfFireOrigin": "Zone 7",
    "classification": "False alarm malfunction",
    "handoverOfficer": "SGT3 Alsyraf T190350",
    "handoverNpc": "Nanyang NPC"
  },
  "confidence": {
    "applianceCallSign": 0.95,
    "locationOfFire": 0.9,
    "fireInvolved": 0.8,
    "methodOfExtinguishment": 0.0,
    "damagesSustained": 0.0,
    "probableCause": 0.85,
    "ignitionSource": 0.0,
    "ignitionFuel": 0.0,
    "eventsCircumstances": 0.7,
    "areaOfFireOrigin": 0.8,
    "classification": 0.85,
    "handoverOfficer": 0.75,
    "handoverNpc": 0.75
  }
}
```
"""

LLM_JSON_WITH_PREAMBLE = (
    "Here is the extracted data in JSON format:\n\n" + LLM_JSON_RESPONSE
)


def test_parse_llm_json_handles_preamble_and_fences():
    parsed = _parse_llm_json(LLM_JSON_WITH_PREAMBLE)
    assert parsed["fields"]["applianceCallSign"] == "LF812"
    assert parsed["fields"]["locationOfFire"] == "7 Gul Ave"


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["fake_extraction"] is True
    assert data["fake_transcription"] is True


def test_health_reports_llm_and_whisper_config():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["llm_base_url"] == settings.llm_base_url
    assert data["llm_model"] == settings.llm_model
    assert data["whisper_model"] == settings.whisper_model
    assert data["whisper_device"] == settings.whisper_device


def test_extract_returns_fake_fields():
    response = client.post(
        "/v1/extract",
        json={
            "text": FAM_STOP_MESSAGE,
            "type": "stop_message",
            "incident_type_name": "False Alarm Malfunction",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "fake"
    assert set(data["fields"].keys()) == set(EXTRACTABLE_FIELDS)
    assert set(data["confidence"].keys()) == set(EXTRACTABLE_FIELDS)
    assert data["fields"]["applianceCallSign"] == "LF812"
    assert data["fields"]["locationOfFire"] == "7 Gul Ave"


def test_transcribe_returns_fake_transcript():
    response = client.post(
        "/v1/transcribe",
        files={"file": ("sample.webm", b"fake-audio-bytes", "audio/webm")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "fake"
    assert data["transcript"]
    assert data["confidence"] == 0.95


def test_extract_llm_parses_preamble_wrapped_json():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": LLM_JSON_WITH_PREAMBLE}}],
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(settings, "use_fake_extraction", False),
        patch("app.extract.httpx.AsyncClient", return_value=mock_client),
    ):
        response = client.post(
            "/v1/extract",
            json={"text": FAM_STOP_MESSAGE, "type": "stop_message"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "ollama"
    assert data["fields"]["applianceCallSign"] == "LF812"


def test_extract_llm_parses_fenced_json():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": LLM_JSON_RESPONSE}}],
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(settings, "use_fake_extraction", False),
        patch("app.extract.httpx.AsyncClient", return_value=mock_client),
    ):
        response = client.post(
            "/v1/extract",
            json={"text": FAM_STOP_MESSAGE, "type": "stop_message"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "ollama"
    assert data["fields"]["applianceCallSign"] == "LF812"
    assert data["fields"]["locationOfFire"] == "7 Gul Ave"


def test_extract_llm_invalid_json_returns_502():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "not valid json at all"}}],
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(settings, "use_fake_extraction", False),
        patch("app.extract.httpx.AsyncClient", return_value=mock_client),
    ):
        response = client.post(
            "/v1/extract",
            json={"text": FAM_STOP_MESSAGE, "type": "stop_message"},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "LLM returned invalid JSON"
