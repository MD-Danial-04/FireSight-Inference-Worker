from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.analyze_photo import (
    SUGGESTED_SECTION_CONFIDENCE_THRESHOLD,
    _normalize_section,
    _resolve_section,
)
from app.config import settings
from app.main import app

client = TestClient(app)

LLM_PHOTO_RESPONSE = """
```json
{
  "caption": "Charring on wooden door frame near the kitchen entrance.",
  "detected_elements": ["door frame charring", "smoke staining"],
  "suggested_section": "burn_patterns",
  "confidence": {
    "caption": 0.88,
    "suggested_section": 0.81
  }
}
```
"""

LLM_LOW_CONFIDENCE_SECTION = """
{
  "caption": "Exterior view of the affected shophouse.",
  "detected_elements": ["building facade"],
  "suggested_section": "incident",
  "confidence": {
    "caption": 0.8,
    "suggested_section": 0.4
  }
}
"""

LLM_INVALID_SECTION = """
{
  "caption": "Vehicle compartment with fire damage.",
  "detected_elements": ["vehicle", "fire damage"],
  "suggested_section": "vehicle",
  "confidence": {
    "caption": 0.9,
    "suggested_section": 0.9
  }
}
"""


def test_normalize_section_rejects_vehicle():
    assert _normalize_section("vehicle") is None
    assert _normalize_section("burn_patterns") == "burn_patterns"
    assert _normalize_section(None) is None


def test_resolve_section_applies_threshold():
    assert _resolve_section("damages", SUGGESTED_SECTION_CONFIDENCE_THRESHOLD) == "damages"
    assert _resolve_section("damages", SUGGESTED_SECTION_CONFIDENCE_THRESHOLD - 0.01) is None
    assert _resolve_section(None, 0.9) is None


def test_analyze_photo_returns_fake_result():
    response = client.post(
        "/v1/analyze-photo",
        files={"file": ("scene.jpg", BytesIO(b"fake-image"), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "fake"
    assert data["suggested_section"] == "burn_patterns"
    assert data["caption"]
    assert isinstance(data["detected_elements"], list)


def test_analyze_photo_rejects_empty_file():
    response = client.post(
        "/v1/analyze-photo",
        files={"file": ("empty.jpg", BytesIO(b""), "image/jpeg")},
    )
    assert response.status_code == 400


def test_analyze_photo_llm_parses_json():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": LLM_PHOTO_RESPONSE},
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(settings, "use_fake_photo_analysis", False),
        patch("app.analyze_photo.httpx.AsyncClient", return_value=mock_client),
    ):
        response = client.post(
            "/v1/analyze-photo",
            files={"file": ("scene.jpg", BytesIO(b"fake-image"), "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "ollama"
    assert data["suggested_section"] == "burn_patterns"
    assert "door frame" in data["caption"].lower()


def test_analyze_photo_llm_nulls_low_confidence_section():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": LLM_LOW_CONFIDENCE_SECTION},
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(settings, "use_fake_photo_analysis", False),
        patch("app.analyze_photo.httpx.AsyncClient", return_value=mock_client),
    ):
        response = client.post(
            "/v1/analyze-photo",
            files={"file": ("scene.jpg", BytesIO(b"fake-image"), "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["suggested_section"] is None


def test_analyze_photo_llm_rejects_invalid_section_value():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": LLM_INVALID_SECTION},
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(settings, "use_fake_photo_analysis", False),
        patch("app.analyze_photo.httpx.AsyncClient", return_value=mock_client),
    ):
        response = client.post(
            "/v1/analyze-photo",
            files={"file": ("scene.jpg", BytesIO(b"fake-image"), "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["suggested_section"] is None
