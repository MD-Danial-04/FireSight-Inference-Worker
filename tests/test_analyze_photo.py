from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.analyze_photo import (
    _build_user_prompt,
    _split_field_notes_excerpt,
)
from app.config import settings
from app.main import app
from app.schemas import AnalyzePhotoContext

client = TestClient(app)

LLM_PHOTO_RESPONSE = """
```json
{
  "caption": "Charring on wooden door frame near the kitchen entrance."
}
```
"""


def test_build_user_prompt_prioritizes_visual_description():
    prompt = _build_user_prompt(
        AnalyzePhotoContext(
            location_of_fire="7 Gull Ave",
            incident_type_name="Fire (Moderate) — Rubbish Chute",
            stop_message_excerpt="LF812 stop at 7 Gull Avenue, black smoke showing",
            field_notes_excerpt=(
                "Photos already logged (describe what is different in THIS image):\n"
                "Photo 9 (IMG_0041) [burn_patterns]: ceiling charring, smoke staining"
            ),
        ),
    )

    assert "Describe ONLY what is visible" in prompt
    assert "CONTEXT (background only" in prompt
    assert "7 Gull Ave" in prompt
    assert "Photos already logged" in prompt
    assert "BACKGROUND (do not copy into caption)" in prompt
    assert "Stop message excerpt:" not in prompt
    assert "Field notes excerpt:" not in prompt


def test_build_user_prompt_omits_stop_message_when_not_provided():
    prompt = _build_user_prompt(
        AnalyzePhotoContext(
            location_of_fire="Blk 1",
            field_notes_excerpt="Short investigator note.",
        ),
    )

    assert "BACKGROUND" not in prompt
    assert "INVESTIGATOR NOTES" in prompt
    assert "Short investigator note." in prompt


def test_split_field_notes_excerpt_separates_prior_photos_block():
    investigator, prior = _split_field_notes_excerpt(
        "Investigator notes here.\n\n"
        "Photos already logged (describe what is different in THIS image):\n"
        "Photo 1 (IMG_001) [burn_patterns]: charring",
    )

    assert investigator == "Investigator notes here."
    assert prior is not None
    assert prior.startswith("Photos already logged")


def test_analyze_photo_returns_fake_result():
    response = client.post(
        "/v1/analyze-photo",
        files={"file": ("scene.jpg", BytesIO(b"fake-image"), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "fake"
    assert data["caption"]


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
    assert "door frame" in data["caption"].lower()
