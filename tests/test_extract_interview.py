from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)

INTERVIEW_TEXT = "My name is John Tan. NRIC S1234567A. Mobile number 91234567."

LLM_RESPONSE = """
```json
{
  "fields": {
    "name": "John Tan",
    "nameChinese": "",
    "designation": "Tenant",
    "nric": "S1234567A",
    "passportNo": "",
    "nationality": "Singaporean",
    "sex": "",
    "age": "",
    "dateAndPlaceOfBirth": "",
    "maritalStatus": "",
    "numberOfChildren": "",
    "citizenshipCertNo": "",
    "vehicleNo": "",
    "address": "",
    "placeOfEmployment": "",
    "contactHome": "",
    "contactMobile": "91234567",
    "contactOffice": "",
    "interviewTakenPlace": "",
    "interpretedBy": ""
  },
  "confidence": {
    "name": 0.95,
    "nameChinese": 0.0,
    "designation": 0.8,
    "nric": 0.99,
    "passportNo": 0.0,
    "nationality": 0.7,
    "sex": 0.0,
    "age": 0.0,
    "dateAndPlaceOfBirth": 0.0,
    "maritalStatus": 0.0,
    "numberOfChildren": 0.0,
    "citizenshipCertNo": 0.0,
    "vehicleNo": 0.0,
    "address": 0.0,
    "placeOfEmployment": 0.0,
    "contactHome": 0.0,
    "contactMobile": 0.9,
    "contactOffice": 0.0,
    "interviewTakenPlace": 0.0,
    "interpretedBy": 0.0
  }
}
```
"""


def test_extract_interview_fake_mode():
    response = client.post(
        "/v1/extract-interview",
        json={"text": INTERVIEW_TEXT, "interview_language": "en"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "fake"
    assert data["fields"]["name"]
    assert "contactMobile" in data["fields"]


def test_extract_interview_llm_parses_json():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": LLM_RESPONSE}}],
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(settings, "use_fake_extraction", False),
        patch("app.extract_interview.httpx.AsyncClient", return_value=mock_client),
    ):
        response = client.post(
            "/v1/extract-interview",
            json={"text": INTERVIEW_TEXT, "interview_language": "en"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "ollama"
    assert data["fields"]["name"] == "John Tan"
    assert data["fields"]["contactMobile"] == "91234567"
