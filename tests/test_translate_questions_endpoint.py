from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE_QUESTIONS = [
    {
        "id": "device-type",
        "prompt": "What type of mobility device is it?",
        "hint": "e.g. PMD",
        "section": "Affected device",
    }
]


def test_translate_interview_questions_returns_fake_translation():
    response = client.post(
        "/v1/translate-interview-questions",
        json={"questions": SAMPLE_QUESTIONS, "interview_language": "zh"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "fake"
    assert data["questions"][0]["prompt_conduct"].startswith("[ZH]")
