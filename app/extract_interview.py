import json

import httpx
from fastapi import HTTPException

from app.config import settings
from app.extract import _parse_llm_json
from app.schemas import (
    ExtractInterviewRequest,
    ExtractInterviewResponse,
    InterviewExtractableField,
)

INTERVIEW_EXTRACTABLE_FIELDS: list[InterviewExtractableField] = [
    "name",
    "nameChinese",
    "designation",
    "nric",
    "passportNo",
    "nationality",
    "sex",
    "age",
    "dateAndPlaceOfBirth",
    "maritalStatus",
    "numberOfChildren",
    "citizenshipCertNo",
    "vehicleNo",
    "address",
    "placeOfEmployment",
    "contactHome",
    "contactMobile",
    "contactOffice",
    "interviewTakenPlace",
    "interpretedBy",
]

SYSTEM_PROMPT = """
You extract structured interviewee profile fields from Singapore fire investigation interview transcripts.
Return ONLY valid JSON with this shape:
{
  "fields": {
    "name": "",
    "nameChinese": "",
    "designation": "",
    "nric": "",
    "passportNo": "",
    "nationality": "",
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
    "contactMobile": "",
    "contactOffice": "",
    "interviewTakenPlace": "",
    "interpretedBy": ""
  },
  "confidence": {
    "...same keys...": 0.0
  }
}

Rules:
- Use empty string when unknown.
- Confidence must be between 0.0 and 1.0.
- Keep extracted values as spoken in transcript; do not invent values.
- NRIC format is usually S/T/F/G + 7 digits + trailing letter.
- Contact numbers in Singapore are usually 8 digits.
""".strip()


def fake_extract_interview(req: ExtractInterviewRequest) -> ExtractInterviewResponse:
    fields = {
        "name": "John Tan",
        "nameChinese": "",
        "designation": "Tenant",
        "nric": "S1234567A",
        "passportNo": "",
        "nationality": "Singaporean",
        "sex": "Male",
        "age": "34",
        "dateAndPlaceOfBirth": "",
        "maritalStatus": "",
        "numberOfChildren": "",
        "citizenshipCertNo": "",
        "vehicleNo": "",
        "address": "Blk 1 Example Street",
        "placeOfEmployment": "",
        "contactHome": "",
        "contactMobile": "91234567",
        "contactOffice": "",
        "interviewTakenPlace": "",
        "interpretedBy": "",
    }
    confidence = {
        key: 0.85 if fields[key] else 0.0
        for key in INTERVIEW_EXTRACTABLE_FIELDS
    }
    if req.text.strip():
        confidence["name"] = 0.9
    return ExtractInterviewResponse(fields=fields, confidence=confidence, source="fake")


async def llm_extract_interview(req: ExtractInterviewRequest) -> ExtractInterviewResponse:
    user_prompt = f"""
Interview language: {req.interview_language}

Transcript:
{req.text}
""".strip()

    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "format": "json",
    }

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.llm_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM unavailable: HTTP {exc.response.status_code}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM unavailable: {exc}",
        ) from exc

    try:
        parsed = _parse_llm_json(content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="LLM returned invalid JSON",
        ) from exc

    fields = {
        key: str(parsed["fields"].get(key, "")).strip()
        for key in INTERVIEW_EXTRACTABLE_FIELDS
    }
    confidence_raw = parsed.get("confidence", {})
    confidence = {
        key: float(confidence_raw.get(key, 0.0))
        for key in INTERVIEW_EXTRACTABLE_FIELDS
    }
    return ExtractInterviewResponse(fields=fields, confidence=confidence, source="ollama")


async def extract_interview_details(
    req: ExtractInterviewRequest,
) -> ExtractInterviewResponse:
    if settings.use_fake_extraction:
        return fake_extract_interview(req)
    return await llm_extract_interview(req)
