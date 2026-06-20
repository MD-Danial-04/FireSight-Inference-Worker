import json
import re

import httpx
from fastapi import HTTPException

from app.config import settings
from app.schemas import ExtractRequest, ExtractResponse, ExtractableField

EXTRACTABLE_FIELDS: list[ExtractableField] = [
    "applianceCallSign",
    "locationOfFire",
    "fireInvolved",
    "methodOfExtinguishment",
    "damagesSustained",
    "probableCause",
    "ignitionSource",
    "ignitionFuel",
    "eventsCircumstances",
    "areaOfFireOrigin",
    "classification",
    "handoverOfficer",
    "handoverNpc",
]

SYSTEM_PROMPT = """
You extract structured fire incident report fields from SCDF stop messages.
Return ONLY valid JSON with this shape:
{
  "fields": {
    "applianceCallSign": "",
    "locationOfFire": "",
    "fireInvolved": "",
    "methodOfExtinguishment": "",
    "damagesSustained": "",
    "probableCause": "",
    "ignitionSource": "",
    "ignitionFuel": "",
    "eventsCircumstances": "",
    "areaOfFireOrigin": "",
    "classification": "",
    "handoverOfficer": "",
    "handoverNpc": ""
  },
  "confidence": {
    "...same keys...": 0.0
  }
}
Use empty string when unknown. Confidence is 0.0 to 1.0.
""".strip()


def _parse_llm_json(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or "fields" not in parsed:
        raise ValueError("missing fields key")
    return parsed


def fake_extract(req: ExtractRequest) -> ExtractResponse:
    return ExtractResponse(
        fields={
            "applianceCallSign": "LF812",
            "locationOfFire": "7 Gul Ave",
            "fireInvolved": req.incident_type_name or "False alarm malfunction",
            "methodOfExtinguishment": "",
            "damagesSustained": "",
            "probableCause": "False alarm malfunction of manual call point",
            "ignitionSource": "",
            "ignitionFuel": "",
            "eventsCircumstances": req.text.strip(),
            "areaOfFireOrigin": "Zone 7",
            "classification": "False alarm malfunction",
            "handoverOfficer": "S3 Alsyraf T190350",
            "handoverNpc": "Nanyang NPC",
        },
        confidence={
            key: 0.99 if key in {"applianceCallSign", "locationOfFire"} else 0.75
            for key in EXTRACTABLE_FIELDS
        },
        source="fake",
    )


async def llm_extract(req: ExtractRequest) -> ExtractResponse:
    user_prompt = f"""
Incident type: {req.incident_type_name or "unknown"}
Input type: {req.type}

Text:
{req.text}
""".strip()

    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
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

    fields = {key: str(parsed["fields"].get(key, "")).strip() for key in EXTRACTABLE_FIELDS}
    confidence_raw = parsed.get("confidence", {})
    confidence = {
        key: float(confidence_raw.get(key, 0.0))
        for key in EXTRACTABLE_FIELDS
    }

    return ExtractResponse(fields=fields, confidence=confidence, source="ollama")


async def extract_fields(req: ExtractRequest) -> ExtractResponse:
    if settings.use_fake_extraction:
        return fake_extract(req)
    return await llm_extract(req)
