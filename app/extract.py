import json
import re

import httpx
from fastapi import HTTPException

from app.config import settings
from app.normalize import apply_field_normalization
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

SCDF extraction rules:
- applianceCallSign: LF plus 3 digits (e.g. LF812). Normalize ASR errors such as LF-A12 or LF A 12 to LF812.
- locationOfFire: Address as stated; fix common mishears (Gall to Gul, Avenue to Ave).
- areaOfFireOrigin: Free text from the message — e.g. Zone 7, living room, kitchen. Do NOT assume zone; extract what is actually said.
- handoverOfficer: Rank abbreviation, name, and service ID. Use SCDF rank forms such as SGT3 (not S3) and SSS for triple S. Decode NATO-spoken IDs to compact alphanumeric (e.g. Tango 1, 9-0-3-5-0 becomes T190350).
- handoverNpc: NPC name (e.g. Nanyang NPC).
- eventsCircumstances: Investigation findings plus any liase line. Stop messages always include liase with a person or role (e.g. Liase with Mr. Zaini, safety officer). Fix ASR mishears such as Liars or Lease to Liase.
- classification: Prefer incident_type_name from the user message when provided.

Example input:
LF812 stop for location at 7 Gul Ave. Case classified as False alarm malfunction of manual call point, at zone 7. Upon investigation no smoke no fire. Liase with Mr. Zaini, safety officer. Case handed over to Sergeant 3 Alsyraf, Tango 1, 9-0-3-5-0 from Nanyang NPC.

Example output fields:
{
  "applianceCallSign": "LF812",
  "locationOfFire": "7 Gul Ave",
  "fireInvolved": "False alarm malfunction",
  "methodOfExtinguishment": "",
  "damagesSustained": "",
  "probableCause": "False alarm malfunction of manual call point",
  "ignitionSource": "",
  "ignitionFuel": "",
  "eventsCircumstances": "No smoke no fire. Liase with Mr. Zaini, safety officer",
  "areaOfFireOrigin": "Zone 7",
  "classification": "False alarm malfunction",
  "handoverOfficer": "SGT3 Alsyraf T190350",
  "handoverNpc": "Nanyang NPC"
}
""".strip()


def _extract_json_text(content: str) -> str:
    text = content.strip()

    fence_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        return fence_match.group(1).strip()

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:]


def _parse_llm_json(content: str) -> dict:
    candidates: list[str] = []
    stripped = content.strip()
    if stripped:
        candidates.append(stripped)
    extracted = _extract_json_text(content)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    last_error: json.JSONDecodeError | ValueError | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict) or "fields" not in parsed:
            last_error = ValueError("missing fields key")
            continue
        return parsed

    if last_error is not None:
        raise last_error
    raise ValueError("missing fields key")


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
            "handoverOfficer": "SGT3 Alsyraf T190350",
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

    fields = {key: str(parsed["fields"].get(key, "")).strip() for key in EXTRACTABLE_FIELDS}
    fields = apply_field_normalization(fields, req.text)
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
