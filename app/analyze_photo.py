import base64
import json
import re

import httpx
from fastapi import HTTPException

from app.config import settings
from app.schemas import (
    AnalyzePhotoContext,
    AnalyzePhotoResponse,
    PhotoAnalysisConfidence,
    SuggestedPhotoSection,
)

SUGGESTED_SECTION_CONFIDENCE_THRESHOLD = 0.7

VALID_SECTIONS: set[str] = {
    "incident",
    "damages",
    "area_of_origin",
    "burn_patterns",
    "evidentiary",
}

SYSTEM_PROMPT = """
You analyze fire-scene investigation photographs for a Singapore Civil Defence Force (SCDF) fire report.
Return ONLY valid JSON with this shape:
{
  "caption": "1-2 sentence investigation-style description for the annex photo log",
  "detected_elements": ["short", "observable", "labels"],
  "suggested_section": "burn_patterns",
  "confidence": {
    "caption": 0.85,
    "suggested_section": 0.78
  }
}

suggested_section must be exactly one of these five values, or null if none apply confidently:
- incident: general scene or overview photos referenced with Annex A
- damages: property or content damage
- area_of_origin: where the fire likely originated
- burn_patterns: char patterns, smoke staining, fire spread indicators
- evidentiary: physical evidence relevant to cause

Do NOT use appliance, vehicle, or other as suggested_section values.
Subject matter such as vehicles or appliances belongs in caption and detected_elements only.

detected_elements: short observable labels only (e.g. "ceiling charring", "smoke staining").
confidence values are 0.0 to 1.0. Use suggested_section null when no section fits confidently.
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


def _parse_photo_analysis_json(content: str) -> dict:
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
        if not isinstance(parsed, dict) or "caption" not in parsed:
            last_error = ValueError("missing caption key")
            continue
        return parsed

    if last_error is not None:
        raise last_error
    raise ValueError("missing caption key")


def _normalize_section(raw: object) -> SuggestedPhotoSection | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"", "null", "none"}:
        return None
    if value in VALID_SECTIONS:
        return value  # type: ignore[return-value]
    return None


def _resolve_section(
    section: SuggestedPhotoSection | None,
    section_confidence: float | None,
) -> SuggestedPhotoSection | None:
    if section is None or section_confidence is None:
        return None
    if section_confidence < SUGGESTED_SECTION_CONFIDENCE_THRESHOLD:
        return None
    return section


def _build_user_prompt(context: AnalyzePhotoContext | None) -> str:
    lines = ["Describe this fire-scene photograph for an investigation annex."]
    if context:
        if context.location_of_fire:
            lines.append(f"Location of fire: {context.location_of_fire}")
        if context.incident_type_name:
            lines.append(f"Incident type: {context.incident_type_name}")
        if context.stop_message_excerpt:
            lines.append(f"Stop message excerpt: {context.stop_message_excerpt}")
        if context.field_notes_excerpt:
            lines.append(f"Field notes excerpt: {context.field_notes_excerpt}")
    return "\n".join(lines)


def fake_analyze_photo(_context: AnalyzePhotoContext | None = None) -> AnalyzePhotoResponse:
    return AnalyzePhotoResponse(
        caption="Charring and smoke staining observed on ceiling lining above the seating area.",
        detected_elements=["ceiling charring", "smoke staining"],
        suggested_section="burn_patterns",
        confidence=PhotoAnalysisConfidence(caption=0.85, suggested_section=0.78),
        source="fake",
    )


def _ollama_base_url() -> str:
    return settings.llm_base_url.rstrip("/").removesuffix("/v1")


async def llm_analyze_photo(
    image_bytes: bytes,
    context: AnalyzePhotoContext | None = None,
) -> AnalyzePhotoResponse:
    image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    user_prompt = _build_user_prompt(context)

    payload = {
        "model": settings.vision_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt, "images": [image_b64]},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{_ollama_base_url()}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Vision model unavailable: HTTP {exc.response.status_code}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Vision model unavailable: {exc}",
        ) from exc
    except (KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Vision model returned an unexpected response",
        ) from exc

    try:
        parsed = _parse_photo_analysis_json(content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Vision model returned invalid JSON",
        ) from exc

    caption = str(parsed.get("caption", "")).strip()
    if not caption:
        raise HTTPException(status_code=502, detail="Vision model returned empty caption")

    detected_raw = parsed.get("detected_elements", [])
    detected_elements: list[str] = []
    if isinstance(detected_raw, list):
        detected_elements = [str(item).strip() for item in detected_raw if str(item).strip()]

    confidence_raw = parsed.get("confidence", {})
    caption_confidence = 0.0
    section_confidence: float | None = None
    if isinstance(confidence_raw, dict):
        try:
            caption_confidence = max(0.0, min(1.0, float(confidence_raw.get("caption", 0.0))))
        except (TypeError, ValueError):
            caption_confidence = 0.0
        raw_section_conf = confidence_raw.get("suggested_section")
        if raw_section_conf is not None:
            try:
                section_confidence = max(0.0, min(1.0, float(raw_section_conf)))
            except (TypeError, ValueError):
                section_confidence = None

    normalized_section = _normalize_section(parsed.get("suggested_section"))
    resolved_section = _resolve_section(normalized_section, section_confidence)

    return AnalyzePhotoResponse(
        caption=caption,
        detected_elements=detected_elements,
        suggested_section=resolved_section,
        confidence=PhotoAnalysisConfidence(
            caption=caption_confidence,
            suggested_section=section_confidence,
        ),
        source="ollama",
    )


async def analyze_photo(
    image_bytes: bytes,
    context: AnalyzePhotoContext | None = None,
) -> AnalyzePhotoResponse:
    if settings.use_fake_photo_analysis:
        return fake_analyze_photo(context)
    return await llm_analyze_photo(image_bytes, context)
