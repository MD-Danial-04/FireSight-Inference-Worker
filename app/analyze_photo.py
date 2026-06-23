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
  "caption": "1-2 sentence description of what is visible in THIS image",
  "detected_elements": ["short", "observable", "labels"],
  "suggested_section": "burn_patterns",
  "confidence": {
    "caption": 0.85,
    "suggested_section": 0.78
  }
}

CAPTION RULES (critical):
- Describe ONLY observable visual facts in THIS photograph: charring, soot, smoke staining, melt damage, burn patterns, heat marks, damaged surfaces, rubbish contents, physical evidence items, etc.
- Do NOT copy or paraphrase stop message text, incident summaries, location narratives, field notes, or prior photo descriptions into the caption.
- Each photo caption must be distinct and specific to what is visible in this frame.

GOOD caption example:
"Heavy soot staining and charring on the rubbish chute door and adjacent wall lining."

BAD caption example (do not output text like this):
"Investigation of a moderate fire incident at 7 Gull Avenue. The fire is believed to have originated in the rubbish chute..."

suggested_section — pick the best report link for THIS image, or null if none fit confidently:
- area_of_origin: Section 5b — visible seat-of-fire or origin indicators (e.g. rubbish chute opening, localized burn seat)
- burn_patterns: Section 5c — char patterns, smoke staining, fire spread or heat indicators
- evidentiary: Section 5d — physical evidence items visible (e.g. ignition source remnants, debris)
- damages: property or content damage visible
- incident: general scene or overview photos

Do NOT use appliance, vehicle, or other as suggested_section values.
Subject matter such as vehicles or appliances belongs in caption and detected_elements only.

detected_elements: short observable labels only (e.g. "ceiling charring", "smoke staining", "rubbish chute door").
confidence values are 0.0 to 1.0. Use suggested_section null when no section fits confidently.
""".strip()

PRIOR_PHOTOS_HEADER = "Photos already logged (describe what is different in THIS image):"
LEGACY_PRIOR_PHOTOS_HEADER = "Prior photo log captions"


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


def _split_field_notes_excerpt(
    excerpt: str | None,
) -> tuple[str | None, str | None]:
    if not excerpt or not excerpt.strip():
        return None, None

    text = excerpt.strip()
    for header in (PRIOR_PHOTOS_HEADER, LEGACY_PRIOR_PHOTOS_HEADER):
        idx = text.find(header)
        if idx != -1:
            investigator_notes = text[:idx].strip() or None
            prior_block = text[idx:].strip()
            return investigator_notes, prior_block

    return text, None


def _truncate_background(text: str, max_length: int = 120) -> str:
    trimmed = text.strip()
    if len(trimmed) <= max_length:
        return trimmed
    return f"{trimmed[:max_length]}…"


def _build_user_prompt(context: AnalyzePhotoContext | None) -> str:
    lines = [
        "Describe ONLY what is visible in this photograph.",
        "Focus on observable fire effects: charring, soot, smoke staining, melt damage, burn patterns, heat marks, and physical evidence visible in the frame.",
        "Do not restate stop message, incident summary, or prior photo descriptions in the caption.",
    ]

    if not context:
        return "\n".join(lines)

    classification_hints: list[str] = []
    if context.location_of_fire:
        classification_hints.append(f"- Location: {context.location_of_fire}")
    if context.incident_type_name:
        classification_hints.append(f"- Incident: {context.incident_type_name}")

    if classification_hints:
        lines.append("")
        lines.append(
            "CLASSIFICATION HINTS (for suggested_section only — do not repeat in caption):",
        )
        lines.extend(classification_hints)

    investigator_notes, prior_photos = _split_field_notes_excerpt(context.field_notes_excerpt)

    if investigator_notes:
        lines.append("")
        lines.append("INVESTIGATOR NOTES (background only — do not repeat in caption):")
        lines.append(investigator_notes)

    if prior_photos:
        lines.append("")
        lines.append(prior_photos)

    if context.stop_message_excerpt:
        lines.append("")
        lines.append("BACKGROUND (do not copy into caption):")
        lines.append(_truncate_background(context.stop_message_excerpt))

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


async def analyze_photo_for_worker(
    image_bytes: bytes,
    context: AnalyzePhotoContext | None = None,
) -> AnalyzePhotoResponse:
    try:
        return await analyze_photo(image_bytes, context)
    except HTTPException as exc:
        raise RuntimeError(exc.detail) from exc
