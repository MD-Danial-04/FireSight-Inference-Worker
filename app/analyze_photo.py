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
    SectionCandidate,
    SectionCandidates,
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
  "caption": "Photo showing ... (2-4 flowing sentences as described below)",
  "detected_elements": ["short", "observable", "labels"],
  "section_candidates": {
    "incident": { "score": 0.2, "reason": null },
    "damages": { "score": 0.6, "reason": "charred furniture visible" },
    "area_of_origin": { "score": 0.82, "reason": "rubbish chute opening" },
    "burn_patterns": { "score": 0.74, "reason": "soot staining on wall" },
    "evidentiary": { "score": 0.3, "reason": null }
  },
  "confidence": { "caption": 0.85 }
}

CAPTION RULES (critical):
- The caption MUST begin with the words "Photo showing ".
- Write 2-4 flowing sentences (a single paragraph, NO labels or headings) that naturally weave in these four aspects:
  1. Overall scene: the general state of the area captured (e.g. heavy charring to upper walls, intact floor level).
  2. Specific evidence: distinct items or details visible (e.g. melted plastic casing on the toaster, distinct "V" pattern on the drywall).
  3. Fire effects: the type and degree of damage using fire-investigation vocabulary (soot deposition, clean burn, calcination, char depth, smoke staining, melt damage, heat marks).
  4. Investigative significance: why this photo matters (e.g. illustrates low-level burns indicating a potential point of origin near the electrical outlet).
- Describe ONLY observable visual facts in THIS photograph. Do NOT copy or paraphrase stop message text, incident summaries, location narratives, field notes, or prior photo descriptions into the caption.
- Do NOT include identifier metadata (image number, file name, date, time, photographer name) in the caption.
- Each photo caption must be distinct and specific to what is visible in this frame.

GOOD caption example:
"Photo showing the upper section of a kitchen wall with heavy charring and soot deposition concentrated near the ceiling line while the lower wall and floor remain comparatively intact. A distinct 'V' burn pattern is visible on the drywall above a wall-mounted electrical outlet, with melted plastic casing on an adjacent appliance. The clean burn and deep char depth at the apex of the pattern, contrasted with lighter smoke staining outward, indicate intense localized heat. This low-level, concentrated damage suggests a potential point of origin at the electrical outlet."

BAD caption example (do not output text like this):
"Investigation of a moderate fire incident at 7 Gull Avenue. The fire is believed to have originated in the rubbish chute..."

section_candidates — score EACH section independently from visible features only (0.0 to 1.0):
- incident: Section 2 — general scene or overview photos
- damages: Section 2 — property or content damage visible
- area_of_origin: Section 5b — seat-of-fire or origin indicators (e.g. rubbish chute opening, localized burn seat)
- burn_patterns: Section 5c — char patterns, smoke staining, fire spread or heat indicators
- evidentiary: Section 5d — physical evidence items visible (e.g. ignition source remnants, debris)

For each section candidate, reason is one short visual phrase based on what you see, or null when score is low.
Score sections independently — a photo may score highly on multiple sections.

detected_elements: short observable labels only (e.g. "ceiling charring", "smoke staining", "rubbish chute door").
confidence.caption is 0.0 to 1.0.
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


def _parse_candidate_value(raw: object) -> SectionCandidate | None:
    if not isinstance(raw, dict):
        return None
    try:
        score = max(0.0, min(1.0, float(raw.get("score", 0))))
    except (TypeError, ValueError):
        return None
    reason_raw = raw.get("reason")
    reason: str | None = None
    if reason_raw is not None:
        reason_str = str(reason_raw).strip()
        if reason_str and reason_str.lower() not in {"null", "none"}:
            reason = reason_str
    return SectionCandidate(score=score, reason=reason)


def _parse_section_candidates(parsed: dict) -> SectionCandidates | None:
    raw = parsed.get("section_candidates")
    if not isinstance(raw, dict):
        return None

    candidates = SectionCandidates(
        incident=_parse_candidate_value(raw.get("incident")),
        damages=_parse_candidate_value(raw.get("damages")),
        area_of_origin=_parse_candidate_value(raw.get("area_of_origin")),
        burn_patterns=_parse_candidate_value(raw.get("burn_patterns")),
        evidentiary=_parse_candidate_value(raw.get("evidentiary")),
    )
    if all(getattr(candidates, section) is None for section in VALID_SECTIONS):
        return None
    return candidates


def _derive_suggested_section(
    candidates: SectionCandidates | None,
) -> tuple[SuggestedPhotoSection | None, float | None]:
    if candidates is None:
        return None, None

    best_section: SuggestedPhotoSection | None = None
    best_score: float | None = None
    for section in VALID_SECTIONS:
        candidate = getattr(candidates, section)
        if candidate is None:
            continue
        if best_score is None or candidate.score > best_score:
            best_section = section  # type: ignore[assignment]
            best_score = candidate.score

    resolved = _resolve_section(best_section, best_score)
    if resolved is None:
        return None, None
    return resolved, best_score


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
        'Write the caption as 2-4 flowing sentences beginning with "Photo showing ", weaving in the overall scene, specific evidence, fire effects, and investigative significance.',
        "Focus on observable fire effects: charring, soot deposition, smoke staining, melt damage, burn patterns, calcination, char depth, heat marks, and physical evidence visible in the frame.",
        "Do not restate stop message, incident summary, or prior photo descriptions in the caption, and do not include identifier metadata (image number, file name, date, time, photographer).",
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
    section_candidates = SectionCandidates(
        incident=SectionCandidate(score=0.15, reason=None),
        damages=SectionCandidate(score=0.4, reason=None),
        area_of_origin=SectionCandidate(score=0.35, reason=None),
        burn_patterns=SectionCandidate(score=0.85, reason="ceiling charring visible"),
        evidentiary=SectionCandidate(score=0.2, reason=None),
    )
    return AnalyzePhotoResponse(
        caption=(
            "Photo showing the ceiling lining above the seating area with charring and "
            "heavy smoke staining concentrated overhead while the lower walls remain "
            "comparatively intact. Soot deposition and clean burn marks are visible across "
            "the ceiling panels directly above the seats. The upward-rising heat damage and "
            "char depth indicate intense fire effects near ceiling level, suggesting fire "
            "spread along the overhead lining away from a lower point of origin."
        ),
        detected_elements=["ceiling charring", "smoke staining"],
        suggested_section="burn_patterns",
        section_candidates=section_candidates,
        confidence=PhotoAnalysisConfidence(caption=0.85, suggested_section=0.85),
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
    if isinstance(confidence_raw, dict):
        try:
            caption_confidence = max(0.0, min(1.0, float(confidence_raw.get("caption", 0.0))))
        except (TypeError, ValueError):
            caption_confidence = 0.0

    section_candidates = _parse_section_candidates(parsed)
    resolved_section, section_confidence = _derive_suggested_section(section_candidates)

    if resolved_section is None:
        legacy_section = _normalize_section(parsed.get("suggested_section"))
        legacy_confidence: float | None = None
        if isinstance(confidence_raw, dict):
            raw_section_conf = confidence_raw.get("suggested_section")
            if raw_section_conf is not None:
                try:
                    legacy_confidence = max(0.0, min(1.0, float(raw_section_conf)))
                except (TypeError, ValueError):
                    legacy_confidence = None
        resolved_section = _resolve_section(legacy_section, legacy_confidence)
        section_confidence = legacy_confidence if resolved_section else None

    return AnalyzePhotoResponse(
        caption=caption,
        detected_elements=detected_elements,
        suggested_section=resolved_section,
        section_candidates=section_candidates,
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
