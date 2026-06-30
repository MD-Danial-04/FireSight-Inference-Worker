import json
import re

import httpx
from fastapi import HTTPException

from app.config import settings
from app.schemas import CleanTranscriptRequest, CleanTranscriptResponse

SYSTEM_PROMPT = """
You clean fire-investigation interview transcripts.

You are given a transcript that may contain BOTH the interviewer/investigator's
questions, prompts and instructions AND the interviewee's answers. Remove every
line or utterance spoken by the interviewer/investigator. Keep ONLY the
interviewee's words.

Rules:
- Keep the interviewee's words EXACTLY as written. Do NOT paraphrase, summarize,
  translate, correct, reorder, or add anything.
- Remove interviewer questions, prompts, acknowledgements and instructions.
- For text formatted with "Q:" and "A:" markers, drop the "Q:" lines entirely
  and unwrap the answers (remove the leading "A:").
- Clean the original-language transcript and the English transcript
  consistently (same content removed from both).
- If the input contains no interviewer content, return it unchanged.

Return ONLY valid JSON with this exact shape:
{
  "transcript_original": "interviewee-only text in the original language",
  "transcript_english": "interviewee-only text in English"
}
""".strip()


def _strip_question_lines(text: str) -> str:
    """Deterministic local cleanup for Q:/A: formatted transcripts."""
    if not text:
        return ""
    kept: list[str] = []
    for line in text.split("\n"):
        if re.match(r"^\s*Q:\s?", line, flags=re.IGNORECASE):
            continue
        kept.append(re.sub(r"^\s*A:\s?", "", line, flags=re.IGNORECASE))
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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


def _parse_clean_json(content: str) -> dict:
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
        if not isinstance(parsed, dict) or "transcript_english" not in parsed:
            last_error = ValueError("missing transcript_english key")
            continue
        return parsed

    if last_error is not None:
        raise last_error
    raise ValueError("missing transcript_english key")


def fake_clean_transcript(req: CleanTranscriptRequest) -> CleanTranscriptResponse:
    return CleanTranscriptResponse(
        transcript_original=_strip_question_lines(req.transcript_original),
        transcript_english=_strip_question_lines(req.transcript_english),
        source="fake",
    )


async def llm_clean_transcript(req: CleanTranscriptRequest) -> CleanTranscriptResponse:
    user_prompt = f"""
Original-language transcript:
{req.transcript_original}

English transcript:
{req.transcript_english}
""".strip()

    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 4096,
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
        parsed = _parse_clean_json(content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="LLM returned invalid JSON",
        ) from exc

    english = str(parsed.get("transcript_english", "")).strip()
    original = str(parsed.get("transcript_original", "")).strip() or english
    # Fall back to the source text if the model returned nothing usable, so a
    # cleanup never silently destroys the statement.
    if not english:
        english = req.transcript_english.strip()
    if not original:
        original = req.transcript_original.strip()

    return CleanTranscriptResponse(
        transcript_original=original,
        transcript_english=english,
        source="ollama",
    )


async def clean_transcript(req: CleanTranscriptRequest) -> CleanTranscriptResponse:
    if settings.use_fake_extraction:
        return fake_clean_transcript(req)
    return await llm_clean_transcript(req)
