import logging

import httpx
from fastapi import HTTPException

from app.config import settings
from app.schemas import InterviewLanguage, TranslationSource

logger = logging.getLogger(__name__)

LANGUAGE_NAMES: dict[InterviewLanguage, str] = {
    "en": "English",
    "ms": "Malay",
    "ta": "Tamil",
    "zh": "Mandarin Chinese",
}

SYSTEM_PROMPT = """
You translate fire investigation interview transcripts into English for Singapore Civil Defence Force reports.
Rules:
- Translate faithfully; do not summarize or omit details.
- Preserve proper names, places, NRIC/passport numbers, vehicle numbers, and times exactly.
- Keep SCDF terminology accurate (e.g. PMD, false alarm, zone).
- Return ONLY the English translation text with no preamble or markdown.
""".strip()


def fake_translate_interview_transcript(
    text: str,
    source_lang: InterviewLanguage,
) -> tuple[str, TranslationSource]:
    if source_lang == "en":
        return text, "none"
    return f"[EN] {text}", "fake"


async def llm_translate_interview_transcript(
    text: str,
    source_lang: InterviewLanguage,
) -> tuple[str, TranslationSource]:
    language_name = LANGUAGE_NAMES[source_lang]
    user_prompt = f"""
Source language: {language_name}

Transcript:
{text}
""".strip()

    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 4096,
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

    translated = content.strip()
    if not translated:
        raise HTTPException(status_code=502, detail="LLM returned empty translation")
    return translated, "ollama"


async def translate_interview_transcript(
    text: str,
    source_lang: InterviewLanguage,
) -> tuple[str, TranslationSource]:
    if source_lang == "en":
        return text, "none"
    if settings.use_fake_extraction:
        return fake_translate_interview_transcript(text, source_lang)
    return await llm_translate_interview_transcript(text, source_lang)
