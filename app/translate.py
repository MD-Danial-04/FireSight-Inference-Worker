import json
import logging
import re

import httpx
from fastapi import HTTPException

from app.config import settings
from app.schemas import (
    InterviewLanguage,
    QuestionTranslationResult,
    TranslateInterviewQuestionInput,
    TranslatedInterviewQuestion,
    TranslationSource,
)

logger = logging.getLogger(__name__)

LANGUAGE_NAMES: dict[InterviewLanguage, str] = {
    "en": "English",
    "ms": "Malay",
    "ta": "Tamil",
    "zh": "Mandarin Chinese",
}

FAKE_LANGUAGE_PREFIX: dict[InterviewLanguage, str] = {
    "en": "",
    "ms": "[MS]",
    "ta": "[TA]",
    "zh": "[ZH]",
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


QUESTIONS_SYSTEM_PROMPT = """
You translate fire investigation interview leading questions for Singapore Civil Defence Force officers.
Return ONLY valid JSON with this shape:
{
  "questions": [
    {
      "id": "question-id-from-input",
      "prompt_conduct": "translated question in target language",
      "hint_conduct": "translated hint or null",
      "section_conduct": "translated section heading or null"
    }
  ]
}

Rules:
- Translate faithfully; keep questions direct and officer-ready.
- Preserve acronyms like PMD, PAB, PMA, OEM, SCDF where appropriate.
- Include one entry per input question (same id).
- hint_conduct and section_conduct may be null when input had no hint/section.
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
    return text[start:] if start != -1 else text


def _fake_conduct_text(text: str, target_lang: InterviewLanguage) -> str:
    prefix = FAKE_LANGUAGE_PREFIX[target_lang]
    return f"{prefix} {text}".strip() if prefix else text


def fake_translate_interview_questions(
    questions: list[TranslateInterviewQuestionInput],
    target_lang: InterviewLanguage,
) -> QuestionTranslationResult:
    if target_lang == "en":
        translated = [
            TranslatedInterviewQuestion(
                id=q.id,
                prompt_conduct=q.prompt,
                hint_conduct=q.hint,
                section_conduct=q.section,
            )
            for q in questions
        ]
        return QuestionTranslationResult(questions=translated, source="fake")

    translated = [
        TranslatedInterviewQuestion(
            id=q.id,
            prompt_conduct=_fake_conduct_text(q.prompt, target_lang),
            hint_conduct=_fake_conduct_text(q.hint, target_lang) if q.hint else None,
            section_conduct=_fake_conduct_text(q.section, target_lang) if q.section else None,
        )
        for q in questions
    ]
    return QuestionTranslationResult(questions=translated, source="fake")


async def llm_translate_interview_questions(
    questions: list[TranslateInterviewQuestionInput],
    target_lang: InterviewLanguage,
) -> QuestionTranslationResult:
    language_name = LANGUAGE_NAMES[target_lang]
    payload_questions = [
        {
            "id": q.id,
            "prompt": q.prompt,
            **({"hint": q.hint} if q.hint else {}),
            **({"section": q.section} if q.section else {}),
        }
        for q in questions
    ]
    user_prompt = f"""
Target language: {language_name}

Questions:
{json.dumps(payload_questions, indent=2)}
""".strip()

    request_payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": QUESTIONS_SYSTEM_PROMPT},
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
                json=request_payload,
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
        parsed = json.loads(_extract_json_text(content))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="LLM returned invalid JSON") from exc

    question_ids = {q.id for q in questions}
    by_id: dict[str, TranslatedInterviewQuestion] = {}
    for item in parsed.get("questions", []):
        if not isinstance(item, dict):
            continue
        qid = str(item.get("id", "")).strip()
        prompt_conduct = str(item.get("prompt_conduct", "")).strip()
        if not qid or qid not in question_ids or not prompt_conduct:
            continue
        hint_raw = item.get("hint_conduct")
        section_raw = item.get("section_conduct")
        by_id[qid] = TranslatedInterviewQuestion(
            id=qid,
            prompt_conduct=prompt_conduct,
            hint_conduct=str(hint_raw).strip() if hint_raw else None,
            section_conduct=str(section_raw).strip() if section_raw else None,
        )

    translated = [
        by_id.get(
            q.id,
            TranslatedInterviewQuestion(
                id=q.id,
                prompt_conduct=q.prompt,
                hint_conduct=q.hint,
                section_conduct=q.section,
            ),
        )
        for q in questions
    ]
    return QuestionTranslationResult(questions=translated, source="ollama")


async def translate_interview_questions(
    questions: list[TranslateInterviewQuestionInput],
    target_lang: InterviewLanguage,
) -> QuestionTranslationResult:
    if target_lang == "en":
        return fake_translate_interview_questions(questions, target_lang)
    if settings.use_fake_extraction:
        return fake_translate_interview_questions(questions, target_lang)
    return await llm_translate_interview_questions(questions, target_lang)
