import json
import re

import httpx
from fastapi import HTTPException

from app.config import settings
from app.schemas import (
    AnalyzeInterviewRequest,
    AnalyzeInterviewResponse,
    FollowUpSuggestion,
    InterviewLanguage,
    QuestionCoverage,
    QuestionCoverageStatus,
)

SYSTEM_PROMPT = """
You analyze fire investigation interview transcripts against a checklist of leading questions.
Return ONLY valid JSON with this shape:
{
  "coverage": [
    {
      "id": "question-id-from-input",
      "status": "answered",
      "evidence": "brief quote or paraphrase from transcript",
      "confidence": 0.9
    }
  ],
  "follow_ups": [
    {
      "related_question_id": "question-id-or-null",
      "prompt": "short follow-up question in English for the report",
      "prompt_conduct": "same follow-up in the conduct language for asking aloud",
      "reason": "why this follow-up is needed"
    }
  ]
}

For each question in the checklist, classify status as exactly one of:
- answered: transcript contains a substantive response
- partial: topic touched but incomplete
- unanswered: no relevant content in transcript
- unclear: response exists but vague, contradictory, or evasive

Include one coverage entry per input question (same id). Confidence is 0.0 to 1.0.

Generate follow_up prompts for partial, unanswered, and unclear items only.
Follow-ups should be short, direct questions an investigator can ask next.
prompt must be English. prompt_conduct must be in the conduct language provided by the user (same as prompt when English).
Do not invent coverage for questions not in the input list.
""".strip()

_FAKE_STATUS_CYCLE: list[QuestionCoverageStatus] = [
    "answered",
    "partial",
    "unanswered",
    "unclear",
]


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


def _parse_analysis_json(content: str) -> dict:
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
        if not isinstance(parsed, dict) or "coverage" not in parsed:
            last_error = ValueError("missing coverage key")
            continue
        return parsed

    if last_error is not None:
        raise last_error
    raise ValueError("missing coverage key")


def _normalize_status(raw: str) -> QuestionCoverageStatus:
    value = raw.strip().lower()
    if value in {"answered", "partial", "unanswered", "unclear"}:
        return value  # type: ignore[return-value]
    return "unanswered"


def _conduct_prompt(english: str, interview_language: InterviewLanguage) -> str:
    if interview_language == "en":
        return english
    prefix = {"ms": "[MS]", "ta": "[TA]", "zh": "[ZH]"}[interview_language]
    return f"{prefix} {english}"


def fake_analyze_interview(req: AnalyzeInterviewRequest) -> AnalyzeInterviewResponse:
    coverage: list[QuestionCoverage] = []
    follow_ups: list[FollowUpSuggestion] = []

    for index, question in enumerate(req.questions):
        status = _FAKE_STATUS_CYCLE[index % len(_FAKE_STATUS_CYCLE)]
        evidence = ""
        confidence = 0.5

        if status == "answered":
            evidence = f"Transcript mentions topic related to: {question.prompt[:60]}"
            confidence = 0.85
        elif status == "partial":
            evidence = "Topic briefly mentioned without full detail"
            confidence = 0.6
        elif status == "unclear":
            evidence = "Vague or inconsistent response detected"
            confidence = 0.55

        coverage.append(
            QuestionCoverage(
                id=question.id,
                status=status,
                evidence=evidence,
                confidence=confidence,
            )
        )

        if status in {"partial", "unanswered", "unclear"}:
            english_prompt = f"Could you clarify: {question.prompt}"
            follow_ups.append(
                FollowUpSuggestion(
                    related_question_id=question.id,
                    prompt=english_prompt,
                    prompt_conduct=_conduct_prompt(english_prompt, req.interview_language),
                    reason=f"Question marked as {status}",
                )
            )

    return AnalyzeInterviewResponse(coverage=coverage, follow_ups=follow_ups, source="fake")


async def llm_analyze_interview(req: AnalyzeInterviewRequest) -> AnalyzeInterviewResponse:
    questions_payload = [
        {
            "id": q.id,
            "prompt": q.prompt,
            **({"hint": q.hint} if q.hint else {}),
        }
        for q in req.questions
    ]

    conduct_language_note = ""
    if req.interview_language != "en":
        conduct_language_note = (
            f"\nConduct language: {req.interview_language} "
            "(follow-up prompt_conduct must be in this language; prompt must be English)."
        )

    user_prompt = f"""
Transcript:
{req.transcript}

Leading questions checklist:
{json.dumps(questions_payload, indent=2)}{conduct_language_note}
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
        parsed = _parse_analysis_json(content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="LLM returned invalid JSON",
        ) from exc

    question_ids = {q.id for q in req.questions}
    coverage_by_id: dict[str, QuestionCoverage] = {}

    for item in parsed.get("coverage", []):
        if not isinstance(item, dict):
            continue
        qid = str(item.get("id", "")).strip()
        if not qid or qid not in question_ids:
            continue
        coverage_by_id[qid] = QuestionCoverage(
            id=qid,
            status=_normalize_status(str(item.get("status", "unanswered"))),
            evidence=str(item.get("evidence", "")).strip(),
            confidence=max(0.0, min(1.0, float(item.get("confidence", 0.0)))),
        )

    coverage = [
        coverage_by_id.get(
            q.id,
            QuestionCoverage(id=q.id, status="unanswered", evidence="", confidence=0.0),
        )
        for q in req.questions
    ]

    follow_ups: list[FollowUpSuggestion] = []
    for item in parsed.get("follow_ups", []):
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            continue
        prompt_conduct = str(item.get("prompt_conduct", "")).strip() or _conduct_prompt(
            prompt, req.interview_language
        )
        related = item.get("related_question_id")
        related_id = str(related).strip() if related else None
        if related_id and related_id not in question_ids:
            related_id = None
        follow_ups.append(
            FollowUpSuggestion(
                related_question_id=related_id,
                prompt=prompt,
                prompt_conduct=prompt_conduct,
                reason=str(item.get("reason", "")).strip(),
            )
        )

    return AnalyzeInterviewResponse(coverage=coverage, follow_ups=follow_ups, source="ollama")


async def analyze_interview_coverage(req: AnalyzeInterviewRequest) -> AnalyzeInterviewResponse:
    if settings.use_fake_extraction:
        return fake_analyze_interview(req)
    return await llm_analyze_interview(req)
