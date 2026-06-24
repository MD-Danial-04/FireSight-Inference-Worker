#!/usr/bin/env python3
"""One-time script to seed multilingual leading-question template files.

Reads English questions from scripts/leading_questions_en.json, translates to
ms/ta/zh via the configured LLM (or fake prefixes when --fake), and writes
TypeScript constant files to the Fire Report Generation App.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402

INPUT_PATH = Path(__file__).resolve().parent / "leading_questions_en.json"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "Fire Report Generation App"
    / "src"
    / "app"
    / "constants"
    / "leadingQuestions"
)

LANGUAGE_NAMES = {
    "ms": "Malay",
    "ta": "Tamil",
    "zh": "Mandarin Chinese",
}

FAKE_LANGUAGE_PREFIX = {
    "ms": "[MS]",
    "ta": "[TA]",
    "zh": "[ZH]",
}

TARGET_LANGS = ("ms", "ta", "zh")
DEFAULT_BATCH_SIZE = 8
MAX_RETRIES = 3

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
- Return a single JSON object only. No markdown fences or commentary.
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


def _parse_json_object(content: str) -> dict:
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
        if not isinstance(parsed, dict):
            last_error = ValueError("expected JSON object")
            continue
        return parsed

    snippet = content[:500].replace("\n", "\\n")
    if last_error is not None:
        raise ValueError(f"Invalid JSON from LLM ({last_error}); snippet: {snippet}") from last_error
    raise ValueError(f"No JSON object found in LLM response; snippet: {snippet}")


def ts_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def ts_loc(en: str, ms: str, ta: str, zh: str) -> str:
    return f"loc({ts_string(en)}, {ts_string(ms)}, {ts_string(ta)}, {ts_string(zh)})"


def _fake_conduct_text(text: str, target_lang: str) -> str:
    prefix = FAKE_LANGUAGE_PREFIX[target_lang]
    return f"{prefix} {text}".strip()


def _chunk_questions(questions: list[dict], batch_size: int) -> list[list[dict]]:
    if batch_size <= 0:
        return [questions]
    return [questions[i : i + batch_size] for i in range(0, len(questions), batch_size)]


def _parse_translation_batch(
    content: str,
    questions: list[dict],
) -> dict[str, dict[str, str | None]]:
    parsed = _parse_json_object(content)
    by_id: dict[str, dict[str, str | None]] = {}
    question_ids = {q["id"] for q in questions}
    for item in parsed.get("questions", []):
        if not isinstance(item, dict):
            continue
        qid = str(item.get("id", "")).strip()
        prompt_conduct = str(item.get("prompt_conduct", "")).strip()
        if not qid or qid not in question_ids or not prompt_conduct:
            continue
        hint_raw = item.get("hint_conduct")
        section_raw = item.get("section_conduct")
        by_id[qid] = {
            "prompt": prompt_conduct,
            "hint": str(hint_raw).strip() if hint_raw else None,
            "section": str(section_raw).strip() if section_raw else None,
        }
    return by_id


async def _llm_translate_batch_once(
    questions: list[dict],
    target_lang: str,
) -> dict[str, dict[str, str | None]]:
    language_name = LANGUAGE_NAMES[target_lang]
    payload_questions = [
        {
            "id": q["id"],
            "prompt": q["prompt"],
            **({"hint": q["hint"]} if q.get("hint") else {}),
            **({"section": q["section"]} if q.get("section") else {}),
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

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=request_payload,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

    return _parse_translation_batch(content, questions)


async def llm_translate_batch(
    questions: list[dict],
    target_lang: str,
    *,
    batch_size: int,
) -> dict[str, dict[str, str | None]]:
    merged: dict[str, dict[str, str | None]] = {}
    chunks = _chunk_questions(questions, batch_size)

    for index, chunk in enumerate(chunks, start=1):
        label = f"{target_lang} batch {index}/{len(chunks)} ({len(chunk)} questions)"
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"    {label} attempt {attempt}")
                batch_result = await _llm_translate_batch_once(chunk, target_lang)
                missing = [q["id"] for q in chunk if q["id"] not in batch_result]
                if missing:
                    raise ValueError(f"missing translations for: {', '.join(missing)}")
                merged.update(batch_result)
                break
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1.5 * attempt)
                continue
        else:
            raise RuntimeError(f"Failed {label}") from last_error

    return merged


def fake_translate_batch(
    questions: list[dict],
    target_lang: str,
) -> dict[str, dict[str, str | None]]:
    by_id: dict[str, dict[str, str | None]] = {}
    for question in questions:
        by_id[question["id"]] = {
            "prompt": _fake_conduct_text(question["prompt"], target_lang),
            "hint": _fake_conduct_text(question["hint"], target_lang)
            if question.get("hint")
            else None,
            "section": _fake_conduct_text(question["section"], target_lang)
            if question.get("section")
            else None,
        }
    return by_id


async def translate_questions_for_lang(
    questions: list[dict],
    target_lang: str,
    *,
    use_fake: bool,
    batch_size: int,
) -> dict[str, dict[str, str | None]]:
    if use_fake:
        return fake_translate_batch(questions, target_lang)
    return await llm_translate_batch(questions, target_lang, batch_size=batch_size)


def render_template_file(
    *,
    title: str,
    title_export_name: str,
    export_name: str,
    questions: list[dict],
    translations: dict[str, dict[str, dict[str, str | None]]],
) -> str:
    lines = [
        'import type { LeadingQuestion } from "./leadingQuestions";',
        'import { loc } from "./leadingQuestions";',
        ""
        f"export const {title_export_name} = {ts_string(title)};",
        "",
        f"export const {export_name}: LeadingQuestion[] = [",
    ]

    for question in questions:
        qid = question["id"]
        section_en = question["section"]
        prompt_en = question["prompt"]
        hint_en = question.get("hint")

        section_ms = translations["ms"][qid]["section"] or section_en
        section_ta = translations["ta"][qid]["section"] or section_en
        section_zh = translations["zh"][qid]["section"] or section_en
        prompt_ms = translations["ms"][qid]["prompt"]
        prompt_ta = translations["ta"][qid]["prompt"]
        prompt_zh = translations["zh"][qid]["prompt"]

        lines.append("  {")
        lines.append(f'    id: {ts_string(qid)},')
        lines.append(
            f"    section: {ts_loc(section_en, section_ms, section_ta, section_zh)},"
        )
        lines.append(
            f"    prompt: {ts_loc(prompt_en, prompt_ms, prompt_ta, prompt_zh)},"
        )

        if hint_en:
            hint_ms = translations["ms"][qid]["hint"] or hint_en
            hint_ta = translations["ta"][qid]["hint"] or hint_en
            hint_zh = translations["zh"][qid]["hint"] or hint_en
            lines.append(
                f"    hint: {ts_loc(hint_en, hint_ms, hint_ta, hint_zh)},"
            )

        lines.append("  },")

    lines.append("];")
    lines.append("")
    return "\n".join(lines)


def output_filename_for_key(key: str) -> str:
    mapping = {
        "amd": "amdLeadingQuestions.ts",
        "vehicle_fire": "vehicleFireLeadingQuestions.ts",
        "lpg": "lpgFireLeadingQuestions.ts",
    }
    return mapping[key]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use [MS]/[TA]/[ZH] prefix fake translations instead of LLM",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Questions per LLM request (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--only",
        choices=["amd", "vehicle_fire", "lpg"],
        help="Translate a single template (useful after a partial run)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated TypeScript constant files",
    )
    args = parser.parse_args()

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if args.only:
        data = {args.only: data[args.only]}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for key, template in data.items():
        questions = template["questions"]
        print(f"Translating {key} ({len(questions)} questions)...")

        translations: dict[str, dict[str, dict[str, str | None]]] = {}
        for lang in TARGET_LANGS:
            print(f"  -> {lang}")
            translations[lang] = await translate_questions_for_lang(
                questions,
                lang,
                use_fake=args.fake,
                batch_size=args.batch_size,
            )

        content = render_template_file(
            title=template["title"],
            title_export_name=template["title_export_name"],
            export_name=template["export_name"],
            questions=questions,
            translations=translations,
        )

        out_path = args.output_dir / output_filename_for_key(key)
        out_path.write_text(content, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
