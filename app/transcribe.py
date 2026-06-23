import asyncio
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import settings
from app.schemas import InterviewLanguage, TranscribeResponse

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

LANGUAGE_INITIAL_PROMPTS: dict[InterviewLanguage, str] = {
    "en": settings.whisper_initial_prompt,
    "ms": "Temubual siasatan kebakaran SCDF. PMD, penggera palsu, zon.",
    "ta": "SCDF தீ விசாரணை நேர்காணல். PMD, தவறான அலாரம், மண்டலம்.",
    "zh": "新加坡民防部队火灾调查访谈。个人代步工具、误报、区域。",
}


def load_whisper_model() -> "WhisperModel":
    from faster_whisper import WhisperModel

    device = settings.whisper_device
    compute_type = settings.whisper_compute_type

    logger.info(
        "Loading Whisper model=%s device=%s compute_type=%s",
        settings.whisper_model,
        device,
        compute_type,
    )
    try:
        model = WhisperModel(
            settings.whisper_model,
            device=device,
            compute_type=compute_type,
        )
    except ValueError as exc:
        if device == "cuda" and "CUDA" in str(exc):
            logger.warning(
                "CTranslate2 has no CUDA support on this system (%s); falling back to cpu/int8",
                exc,
            )
            model = WhisperModel(
                settings.whisper_model,
                device="cpu",
                compute_type="int8",
            )
        else:
            raise
    logger.info("Whisper model loaded")
    return model


def _transcribe_file(
    model: "WhisperModel",
    audio_path: str,
    *,
    language: InterviewLanguage | None = None,
) -> TranscribeResponse:
    lang: InterviewLanguage = language or settings.whisper_language  # type: ignore[assignment]
    transcribe_kwargs: dict = {
        "language": lang,
        "vad_filter": settings.whisper_vad_filter,
        "beam_size": settings.whisper_beam_size,
        "condition_on_previous_text": settings.whisper_condition_on_previous_text,
    }
    initial_prompt = LANGUAGE_INITIAL_PROMPTS.get(lang) or settings.whisper_initial_prompt
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt

    segments, _info = model.transcribe(audio_path, **transcribe_kwargs)
    segment_list = list(segments)
    transcript = " ".join(segment.text.strip() for segment in segment_list).strip()

    confidence: float | None = None
    if segment_list:
        avg_logprob = sum(segment.avg_logprob for segment in segment_list) / len(segment_list)
        confidence = min(1.0, max(0.0, math.exp(avg_logprob)))

    return TranscribeResponse(
        transcript_original=transcript,
        transcript_english=transcript,
        interview_language=lang,
        confidence=confidence,
        source="whisper",
        translation_source="none",
    )


def _suffix_from_filename(filename: str | None) -> str:
    if not filename:
        return ".webm"
    suffix = Path(filename).suffix
    return suffix if suffix else ".webm"


def _transcribe_bytes_sync(
    model: "WhisperModel",
    audio_bytes: bytes,
    filename: str | None,
    *,
    language: InterviewLanguage | None = None,
) -> TranscribeResponse:
    suffix = _suffix_from_filename(filename)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        return _transcribe_file(model, tmp_path, language=language)
    finally:
        os.unlink(tmp_path)


async def transcribe_audio(
    model: "WhisperModel",
    audio_bytes: bytes,
    filename: str | None = None,
    *,
    language: InterviewLanguage | None = None,
) -> TranscribeResponse:
    return await asyncio.to_thread(
        _transcribe_bytes_sync,
        model,
        audio_bytes,
        filename,
        language=language,
    )
