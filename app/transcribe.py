import asyncio
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import settings
from app.schemas import TranscribeResponse

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


def load_whisper_model() -> "WhisperModel":
    from faster_whisper import WhisperModel

    logger.info(
        "Loading Whisper model=%s device=%s compute_type=%s",
        settings.whisper_model,
        settings.whisper_device,
        settings.whisper_compute_type,
    )
    model = WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    logger.info("Whisper model loaded")
    return model


def _transcribe_file(model: "WhisperModel", audio_path: str) -> TranscribeResponse:
    transcribe_kwargs: dict = {"language": settings.whisper_language}
    if settings.whisper_initial_prompt:
        transcribe_kwargs["initial_prompt"] = settings.whisper_initial_prompt

    segments, _info = model.transcribe(audio_path, **transcribe_kwargs)
    segment_list = list(segments)
    transcript = " ".join(segment.text.strip() for segment in segment_list).strip()

    confidence: float | None = None
    if segment_list:
        avg_logprob = sum(segment.avg_logprob for segment in segment_list) / len(segment_list)
        confidence = min(1.0, max(0.0, math.exp(avg_logprob)))

    return TranscribeResponse(
        transcript=transcript,
        confidence=confidence,
        source="whisper",
    )


def _suffix_from_filename(filename: str | None) -> str:
    if not filename:
        return ".webm"
    suffix = Path(filename).suffix
    return suffix if suffix else ".webm"


def _transcribe_bytes_sync(model: "WhisperModel", audio_bytes: bytes, filename: str | None) -> TranscribeResponse:
    suffix = _suffix_from_filename(filename)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        return _transcribe_file(model, tmp_path)
    finally:
        os.unlink(tmp_path)


async def transcribe_audio(
    model: "WhisperModel",
    audio_bytes: bytes,
    filename: str | None = None,
) -> TranscribeResponse:
    return await asyncio.to_thread(_transcribe_bytes_sync, model, audio_bytes, filename)
