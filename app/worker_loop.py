import asyncio
import logging
from uuid import UUID

from app.analyze_interview import analyze_interview_coverage
from app.analyze_photo import analyze_photo_for_worker
from app.clean_transcript import clean_transcript
from app.coordinator_client import CoordinatorClient
from app.config import settings
from app.extract import extract_fields
from app.extract_interview import extract_interview_details
from app.schemas import (
    AnalyzeInterviewRequest,
    AnalyzePhotoContext,
    CleanTranscriptRequest,
    ExtractRequest,
    ExtractInterviewRequest,
    InterviewLanguage,
    InterviewQuestion,
    TranscribeResponse,
)
from app.transcribe import transcribe_audio
from app.translate import translate_interview_transcript

logger = logging.getLogger(__name__)


async def run_worker_loop(
    coordinator: CoordinatorClient,
    whisper_model,
    *,
    stop_event: asyncio.Event,
) -> None:
    logger.info("Worker loop started (poll interval=%ss)", settings.worker_poll_interval_sec)
    while not stop_event.is_set():
        try:
            claim = await coordinator.claim()
            if claim is None:
                await asyncio.sleep(settings.worker_poll_interval_sec)
                continue

            job_id = UUID(claim["job_id"])
            phase = claim.get("phase", "transcribe")
            message_type = claim.get("message_type", "stop_message")
            incident_type_name = claim.get("incident_type_name")
            logger.info("Claimed job %s (phase=%s type=%s)", job_id, phase, message_type)

            try:
                if phase == "analyze_interview":
                    transcript_text = (claim.get("transcript") or "").strip()
                    questions_raw = claim.get("analysis_questions") or []
                    if not transcript_text:
                        raise RuntimeError("Analyze interview claim missing transcript")
                    if not questions_raw:
                        raise RuntimeError("Analyze interview claim missing questions")
                    questions = [
                        InterviewQuestion.model_validate(q) for q in questions_raw
                    ]
                    interview_language: InterviewLanguage = claim.get("interview_language") or "en"
                    analysis_response = await analyze_interview_coverage(
                        AnalyzeInterviewRequest(
                            transcript=transcript_text,
                            questions=questions,
                            interview_language=interview_language,
                        )
                    )
                    await coordinator.complete_analysis(
                        job_id,
                        result=analysis_response.model_dump(mode="json"),
                    )
                elif phase == "clean_transcript":
                    original = (claim.get("transcript_original") or "").strip()
                    english = (
                        claim.get("transcript_english")
                        or claim.get("transcript")
                        or ""
                    ).strip()
                    if not original and not english:
                        raise RuntimeError("Clean transcript claim missing transcript")
                    interview_language: InterviewLanguage = (
                        claim.get("interview_language") or "en"
                    )
                    clean_response = await clean_transcript(
                        CleanTranscriptRequest(
                            transcript_original=original or english,
                            transcript_english=english or original,
                            interview_language=interview_language,
                        )
                    )
                    await coordinator.complete_clean_transcript(
                        job_id,
                        transcript_original=clean_response.transcript_original,
                        transcript_english=clean_response.transcript_english,
                    )
                elif phase == "analyze_photo":
                    image_bytes, _filename = await coordinator.download_image(job_id)
                    if not image_bytes:
                        raise RuntimeError("Analyze photo claim missing image")
                    ctx_raw = claim.get("photo_context") or {}
                    context = AnalyzePhotoContext.model_validate(ctx_raw)
                    photo_response = await analyze_photo_for_worker(image_bytes, context)
                    await coordinator.complete_photo_analysis(
                        job_id,
                        result=photo_response.model_dump(mode="json"),
                    )
                elif phase == "extract":
                    transcript_text = (claim.get("transcript") or "").strip()
                    if not transcript_text:
                        raise RuntimeError("Extract claim missing transcript")
                    if message_type == "interview":
                        extract_response = await extract_interview_details(
                            ExtractInterviewRequest(
                                text=transcript_text,
                                interview_language=claim.get("interview_language") or "en",
                            )
                        )
                        await coordinator.complete_extraction(
                            job_id,
                            interview_details=extract_response.model_dump(mode="json"),
                        )
                    else:
                        extract_response = await extract_fields(
                            ExtractRequest(
                                text=transcript_text,
                                type=message_type,
                                incident_type_name=incident_type_name,
                            )
                        )
                        await coordinator.complete_extraction(
                            job_id,
                            result=extract_response.model_dump(mode="json"),
                        )
                else:
                    interview_language: InterviewLanguage = claim.get("interview_language") or "en"
                    audio_bytes, filename = await coordinator.download_audio(job_id)
                    transcript_response = await _transcribe(
                        whisper_model,
                        audio_bytes,
                        filename,
                        interview_language=interview_language,
                    )
                    await coordinator.complete_transcription(
                        job_id,
                        transcript=transcript_response.transcript_english,
                        transcript_original=transcript_response.transcript_original,
                        transcript_english=transcript_response.transcript_english,
                        interview_language=transcript_response.interview_language,
                    )
                logger.info("Completed job %s", job_id)
            except Exception as exc:
                logger.exception("Job %s failed", job_id)
                try:
                    await coordinator.fail(job_id, error=str(exc))
                except Exception:
                    logger.exception("Failed to report job %s failure", job_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Worker loop error")
            await asyncio.sleep(settings.worker_poll_interval_sec)


async def _transcribe(
    whisper_model,
    audio_bytes: bytes,
    filename: str,
    *,
    interview_language: InterviewLanguage = "en",
) -> TranscribeResponse:
    if settings.use_fake_transcription:
        original = "LF812 stop for location at 7 Gul Ave. Case classified as false alarm malfunction."
        english, translation_source = await translate_interview_transcript(
            original,
            interview_language,
        )
        return TranscribeResponse(
            transcript_original=original,
            transcript_english=english,
            interview_language=interview_language,
            confidence=0.95,
            source="fake",
            translation_source=translation_source,
        )
    if whisper_model is None:
        raise RuntimeError("Whisper model not loaded")
    response = await transcribe_audio(
        whisper_model,
        audio_bytes,
        filename,
        language=interview_language,
    )
    if interview_language != "en":
        english, translation_source = await translate_interview_transcript(
            response.transcript_original,
            interview_language,
        )
        response = response.model_copy(
            update={
                "transcript_english": english,
                "translation_source": translation_source,
            }
        )
    return response
