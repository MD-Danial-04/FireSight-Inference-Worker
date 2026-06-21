import asyncio
import logging
from uuid import UUID

from app.coordinator_client import CoordinatorClient
from app.config import settings
from app.extract import extract_fields
from app.schemas import ExtractRequest, TranscribeResponse
from app.transcribe import transcribe_audio

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
            message_type = claim.get("message_type", "stop_message")
            incident_type_name = claim.get("incident_type_name")
            logger.info("Claimed job %s (type=%s)", job_id, message_type)

            try:
                audio_bytes, filename = await coordinator.download_audio(job_id)
                transcript_response = await _transcribe(whisper_model, audio_bytes, filename)
                extract_response = await extract_fields(
                    ExtractRequest(
                        text=transcript_response.transcript,
                        type=message_type,
                        incident_type_name=incident_type_name,
                    )
                )
                await coordinator.complete(
                    job_id,
                    transcript=transcript_response.transcript,
                    result=extract_response.model_dump(mode="json"),
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


async def _transcribe(whisper_model, audio_bytes: bytes, filename: str) -> TranscribeResponse:
    if settings.use_fake_transcription:
        return TranscribeResponse(
            transcript="LF812 stop for location at 7 Gul Ave. Case classified as false alarm malfunction.",
            confidence=0.95,
            source="fake",
        )
    if whisper_model is None:
        raise RuntimeError("Whisper model not loaded")
    return await transcribe_audio(whisper_model, audio_bytes, filename)
