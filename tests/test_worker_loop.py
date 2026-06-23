import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.schemas import (
    AnalyzeInterviewResponse,
    AnalyzePhotoResponse,
    ExtractResponse,
    PhotoAnalysisConfidence,
    QuestionCoverage,
    TranscribeResponse,
)
from app.worker_loop import run_worker_loop


def test_worker_loop_processes_transcribe_claim():
    job_id = uuid4()
    stop_event = asyncio.Event()
    coordinator = AsyncMock()
    coordinator.claim = AsyncMock(
        side_effect=[
            {
                "job_id": str(job_id),
                "phase": "transcribe",
                "message_type": "stop_message",
                "incident_type_name": "Fire",
            },
            None,
        ]
    )
    coordinator.download_audio = AsyncMock(return_value=(b"audio-bytes", "sample.wav"))
    coordinator.complete_transcription = AsyncMock()
    coordinator.complete_extraction = AsyncMock()
    coordinator.complete_analysis = AsyncMock()
    coordinator.fail = AsyncMock()

    fake_transcript = TranscribeResponse(
        transcript_original="LF812 stop for location.",
        transcript_english="LF812 stop for location.",
        interview_language="en",
        confidence=0.9,
        source="fake",
        translation_source="none",
    )
    fake_extract = ExtractResponse(
        fields={"applianceCallSign": "LF812"},
        confidence={"applianceCallSign": 0.95},
        source="fake",
    )

    async def run_once():
        with (
            patch("app.worker_loop._transcribe", AsyncMock(return_value=fake_transcript)),
            patch("app.worker_loop.extract_fields", AsyncMock(return_value=fake_extract)),
            patch("app.worker_loop.settings") as mock_settings,
        ):
            mock_settings.worker_poll_interval_sec = 0.01
            mock_settings.use_fake_transcription = True

            task = asyncio.create_task(
                run_worker_loop(coordinator, whisper_model=None, stop_event=stop_event)
            )
            await asyncio.sleep(0.05)
            stop_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run_once())

    coordinator.download_audio.assert_called_once()
    coordinator.complete_transcription.assert_called_once()
    coordinator.complete_extraction.assert_not_called()
    call_kwargs = coordinator.complete_transcription.call_args.kwargs
    assert call_kwargs["transcript"] == "LF812 stop for location."
    assert call_kwargs["transcript_english"] == "LF812 stop for location."
    assert call_kwargs["transcript_original"] == "LF812 stop for location."
    assert call_kwargs["interview_language"] == "en"


def test_worker_loop_processes_transcribe_claim_with_interview_language():
    job_id = uuid4()
    stop_event = asyncio.Event()
    coordinator = AsyncMock()
    coordinator.claim = AsyncMock(
        side_effect=[
            {
                "job_id": str(job_id),
                "phase": "transcribe",
                "message_type": "field_notes",
                "interview_language": "ms",
            },
            None,
        ]
    )
    coordinator.download_audio = AsyncMock(return_value=(b"audio-bytes", "sample.webm"))
    coordinator.complete_transcription = AsyncMock()
    coordinator.fail = AsyncMock()

    fake_transcript = TranscribeResponse(
        transcript_original="Kebakaran di dapur.",
        transcript_english="[EN] Kebakaran di dapur.",
        interview_language="ms",
        confidence=0.9,
        source="fake",
        translation_source="fake",
    )

    async def run_once():
        with (
            patch("app.worker_loop._transcribe", AsyncMock(return_value=fake_transcript)),
            patch("app.worker_loop.settings") as mock_settings,
        ):
            mock_settings.worker_poll_interval_sec = 0.01
            mock_settings.use_fake_transcription = True

            task = asyncio.create_task(
                run_worker_loop(coordinator, whisper_model=None, stop_event=stop_event)
            )
            await asyncio.sleep(0.05)
            stop_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run_once())

    call_kwargs = coordinator.complete_transcription.call_args.kwargs
    assert call_kwargs["interview_language"] == "ms"
    assert call_kwargs["transcript_original"] == "Kebakaran di dapur."
    assert call_kwargs["transcript_english"] == "[EN] Kebakaran di dapur."


def test_worker_loop_processes_extract_claim():
    job_id = uuid4()
    stop_event = asyncio.Event()
    coordinator = AsyncMock()
    coordinator.claim = AsyncMock(
        side_effect=[
            {
                "job_id": str(job_id),
                "phase": "extract",
                "message_type": "stop_message",
                "incident_type_name": "Fire",
                "transcript": "LF812 stop for location.",
            },
            None,
        ]
    )
    coordinator.download_audio = AsyncMock()
    coordinator.complete_transcription = AsyncMock()
    coordinator.complete_extraction = AsyncMock()
    coordinator.complete_analysis = AsyncMock()
    coordinator.fail = AsyncMock()

    fake_extract = ExtractResponse(
        fields={"applianceCallSign": "LF812"},
        confidence={"applianceCallSign": 0.95},
        source="fake",
    )

    async def run_once():
        with (
            patch("app.worker_loop.extract_fields", AsyncMock(return_value=fake_extract)),
            patch("app.worker_loop.settings") as mock_settings,
        ):
            mock_settings.worker_poll_interval_sec = 0.01
            mock_settings.use_fake_transcription = True

            task = asyncio.create_task(
                run_worker_loop(coordinator, whisper_model=None, stop_event=stop_event)
            )
            await asyncio.sleep(0.05)
            stop_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run_once())

    coordinator.download_audio.assert_not_called()
    coordinator.complete_transcription.assert_not_called()
    coordinator.complete_extraction.assert_called_once()
    call_kwargs = coordinator.complete_extraction.call_args.kwargs
    assert call_kwargs["result"]["fields"]["applianceCallSign"] == "LF812"


def test_worker_loop_processes_analyze_interview_claim():
    job_id = uuid4()
    stop_event = asyncio.Event()
    coordinator = AsyncMock()
    coordinator.claim = AsyncMock(
        side_effect=[
            {
                "job_id": str(job_id),
                "phase": "analyze_interview",
                "message_type": "field_notes",
                "transcript": "The device is a PMD.",
                "analysis_questions": [
                    {"id": "device-type", "prompt": "What type of mobility device is it?"},
                ],
            },
            None,
        ]
    )
    coordinator.download_audio = AsyncMock()
    coordinator.complete_transcription = AsyncMock()
    coordinator.complete_extraction = AsyncMock()
    coordinator.complete_analysis = AsyncMock()
    coordinator.fail = AsyncMock()

    fake_analysis = AnalyzeInterviewResponse(
        coverage=[
            QuestionCoverage(
                id="device-type",
                status="answered",
                evidence="PMD",
                confidence=0.9,
            )
        ],
        follow_ups=[],
        source="fake",
    )

    async def run_once():
        with (
            patch(
                "app.worker_loop.analyze_interview_coverage",
                AsyncMock(return_value=fake_analysis),
            ),
            patch("app.worker_loop.settings") as mock_settings,
        ):
            mock_settings.worker_poll_interval_sec = 0.01
            mock_settings.use_fake_transcription = True

            task = asyncio.create_task(
                run_worker_loop(coordinator, whisper_model=None, stop_event=stop_event)
            )
            await asyncio.sleep(0.05)
            stop_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run_once())

    coordinator.download_audio.assert_not_called()
    coordinator.complete_transcription.assert_not_called()
    coordinator.complete_extraction.assert_not_called()
    coordinator.complete_analysis.assert_called_once()
    call_kwargs = coordinator.complete_analysis.call_args.kwargs
    assert call_kwargs["result"]["coverage"][0]["id"] == "device-type"


def test_worker_loop_processes_analyze_photo_claim():
    job_id = uuid4()
    stop_event = asyncio.Event()
    coordinator = AsyncMock()
    coordinator.claim = AsyncMock(
        side_effect=[
            {
                "job_id": str(job_id),
                "phase": "analyze_photo",
                "message_type": "field_notes",
                "photo_context": {
                    "location_of_fire": "7 Gul Ave",
                    "incident_type_name": "Structure Fire",
                },
            },
            None,
        ]
    )
    coordinator.download_image = AsyncMock(return_value=(b"image-bytes", "scene.jpg"))
    coordinator.complete_photo_analysis = AsyncMock()
    coordinator.download_audio = AsyncMock()
    coordinator.complete_transcription = AsyncMock()
    coordinator.complete_extraction = AsyncMock()
    coordinator.complete_analysis = AsyncMock()
    coordinator.fail = AsyncMock()

    fake_photo = AnalyzePhotoResponse(
        caption="Charring on ceiling lining.",
        detected_elements=["ceiling charring"],
        suggested_section="burn_patterns",
        confidence=PhotoAnalysisConfidence(caption=0.85, suggested_section=0.78),
        source="fake",
    )

    async def run_once():
        with (
            patch(
                "app.worker_loop.analyze_photo_for_worker",
                AsyncMock(return_value=fake_photo),
            ),
            patch("app.worker_loop.settings") as mock_settings,
        ):
            mock_settings.worker_poll_interval_sec = 0.01
            mock_settings.use_fake_transcription = True

            task = asyncio.create_task(
                run_worker_loop(coordinator, whisper_model=None, stop_event=stop_event)
            )
            await asyncio.sleep(0.05)
            stop_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run_once())

    coordinator.download_image.assert_called_once()
    coordinator.complete_photo_analysis.assert_called_once()
    call_kwargs = coordinator.complete_photo_analysis.call_args.kwargs
    assert call_kwargs["result"]["suggested_section"] == "burn_patterns"
