import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.schemas import ExtractResponse, TranscribeResponse
from app.worker_loop import run_worker_loop


def test_worker_loop_processes_claimed_job():
    job_id = uuid4()
    stop_event = asyncio.Event()
    coordinator = AsyncMock()
    coordinator.claim = AsyncMock(
        side_effect=[
            {
                "job_id": str(job_id),
                "message_type": "stop_message",
                "incident_type_name": "Fire",
            },
            None,
        ]
    )
    coordinator.download_audio = AsyncMock(return_value=(b"audio-bytes", "sample.wav"))
    coordinator.complete = AsyncMock()
    coordinator.fail = AsyncMock()

    fake_transcript = TranscribeResponse(
        transcript="LF812 stop for location.",
        confidence=0.9,
        source="fake",
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
    coordinator.complete.assert_called_once()
    call_kwargs = coordinator.complete.call_args.kwargs
    assert call_kwargs["transcript"] == "LF812 stop for location."
    assert call_kwargs["result"]["fields"]["applianceCallSign"] == "LF812"
