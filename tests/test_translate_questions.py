import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.schemas import QuestionTranslationResult, TranslatedInterviewQuestion, TranslateInterviewQuestionInput
from app.translate import fake_translate_interview_questions, translate_interview_questions


def test_fake_translate_interview_questions_non_english():
    questions = [
        TranslateInterviewQuestionInput(
            id="device-type",
            prompt="What type of mobility device is it?",
            hint="e.g. PMD",
            section="Affected device",
        )
    ]

    async def run():
        result = await translate_interview_questions(questions, "ms")
        assert result.source == "fake"
        assert result.questions[0].prompt_conduct.startswith("[MS]")
        assert result.questions[0].hint_conduct is not None
        assert result.questions[0].section_conduct is not None

    asyncio.run(run())


def test_fake_translate_interview_questions_english_identity():
    questions = [
        TranslateInterviewQuestionInput(
            id="device-type",
            prompt="What type of mobility device is it?",
        )
    ]
    result = fake_translate_interview_questions(questions, "en")
    assert result.questions[0].prompt_conduct == questions[0].prompt


def test_worker_loop_processes_translate_questions_claim():
    job_id = uuid4()
    stop_event = asyncio.Event()
    coordinator = AsyncMock()
    coordinator.claim = AsyncMock(
        side_effect=[
            {
                "job_id": str(job_id),
                "phase": "translate_questions",
                "message_type": "field_notes",
                "interview_language": "ta",
                "analysis_questions": [
                    {
                        "id": "device-type",
                        "prompt": "What type of mobility device is it?",
                        "section": "Affected device",
                    }
                ],
            },
            None,
        ]
    )
    coordinator.complete_question_translation = AsyncMock()
    coordinator.fail = AsyncMock()

    fake_result = QuestionTranslationResult(
        questions=[
            TranslatedInterviewQuestion(
                id="device-type",
                prompt_conduct="[TA] What type of mobility device is it?",
                section_conduct="[TA] Affected device",
            )
        ],
        source="fake",
    )

    async def run_once():
        with (
            patch(
                "app.worker_loop.translate_interview_questions",
                AsyncMock(return_value=fake_result),
            ),
            patch("app.worker_loop.settings") as mock_settings,
        ):
            from app.worker_loop import run_worker_loop

            mock_settings.worker_poll_interval_sec = 0.01
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

    coordinator.complete_question_translation.assert_called_once()
    call_kwargs = coordinator.complete_question_translation.call_args.kwargs
    assert call_kwargs["result"]["questions"][0]["id"] == "device-type"
