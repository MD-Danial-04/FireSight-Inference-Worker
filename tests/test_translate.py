import asyncio

from app.translate import fake_translate_interview_transcript, translate_interview_transcript
from app.worker_loop import _transcribe


def test_translate_english_skips_translation():
    async def run():
        text = "The fire started in the kitchen."
        result, source = await translate_interview_transcript(text, "en")
        assert result == text
        assert source == "none"

    asyncio.run(run())


def test_fake_translate_non_english_prefixes():
    async def run():
        for lang in ("ms", "ta", "zh"):
            text = "Kebakaran bermula di dapur."
            result, source = fake_translate_interview_transcript(text, lang)
            assert result == f"[EN] {text}"
            assert source == "fake"

    asyncio.run(run())


def test_fake_transcribe_all_languages():
    async def run():
        for lang in ("en", "ms", "ta", "zh"):
            response = await _transcribe(
                None,
                b"audio",
                "sample.webm",
                interview_language=lang,
            )
            assert response.transcript_original
            assert response.transcript_english
            assert response.interview_language == lang
            if lang == "en":
                assert response.translation_source == "none"
                assert response.transcript_original == response.transcript_english
            else:
                assert response.translation_source == "fake"
                assert response.transcript_english.startswith("[EN] ")

    asyncio.run(run())
