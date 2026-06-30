import logging
from uuid import UUID

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class CoordinatorClient:
    def __init__(self) -> None:
        self._base = settings.coordinator_base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {settings.coordinator_worker_api_key}"}

    async def claim(self) -> dict | None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base}/v1/worker/claim",
                headers=self._headers,
            )
            response.raise_for_status()
            data = response.json()
            return data if data else None

    async def download_audio(self, job_id: UUID) -> tuple[bytes, str]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(
                f"{self._base}/v1/worker/jobs/{job_id}/audio",
                headers=self._headers,
            )
            response.raise_for_status()
            content_disposition = response.headers.get("content-disposition", "")
            filename = "audio.webm"
            if "filename=" in content_disposition:
                filename = content_disposition.split("filename=", 1)[1].strip('"')
            return response.content, filename

    async def complete_transcription(
        self,
        job_id: UUID,
        *,
        transcript: str,
        transcript_original: str | None = None,
        transcript_english: str | None = None,
        interview_language: str | None = None,
    ) -> None:
        payload: dict = {"transcript": transcript}
        if transcript_original is not None:
            payload["transcript_original"] = transcript_original
        if transcript_english is not None:
            payload["transcript_english"] = transcript_english
        if interview_language is not None:
            payload["interview_language"] = interview_language
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base}/v1/worker/jobs/{job_id}/transcribe",
                headers=self._headers,
                json=payload,
            )
            response.raise_for_status()

    async def complete_extraction(
        self,
        job_id: UUID,
        *,
        result: dict | None = None,
        interview_details: dict | None = None,
    ) -> None:
        payload: dict = {}
        if result is not None:
            payload["result"] = result
        if interview_details is not None:
            payload["interview_details"] = interview_details
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base}/v1/worker/jobs/{job_id}/complete-extraction",
                headers=self._headers,
                json=payload,
            )
            response.raise_for_status()

    async def complete_analysis(self, job_id: UUID, *, result: dict) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base}/v1/worker/jobs/{job_id}/complete-analysis",
                headers=self._headers,
                json={"result": result},
            )
            response.raise_for_status()

    async def complete_clean_transcript(
        self,
        job_id: UUID,
        *,
        transcript_original: str,
        transcript_english: str,
    ) -> None:
        payload = {
            "transcript_original": transcript_original,
            "transcript_english": transcript_english,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base}/v1/worker/jobs/{job_id}/complete-clean-transcript",
                headers=self._headers,
                json=payload,
            )
            response.raise_for_status()

    async def download_image(self, job_id: UUID) -> tuple[bytes, str]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(
                f"{self._base}/v1/worker/jobs/{job_id}/image",
                headers=self._headers,
            )
            response.raise_for_status()
            content_disposition = response.headers.get("content-disposition", "")
            filename = "photo.jpg"
            if "filename=" in content_disposition:
                filename = content_disposition.split("filename=", 1)[1].strip('"')
            return response.content, filename

    async def complete_photo_analysis(self, job_id: UUID, *, result: dict) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base}/v1/worker/jobs/{job_id}/complete-photo-analysis",
                headers=self._headers,
                json={"result": result},
            )
            response.raise_for_status()

    async def fail(self, job_id: UUID, *, error: str) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base}/v1/worker/jobs/{job_id}/fail",
                headers=self._headers,
                json={"error": error},
            )
            response.raise_for_status()


def get_coordinator_client() -> CoordinatorClient:
    return CoordinatorClient()
