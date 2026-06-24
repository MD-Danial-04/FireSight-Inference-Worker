import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.analyze_interview import analyze_interview_coverage
from app.analyze_photo import analyze_photo
from app.coordinator_client import get_coordinator_client
from app.config import settings
from app.extract import extract_fields
from app.extract_interview import extract_interview_details
from app.schemas import (
    AnalyzeInterviewRequest,
    AnalyzeInterviewResponse,
    AnalyzePhotoContext,
    AnalyzePhotoResponse,
    ExtractRequest,
    ExtractResponse,
    ExtractInterviewRequest,
    ExtractInterviewResponse,
    HealthResponse,
    InterviewLanguage,
    TranscribeResponse,
)
from app.transcribe import load_whisper_model
from app.worker_loop import _transcribe, run_worker_loop

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.use_fake_transcription:
        app.state.whisper_model = None
    else:
        app.state.whisper_model = load_whisper_model()

    worker_task: asyncio.Task | None = None
    stop_event = asyncio.Event()
    if settings.worker_enabled:
        coordinator = get_coordinator_client()
        worker_task = asyncio.create_task(
            run_worker_loop(coordinator, app.state.whisper_model, stop_event=stop_event)
        )

    yield

    if worker_task is not None:
        stop_event.set()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
    app.state.whisper_model = None


app = FastAPI(title="FireSight Inference Worker", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        fake_extraction=settings.use_fake_extraction,
        fake_transcription=settings.use_fake_transcription,
        fake_photo_analysis=settings.use_fake_photo_analysis,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        vision_model=settings.vision_model,
        whisper_model=settings.whisper_model,
        whisper_device=settings.whisper_device,
    )


@app.post("/v1/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    return await extract_fields(req)


@app.post("/v1/extract-interview", response_model=ExtractInterviewResponse)
async def extract_interview(req: ExtractInterviewRequest) -> ExtractInterviewResponse:
    return await extract_interview_details(req)


@app.post("/v1/analyze-interview", response_model=AnalyzeInterviewResponse)
async def analyze_interview(req: AnalyzeInterviewRequest) -> AnalyzeInterviewResponse:
    return await analyze_interview_coverage(req)


@app.post("/v1/analyze-photo", response_model=AnalyzePhotoResponse)
async def analyze_photo_endpoint(
    file: UploadFile = File(...),
    location_of_fire: str | None = None,
    incident_type_name: str | None = None,
    stop_message_excerpt: str | None = None,
    field_notes_excerpt: str | None = None,
) -> AnalyzePhotoResponse:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    context = AnalyzePhotoContext(
        location_of_fire=location_of_fire,
        incident_type_name=incident_type_name,
        stop_message_excerpt=stop_message_excerpt,
        field_notes_excerpt=field_notes_excerpt,
    )
    return await analyze_photo(image_bytes, context)


@app.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    interview_language: InterviewLanguage = Form("en"),
) -> TranscribeResponse:
    audio_bytes = await file.read()
    whisper_model = getattr(app.state, "whisper_model", None)
    return await _transcribe(
        whisper_model,
        audio_bytes,
        file.filename or "audio.webm",
        interview_language=interview_language,
    )
