import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.coordinator_client import get_coordinator_client
from app.config import settings
from app.extract import extract_fields
from app.schemas import ExtractRequest, ExtractResponse, HealthResponse, TranscribeResponse
from app.transcribe import load_whisper_model, transcribe_audio
from app.worker_loop import run_worker_loop

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
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        whisper_model=settings.whisper_model,
        whisper_device=settings.whisper_device,
    )


@app.post("/v1/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    return await extract_fields(req)


@app.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscribeResponse:
    if settings.use_fake_transcription:
        return TranscribeResponse(
            transcript="LF812 stop for location at 7 Gul Ave. Case classified as false alarm malfunction.",
            confidence=0.95,
            source="fake",
        )

    model = app.state.whisper_model
    if model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")

    audio_bytes = await file.read()
    return await transcribe_audio(model, audio_bytes, file.filename)
