from typing import Literal

from pydantic import BaseModel, Field


ExtractableField = Literal[
    "applianceCallSign",
    "locationOfFire",
    "fireInvolved",
    "methodOfExtinguishment",
    "damagesSustained",
    "probableCause",
    "ignitionSource",
    "ignitionFuel",
    "eventsCircumstances",
    "areaOfFireOrigin",
    "classification",
    "handoverOfficer",
    "handoverNpc",
]


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Stop message or field notes")
    type: Literal["stop_message", "field_notes"] = "stop_message"
    incident_type_name: str | None = None


class ExtractResponse(BaseModel):
    fields: dict[ExtractableField, str]
    confidence: dict[ExtractableField, float]
    source: Literal["fake", "ollama", "nim", "regex_fallback"] = "fake"


class TranscribeResponse(BaseModel):
    transcript: str
    confidence: float | None = None
    source: Literal["fake", "whisper"] = "fake"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    fake_extraction: bool
    fake_transcription: bool
    llm_base_url: str
    llm_model: str
    whisper_model: str
    whisper_device: str
