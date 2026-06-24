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


InterviewLanguage = Literal["en", "ms", "ta", "zh"]
TranslationSource = Literal["none", "fake", "ollama", "nim"]


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Stop message or field notes")
    type: Literal["stop_message", "field_notes"] = "stop_message"
    incident_type_name: str | None = None


class ExtractResponse(BaseModel):
    fields: dict[ExtractableField, str]
    confidence: dict[ExtractableField, float]
    source: Literal["fake", "ollama", "nim", "regex_fallback"] = "fake"


class TranscribeResponse(BaseModel):
    transcript_original: str
    transcript_english: str
    interview_language: InterviewLanguage = "en"
    confidence: float | None = None
    source: Literal["fake", "whisper"] = "fake"
    translation_source: TranslationSource = "none"

    @property
    def transcript(self) -> str:
        return self.transcript_english


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    fake_extraction: bool
    fake_transcription: bool
    fake_photo_analysis: bool
    llm_base_url: str
    llm_model: str
    vision_model: str
    whisper_model: str
    whisper_device: str


QuestionCoverageStatus = Literal["answered", "partial", "unanswered", "unclear"]


class InterviewQuestion(BaseModel):
    id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    hint: str | None = None
    section: str | None = None


class AnalyzeInterviewRequest(BaseModel):
    transcript: str = Field(..., min_length=1)
    questions: list[InterviewQuestion] = Field(..., min_length=1)
    interview_language: InterviewLanguage = "en"


class QuestionCoverage(BaseModel):
    id: str
    status: QuestionCoverageStatus
    evidence: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)


class FollowUpSuggestion(BaseModel):
    related_question_id: str | None = None
    prompt: str
    prompt_conduct: str
    reason: str


class AnalyzeInterviewResponse(BaseModel):
    coverage: list[QuestionCoverage]
    follow_ups: list[FollowUpSuggestion]
    source: Literal["fake", "ollama", "nim"] = "fake"


SuggestedPhotoSection = Literal[
    "incident",
    "damages",
    "area_of_origin",
    "burn_patterns",
    "evidentiary",
]

PhotoAnalysisSource = Literal["fake", "ollama", "nim"]


class PhotoAnalysisConfidence(BaseModel):
    caption: float = Field(..., ge=0.0, le=1.0)
    suggested_section: float | None = Field(default=None, ge=0.0, le=1.0)


class SectionCandidate(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str | None = None


class SectionCandidates(BaseModel):
    incident: SectionCandidate | None = None
    damages: SectionCandidate | None = None
    area_of_origin: SectionCandidate | None = None
    burn_patterns: SectionCandidate | None = None
    evidentiary: SectionCandidate | None = None


class AnalyzePhotoResponse(BaseModel):
    caption: str
    detected_elements: list[str] = Field(default_factory=list)
    suggested_section: SuggestedPhotoSection | None = None
    section_candidates: SectionCandidates | None = None
    confidence: PhotoAnalysisConfidence
    source: PhotoAnalysisSource = "fake"


class AnalyzePhotoContext(BaseModel):
    location_of_fire: str | None = None
    incident_type_name: str | None = None
    stop_message_excerpt: str | None = None
    field_notes_excerpt: str | None = None
