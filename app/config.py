from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    use_fake_extraction: bool = True
    use_fake_transcription: bool = True

    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "NIM_LLM_BASE_URL"),
    )
    llm_model: str = Field(
        default="llama3.1:8b",
        validation_alias=AliasChoices("LLM_MODEL", "NIM_LLM_MODEL"),
    )
    llm_api_key: str = Field(
        default="ollama",
        validation_alias=AliasChoices("LLM_API_KEY", "NIM_API_KEY"),
    )

    whisper_model: str = "base"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_language: str = "en"
    whisper_initial_prompt: str = ""


settings = Settings()
