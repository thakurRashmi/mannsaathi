"""
Centralized configuration. Reads from environment / .env file.

Why a settings class instead of os.getenv() everywhere:
  - Single source of truth, typed, validated at startup
  - Easy to mock in tests
  - Pydantic catches missing/malformed config before the app boots
"""
from typing import List, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM provider switch ----
    # "ollama"  -> local, privacy-first (default; works on corp networks)
    # "gemini"  -> cloud, fast (only on networks that allow Google AI APIs)
    llm_provider: Literal["ollama", "gemini"] = Field(
        default="ollama", alias="LLM_PROVIDER"
    )

    # ---- Ollama ----
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="gemma2:2b", alias="OLLAMA_MODEL")

    # ---- Gemini (only used when llm_provider == "gemini") ----
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash-exp", alias="GEMINI_MODEL")

    # ---- Server ----
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ---- CORS ----
    allowed_origins_raw: str = Field(
        default="http://localhost:3000",
        alias="ALLOWED_ORIGINS",
    )

    @property
    def allowed_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins_raw.split(",") if o.strip()]


# Singleton instance imported throughout the app.
settings = Settings()
