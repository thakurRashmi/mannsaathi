"""
FastAPI entrypoint. Wires together config, middleware, and routes.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.core.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("mannsaathi")

app = FastAPI(
    title="MannSaathi API",
    description="Multi-agent mental wellness companion (backend).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe for Docker / Kubernetes."""
    return {
        "status": "ok",
        "service": "mannsaathi-backend",
        "llm_provider": settings.llm_provider,
    }


@app.on_event("startup")
async def on_startup() -> None:
    log.info(
        "MannSaathi backend starting up :: llm_provider=%s",
        settings.llm_provider,
    )
    if settings.llm_provider == "gemini" and not settings.google_api_key:
        log.warning(
            "LLM_PROVIDER=gemini but GOOGLE_API_KEY is empty — chat will fail"
        )
    if settings.llm_provider == "ollama":
        log.info(
            "Using Ollama at %s with model %s",
            settings.ollama_base_url,
            settings.ollama_model,
        )
