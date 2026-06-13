"""
LLM provider abstraction.

Why this file exists:
  Every agent in this app should get its LLM through `get_chat_model()`,
  not by directly instantiating ChatOllama / ChatGoogleGenerativeAI.
  That way, switching providers (e.g. Ollama -> Gemini for prod) is a
  one-line config change, not a code-wide refactor.

  This is a small example of the Dependency Inversion principle:
  agents depend on the abstract "chat model" interface, not the concrete
  provider implementation.
"""
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from app.core.config import settings


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """
    Returns a configured chat model based on LLM_PROVIDER env var.

    Cached so we don't re-construct the client (and its HTTP connection
    pool) on every request.
    """
    provider = settings.llm_provider

    if provider == "ollama":
        # Local LLM via Ollama daemon. No API key, no network call,
        # data never leaves the host. Ideal for sensitive use cases.
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.7,  # warm but coherent — not too creative
            # Keep responses bounded so the chat stays snappy.
            num_predict=400,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.google_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=gemini but GOOGLE_API_KEY is empty. "
                "Set it in .env or switch LLM_PROVIDER to 'ollama'."
            )
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.7,
            max_output_tokens=400,
        )

    raise RuntimeError(f"Unknown LLM_PROVIDER: {provider!r}")
