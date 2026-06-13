# MannSaathi (मन साथी)

A multi-agent AI mental wellness companion. **Not** a therapist replacement — a safe, structured "0-to-1" support layer for people who need to talk it out.

## What it does

- Conversational support powered by specialized AI agents (Listener, Reflector, Resource-router)
- Safety-critical crisis detection that routes high-risk users to human helplines instantly
- Hybrid architecture: deterministic rules guarantee crisis paths cannot be overridden by LLM hallucinations

## Tech stack

- **Frontend:** Next.js 14 + Tailwind CSS
- **Backend:** FastAPI (Python 3.11)
- **AI orchestration:** LangGraph + LangChain
- **LLM:** Google Gemini 2.0 Flash (free tier)
- **State:** SQLite (session) + pgvector (long-term memory, later)
- **Deploy:** Docker Compose locally; Vercel + Railway for prod

## Local development

```bash
# 1. Copy env file and add your Gemini API key
cp .env.example .env
# edit .env and paste GOOGLE_API_KEY

# 2. Bring everything up
docker compose up --build

# 3. Open the app
open http://localhost:3000
```

Backend runs on `:8000`, frontend on `:3000`.

## Project structure

```
mannsaathi/
├── backend/              # FastAPI + LangGraph
│   ├── app/
│   │   ├── agents/       # Listener, Reflector, Crisis Detector
│   │   ├── api/          # FastAPI routes
│   │   ├── core/         # Config, logging, safety rules
│   │   └── db/           # Session storage
│   └── tests/            # Eval harness + unit tests
├── frontend/             # Next.js app
└── docker-compose.yml
```

## Safety notice

MannSaathi is an AI companion, not medical advice or therapy. In crisis, call:

- **iCall:** 9152987821
- **Vandrevala Foundation:** 1860-2662-345
- **AASRA:** 9820466726
