# MannSaathi (मन साथी)

> Multi-agent AI mental wellness companion. Runs Gemma 2B locally via Ollama. Hybrid rules + LLM crisis-detection gate routes users to real helplines.

**Status:** ✅ Working MVP. Local development on Docker.

---

## Why this exists

In India, roughly 1 in 7 people experiences a mental health issue — and only a fraction get any kind of support. The gap isn't just access to therapists. It's the much bigger group of people who never seek help at all, often because *"is this even worth talking about?"*

MannSaathi is built for that middle space:
- A calm listener for venting, processing a rough day, or feeling lonely
- A pattern-noticing partner when you want to go deeper (*"why does this keep happening to me?"*)
- A safety layer that recognizes real crisis language and routes users to vetted human helplines

It's intentionally narrow: low-friction, available at 2am, on-device. For anything clinical, it surfaces the right phone numbers.

---

## What's interesting (engineering-wise)

### 🛡️ Safety-critical hybrid crisis detection

The hardest engineering problem here is **not** the AI — it's making sure the AI can never cause harm. The crisis-detection gate uses two parallel layers, OR-merged:

```
        user message
             ▼
    ┌────────┴────────┐
    │                 │
[rules layer]    [LLM layer]
 50+ regexes     structured JSON
 <1ms latency    25s hard timeout
 zero-FN bias    catches euphemisms
    │                 │
    └────────┬────────┘
             ▼
        OR-merge
             ▼
   if crisis → hard-coded helpline text  (NO LLM in the user-facing reply)
   if safe   → continue to specialized agents
```

**Architectural rules of the safety system:**
1. **OR-gate, not AND-gate** — either layer firing routes to helpline
2. **Rules run first, in microseconds** — short-circuits the LLM call when obvious
3. **The user-facing crisis text is never LLM-generated** — it's vetted, hard-coded
4. **LLM failures fail-safe to "not crisis"** — rules layer is the safety floor
5. **`is_crisis: False` is never overridden by the LLM if rules said `True`**

### 🤖 LangGraph multi-agent orchestration

When the gate decides it's safe to talk, a LangGraph state graph routes the message to one of three specialized agents:

```
                  user_message
                       ▼
                  ┌─────────────┐
                  │ crisis_gate │
                  └──────┬──────┘
                         │ safe?
                         ▼
                  ┌─────────────┐
                  │   triage    │  (structured JSON classifier)
                  └─┬─────┬─────┘
                    │     │     │
                    ▼     ▼     ▼
              [listener] [reflector] [advice_redirect]
                    │     │     │
                    └─────┼─────┘
                          ▼
                         END
```

| Agent | When it fires | Style |
|---|---|---|
| **Listener** | Venting, sharing feelings | Short, reflective, one gentle question |
| **Reflector** | "Why does this always happen?" | Notices patterns, tentative language ("I wonder if…") |
| **Advice-redirect** | "What medication should I take?" | Honest scope statement, warm redirect |

Each agent has one focused prompt — not a mega-prompt trying to do everything.

### 🔒 Privacy by design — local LLM inference

The LLM runs **entirely on your machine** via Ollama (Gemma 2B). Conversations never leave the device. This isn't just a clever workaround for corporate firewalls — it's the right design for emotional disclosures.

The codebase is provider-agnostic: a single env var (`LLM_PROVIDER=ollama|gemini`) swaps backends. The architecture survives the change.

### 📊 Eval harness with zero-FN gating

A ~140-case labeled dataset (`tests/eval/cases.jsonl`) with six categories — direct crisis, euphemism, figurative, normal distress, discussion, casual. Pytest gates merges on **zero false-negatives** for the rules layer.

```bash
# Fast rules-only eval (every commit, in CI):
python -m tests.eval.runner --mode rules_only

# Slow full-gate eval (rules + LLM; before releases):
RUN_FULL_GATE_EVAL=1 pytest tests/test_crisis_eval.py
```

Sample output:

```
=========================================================================
 MannSaathi crisis-detection eval — mode: rules_only
=========================================================================
  total cases       : 140
  duration          : 0.00s (avg 0.0 ms / case)

  TP (real crisis caught)   : 40
  TN (safe correctly safe)  : 75
  FP (safe flagged crisis)  : 0
  FN (CRISIS MISSED)        : 25   (all euphemisms — LLM layer's job)

  precision : 100.00%
  recall    :  61.54%
  accuracy  :  82.14%
```

(Full-gate metrics — including how the LLM layer closes the euphemism gap — generated in the `docs/eval/` directory.)

---

## Tech stack

- **Frontend:** Next.js 14 + TailwindCSS
- **Backend:** FastAPI (Python 3.11), async-first
- **AI orchestration:** LangGraph + LangChain
- **LLM:** Google Gemma 2B (local via Ollama; Gemini optional)
- **State:** TypedDict graph state (LangGraph); SQLite session memory planned
- **Deploy:** Docker Compose locally; Vercel + Railway for prod (planned)

---

## Local development

### Prerequisites

- Docker Desktop
- [Ollama](https://ollama.com) running on the host with `gemma2:2b` pulled:
  ```bash
  brew install ollama
  brew services start ollama
  ollama pull gemma2:2b
  ```
- Ollama listening on all interfaces (so the backend container can reach it):
  ```bash
  launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
  brew services restart ollama
  ```

### Run

```bash
git clone <repo>
cd mannsaathi
cp .env.example .env   # defaults work; no API key needed for Ollama
docker compose up --build
open http://localhost:3000
```

### Tests

```bash
# Unit tests (53 rules-layer tests):
docker exec mannsaathi-backend python3 -m pytest tests/test_crisis_rules.py

# Eval harness (rules-only — fast):
docker exec mannsaathi-backend python3 -m pytest tests/test_crisis_eval.py

# Full-gate eval (slow, needs Ollama up):
docker exec mannsaathi-backend env RUN_FULL_GATE_EVAL=1 \
    python3 -m pytest tests/test_crisis_eval.py -v
```

---

## Safety notice

MannSaathi is an AI companion, not medical advice or therapy. In crisis, call:

- 📞 **iCall** (free, confidential): **9152987821**
- 📞 **Vandrevala Foundation** (24/7): **1860-2662-345**
- 📞 **AASRA** (24/7): **9820466726**
- 📞 **Emergency**: **112** (India)

---

## Project structure

```
mannsaathi/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── crisis_rules.py   # deterministic safety layer
│   │   │   ├── crisis_llm.py     # LLM safety classifier
│   │   │   ├── crisis_gate.py    # OR-merge orchestrator
│   │   │   ├── triage.py         # routes safe messages to one of three agents
│   │   │   ├── listener.py       # default — short reflective listening
│   │   │   ├── reflector.py      # deeper-probe agent
│   │   │   ├── advice_redirect.py
│   │   │   ├── graph.py          # the LangGraph wiring all of the above
│   │   │   ├── state.py          # typed GraphState
│   │   │   └── llm.py            # provider-agnostic chat-model factory
│   │   ├── api/chat.py
│   │   ├── core/config.py
│   │   └── main.py
│   └── tests/
│       ├── test_crisis_rules.py   # 53 unit tests on the safety floor
│       ├── test_crisis_eval.py    # eval harness pytest integration
│       └── eval/
│           ├── cases.jsonl        # 140 labeled cases, 6 categories
│           ├── runner.py          # eval runner with precision/recall/F1
│           └── report_markdown.py # render eval as Markdown table
├── frontend/                      # Next.js app
└── docker-compose.yml
```
