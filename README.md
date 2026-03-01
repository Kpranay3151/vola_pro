# Vola Pro — Tabular Data Agentic AI Pipeline

A production-grade AI pipeline that analyzes personal financial transactions using natural language. Give it a user ID and a question — it fetches, reasons, visualizes, and responds, all protected by multi-layer guardrails and accelerated by per-user caching.

## Overview

Managing personal finances shouldn't require spreadsheets or SQL. Vola Pro lets users ask plain-English questions — *"What did I spend the most on?"*, *"Am I saving money?"* — and get grounded, chart-backed answers in seconds. Built for fintech platforms, banking apps, or any product that wants to turn raw transaction data into conversational financial intelligence.

## Architecture

```
                              Vola Pro — Pipeline Flow
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                                                                          │
 │   User Prompt                                                            │
 │       │                                                                  │
 │       ▼                                                                  │
 │   ┌──────────────────┐     ┌─────────────────┐                          │
 │   │  INPUT GUARDRAILS │     │   KV CACHE       │                         │
 │   │                  │     │                 │                           │
 │   │ • Injection      │     │ • User Profile  │                           │
 │   │ • Scope Check    │     │ • Query History │                           │
 │   │ • Cross-User     │     │ • Viz State     │                           │
 │   │ • Length Limit    │     │                 │                           │
 │   └────────┬─────────┘     └────────┬────────┘                          │
 │            │                        │                                    │
 │            ▼                        ▼                                    │
 │   ┌─────────────────────────────────────────┐                           │
 │   │         CONTEXT ASSEMBLY                 │                           │
 │   │  System Prompt + Profile + Few-Shot +    │                           │
 │   │  Data Summary + User Query               │                           │
 │   └────────────────┬────────────────────────┘                           │
 │                    │                                                     │
 │                    ▼                                                     │
 │   ┌─────────────────────────────────────────┐                           │
 │   │         LLM REASONING                    │                           │
 │   │  Gemini / OpenRouter (free tier)         │                           │
 │   │  + Tool Calling (JSON schemas)           │                           │
 │   └──────┬──────────────────┬───────────────┘                           │
 │          │                  │                                            │
 │     Text Response     Tool Calls                                        │
 │          │                  │                                            │
 │          │                  ▼                                            │
 │          │         ┌────────────────┐                                   │
 │          │         │ TOOL DISPATCH   │                                   │
 │          │         │                │                                    │
 │          │         │ • Trend Line   │──── ./output/*.png                 │
 │          │         │ • Donut Chart  │                                    │
 │          │         │ • Income Bars  │                                    │
 │          │         └────────────────┘                                   │
 │          │                  │                                            │
 │          ▼                  ▼                                            │
 │   ┌─────────────────────────────────────────┐                           │
 │   │        OUTPUT GUARDRAILS                 │                           │
 │   │  • Hallucination Check (numbers + dates) │                           │
 │   │  • Toxicity Filter                       │                           │
 │   │  • Confidence Gating                     │                           │
 │   └────────────────┬────────────────────────┘                           │
 │                    │                                                     │
 │                    ▼                                                     │
 │   ┌─────────────────────────────────────────┐                           │
 │   │        STRUCTURED RESPONSE               │                           │
 │   │  { response, charts, latency, flags }    │                           │
 │   └─────────────────────────────────────────┘                           │
 │                                                                          │
 │   ── Operational Layer ──────────────────────────────────                │
 │   Token Budget | Circuit Breaker | Audit Logging | Graceful Fallback     │
 └──────────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Natural Language Querying** — Ask questions in plain English; the pipeline translates to Pandas operations and generates grounded responses.
- **Autonomous Visualization** — The LLM decides which charts to produce via tool calling: monthly trends, category breakdowns, and income vs. expense comparisons.
- **User-Specific KV Cache** — Profile, query history, and visualization state cached per user for instant, context-aware follow-ups.
- **Multi-Layer Guardrails** — Input (injection detection, scope enforcement, cross-user leakage prevention), Output (hallucination verification against real data, toxicity filtering, confidence gating), and Operational (token budgets, circuit breaker, audit logging).
- **Multi-Provider LLM** — Gemini primary + OpenRouter fallback with smart retry, exponential backoff, and model fallback chain.
- **Graceful Degradation** — If the LLM is unreachable, falls back to direct DataFrame stats. Malformed tool calls are retried with corrective hints.
- **Interactive Web UI** — Streamlit chat interface for real-time testing with user switching, inline charts, and guardrail visibility.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Data Processing | Pandas |
| Visualizations | Matplotlib (dark theme, PNG output) |
| LLM Providers | Google Gemini, OpenRouter (free tier) |
| LLM Protocol | OpenAI-compatible Chat Completions + Tool Calling |
| Web UI | Streamlit |
| Testing | Pytest (52 tests) |
| Audit | JSON-L logging with PII hashing (SHA-256) |

## Project Structure

```
vola_pro/
├── src/                          # Core pipeline modules
│   ├── pipeline.py               #   Main orchestrator (4-stage pipeline)
│   ├── llm_client.py             #   Multi-provider LLM client with retry
│   ├── guardrails.py             #   Input, Output, and Operational guardrails
│   ├── tool_registry.py          #   Tool schemas + dispatch for visualizations
│   ├── visualizations.py         #   3 chart types (trend, donut, income bars)
│   ├── cache.py                  #   User-specific KV cache manager
│   ├── context.py                #   LLM prompt builder (system + profile + few-shot)
│   ├── audit.py                  #   JSON-L audit logger (PII-safe)
│   └── utils.py                  #   Data loading, category parsing, helpers
│
├── ui/                           # Streamlit web interface
│   ├── chat.py                   #   Chat interface with inline chart rendering
│   ├── sidebar.py                #   User selector, profile card, example queries
│   ├── metadata.py               #   Latency, cache, guardrail flag badges
│   ├── session.py                #   Per-user session state management
│   └── config.py                 #   UI constants and guardrail severity map
│
├── tests/                        # Test suite (52 tests)
│   ├── test_cache.py             #   11 cache tests
│   ├── test_guardrails.py        #   23 guardrail tests
│   ├── test_pipeline_integration.py  #  9 pipeline integration tests
│   └── test_visualizations.py    #   9 visualization tests
│
├── app.py                        # Streamlit entry point
├── main.py                       # CLI demo runner (5 test queries)
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
└── .gitignore
```

## Setup & Installation

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd vola_pro
pip install -r requirements.txt
```

### 2. Configure API keys

Copy the example env file and add your key(s):

```bash
cp .env.example .env
```

Edit `.env` with at least one of:

```
OPENROUTER_API_KEY=your_openrouter_key_here
GEMINI_API_KEY=your_gemini_key_here
```

- **OpenRouter** — Get a free key at [openrouter.ai](https://openrouter.ai/)
- **Gemini** — Get a free key at [aistudio.google.com](https://aistudio.google.com/)

Either key works standalone. If both are set, Gemini is primary with OpenRouter as fallback.

### 3. Run the CLI demo

```bash
python main.py
```

This executes 5 test queries across 2 users (category breakdown, spending trend, savings analysis, prompt injection test, cross-user leakage test) and verifies cache hit behavior.

### 4. Run the Web UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Select a user, type queries, and see responses with inline charts and guardrail badges.

### 5. Run tests

```bash
pytest tests/ -v
```

Expected: 52/52 passing.
