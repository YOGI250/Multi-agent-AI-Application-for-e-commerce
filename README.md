# E-Commerce AI Support System

A multi-agent AI customer support system built with LangGraph, FastAPI, and LangFuse. Three specialized agents handle order tracking, product recommendations, and support ticket management, all orchestrated through an LLM-based intent router.

## Architecture

- **Intent Router** — LLM-based classifier routes each request to the correct agent
- **Order Agent** — Handles order status, tracking, and delivery queries (with Shipment Tracking subgraph)
- **Product Agent** — Handles product search and recommendations (with Product Enrichment subgraph)
- **Support Agent** — Handles complaints, refunds, and ticket creation (with Escalation Handler subgraph)

Each agent is a LangGraph `StateGraph` with 6–8 nodes (validate → reason → tool call → subgraph → format → respond) connected by conditional edges. The LLM only runs in reasoning nodes; all other nodes are deterministic.

## Prerequisites

- Docker and Docker Compose
- Python 3.12 (for local dev)
- Groq API key (free at [console.groq.com](https://console.groq.com))

## Setup

### 1. Clone and configure environment

```bash
git clone https://github.com/YOGI250/Multi-agent-AI-Application-for-e-commerce.git
cd Multi-agent-AI-Application-for-e-commerce
cp .env.example .env
```

Edit `.env` and fill in your values:

```bash
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_CLIENT_ID=your_google_oauth_client_id
GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret
```

All other values have working defaults for local development.

### 2. Start the full stack

```bash
docker compose up -d
```

This brings up:

| Service | URL |
|---|---|
| FastAPI app | http://localhost:8000 |
| FastAPI docs | http://localhost:8000/docs |
| LangFuse UI | http://localhost:3000 |
| Grafana dashboard | http://localhost:3001 |
| Prometheus metrics | http://localhost:9090 |

### 3. Seed the database (first time only)

```bash
docker exec ecommerce-fastapi python database/import_kaggle_data.py
```

### 4. Seed LangFuse (first time only)

```bash
source venv/bin/activate

# Push versioned prompts for all 3 agents
python eval/seed_prompts.py

# Create the evaluation dataset in LangFuse
python eval/seed_dataset.py

# Register LLM-as-judge evaluators
python eval/setup_langfuse_evaluators.py
```

## Running the Application

### Via Docker Compose (recommended)

```bash
docker compose up -d        # start
docker compose logs -f      # follow logs
docker compose down         # stop
```

### Local development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Using the Chat API

```bash
# Send a chat message (guest session)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my order ORD-ABC123-1?"}'

# Send with session continuity (conversation history is preserved)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session-123" \
  -d '{"message": "Show me budget laptops under 50000"}'
```

Or open the frontend at **http://localhost:8000/app**

## Session Management

- Pass `X-Session-ID` in the request header to maintain conversation continuity across requests
- The session ID is returned in every response body so clients can persist it
- Sessions expire after **30 minutes** of inactivity (configurable via `SESSION_EXPIRY_MINUTES` in `.env`)
- Conversation history, session metadata, and LangFuse trace IDs are stored in the `users`, `sessions`, and `messages` database tables

## Authentication

Google OAuth 2.0 is supported. The frontend sends a Google ID token in the `Authorization: Bearer <token>` header. The API validates it via `google.oauth2.id_token.verify_oauth2_token()` and resolves or creates the user record. Guest access (no token) is also supported.

## Accessing LangFuse

1. Open **http://localhost:3000**
2. Login: `admin@langfuse.com` / `password` (set via `LANGFUSE_INIT_USER_*` in `.env`)
3. **Tracing** — every request creates a trace with spans for each agent node, tool calls, and LLM generations. Token usage and cost are recorded at trace level.
4. **Prompts** — all agent system prompts are versioned here. Update a prompt version in LangFuse without redeploying code.
5. **Datasets** — `ecommerce-eval-dataset` contains evaluation test cases. Each eval run adds a new experiment entry.
6. **Scores** — every response is scored for `answer_relevancy`, `task_completion`, and `correctness` automatically after each request.

## Accessing the Monitoring Dashboard

1. Open **http://localhost:3001**
2. Login: `admin` / password from `GRAFANA_ADMIN_PASSWORD` in `.env` (default: `admin`)
3. Go to **Dashboards → E-Commerce AI Support**

The dashboard shows:

| Panel | What it shows |
|---|---|
| Request Volume Over Time | Requests per minute by agent |
| Agent Selection Distribution | Pie chart of agent usage |
| Average Latency Per Agent | Response time trends |
| Error Rate | Errors over time |
| LLM Token Consumption | Input/output token usage over time |
| Application Logs | Live logs from Loki, filterable by `agent_used`, `session_id`, `request_id`, `level` |

### Prometheus Metrics

The app exposes these custom metrics at `/metrics`:

| Metric | Type | Labels |
|---|---|---|
| `chat_requests_total` | Counter | `agent_used`, `status` |
| `chat_request_latency_seconds` | Histogram | `agent_used` |
| `llm_tokens_total` | Counter | `agent_used`, `token_type` |
| `chat_errors_total` | Counter | `error_type`, `agent_used` |

### Log Ingestion

In Docker Compose, Promtail ships container logs to Loki. In Kubernetes (Helm deploy), Promtail runs as a DaemonSet collecting pod logs from `/var/log/pods/`. JSON log fields (`level`, `agent_used`, `session_id`, `request_id`) are extracted as Loki labels for filtering.

## Running Tests

```bash
source venv/bin/activate
pytest --cov=. --cov-fail-under=85 -v
```

## Running LLM Evaluation

```bash
# Mocked (CI mode — no real LLM calls)
python eval/run_eval.py

# Live (calls real LLM via FastAPI)
RUN_LIVE_EVAL=true python eval/run_eval.py
```

Reports are saved to `eval/reports/` and pushed to LangFuse under `ecommerce-eval-dataset`.

## CI/CD Pipeline

The GitHub Actions pipeline runs on every push to `main`:

| Stage | Runner | What it does |
|---|---|---|
| Stage 1 — Lint & Format | ubuntu-latest | flake8 + black --check |
| Stage 2 — Unit Tests | ubuntu-latest | pytest with ≥85% coverage |
| Stage 3 — LLM Evaluation | ubuntu-latest | Mocked eval, uploads report as artifact |
| Stage 4 — Docker Build | ubuntu-latest | Builds image (no push) |
| Stage 5 — Push to Registry | ubuntu-latest | Pushes to Docker Hub (`main` only) |
| Stage 6 — Deploy to Kubernetes | self-hosted | Helm upgrade, smoke test `/health`, auto-rollback on failure |

Stage 6 runs on a self-hosted runner on the local machine where minikube is running. On smoke test failure, `helm rollback` is triggered automatically.

## Project Structure

```
├── agents/              # LangGraph agent definitions (intent router + 3 agents)
├── api/                 # FastAPI routes, schemas, session management
├── config/              # Pydantic settings, all config from .env
├── database/            # SQLAlchemy models, connection, data import scripts
├── eval/                # LLM evaluation pipeline, dataset, prompt seeding scripts
├── helm/                # Kubernetes Helm chart (deployment, Loki, Promtail, Grafana)
├── langfuse_helpers/    # Tracing, scoring, and evaluation helpers
├── monitoring/          # Prometheus config, Grafana dashboards, Loki/Promtail configs
├── services/            # Mock APIs for orders, products, and support
├── subgraphs/           # Nested LangGraph subgraphs (shipment, enrichment, escalation)
├── tests/               # Unit tests
├── tools/               # LangChain tool definitions with LangFuse span instrumentation
└── utils/               # Shared utilities (session memory, LangFuse context propagation)
```

## Environment Variables

See `.env.example` for all available configuration options.
