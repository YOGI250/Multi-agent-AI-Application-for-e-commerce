# E-Commerce AI Support System

A multi-agent AI customer support system built with LangGraph, FastAPI, and LangFuse. Three specialized agents handle order tracking, product recommendations, and support ticket management, all orchestrated through an LLM-based intent router.

## Architecture

- **Intent Router** — LLM-based classifier routes requests to the correct agent
- **Order Agent** — Handles order status, tracking, and delivery queries (with Shipment Tracking subgraph)
- **Product Agent** — Handles product search and recommendations (with Product Enrichment subgraph)
- **Support Agent** — Handles complaints, refunds, and ticket creation (with Escalation Handler subgraph)

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
# Send a chat message (guest)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my order ORD-ABC123-1?"}'

# Send with session continuity
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session-123" \
  -d '{"message": "Show me budget laptops under 50000"}'
```

Or open the frontend at **http://localhost:8000/app**

## Accessing LangFuse

1. Open **http://localhost:3000**
2. Login: `admin@langfuse.com` / `password` (set via `LANGFUSE_INIT_USER_*` in `.env`)
3. Navigate to **Tracing** to see traces, spans, and generations for every request
4. Navigate to **Prompts** to manage and version agent prompts
5. Navigate to **Datasets** to view evaluation datasets and run history

## Accessing the Monitoring Dashboard

1. Open **http://localhost:3001**
2. Login: `admin` / password from `GRAFANA_ADMIN_PASSWORD` in `.env` (default: `admin`)
3. Go to **Dashboards → E-Commerce AI Support**

The dashboard shows:
- **Request Volume Over Time** — requests per minute by agent
- **Agent Selection Distribution** — pie chart of agent usage
- **Average Latency Per Agent** — response time trends
- **Error Rate** — errors over time
- **LLM Token Consumption** — token usage over time
- **Application Logs** — live logs filterable by `agent_used`, `session_id`, `request_id`, `level`

## Running Tests

```bash
source venv/bin/activate
pytest --cov=. --cov-fail-under=85 -v
```

## Running LLM Evaluation

```bash
# Mocked (CI mode)
python eval/run_eval.py

# Live (calls real LLM)
RUN_LIVE_EVAL=true python eval/run_eval.py
```

Reports are saved to `eval/reports/`.

## CI/CD Pipeline

The GitHub Actions pipeline runs on every push to `main` and on pull requests:

1. **Stage 1** — Lint (flake8) and format check (black)
2. **Stage 2** — Unit tests with ≥85% coverage
3. **Stage 3** — LLM evaluation with mocked responses
4. **Stage 4** — Docker image build
5. **Stage 5** — Push to Docker Hub (main only)
6. **Stage 6** — Deploy to Kubernetes via Helm (main only)

## Project Structure

```
├── agents/          # LangGraph agent definitions
├── api/             # FastAPI routes and schemas
├── config/          # Settings and environment config
├── database/        # SQLAlchemy models and connection
├── eval/            # LLM evaluation pipeline
├── helm/            # Kubernetes Helm chart
├── langfuse_helpers/# LangFuse tracing, scoring, evaluation
├── monitoring/      # Grafana dashboards, Prometheus, Loki
├── services/        # Mock API services
├── subgraphs/       # LangGraph subgraphs (nested agents)
├── tools/           # LangChain tool definitions
├── utils/           # Shared utilities
└── tests/           # Unit tests
```

## Environment Variables

See `.env.example` for all available configuration options.
