# LangGraph Explicit Nodes (CLI + Web UI)

This app replicates the **explicit-node** approach from `langgraph_explicit_vs_react.ipynb`:
- `router_node` classifies user input into `weather` or `joke`
- conditional edge routes to `weather_node` or `joke_node`
- each terminal node writes `final_answer`

No LangChain agent/tool abstractions are used. It uses:
- `langgraph` for graph orchestration
- `openai` SDK for LLM calls
- `requests` for OpenWeather API

## Files
- `app.py`: dual-mode entrypoint
  - CLI mode for terminal chat
  - Web mode (FastAPI + browser UI)
- `core.py`: LangGraph explicit flow (router + weather/joke nodes)
- `templates/index.html`: browser UI page served at `/`
- `requirements.txt`: dependencies

## Environment
Set these in your env file:
- `OPENAI_API_KEY`
- `OPENWEATHER_API_KEY`
- optional: `OPENAI_MODEL` (default: `gpt-4o-mini`)

## Setup
```bash
cd langgraph_explicit_fastapi
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export ENV_FILE="/absolute/path/to/your/.env"
```

## Run in terminal (CLI)
```bash
python app.py --mode cli
```

Example terminal prompt:
- `What is the weather in Tokyo?`
- `Tell me a joke about Python`
- type `exit` to quit

## Run web API + UI
```bash
python app.py --mode web --host 0.0.0.0 --port 8101 --reload
```

Open in browser:
- `http://localhost:8101/` (UI page)

API endpoints:
- `GET /health`
- `POST /chat`

## Test API from terminal
```bash
curl -X POST http://localhost:8101/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the weather in Tokyo?"}'
```

```bash
curl -X POST http://localhost:8101/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Tell me a joke about Python programmers"}'
```

Health check:
```bash
curl http://localhost:8101/health
```
