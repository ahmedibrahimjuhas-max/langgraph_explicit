import argparse
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    from .core import graph
except ImportError:
    from core import graph


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    method: Literal["explicit"]
    intent: Literal["weather", "joke"]
    answer: str
    city: Optional[str] = None
    topic: Optional[str] = None


app = FastAPI(title="LangGraph Explicit Nodes API", version="1.0.0")
BASE_DIR = Path(__file__).resolve().parent
UI_FILE = BASE_DIR / "templates" / "index.html"


def run_graph(question: str) -> ChatResponse:
    cleaned = question.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = graph.invoke(
        {
            "user_input": cleaned,
            "intent": "joke",
            "city": None,
            "topic": None,
            "final_answer": None,
        }
    )

    return ChatResponse(
        method="explicit",
        intent=result["intent"],
        answer=result.get("final_answer") or "No answer generated.",
        city=result.get("city"),
        topic=result.get("topic"),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    if not UI_FILE.exists():
        raise HTTPException(status_code=500, detail=f"UI file not found: {UI_FILE}")
    return UI_FILE.read_text(encoding="utf-8")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    return run_graph(payload.question)


def run_cli() -> None:
    print("LangGraph Explicit Chat (type 'exit' to quit)")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        if not user_input:
            continue

        try:
            result = run_graph(user_input)
            print(f"Intent: {result.intent}")
            if result.city:
                print(f"City: {result.city}")
            if result.topic:
                print(f"Topic: {result.topic}")
            print(f"Assistant: {result.answer}")
        except HTTPException as error:
            print(f"Error: {error.detail}")
        except Exception as error:
            print(f"Error: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph explicit app")
    parser.add_argument(
        "--mode",
        choices=["cli", "web"],
        default="cli",
        help="Run interactive terminal mode or web server mode.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Web host for --mode web.")
    parser.add_argument("--port", type=int, default=8101, help="Web port for --mode web.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload in web mode.",
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    else:
        uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
