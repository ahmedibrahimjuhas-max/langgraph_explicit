import os
from pathlib import Path
from typing import Literal, Optional, TypedDict

import requests
from dotenv import load_dotenv
from openai import OpenAI

from langgraph.graph import END, StateGraph


BASE_DIR = Path(__file__).resolve().parent


def load_environment() -> None:
    env_file = os.getenv("ENV_FILE")
    if env_file:
        load_dotenv(env_file, override=False)
        return
    load_dotenv(BASE_DIR / ".env", override=False)
    load_dotenv(override=False)


load_environment()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Set it in ENV_FILE or environment.")
if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY not found. Set it in ENV_FILE or environment.")

client = OpenAI(api_key=OPENAI_API_KEY)


class ExplicitState(TypedDict):
    user_input: str
    intent: Literal["weather", "joke"]
    city: Optional[str]
    topic: Optional[str]
    final_answer: Optional[str]


def llm_text(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    result = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (result.choices[0].message.content or "").strip()


def router_node(state: ExplicitState) -> ExplicitState:
    classification = llm_text(
        system_prompt=(
            "Classify the user message into weather or joke. "
            "Return exactly these labeled lines and nothing else."
        ),
        user_prompt=(
            "Format:\n"
            "INTENT: weather|joke\n"
            "CITY: <city or empty>\n"
            "TOPIC: <topic or general>\n\n"
            f"Message: {state['user_input']}"
        ),
        temperature=0,
    )

    intent: Literal["weather", "joke"] = "joke"
    city: Optional[str] = None
    topic: Optional[str] = "general"

    for line in classification.splitlines():
        text = line.strip()
        if text.startswith("INTENT:"):
            parsed = text.split(":", 1)[1].strip().lower()
            intent = "weather" if parsed == "weather" else "joke"
        elif text.startswith("CITY:"):
            parsed = text.split(":", 1)[1].strip()
            city = parsed or None
        elif text.startswith("TOPIC:"):
            parsed = text.split(":", 1)[1].strip()
            topic = parsed or "general"

    return {**state, "intent": intent, "city": city, "topic": topic}


def weather_node(state: ExplicitState) -> ExplicitState:
    city = (state.get("city") or "").strip()
    if not city:
        return {
            **state,
            "final_answer": "Please include a city so I can check the weather.",
        }

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
        timeout=20,
    )
    payload = response.json()

    if response.status_code != 200:
        details = payload.get("message", "Unknown error") if isinstance(payload, dict) else str(payload)
        return {
            **state,
            "final_answer": f"I could not fetch weather for '{city}'. API returned: {details}.",
        }

    summary = (
        f"{city}: {payload['weather'][0]['description']}, "
        f"{payload['main']['temp']} deg C, humidity {payload['main']['humidity']}%."
    )

    answer = llm_text(
        system_prompt="You are a concise assistant. Use the provided weather summary only.",
        user_prompt=(
            f"User asked: {state['user_input']}\n"
            f"Weather summary: {summary}\n"
            "Write a short friendly answer."
        ),
        temperature=0.3,
    )
    return {**state, "final_answer": answer}


def joke_node(state: ExplicitState) -> ExplicitState:
    topic = (state.get("topic") or "general").strip()
    answer = llm_text(
        system_prompt="Tell one short, clean joke.",
        user_prompt=f"Topic: {topic}",
        temperature=0.8,
    )
    return {**state, "final_answer": answer}


def route_decision(state: ExplicitState) -> str:
    return "weather_node" if state["intent"] == "weather" else "joke_node"


builder = StateGraph(ExplicitState)
builder.add_node("router_node", router_node)
builder.add_node("weather_node", weather_node)
builder.add_node("joke_node", joke_node)
builder.set_entry_point("router_node")
builder.add_conditional_edges(
    "router_node",
    route_decision,
    {"weather_node": "weather_node", "joke_node": "joke_node"},
)
builder.add_edge("weather_node", END)
builder.add_edge("joke_node", END)
graph = builder.compile()
