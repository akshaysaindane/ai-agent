from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

from src.agent import run_agent

app = FastAPI(title="AI Scholarship Agent")


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "AI Scholarship Agent is running!"
    }


@app.post("/chat")
def chat(data: Question):
    answer = run_agent(data.question)

    return {
        "question": data.question,
        "answer": answer
    }


app.mount("/ui", StaticFiles(directory="src/static", html=True), name="static")