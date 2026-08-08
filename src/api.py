from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.agent import run_agent

app = FastAPI(title="AI Scholarship Agent")


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse("src/static/index.html")


@app.post("/chat")
def chat(data: Question):
    answer = run_agent(data.question)

    return {
        "question": data.question,
        "answer": answer
    }