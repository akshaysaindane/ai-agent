from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import run_agent

app = FastAPI()


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "AI Agent Running"}


@app.post("/chat")
def chat(data: Question):
    return {
        "answer": run_agent(data.question)
    }