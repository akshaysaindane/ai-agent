from src.rag import search_pdf
from src.llm import ask_llm


def run_agent(question):
    keywords = [
        "scholarship",
        "document",
        "eligibility",
        "income",
        "benefit",
        "deadline",
        "aadhaar",
        "bank",
        "marks",
    ]

    if any(word in question.lower() for word in keywords):
        print("📄 Using RAG Tool...")
        return search_pdf(question)

    print("🤖 Using Groq LLM...")
    return ask_llm(question)