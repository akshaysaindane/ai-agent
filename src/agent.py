from src.rag import search_pdf
from src.llm import ask_llm


def run_agent(question):

    keywords = [
        "scholarship",
        "document",
        "eligibility",
        "income",
        "benefit",
        "benefits",
        "deadline",
        "aadhaar",
        "bank",
        "marks",
        "amount",
        "apply",
        "application",
        "criteria",
    ]

    question_lower = question.lower()

    if any(word in question_lower for word in keywords):

        print("📄 Using RAG Tool...")

        context = search_pdf(question)

        if context == "No relevant information found.":
            return ask_llm(question)

        print("🤖 Using Groq LLM with RAG context...")

        return ask_llm(question, context)

    print("🤖 Using Groq LLM...")

    return ask_llm(question)