from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)


def ask_llm(question, context=None):

    if context:
        prompt = f"""
You are an intelligent AI assistant.

Answer the user's question using the information provided in the context.

IMPORTANT RULES:
- Answer only what the user asked.
- Do not unnecessarily provide the entire scholarship information.
- If the user asks about benefits, give only the benefits.
- If the user asks about eligibility, give only eligibility.
- If the user asks about deadline, give only the deadline.
- If the context does not contain the answer, clearly say that the information is not available in the provided document.
- Do not make up scholarship information.

Context:
{context}

User Question:
{question}

Give a clear and concise answer.
"""

    else:
        prompt = f"""
You are a helpful general-purpose AI assistant.

Answer the user's question clearly and naturally.

User Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content