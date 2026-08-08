from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq


load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)


def ask_llm(question):
    response = llm.invoke(question)
    return response.content