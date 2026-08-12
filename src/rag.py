import re
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


doc_data = None


def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def load_documents():
    global doc_data

    if doc_data is not None:
        return doc_data

    pdf_path = Path(__file__).resolve().parent.parent / "documents" / "scholarship_info.pdf"

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    data = []

    for doc in docs:
        words = tokenize(doc.page_content)
        word_counts = Counter(words)

        data.append({
            "text": doc.page_content,
            "words": word_counts
        })

    doc_data = data

    return doc_data


def search_pdf(question):
    documents = load_documents()

    question_words = set(tokenize(question))

    if not question_words:
        return "No relevant information found."

    scored_docs = []

    for item in documents:
        score = sum(
            count
            for word, count in item["words"].items()
            if word in question_words
        )

        if score > 0:
            scored_docs.append((score, item["text"]))

    scored_docs.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    if not scored_docs:
        return "No relevant information found."

    return "\n\n".join(
        text for score, text in scored_docs[:3]
    )