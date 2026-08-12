import re
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load PDF
loader = PyPDFLoader("documents/scholarship_info.pdf")
documents = loader.load()


# Split PDF into smaller sections
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)


def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


# Prepare searchable text
doc_data = []

for doc in docs:
    words = tokenize(doc.page_content)
    word_counts = Counter(words)

    doc_data.append({
        "text": doc.page_content,
        "words": word_counts
    })


def search_pdf(question):
    question_words = set(tokenize(question))

    if not question_words:
        return "No relevant information found."

    scored_docs = []

    for item in doc_data:
        score = sum(
            count
            for word, count in item["words"].items()
            if word in question_words
        )

        if score > 0:
            scored_docs.append((score, item["text"]))

    # Highest matching sections first
    scored_docs.sort(reverse=True, key=lambda x: x[0])

    if not scored_docs:
        return "No relevant information found."

    # Return top 3 results
    return "\n\n".join(
        text for score, text in scored_docs[:3]
    )