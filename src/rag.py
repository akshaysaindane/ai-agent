from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


loader = PyPDFLoader("documents/scholarship_info.pdf")
documents = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


db = FAISS.from_documents(docs, embedding)


retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


def search_pdf(question):
    results = retriever.invoke(question)

    if not results:
        return "No relevant information found."

    return "\n\n".join(
        doc.page_content for doc in results
    )