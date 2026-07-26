from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("documents/scholarship_info.pdf")

documents = loader.load()

print("Total Pages:", len(documents))
print("\nFirst Page:\n")
print(documents[0].page_content)