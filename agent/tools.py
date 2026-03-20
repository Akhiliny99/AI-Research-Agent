import os
from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.documents import Document

web_search_tool = TavilySearch(
    max_results=4,
    search_depth="advanced",
    include_answer=True,
    name="web_search",
    description="Search the web for current information on any topic."
)

# ── RAG Tool ─────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_documents(docs_folder: str = "docs") -> list[Document]:
    """Load all .txt and .pdf files from the docs folder."""
    documents = []

    for filename in os.listdir(docs_folder):
        filepath = os.path.join(docs_folder, filename)

        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append(Document(
                page_content=text,
                metadata={"source": filename}
            ))

        elif filename.endswith(".pdf"):
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Could not load {filename}: {e}")

    return documents


def build_vectorstore(docs_folder: str = "docs"):
    """Build FAISS vectorstore from documents in docs folder."""
    documents = load_documents(docs_folder)

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


@tool
def rag_search(query: str) -> str:
    """Search uploaded documents for relevant information using semantic similarity."""
    vectorstore = build_vectorstore()

    if vectorstore is None:
        return "No documents found in the docs folder. Please add .txt or .pdf files."

    results = vectorstore.similarity_search(query, k=4)

    if not results:
        return "No relevant information found in documents."

    output = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        output.append(f"[Source {i}: {source}]\n{doc.page_content}")

    return "\n\n".join(output)


# ── Tool list exported to agent ───────────────────────────────
tools = [web_search_tool, rag_search]