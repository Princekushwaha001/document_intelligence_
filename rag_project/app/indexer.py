import asyncio
import os
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_documents(data_folder: str) -> List[Document]:
    """
    Load all .txt and .pdf files from the given folder.
    Returns a list of LangChain Document objects.
    """
    documents: List[Document] = []

    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)

        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())
            print(f"  [indexer] Loaded: {filename}")

        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            print(f"  [indexer] Loaded: {filename}")

    return documents


async def build_vector_store(data_folder: str) -> FAISS:
    """
    Full indexing pipeline (async):
      1. Load documents from data_folder
      2. Split them into smaller chunks
      3. Embed the chunks using OpenAI 
      4. Store in a FAISS in-memory vector store

    Returns the ready FAISS vector store.
    """
    # Step 1: Load all documents (fast, local disk — sync is fine here)
    documents: List[Document] = load_documents(data_folder)

    if not documents:
        raise ValueError(f"No .txt or .pdf files found in '{data_folder}'. Please add documents.")

    # Step 2: Split documents into smaller overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks: List[Document] = splitter.split_documents(documents)
    print(f"  [indexer] Created {len(chunks)} chunks from {len(documents)} document pages.")

    # Step 3 + 4: Embed chunks and store in FAISS
    # HuggingFaceEmbeddings runs locally (no API key needed).
    # FAISS.from_documents() is blocking — asyncio.to_thread() keeps the event loop free.
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )
    vector_store: FAISS = await asyncio.to_thread(
        FAISS.from_documents, chunks, embeddings
    )
    print(f"  [indexer] FAISS vector store is ready.")

    return vector_store
