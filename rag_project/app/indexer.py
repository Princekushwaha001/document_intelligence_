
"""
indexer.py
----------
Document loading and vector store indexing pipeline.

Responsibility:
    1. Scan the data/ folder and load every .txt and .pdf file.
    2. Split the loaded text into small overlapping chunks so that
       the retriever can find precise, relevant passages.
    3. Convert each chunk into a numeric vector (embedding) using a
       local HuggingFace model — no API key or internet call required.
    4. Store all vectors in a FAISS in-memory index that can be
       searched by similarity at query time.

Public functions:
    load_documents(data_folder)  — load raw documents from disk
    build_vector_store(data_folder) — full pipeline, returns FAISS index
"""


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
    Scan *data_folder* and load every supported file into LangChain Documents.

    Supported formats:
        - .txt  — loaded with TextLoader (UTF-8 encoding)
        - .pdf  — loaded with PyPDFLoader (one Document per page)

    Any other file extension is silently skipped.

    Args:
        data_folder (str): Path to the folder that contains the source files,
                           e.g. "data".

    Returns:
        List[Document]: A flat list of LangChain Document objects.
                        Each Document has a .page_content string and
                        a .metadata dict that includes the 'source' filename.

    Example:
        >>> docs = load_documents("data")
        >>> print(docs[0].metadata["source"])
        data/company_handbook.txt
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
    Run the full indexing pipeline and return a ready-to-query FAISS index.

    Pipeline steps:
        1. load_documents()   — read .txt / .pdf files from data_folder
        2. Split              — RecursiveCharacterTextSplitter
                                (chunk_size=500, chunk_overlap=50)
        3. Embed              — HuggingFace 'all-MiniLM-L6-v2' (local, free)
        4. Index              — FAISS.from_documents() stored in RAM only

    The resulting index lives entirely in memory. It is rebuilt every time
    the server starts. Nothing is written to disk.

    Args:
        data_folder (str): Path to the folder containing source documents,
                           e.g. "data".

    Returns:
        FAISS: An in-memory FAISS vector store ready for similarity search.

    Raises:
        ValueError: If no supported documents are found in data_folder.

    Example:
        >>> vector_store = await build_vector_store("data")
        >>> # vector_store is now ready to use as a retriever
    """
    # Step 1 — Load raw documents from disk
    documents: List[Document] = load_documents(data_folder)

    if not documents:
        raise ValueError(f"No .txt or .pdf files found in '{data_folder}'. Please add documents.")

    # Step 2 — Split into small overlapping chunks
    # chunk_size=500  : each chunk is at most 500 characters
    # chunk_overlap=50: consecutive chunks share 50 characters so context
    #                   is not lost at chunk boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks: List[Document] = splitter.split_documents(documents)
    print(f"  [indexer] Created {len(chunks)} chunks from {len(documents)} document pages.")

    # Step 3 + 4 — Embed and index
    # HuggingFace 'all-MiniLM-L6-v2': small (80 MB), fast, no API key needed.
    # asyncio.to_thread() prevents the blocking FAISS call from freezing the
    # async event loop during server startup.
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )
    vector_store: FAISS = await asyncio.to_thread(
        FAISS.from_documents, chunks, embeddings
    )
    print(f"  [indexer] FAISS vector store is ready.")

    return vector_store
