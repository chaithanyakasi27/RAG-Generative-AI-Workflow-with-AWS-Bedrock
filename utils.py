"""
utils.py
--------
Utility functions for document loading, chunking, and FAISS persistence
(using AWS Titan embeddings for Bedrock).
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from llm_wrapper import TitanEmbeddings


# -------- Document Helpers --------

def load_text_files(uploaded_files):
    """Load plain text files and return as list of strings."""
    docs = []
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        docs.append(text)
    return docs


def load_pdf_files(uploaded_files):
    """Extract text from PDFs using PyPDFLoader (one chunk per page)."""
    docs = []
    for file in uploaded_files:
        file_path = f"temp_{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.read())
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        for page in pages:
            docs.append(page.page_content)
        os.remove(file_path)  # cleanup temp file
    return docs


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks for embeddings.
    
    Args:
        text (str): Input text.
        chunk_size (int): Max words per chunk.
        overlap (int): Number of words overlapping between chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# -------- FAISS Persistence Helpers --------

def save_faiss_store(vectorstore, path="faiss_store"):
    """
    Save FAISS vectorstore to disk (saves index.faiss + index.pkl).
    """
    vectorstore.save_local(path)


def load_faiss_store(path="faiss_store", region="us-east-1"):
    """
    Load FAISS vectorstore from disk, or return None if not found.
    """
    embeddings = TitanEmbeddings(region_name=region)
    if os.path.exists(path):
        try:
            return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            return None
    return None


def build_or_load_faiss(docs, path="faiss_store", region="us-east-1"):
    """
    Load FAISS if exists, else build new from docs.
    
    Args:
        docs (list[str]): List of text docs/chunks.
        path (str): Directory path to FAISS index.
        region (str): AWS region for Titan embeddings.
    """
    vectorstore = load_faiss_store(path, region)
    if vectorstore is None and docs:
        vectorstore = FAISS.from_texts(docs, TitanEmbeddings(region_name=region))
        save_faiss_store(vectorstore, path)
    return vectorstore
    return None