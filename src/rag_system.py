"""
RAG System — FAISS vector store built from rag_manual.json.

Uses sentence-transformers for embeddings (no API key needed locally).
Falls back to keyword search if FAISS is unavailable.

Usage:
    from src.rag_system import RAGSystem
    rag = RAGSystem()
    context = rag.retrieve_context("How to handle class imbalance?", k=3)
"""

import json
from pathlib import Path
from typing import List, Optional


RAG_MANUAL_PATH = Path(__file__).parent.parent / "rag_manual.json"


class RAGSystem:
    def __init__(self, rag_manual_path: Optional[str] = None):
        path = rag_manual_path or str(RAG_MANUAL_PATH)
        self.rag_manual = self._load_manual(path)
        self.documents = self._build_documents()
        self._vectorstore = None
        self._embeddings = None
        self._build_vectorstore()

    # ── Public API ──────────────────────────────────────────────────────────────

    def retrieve_context(self, query: str, k: int = 3) -> List[str]:
        if self._vectorstore is not None:
            return self._faiss_search(query, k)
        return self._keyword_search(query, k)

    def get_all_nodes(self):
        return self.rag_manual["tree"]["nodes"]

    # ── Internal ────────────────────────────────────────────────────────────────

    def _load_manual(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_documents(self) -> List[str]:
        docs = []
        for node in self.rag_manual["tree"]["nodes"]:
            doc = (
                f"Question: {node['question']}\n"
                f"Answer: {node['answer']}\n"
                f"Tags: {', '.join(node.get('tags', []))}\n"
                f"Code: {node.get('code_snippet', '')}\n"
                f"Metrics Impact: {node.get('metrics_impact', '')}"
            )
            docs.append(doc)
        return docs

    def _build_vectorstore(self):
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = splitter.create_documents(self.documents)
            self._vectorstore = FAISS.from_documents(texts, self._embeddings)
        except Exception as e:
            # Fallback to keyword search — no FAISS dependency required
            print(f"[RAGSystem] FAISS unavailable ({e}), using keyword search fallback.")
            self._vectorstore = None

    def _faiss_search(self, query: str, k: int) -> List[str]:
        docs = self._vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def _keyword_search(self, query: str, k: int) -> List[str]:
        """Simple TF-style keyword overlap search as fallback."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            score = len(query_words & doc_words)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]
