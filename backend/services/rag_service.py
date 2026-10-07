"""
Domain Knowledge Retrieval Service (RAG Service).
Encapsulates TF-IDF vector retrieval and domain terminology indexing without changing core algorithms.
"""

from typing import List, Dict, Any, Optional
from backend.rag_engine import rag_engine, KnowledgeChunk


class RAGService:
    """Wraps the underlying TF-IDF knowledge base and terminology engine."""

    def __init__(self, engine=rag_engine):
        self._engine = engine

    def retrieve(self, query: str, domain: str = "all", top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks for a given query and domain filter."""
        return self._engine.retrieve(query=query, domain=domain, top_k=top_k)

    def get_domains(self) -> List[str]:
        """Return all available knowledge domains."""
        return self._engine.get_domains()

    def get_all_terms(self) -> List[Dict[str, Any]]:
        """Return all indexed domain terminology."""
        return self._engine.get_all_terms()

    @property
    def chunks(self) -> List[KnowledgeChunk]:
        """Direct access to indexed chunks for counting and introspection."""
        return self._engine.chunks

    def add_custom_term(self, term: str, definition: str, domain: str = "custom") -> KnowledgeChunk:
        """Add and dynamically re-index a custom domain term."""
        return self._engine.add_custom_term(term=term, definition=definition, domain=domain)


# Process-level singleton instance
rag_service = RAGService()
