"""
RAG Engine for Real-Time AI Translator.
Handles knowledge ingestion, vector embedding, semantic similarity retrieval, and source grounding.
"""

import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeChunk:
    def __init__(self, term: str, definition: str, domain: str, source_file: str = ""):
        self.term = term.strip()
        self.definition = definition.strip()
        self.domain = domain.lower()
        self.source_file = source_file
        # Full text used for semantic embedding and retrieval
        self.full_text = f"{self.term}: {self.definition}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "definition": self.definition,
            "domain": self.domain,
            "source_file": self.source_file,
            "text": self.full_text
        }


class RAGEngine:
    """
    Vector-based RAG retrieval engine using TF-IDF n-gram embeddings and Cosine Similarity.
    Provides semantic retrieval, source document tracking, and confidence scoring.
    """

    def __init__(self, knowledge_dir: Optional[str] = None):
        if knowledge_dir is None:
            knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
        self.knowledge_dir = knowledge_dir
        self.chunks: List[KnowledgeChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.load_knowledge_base()

    def _parse_file(self, filepath: str, domain: str) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []
        if not os.path.exists(filepath):
            return chunks

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        raw_entries = re.split(r"\n\s*\n", content)
        for entry in raw_entries:
            entry = entry.strip()
            if not entry or entry.startswith("#"):
                continue

            if ":" in entry:
                parts = entry.split(":", 1)
                term = parts[0].strip().lstrip("#").strip()
                definition = parts[1].strip()
                if term and definition:
                    chunks.append(KnowledgeChunk(
                        term=term,
                        definition=definition,
                        domain=domain,
                        source_file=os.path.basename(filepath)
                    ))
            else:
                lines = entry.splitlines()
                if lines:
                    term = lines[0].strip()
                    definition = " ".join([l.strip() for l in lines[1:]]).strip()
                    if term and definition:
                        chunks.append(KnowledgeChunk(
                            term=term,
                            definition=definition,
                            domain=domain,
                            source_file=os.path.basename(filepath)
                        ))
        return chunks

    def load_knowledge_base(self):
        """Loads and indexes all documents in the knowledge directory."""
        self.chunks = []
        if os.path.exists(self.knowledge_dir):
            for fname in os.listdir(self.knowledge_dir):
                fpath = os.path.join(self.knowledge_dir, fname)
                if os.path.isfile(fpath) and fname.endswith((".txt", ".md")):
                    domain = fname.replace("_terms.txt", "").replace(".txt", "").replace(".md", "")
                    parsed = self._parse_file(fpath, domain)
                    self.chunks.extend(parsed)

        self._build_index()

    def _build_index(self):
        """Builds TF-IDF vector embeddings matrix for all loaded chunks."""
        if not self.chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        corpus = [c.full_text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def add_custom_term(self, term: str, definition: str, domain: str = "custom") -> KnowledgeChunk:
        """Dynamically add a custom term to the knowledge base and re-index."""
        chunk = KnowledgeChunk(term=term, definition=definition, domain=domain, source_file="custom_input")
        self.chunks.append(chunk)
        self._build_index()
        return chunk

    def retrieve(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 3,
        threshold: float = 0.30
    ) -> Dict[str, Any]:
        """
        Retrieves top_k relevant knowledge entries for the given transcript query.
        Returns:
            {
                "chunks": [{"term": ..., "definition": ..., "similarity": 0.87, "source_file": ...}],
                "sources_used": ["technical_terms.txt"],
                "total_matches": 3
            }
        """
        query = (query or "").strip()
        if not query or not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            return {"chunks": [], "sources_used": [], "total_matches": 0}

        query_vec = self.vectorizer.transform([query])
        raw_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        query_words = set(re.findall(r"\w+", query.lower()))
        scored_candidates = []

        for idx, chunk in enumerate(self.chunks):
            if domain and domain.lower() != "all" and chunk.domain != domain.lower():
                continue

            raw_sim = float(raw_similarities[idx])
            term_lower = chunk.term.lower()
            term_words = set(re.findall(r"\w+", term_lower))

            # Calculate semantic similarity score
            # Exact phrase match in query
            if term_lower in query.lower():
                sim = min(0.95, max(0.85, 0.80 + raw_sim * 0.5))
            # Direct word match of key term
            elif term_words and term_words.issubset(query_words):
                sim = min(0.92, max(0.80, 0.75 + raw_sim * 0.4))
            elif term_words.intersection(query_words):
                sim = min(0.84, max(0.65, 0.60 + raw_sim * 0.4))
            else:
                # Raw cosine similarity without direct keyword hit
                sim = raw_sim * 1.5

            # Apply strict threshold so unrelated terms (e.g. 0.09) are not returned
            if sim >= threshold:
                item = chunk.to_dict()
                item["score"] = round(float(sim), 2)
                item["similarity"] = round(float(sim), 2)
                scored_candidates.append(item)

        # Sort by similarity descending
        scored_candidates.sort(key=lambda x: x["similarity"], reverse=True)
        top_chunks = scored_candidates[:top_k]

        # Collect unique source files used for grounding
        sources = sorted(list(set(c["source_file"] for c in top_chunks if c.get("source_file"))))

        return {
            "chunks": top_chunks,
            "sources_used": sources,
            "total_matches": len(top_chunks)
        }

    def format_context_for_prompt(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into a clean context section for the LLM prompt."""
        if not retrieved_chunks:
            return "No specific domain terms retrieved for this sentence."

        lines = []
        for item in retrieved_chunks:
            lines.append(f"• {item['term']}: {item['definition']}")
        return "\n".join(lines)

    def get_domains(self) -> List[str]:
        """Returns unique domains currently available in the knowledge base."""
        domains = set(c.domain for c in self.chunks)
        return sorted(list(domains))

    def get_all_terms(self) -> List[Dict[str, Any]]:
        """Returns all terms currently in the knowledge base."""
        return [c.to_dict() for c in self.chunks]


# Global singleton instance
rag_engine = RAGEngine()
