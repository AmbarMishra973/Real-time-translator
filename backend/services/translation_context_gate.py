"""
Deterministic Translation Context & Selective RAG Gating Layer.
Evaluates input transcripts to decide whether domain RAG context should be injected,
preventing unnecessary context dilution on general conversational speech while
preserving grounded technical and domain terminology.
"""

import re
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from backend.services.rag_service import rag_service, RAGService


# Canonical technical, corporate, and medical vocabulary across English, Hinglish, and Hindi (Devanagari)
CANONICAL_TECHNICAL_TERMS: Set[str] = {
    # Cloud, Infrastructure & Containers
    "docker", "kubernetes", "k8s", "container", "containers", "pod", "pods",
    "cluster", "clusters", "cloud", "aws", "azure", "gcp",
    # Architecture, APIs & Protocols
    "api", "apis", "rest", "restful", "http", "https", "grpc", "websocket",
    "microservices", "microservice", "backend", "frontend", "architecture",
    "jwt", "oauth", "token", "payload", "endpoint", "endpoints", "server", "servers",
    # AI, ML & NLP
    "rag", "rag architecture", "llm", "large language model", "vector database",
    "vector embeddings", "prompt engineering", "transformer", "nlp", "asr", "stt", "tts",
    "cosine similarity", "hallucination", "hallucinations", "fine-tuning",
    # Databases & Backend Stacks
    "database", "databases", "sql", "nosql", "postgresql", "postgres", "mysql",
    "mongodb", "redis", "caching", "cache", "fastapi", "spring boot", "django",
    "flask", "node", "nodejs", "python", "java", "typescript",
    # Engineering & DevOps
    "ci/cd", "deployment", "deploy", "latency", "throughput", "pipeline", "pipelines",
    # Business & Corporate
    "roi", "sprint", "kpi", "stakeholders", "deliverables", "quarterly review",
    "bandwidth", "scope creep", "burn rate", "synergy",
    # Medical & Healthcare
    "hypertension", "diagnosis", "prescription", "triage", "outpatient",
    "prognosis", "contraindication", "chronic disease",
    # Devanagari Transliterated Loanwords (Hindi -> English Translation Grounding)
    "एपीआई", "डॉकर", "कुबेरनेट्स", "माइक्रोसर्विसेज", "माइक्रोसर्विस",
    "कंटेनर", "कंटेनर्स", "डेटाबेस", "बैकएंड", "फ्रंटएंड", "आरएजी",
    "एलएलएम", "आर्किटेक्चर", "सर्वर", "क्लाउड", "लेटेंसी", "कैश"
}


@dataclass
class GateDecision:
    """Represents a deterministic gating decision for translation RAG context injection."""
    decision: str  # "USE_RAG" or "NO_RAG"
    use_rag: bool
    matched_terms: List[str] = field(default_factory=list)
    technical_density: float = 0.0
    retrieval_score: float = 0.0
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    reason: str = ""
    gate_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "use_rag": self.use_rag,
            "matched_terms": self.matched_terms,
            "technical_density": round(self.technical_density, 3),
            "retrieval_score": round(self.retrieval_score, 3),
            "total_chunks": len(self.chunks),
            "sources_used": self.sources_used,
            "reason": self.reason,
            "gate_latency_ms": round(self.gate_latency_ms, 3)
        }


class TranslationContextGate:
    """
    Deterministic, sub-millisecond gating policy for selective RAG injection.
    Signals evaluated:
    1. Exact domain terminology matching with word boundary protection
    2. Meaningful technical density ratio
    3. Grounded TF-IDF retrieval relevance score threshold
    """

    def __init__(
        self,
        rag_svc: Optional[RAGService] = None,
        relevance_threshold: float = 0.35,
        extra_terms: Optional[Set[str]] = None
    ):
        self.rag_service = rag_svc or rag_service
        self.relevance_threshold = relevance_threshold
        self.terms: Set[str] = set(CANONICAL_TECHNICAL_TERMS)
        if extra_terms:
            self.terms.update(t.lower() for t in extra_terms)
        self._load_indexed_terms()
        self._rebuild_pattern()

    def _load_indexed_terms(self):
        """Augments terminology dictionary with all indexed chunks from active RAG service."""
        try:
            indexed = self.rag_service.get_all_terms()
            for item in indexed:
                term = item.get("term", "").strip().lower()
                if term:
                    self.terms.add(term)
        except Exception:
            pass

    def _rebuild_pattern(self):
        """Compiles a single unified boundary-protected regex pattern for all registered terms."""
        if not self.terms:
            self._compiled_pattern = None
            return
        sorted_terms = sorted(self.terms, key=len, reverse=True)
        escaped = "|".join(re.escape(t) for t in sorted_terms)
        self._compiled_pattern = re.compile(
            rf"(?<![\w\u0900-\u097F])(?:{escaped})(?![\w\u0900-\u097F])",
            re.IGNORECASE
        )

    def detect_technical_terms(self, text: str) -> List[str]:
        """
        Detects domain/technical terminology using boundary-protected regular expressions.
        Prevents subword false positives (e.g. 'api' in 'capital', 'rag' in 'storage', 'rest' in 'interest').
        """
        if not text or not self._compiled_pattern:
            return []

        matches = self._compiled_pattern.findall(text)
        # Deduplicate while preserving matched terms
        return list(dict.fromkeys(m.lower() for m in matches))

    def calculate_technical_density(self, text: str, matched_terms: List[str]) -> float:
        """Computes technical density as matched terminology count relative to meaningful words."""
        tokens = [t for t in re.findall(r"[\w\u0900-\u097F]+", text.lower()) if len(t) > 1]
        if not tokens:
            return 0.0
        return min(1.0, len(matched_terms) / float(len(tokens)))

    def evaluate(
        self,
        text: str,
        domain: str = "all",
        top_k: int = 3
    ) -> GateDecision:
        """
        Evaluates input text and returns deterministic GateDecision in sub-millisecond time.
        Policy:
        - If text is empty or has no alphanumeric characters -> NO_RAG
        - If no domain terms matched -> NO_RAG (Zero retrieval overhead on conversational speech)
        - If domain terms matched -> Query RAG retriever
            - If top chunk score >= threshold -> USE_RAG
            - If top chunk score < threshold -> NO_RAG (Context was ungrounded/low confidence)
        """
        t0 = time.perf_counter()
        raw_text = (text or "").strip()

        # 1. Empty or whitespace check
        if not raw_text or not any(c.isalnum() for c in raw_text):
            lat = (time.perf_counter() - t0) * 1000.0
            return GateDecision(
                decision="NO_RAG",
                use_rag=False,
                reason="empty_or_whitespace_input",
                gate_latency_ms=lat
            )

        # 2. Domain Terminology Matching
        matched = self.detect_technical_terms(raw_text)
        density = self.calculate_technical_density(raw_text, matched)

        if not matched:
            lat = (time.perf_counter() - t0) * 1000.0
            return GateDecision(
                decision="NO_RAG",
                use_rag=False,
                matched_terms=[],
                technical_density=0.0,
                retrieval_score=0.0,
                chunks=[],
                sources_used=[],
                reason="no_domain_terminology_detected",
                gate_latency_ms=lat
            )

        # 3. Terminology Detected: Verify Retrieval Grounding
        try:
            rag_res = self.rag_service.retrieve(query=raw_text, domain=domain, top_k=top_k)
            chunks = rag_res.get("chunks", [])
            sources = rag_res.get("sources_used", [])
            top_score = max([float(c.get("score", c.get("similarity", 0.0))) for c in chunks], default=0.0)
        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000.0
            return GateDecision(
                decision="NO_RAG",
                use_rag=False,
                matched_terms=matched,
                technical_density=density,
                retrieval_score=0.0,
                chunks=[],
                sources_used=[],
                reason=f"retrieval_service_error_{e}",
                gate_latency_ms=lat
            )

        lat = (time.perf_counter() - t0) * 1000.0

        # 4. Gating Threshold Decision
        if chunks and top_score >= self.relevance_threshold:
            return GateDecision(
                decision="USE_RAG",
                use_rag=True,
                matched_terms=matched,
                technical_density=density,
                retrieval_score=top_score,
                chunks=chunks,
                sources_used=sources,
                reason="domain_term_detected_with_grounded_context",
                gate_latency_ms=lat
            )
        else:
            return GateDecision(
                decision="NO_RAG",
                use_rag=False,
                matched_terms=matched,
                technical_density=density,
                retrieval_score=top_score,
                chunks=[],
                sources_used=sources,
                reason="retrieval_score_below_threshold",
                gate_latency_ms=lat
            )


# Global singleton instance
translation_context_gate = TranslationContextGate()
