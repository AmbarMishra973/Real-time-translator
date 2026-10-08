"""
Phase 3: Hypothesis Stabilization / Local Agreement Service for Streaming STT.

Provides deterministic, lightweight hypothesis management for incremental ASR partials:
  - Distinguishes STABLE PREFIX from UNSTABLE SUFFIX based on consecutive agreement.
  - Mitigates UI transcript jitter and churn across evolving streaming windows.
  - Final transcript remains 100% authoritative; stabilized partials never contaminate final STT.
  - Fully reversible via HYPOTHESIS_STABILIZATION_ENABLED feature flag.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class HypothesisConfig:
    """Configuration parameters for Hypothesis Stabilization / Local Agreement."""
    enabled: bool = field(
        default_factory=lambda: os.getenv("HYPOTHESIS_STABILIZATION_ENABLED", "false").strip().lower() == "true"
    )
    min_agreements: int = field(
        default_factory=lambda: int(os.getenv("STABILITY_MIN_AGREEMENTS", "2"))
    )
    min_tokens: int = field(
        default_factory=lambda: int(os.getenv("STABILITY_MIN_TOKENS", "1"))
    )


@dataclass
class HypothesisResult:
    """Result of processing an incremental partial hypothesis."""
    stable_text: str
    unstable_text: str
    full_text: str
    stable_tokens: List[str]
    unstable_tokens: List[str]
    is_revised: bool
    token_churn: int
    stable_ratio: float
    total_partials: int
    total_revisions: int
    processing_ms: float


class HypothesisSessionState:
    """Encapsulates session-isolated state for hypothesis stabilization."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.stable_tokens: List[str] = []
        self.unstable_tokens: List[str] = []
        self.pending_tokens: List[str] = []
        self.pending_streaks: List[int] = []
        self.last_displayed_tokens: List[str] = []
        self.raw_hypotheses: List[str] = []
        self.displayed_history: List[str] = []

        # Metrics
        self.total_partials: int = 0
        self.total_revisions: int = 0
        self.total_token_churn: int = 0
        self.token_first_seen_ts: Dict[int, float] = {}
        self.token_stable_ts: Dict[int, float] = {}
        self.token_stability_latencies: List[float] = []
        self.turn_start_ts: float = time.perf_counter()

    def reset_for_next_turn(self) -> None:
        """Resets stabilization tracking cleanly for a subsequent utterance."""
        self.stable_tokens.clear()
        self.unstable_tokens.clear()
        self.pending_tokens.clear()
        self.pending_streaks.clear()
        self.last_displayed_tokens.clear()
        self.raw_hypotheses.clear()
        self.displayed_history.clear()
        self.total_partials = 0
        self.total_revisions = 0
        self.total_token_churn = 0
        self.token_first_seen_ts.clear()
        self.token_stable_ts.clear()
        self.token_stability_latencies.clear()
        self.turn_start_ts = time.perf_counter()


class HypothesisService:
    """
    Lightweight, deterministic Local Agreement Hypothesis Manager.
    Evaluates incoming Whisper partial transcripts against historical agreement streaks
    to lock in high-confidence prefixes while preserving trailing unstable tokens.
    """

    def __init__(self, config: Optional[HypothesisConfig] = None):
        self.config = config or HypothesisConfig()

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Splits transcript into word tokens preserving Unicode (Devanagari, Latin, etc.)."""
        if not text:
            return []
        return [t for t in text.strip().split() if t]

    @staticmethod
    def normalize_for_comparison(token: str) -> str:
        """
        Strips peripheral punctuation and lowercases token for comparison
        while preserving internal letters, numbers, hyphens, and Unicode script.
        """
        cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', token.strip(), flags=re.UNICODE)
        return cleaned.lower()

    def create_session_state(self, session_id: str) -> HypothesisSessionState:
        """Creates an isolated session state tracker."""
        return HypothesisSessionState(session_id=session_id)

    def process_hypothesis(
        self,
        raw_hypothesis: str,
        state: HypothesisSessionState,
        min_agreements: Optional[int] = None
    ) -> HypothesisResult:
        """
        Processes incoming incremental hypothesis text through Local Agreement logic.
        Updates state.stable_tokens and state.unstable_tokens.
        """
        t0 = time.perf_counter()
        required_agreements = min_agreements if min_agreements is not None else self.config.min_agreements
        clean_raw = (raw_hypothesis or "").strip()
        state.raw_hypotheses.append(clean_raw)
        state.total_partials += 1

        tokens = self.tokenize(clean_raw)
        stable_len = len(state.stable_tokens)

        # Candidate tokens after the already-locked stable prefix
        candidate_tokens = tokens[stable_len:] if len(tokens) >= stable_len else []

        new_promotions: List[str] = []
        new_pending_tokens: List[str] = []
        new_pending_streaks: List[int] = []

        now = time.perf_counter()

        for idx, cand in enumerate(candidate_tokens):
            norm_cand = self.normalize_for_comparison(cand)
            global_token_idx = stable_len + idx

            if global_token_idx not in state.token_first_seen_ts:
                state.token_first_seen_ts[global_token_idx] = now

            # Check if this matches the previously pending token at the same position
            if idx < len(state.pending_tokens):
                prev_pending = state.pending_tokens[idx]
                norm_prev = self.normalize_for_comparison(prev_pending)
                if norm_cand == norm_prev:
                    streak = state.pending_streaks[idx] + 1
                else:
                    streak = 1
            else:
                streak = 1

            if streak >= required_agreements and (len(new_pending_tokens) == 0):
                # Consecutive agreement threshold met and all preceding candidates were promoted
                new_promotions.append(cand)
                state.token_stable_ts[global_token_idx] = now
                first_seen = state.token_first_seen_ts.get(global_token_idx, now)
                state.token_stability_latencies.append((now - first_seen) * 1000.0)
            else:
                new_pending_tokens.append(cand)
                new_pending_streaks.append(streak)

        # Commit promotions to stable prefix
        if new_promotions:
            state.stable_tokens.extend(new_promotions)

        state.pending_tokens = new_pending_tokens
        state.pending_streaks = new_pending_streaks
        state.unstable_tokens = new_pending_tokens

        # Format stable and unstable presentation strings
        stable_text = " ".join(state.stable_tokens).strip()
        unstable_text = " ".join(state.unstable_tokens).strip()

        if stable_text and unstable_text:
            full_text = f"{stable_text} {unstable_text}"
        elif stable_text:
            full_text = stable_text
        else:
            full_text = unstable_text

        displayed_tokens = self.tokenize(full_text)

        # Calculate Token Churn and Revision Metrics compared to previous displayed tokens
        is_revised = False
        token_churn = 0

        if state.last_displayed_tokens:
            prev_toks = state.last_displayed_tokens
            # Find common prefix length
            common_prefix_len = 0
            for p, c in zip(prev_toks, displayed_tokens):
                if self.normalize_for_comparison(p) == self.normalize_for_comparison(c):
                    common_prefix_len += 1
                else:
                    break

            if common_prefix_len < len(prev_toks):
                # Previously displayed tokens were retracted or replaced
                is_revised = True
                token_churn = len(prev_toks) - common_prefix_len
                state.total_revisions += 1
                state.total_token_churn += token_churn

        state.last_displayed_tokens = displayed_tokens
        state.displayed_history.append(full_text)

        stable_ratio = round(len(state.stable_tokens) / max(1, len(displayed_tokens)), 3)
        processing_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return HypothesisResult(
            stable_text=stable_text,
            unstable_text=unstable_text,
            full_text=full_text,
            stable_tokens=list(state.stable_tokens),
            unstable_tokens=list(state.unstable_tokens),
            is_revised=is_revised,
            token_churn=token_churn,
            stable_ratio=stable_ratio,
            total_partials=state.total_partials,
            total_revisions=state.total_revisions,
            processing_ms=processing_ms
        )

    def reconcile_final(
        self,
        final_transcript: str,
        state: HypothesisSessionState
    ) -> Dict[str, Any]:
        """
        Reconciles the final authoritative STT transcript with the session's
        stabilized partial display history.
        The final transcript is 100% authoritative and is NEVER overridden.
        Computes final reconciliation distance (word error rate / token edit distance).
        """
        final_clean = (final_transcript or "").strip()
        last_displayed = state.displayed_history[-1] if state.displayed_history else ""

        final_toks = self.tokenize(final_clean)
        disp_toks = self.tokenize(last_displayed)

        # Compute simple token edit distance between last displayed partial and authoritative final
        dist = self._token_levenshtein(disp_toks, final_toks)
        reconciliation_wer = round(dist / max(1, len(final_toks)), 4) if final_toks else 0.0

        return {
            "authoritative_final": final_clean,
            "last_displayed_partial": last_displayed,
            "stable_prefix_at_end": " ".join(state.stable_tokens),
            "unstable_suffix_at_end": " ".join(state.unstable_tokens),
            "reconciliation_token_distance": dist,
            "reconciliation_wer": reconciliation_wer,
            "total_partials_emitted": state.total_partials,
            "total_revisions_incurred": state.total_revisions,
            "total_token_churn": state.total_token_churn,
            "revision_rate": round(state.total_revisions / max(1, state.total_partials), 3),
            "mean_time_to_stable_ms": (
                round(sum(state.token_stability_latencies) / len(state.token_stability_latencies), 2)
                if state.token_stability_latencies else 0.0
            )
        }

    @staticmethod
    def _token_levenshtein(seq1: List[str], seq2: List[str]) -> int:
        """Computes Levenshtein distance across token sequences."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if HypothesisService.normalize_for_comparison(seq1[i - 1]) == HypothesisService.normalize_for_comparison(seq2[j - 1]) else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,       # deletion
                    dp[i][j - 1] + 1,       # insertion
                    dp[i - 1][j - 1] + cost # substitution
                )
        return dp[m][n]


# Process-level singleton instance
hypothesis_service = HypothesisService()
