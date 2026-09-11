"""
Automated Verification Suite for Real-Time AI Translator + RAG
Updated for:
  - Realistic similarity scores & non-empty/empty filtering
  - Source grounding tracking (sources_used)
  - Real processing latency benchmarking metrics (stt_s, rag_s, llm_s, total_s)
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import asyncio
from fastapi.testclient import TestClient
from backend.server import app
from backend.rag_engine import rag_engine
from backend.llm_translator import llm_translator

def test_rag_retrieval():
    print("\n--- TEST 1: RAG Semantic Retrieval & Grounding ---")
    query = "We need to implement RAG architecture with vector database and LLM context"
    res = rag_engine.retrieve(query, domain="all", top_k=3)
    chunks = res["chunks"]
    sources = res["sources_used"]
    print(f"Query: '{query}'")
    print(f"Retrieved {len(chunks)} relevant chunks:")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c['term']} (Similarity: {c['similarity']}): {c['definition'][:50]}...")
    print(f"Sources Used: {sources}")

    assert len(chunks) >= 2, "Expected at least 2 relevant matches"
    assert any(c['term'] in ['RAG', 'RAG Architecture', 'Vector Database', 'LLM Context'] for c in chunks)
    assert len(sources) > 0, "Expected at least one source file used for grounding"

    # Test casual sentence with NO domain terms
    casual_res = rag_engine.retrieve("Hello how are you doing today", domain="all")
    print(f"Casual Query matches: {len(casual_res['chunks'])} (Expected 0 to avoid false positives)")
    assert len(casual_res['chunks']) == 0, "Casual queries should not return spurious low-similarity matches"
    print("✓ RAG Retrieval & Precision Test PASSED!")


def test_llm_translation_and_history():
    print("\n--- TEST 2: LLM Translation, Grounding & History ---")
    session = "test_verify_session_v2"

    t1 = "I have a meeting tomorrow regarding the deployment."
    res1 = llm_translator.translate(t1, source_lang="en", target_lang="hi", session_id=session)
    print(f"Turn 1 Source: '{t1}'")
    print(f"Turn 1 Hindi: '{res1['translated_text']}'")
    print(f"Turn 1 Provider: {res1['provider']} (Latency: {res1['latency_s']}s)")
    assert "no api key configured" not in res1['provider'].lower(), "Provider label should not display raw fallback warning"

    t2 = "It is with the development team and uses Kubernetes."
    res2 = llm_translator.translate(t2, source_lang="en", target_lang="hi", session_id=session)
    print(f"Turn 2 Source: '{t2}'")
    print(f"Turn 2 Hindi: '{res2['translated_text']}'")
    print(f"History Length: {len(res2['history'])}")
    assert len(res2['history']) == 2, "History should retain both turns"
    print("✓ LLM Translation & History Test PASSED!")


def test_fastapi_endpoints():
    print("\n--- TEST 3: FastAPI Backend Endpoints & Real Latency Metrics ---")
    client = TestClient(app)

    # 1. Root
    root_res = client.get("/")
    assert root_res.status_code == 200
    data = root_res.json()
    print(f"Root API Status: {data['status']}, Active Mode: {data['llm_status']['active_mode']}")

    # 2. Translate Endpoint with Metrics & Sources
    form_data = {
        "text": "The microservices architecture will reduce latency.",
        "source_lang": "en",
        "target_lang": "hi",
        "session_id": "api_test_v2",
        "domain": "technical"
    }
    trans_res = client.post("/translate", data=form_data)
    assert trans_res.status_code == 200
    trans_data = trans_res.json()
    print(f"API Translation: '{trans_data['translated_text']}'")
    print(f"API Sources Used: {trans_data.get('sources_used', [])}")
    print(f"Real Latency Metrics: {trans_data.get('metrics', {})}")
    assert "metrics" in trans_data, "Response must include execution latency metrics"
    assert "rag_s" in trans_data["metrics"] and "llm_s" in trans_data["metrics"]

    # 3. TTS Endpoint
    tts_res = client.post("/tts", data={"text": "नमस्ते दुनिया", "target_lang": "hi"})
    assert tts_res.status_code == 200
    assert "X-TTS-Latency" in tts_res.headers
    print(f"TTS Latency Header: {tts_res.headers['X-TTS-Latency']}s, Audio bytes: {len(tts_res.content)}")

    print("✓ All FastAPI Endpoint & Latency Tests PASSED!")


if __name__ == "__main__":
    test_rag_retrieval()
    test_llm_translation_and_history()
    test_fastapi_endpoints()
    print("\n🎉 ALL TESTS PASSED! Clean, grounded pipeline ready for interview demo.")
