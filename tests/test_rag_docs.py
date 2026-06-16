"""T09 · spec 06,07 — TDD Green."""
import pytest

def test_rag_docs():
    from snct.knowledge.rag_docs import retrieve
    ev = retrieve("DG 격리 규칙", k=3)
    assert isinstance(ev, list) and ev and ev[0]["type"] == "doc"
