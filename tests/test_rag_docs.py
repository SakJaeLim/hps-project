"""T09 · spec 06,07 — TDD Red(미구현). 구현되면 xfail 제거 → Green 이어야 함."""
import pytest

@pytest.mark.xfail(reason="TDD Red — T09 미구현", strict=False)
def test_rag_docs():
    from snct.knowledge.rag_docs import retrieve
    ev = retrieve("DG 격리 규칙", k=3)
    assert isinstance(ev, list) and ev and ev[0]["type"] == "doc"
