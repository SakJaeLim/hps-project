"""T10 · spec 07 — TDD Green."""
import pytest

def test_graphrag():
    from snct.knowledge import graphrag
    assert "rehandling_conflict" in graphrag.TEMPLATES
    res = graphrag.ask("이 컨테이너 위에 뭐가 쌓였나")
    assert res["sources"][0]["type"] == "graph"
