"""T10 · spec 07 — TDD Red(미구현). 구현되면 xfail 제거 → Green 이어야 함."""
import pytest

@pytest.mark.xfail(reason="TDD Red — T10 미구현", strict=False)
def test_graphrag():
    from snct.knowledge import graphrag
    assert "rehandling_conflict" in graphrag.TEMPLATES
    res = graphrag.ask("이 컨테이너 위에 뭐가 쌓였나")
    assert res["sources"][0]["type"] == "graph"
