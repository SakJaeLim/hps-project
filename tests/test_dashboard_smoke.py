"""T14 · spec 00 — TDD Red(미구현). 구현되면 xfail 제거 → Green 이어야 함."""
import pytest

def test_dashboard_smoke():
    import importlib.util
    assert importlib.util.find_spec("streamlit") is not None  # 대시보드 기동 가능
