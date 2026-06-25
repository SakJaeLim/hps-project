import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "rl_data: 실제 강화학습 결과 자료(data/강화학습 결과 자료)가 있어야 통과"
    )


def _rl_data_available() -> bool:
    root = pathlib.Path(__file__).resolve().parents[1]
    return (root / "data" / "강화학습 결과 자료").is_dir()


def pytest_collection_modifyitems(config, items):
    """RL 결과 자료가 없는 환경에서는 rl_data 마커 테스트를 skip (git 미추적 데이터)."""
    if _rl_data_available():
        return
    import pytest
    skip = pytest.mark.skip(reason="RL 결과 자료(data/강화학습 결과 자료) 없음 — git 미추적")
    for item in items:
        if "rl_data" in item.keywords:
            item.add_marker(skip)
