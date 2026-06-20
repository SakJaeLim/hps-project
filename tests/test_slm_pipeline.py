import os
import sys
import json
from pathlib import Path

# Ensure src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BASE_DIR = Path(os.environ.get("SNCT_BASE_DIR", PROJECT_ROOT))

def test_gen_sft_output_exists():
    # Verify that Phase 2 data files exist and have correct formats
    data_dir = BASE_DIR / "data" / "simulated"
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    golden_file = data_dir / "eval_golden.jsonl"
    
    assert os.path.exists(train_file)
    assert os.path.exists(val_file)
    assert os.path.exists(golden_file)
    
    # Check one item format
    with open(golden_file, 'r', encoding='utf-8') as f:
        line = f.readline()
        item = json.loads(line)
        assert "type" in item
        assert "instruction" in item
        assert "input" in item
        assert "output" in item

def test_eval_metrics_run():
    # Import eval metrics
    from snct.eval.eval_metrics import run_evaluation
    
    golden_file = BASE_DIR / "data" / "simulated" / "eval_golden.jsonl"
    report_csv = BASE_DIR / "data" / "simulated" / "test_eval_report.csv"
    
    # Run offline evaluation with mock (by passing empty/none model path)
    run_evaluation(
        golden_path=str(golden_file),
        base_model_path=None,
        ft_model_path=None,
        output_csv=str(report_csv)
    )
    
    assert os.path.exists(report_csv)
    
    # Clean up test output
    if os.path.exists(report_csv):
        os.remove(report_csv)

def test_api_routes():
    # Test FastAPI API endpoints directly
    from fastapi.testclient import TestClient
    from snct.api.app_api import app
    
    client = TestClient(app)
    
    # Test compare endpoint
    response = client.post("/compare", json={"prompt": "DG 위험물 적재 규정"})
    assert response.status_code == 200
    data = response.json()
    assert "base_text" in data
    assert "finetuned_text" in data
    assert "terms" in data
    
    # Test metrics endpoint
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "quant" in data
    assert "qual" in data
    
    # Test feedback endpoint
    response = client.post("/feedback", json={"qid": "DG 위험물 적재 규정", "vote": "up"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
