import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock

sys.path.append(str(Path(__file__).resolve().parent.parent))

@pytest.fixture(autouse=True)
def mock_llm():
    with patch("agent.llm.safe_generate_text") as mock_llm:
        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.text = "Mocked explanation text"
        mock_llm.return_value = mock_result
        yield mock_llm

def check_for_errors(app):
    for err in app.error:
        assert "AutoML Error" not in err.value
        assert "Could not build" not in err.value
    assert not app.exception

def test_app_headless_full(mock_llm, tmp_path):
    tasks = [
        ("regression", "target"),
        ("classification", "target"),
        ("clustering", "None") # UI target for clustering is "None"
    ]
    
    for task, tgt in tasks:
        # 1. Create a dummy dataset
        if task == "classification":
            df = pd.DataFrame({
                "target": [0, 1] * 25,
                "feat1": np.random.randn(50),
                "feat2": np.random.rand(50)
            })
        else:
            df = pd.DataFrame({
                "target": np.random.rand(50),
                "feat1": np.random.randn(50),
                "feat2": np.random.rand(50)
            })
            
        csv_path = tmp_path / f"test_data_{task}.csv"
        df.to_csv(csv_path, index=False)
        
        # 2. Start app
        app = AppTest.from_file("../app/main.py")
        app.run(timeout=120)
        check_for_errors(app)
        
        # Inject data
        app.session_state["df"] = df
        app.session_state["df_path"] = str(csv_path)
        app.run()
        check_for_errors(app)
        
        # Set target and run AutoML
        app.session_state["target"] = None if tgt == "None" else tgt
        app.session_state["task_type"] = task
        app.run()
        
        run_btns = [btn for btn in app.button if "Run Algorithm Comparison" in btn.label]
        if run_btns:
            run_btns[0].click().run(timeout=120)
            
        check_for_errors(app)
        
        # Open Fit Check (if rendered)
        eval_btns = [btn for btn in app.button if "Check fit" in btn.label or "Run on a sample" in btn.label]
        if eval_btns:
            eval_btns[0].click().run()
            check_for_errors(app)
            
        # Verify Clustering Insights section if clustering
        if task == "clustering":
            # Check if Clustering Insights is in subheaders
            subheaders = [h.value for h in app.subheader]
            assert any("Clustering Insights" in sh for sh in subheaders), "Clustering Insights missing!"
            
        # Clear Results
        clear_btns = [btn for btn in app.button if "Clear Results" in btn.label]
        if clear_btns:
            clear_btns[0].click().run()
            check_for_errors(app)
