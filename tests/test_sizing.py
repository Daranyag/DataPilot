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

# The user asked for sizes: 10, 30, 100, 500, 2000, 10000
# And tasks: classification, regression, clustering
# And Quick Demo Mode: on, off
SIZES = [10, 30, 100, 500, 2000, 10000]
TASKS = ["classification", "regression", "clustering"]

@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("quick_mode", [True, False])
def test_app_sizing(size, task, quick_mode, tmp_path):
    # 1. Create a dataset
    if task == "classification":
        # we need at least some classes, let's just make it random binary
        df = pd.DataFrame({
            "target": np.random.choice([0, 1], size=size),
            "feat1": np.random.randn(size),
            "feat2": np.random.rand(size)
        })
        # ensure both classes exist if size >= 2
        if size >= 2:
            df.loc[0, "target"] = 0
            df.loc[1, "target"] = 1
    else:
        df = pd.DataFrame({
            "target": np.random.rand(size),
            "feat1": np.random.randn(size),
            "feat2": np.random.rand(size)
        })
        
    csv_path = tmp_path / f"test_data_{size}.csv"
    df.to_csv(csv_path, index=False)
    
    # 2. Start app
    app = AppTest.from_file("../app/main.py")
    app.run(timeout=120)
    
    # Inject data
    app.session_state["df"] = df
    app.session_state["df_path"] = str(csv_path)
    
    # Set target and run AutoML
    app.session_state["target"] = None if task == "clustering" else "target"
    app.session_state["task_type"] = task
    
    app.run()
    
    # Look for the Quick Demo Mode checkbox (if it exists, toggle it)
    for checkbox in app.checkbox:
        if "Quick Demo Mode" in checkbox.label:
            checkbox.set_value(quick_mode).run()
            
    run_btns = [btn for btn in app.button if "Run Algorithm Comparison" in btn.label]
    if run_btns:
        run_btns[0].click().run(timeout=120)
        
    check_for_errors(app)
    
    # If size < 20, we expect an error block stating "Data Problem: Dataset has only"
    if size < 20:
        found_data_problem = any("Data Problem" in err.value for err in app.error)
        assert found_data_problem, "Expected a friendly data problem error for size < 20"
