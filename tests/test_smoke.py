import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append(str(Path(__file__).resolve().parent.parent))
from automl.evaluation import evaluate_models
from automl.fit_check import gather_evidence, generate_explanation

@pytest.fixture(autouse=True)
def mock_llm_and_network():
    # Mock any LLM calls so we don't need network
    with patch("agent.llm.safe_generate_text") as mock_llm:
        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.text = "Mocked explanation text"
        mock_llm.return_value = mock_result
        yield mock_llm

def test_smoke_regression_messy():
    # Regression with text and NaN
    df = pd.DataFrame({
        "target": np.random.rand(50),
        "feat1": np.random.randn(50),
        "text_col": ["cat", "dog", "bird", "fish", "elephant"] * 10,
        "nan_col": [np.nan if i % 5 == 0 else float(i) for i in range(50)]
    })
    results, metric, meta = evaluate_models(df, "regression", "target", total_time_budget=30, algo_time_budget=5)
    assert not results.empty
    assert "Status" in results.columns
    assert all(results["Status"].notna())
    
    # Check fit check doesn't crash on winner
    winner = results["Model"].iloc[0]
    ev = gather_evidence(winner, df, "target")
    ex = generate_explanation(ev)
    assert "text" in ex

def test_smoke_binary_classification():
    # Binary classification
    df = pd.DataFrame({
        "target": [0, 1] * 25,
        "feat1": np.random.randn(50),
        "feat2": np.random.rand(50)
    })
    results, metric, meta = evaluate_models(df, "classification", "target", total_time_budget=30, algo_time_budget=5)
    assert not results.empty
    assert "Status" in results.columns

def test_smoke_multiclass_rare():
    # Multi-class with rare class (only 1 instance of class 2)
    target = [0] * 24 + [1] * 25 + [2] * 1
    df = pd.DataFrame({
        "target": target,
        "feat1": np.random.randn(50)
    })
    results, metric, meta = evaluate_models(df, "classification", "target", total_time_budget=30, algo_time_budget=5)
    assert not results.empty

def test_smoke_clustering():
    # Clustering (target None)
    df = pd.DataFrame({
        "feat1": np.random.randn(50),
        "feat2": np.random.rand(50)
    })
    results, metric, meta = evaluate_models(df, "clustering", None, total_time_budget=30, algo_time_budget=5)
    assert not results.empty
    assert "Status" in results.columns

def test_smoke_duplicates_and_id():
    # Duplicates and ID-like column
    df = pd.DataFrame({
        "target": np.random.rand(50),
        "id_col": range(50),
        "feat1": [1.5] * 50
    })
    # Add duplicates
    df = pd.concat([df, df.iloc[:10]])
    results, metric, meta = evaluate_models(df, "regression", "target", total_time_budget=30, algo_time_budget=5)
    assert not results.empty
    
def test_smoke_target_leakage():
    # Column giving away target
    df = pd.DataFrame({
        "target": np.random.rand(50),
        "feat1": np.random.randn(50)
    })
    df["leaky_col"] = df["target"] * 2 + 0.1
    results, metric, meta = evaluate_models(df, "regression", "target", total_time_budget=30, algo_time_budget=5)
    assert not results.empty
    # We should have excluded leaky_col internally or marked suspicious
