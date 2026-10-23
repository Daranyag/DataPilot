import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from automl.evaluation import evaluate_models

def test_robustness_classification():
    df = pd.DataFrame({
        'num1': np.random.randn(100),
        'num2': np.random.randn(100),
        'text_col': ['A', 'B', 'C', 'A'] * 25,
        'missing_col': [1.0, np.nan] * 50,
        'target': ['Yes', 'No'] * 50
    })
    
    results, metric, meta = evaluate_models(df, "classification", "target", total_time_budget=30)
    assert not results.empty
    assert 'Status' in results.columns

def test_robustness_regression():
    df = pd.DataFrame({
        'num1': np.random.randn(100),
        'num2': np.random.randn(100),
        'text_col': ['X', 'Y', 'Z', 'X'] * 25,
        'missing_col': [5.0, np.nan] * 50,
        'target': np.random.randn(100)
    })
    
    results, metric, meta = evaluate_models(df, "regression", "target", total_time_budget=30)
    assert not results.empty
    assert 'Status' in results.columns

def test_robustness_clustering():
    df = pd.DataFrame({
        'num1': np.random.randn(100),
        'num2': np.random.randn(100),
        'text_col': ['Cat', 'Dog', 'Bird', 'Fish'] * 25,
        'missing_col': [10.0, np.nan] * 50
    })
    
    results, metric, meta = evaluate_models(df, "clustering", None, total_time_budget=30)
    assert not results.empty
    assert 'Status' in results.columns
