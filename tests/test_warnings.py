import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from automl.leakage import detect_leakage_and_warnings

def test_warnings_imbalance():
    # 90% class 1, 10% class 0 -> large effect
    df = pd.DataFrame({
        'feature': np.random.randn(100),
        'target': [1]*90 + [0]*10
    })
    warnings, _, _ = detect_leakage_and_warnings(df, "target", "classification")
    w = next(w for w in warnings if w['type'] == 'imbalance')
    assert w['percentage'] == 0.90
    assert w['effect'] == 'large'

def test_warnings_duplicates():
    # 20 identical rows out of 100 -> large effect
    base_row = pd.DataFrame({'feature': [1.5], 'target': [1]})
    df = pd.concat([pd.DataFrame({'feature': np.random.randn(80), 'target': np.random.choice([0,1], 80)})] + [base_row]*20, ignore_index=True)
    warnings, _, _ = detect_leakage_and_warnings(df, "target", "classification")
    w = next(w for w in warnings if w['type'] == 'duplicates')
    # 19 duplicates out of 100 = 0.19
    assert w['percentage'] > 0.10
    assert w['effect'] == 'large'
    assert w['action_type'] == 'drop_rows'

def test_warnings_empty_constant_id():
    df = pd.DataFrame({
        'empty': [np.nan]*95 + [1, 2, 3, 4, 5],
        'const': [5.0]*100,
        'id_col': [f"ID_{i}" for i in range(100)],
        'target': np.random.choice([0,1], 100)
    })
    warnings, _, _ = detect_leakage_and_warnings(df, "target", "classification")
    w_empty = next(w for w in warnings if w['type'] == 'empty_cols')
    w_const = next(w for w in warnings if w['type'] == 'constant')
    w_id = next(w for w in warnings if w['type'] == 'id_column')
    
    assert w_empty['percentage'] == 0.95
    assert w_empty['effect'] == 'large'
    assert w_const['effect'] == 'moderate'
    assert w_id['effect'] == 'large'

def test_warnings_leakage():
    target = np.random.choice([0,1], 100)
    df = pd.DataFrame({
        'feature': np.random.randn(100),
        'leak': target, # Perfect copy
        'target': target
    })
    warnings, _, _ = detect_leakage_and_warnings(df, "target", "classification")
    w_leak = next(w for w in warnings if w['type'] == 'leakage')
    assert w_leak['percentage'] == 1.0
    assert w_leak['effect'] == 'large'
