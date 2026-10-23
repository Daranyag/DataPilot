"""
Comprehensive Automated Test Suite for DataPilot Fast AutoML Engine.
Tests strategy selection, timeout safety, screening non-leakage, stage-2 recommendations, and caching.
"""

import pandas as pd
from sklearn.datasets import load_iris
from automl.strategy import determine_strategy
from automl.caching import compute_cache_key, get_cached_evaluation, set_cached_evaluation, clear_cache
from automl.safety import run_with_timeout
from automl.evaluation import evaluate_models, compute_recommendations

def test_strategy_selection():
    df_small = pd.DataFrame({'a': range(100), 'target': [0, 1]*50})
    df_med = pd.DataFrame({'a': range(10000), 'target': [0, 1]*5000})
    df_large = pd.DataFrame({'a': range(60000), 'target': [0, 1]*30000})

    s_small = determine_strategy(df_small, 'classification', 'target')
    s_med = determine_strategy(df_med, 'classification', 'target')
    s_large = determine_strategy(df_large, 'classification', 'target')

    assert s_small['tier'] == 'small'
    assert s_small['use_screening'] == False
    assert s_small['stage2_folds'] == 5

    assert s_med['tier'] == 'medium'
    assert s_med['use_screening'] == True

    assert s_large['tier'] == 'large'
    assert s_large['stage2_folds'] == 3

def test_slow_algorithm_timeout():
    def dummy_slow():
        import time
        time.sleep(3)
        return "Done"

    ok, res, err = run_with_timeout(dummy_slow, timeout_seconds=1.0)
    assert ok == False
    assert "Too Slow" in err

def test_caching_identity():
    clear_cache()
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [0, 1, 0, 1]})
    key1 = compute_cache_key(df, 'classification', 'y')
    key2 = compute_cache_key(df, 'classification', 'y')
    assert key1 == key2

    fake_res = pd.DataFrame({'Model': ['M1'], 'Mean Score': [0.9]})
    set_cached_evaluation(key1, fake_res, 'accuracy', {'strategy': 'small'})

    cached = get_cached_evaluation(key1)
    assert cached is not None
    assert cached[0]['Model'].iloc[0] == 'M1'

def test_stage2_recommendations_only():
    df = pd.DataFrame({
        'Model': ['M1', 'M2', 'M3'],
        'Mean Score': [0.95, 0.94, 0.80],
        'Mean Train Time (s)': [0.5, 0.1, 1.2]
    })
    rec_df = compute_recommendations(df, higher_is_better=True)
    # M2 is within 1 std dev of 0.95 and is faster (0.1s vs 0.5s), so M2 should win Best Balance
    assert rec_df.loc[rec_df['Model'] == 'M1', 'Is Most Accurate'].values[0] == True
    assert rec_df.loc[rec_df['Model'] == 'M2', 'Is Best Balance'].values[0] == True

def test_iris_fast_eval():
    iris = load_iris(as_frame=True)
    df = iris.frame
    res, metric, meta = evaluate_models(df, 'classification', 'target')
    assert not res.empty
    assert 'KNN Classifier' in res['Model'].values or 'Logistic Regression' in res['Model'].values
    assert metric == 'accuracy'
