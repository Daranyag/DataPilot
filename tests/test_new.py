import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

from automl.data_prep import optimize_and_clean_data
from automl.evaluation import evaluate_models
import warnings

warnings.filterwarnings('ignore')

def test_data_prep_no_target():
    # Make blobs and add text/missing columns
    X, _ = make_blobs(n_samples=200, centers=3, n_features=3, random_state=42)
    df = pd.DataFrame(X, columns=['f1', 'f2', 'f3'])
    
    # Add text column with <20 unique
    df['text_cat'] = np.random.choice(['A', 'B', 'C', 'D'], size=200)
    
    # Add text column with >20 unique (freq encode)
    df['text_freq'] = np.random.choice([f'Val_{i}' for i in range(50)], size=200)
    
    # Add missing values
    df.loc[0:10, 'f1'] = np.nan
    
    # Add ID column
    df['id_col'] = [f'ID_{i}' for i in range(200)]
    
    df_clean, summary = optimize_and_clean_data(df, target_col=None)
    
    # Check no missing
    assert df_clean.isna().sum().sum() == 0
    # Check id column dropped
    assert 'id_col' not in df_clean.columns
    # Check text_cat one-hot encoded
    assert any(c.startswith('text_cat_') for c in df_clean.columns)
    assert 'text_cat' not in df_clean.columns
    # Check text_freq freq encoded (now float)
    assert df_clean['text_freq'].dtype in [np.float32, np.float64]
    
    # Check summary
    assert 'id_col' in str(summary['dropped_cols'])
    assert 'text_cat' in summary['encoded_onehot']
    assert 'text_freq' in summary['encoded_freq']

def test_clustering_evaluation_basic():
    X, _ = make_blobs(n_samples=150, centers=2, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
    
    res, metric, meta = evaluate_models(df, "clustering", None, total_time_budget=30, algo_time_budget=10)
    
    assert metric == "silhouette"
    assert "Clusters Found" in res.columns
    assert "Cluster Sizes" in res.columns
    assert "Noise Points" in res.columns
    
    # Ensure no placeholders
    for score in res['Mean Score']:
        assert pd.isna(score) or score > -1.0 # Silhouette should be valid or NaN
        
def test_clustering_identical_rows():
    # If all rows are identical, clustering fails
    df = pd.DataFrame(np.ones((100, 3)), columns=['f1', 'f2', 'f3'])
    
    try:
        res, metric, meta = evaluate_models(df, "clustering", None, total_time_budget=30, algo_time_budget=10)
    except ValueError as e:
        # data prep might fail since all cols are constant
        assert "no usable columns left" in str(e).lower()
