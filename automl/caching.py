"""
In-Memory Hash-Based Caching Module for DataPilot AutoML.
Ensures instant responses when repeating identical evaluation runs.
"""

import hashlib
import pandas as pd
from typing import Optional, Tuple, Dict, Any

_EVALUATION_CACHE: Dict[str, Tuple[pd.DataFrame, str, Dict[str, Any]]] = {}

def compute_cache_key(
    df: pd.DataFrame, 
    task_type: str, 
    target_col: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Computes a deterministic MD5 hash based on:
    - Dataset content & shape
    - Task type & target column
    - Evaluation settings / parameters
    """
    hasher = hashlib.md5()
    
    # Hash dataset structure & values
    try:
        data_hash = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        hasher.update(data_hash)
    except Exception:
        # Fallback hash if unhashable complex types present
        hasher.update(f"{df.shape}_{list(df.columns)}".encode('utf-8'))
        
    # Hash metadata
    hasher.update(str(task_type).encode('utf-8'))
    hasher.update(str(target_col).encode('utf-8'))
    
    if extra_params:
        hasher.update(str(sorted(extra_params.items())).encode('utf-8'))
        
    return hasher.hexdigest()

def get_cached_evaluation(cache_key: str) -> Optional[Tuple[pd.DataFrame, str, Dict[str, Any]]]:
    """Retrieves cached results if available."""
    return _EVALUATION_CACHE.get(cache_key)

def set_cached_evaluation(
    cache_key: str, 
    results_df: pd.DataFrame, 
    primary_metric: str,
    meta_info: Dict[str, Any]
) -> None:
    """Stores evaluation results in cache."""
    _EVALUATION_CACHE[cache_key] = (results_df.copy(), primary_metric, meta_info)

def clear_cache() -> None:
    """Clears all cached evaluation results."""
    _EVALUATION_CACHE.clear()
