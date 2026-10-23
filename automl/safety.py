"""
Safety Nets, Execution Time Predictor, Row Capping, and Timeout Module for DataPilot AutoML.
Prevents slow or infinite execution loops from hanging the application.
"""

import time
import concurrent.futures
import pandas as pd
from typing import Tuple, Optional, Any, Dict
from .config import ROW_CAPS, DEFAULT_ALGO_TIME_BUDGET

def apply_row_cap(
    model_name: str, 
    X: pd.DataFrame, 
    y: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series], str]:
    """
    Applies row caps for algorithms that scale poorly (e.g. O(N^2) or heavy memory).
    Returns (X_capped, y_capped, label_annotation)
    """
    n_rows = len(X)
    cap = None
    
    for key, max_rows in ROW_CAPS.items():
        if key.lower() in model_name.lower():
            cap = max_rows
            break
            
    if cap and n_rows > cap:
        sample_indices = X.sample(n=cap, random_state=42).index
        X_sub = X.loc[sample_indices]
        y_sub = y.loc[sample_indices] if y is not None else None
        annotation = f"Trained on sample of {cap:,} rows"
        return X_sub, y_sub, annotation
        
    return X, y, ""

def predict_execution_time(
    model_name: str,
    pipeline: Any,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    target_rows: int,
    budget_seconds: float = DEFAULT_ALGO_TIME_BUDGET
) -> Tuple[bool, float, str]:
    if len(X) < 150: return False, 0.0, ""
    
    # Time at two sizes to remove startup cost
    s1, s2 = 100, min(500, len(X))
    try:
        if y is not None:
            # Stratified sample
            X1 = X.groupby(y, group_keys=False).apply(lambda s: s.sample(min(len(s), max(1, int(s1 * len(s) / len(X)))), random_state=42))
        else:
            X1 = X.sample(n=s1, random_state=42)
        y1 = y.loc[X1.index] if y is not None else None
        
        t0 = time.perf_counter()
        if y1 is not None: pipeline.fit(X1, y1)
        else: pipeline.fit(X1)
        t1 = max(time.perf_counter() - t0, 0.001)
        
        if y is not None:
            X2 = X.groupby(y, group_keys=False).apply(lambda s: s.sample(min(len(s), max(1, int(s2 * len(s) / len(X)))), random_state=42))
        else:
            X2 = X.sample(n=s2, random_state=42)
        y2 = y.loc[X2.index] if y is not None else None
        
        t0 = time.perf_counter()
        if y2 is not None: pipeline.fit(X2, y2)
        else: pipeline.fit(X2)
        t2 = max(time.perf_counter() - t0, 0.001)
    except Exception as e:
        import logging
        logging.error(f"Prediction run failed: {e}")
        return True, budget_seconds + 1.0, f"skipped: execution error on sample ({str(e)[:50]})"

    
    slope = (t2 - t1) / max(1, s2 - s1)
    if slope < 0: slope = t2 / s2
    
    # Margin
    name_lower = model_name.lower()
    if any(k in name_lower for k in ['svm', 'svr', 'hierarchical', 'dbscan']):
        slope = slope * ((target_rows / s2) ** 1.2)
    elif any(k in name_lower for k in ['random forest', 'gradient boosting']):
        slope = slope * 1.5
        
    predicted_time = t2 + slope * (target_rows - s2)
    
    if predicted_time > budget_seconds:
        return True, predicted_time, f"skipped: predicted too slow (about {predicted_time/60.0:.1f} minutes)"
    return False, predicted_time, ""


def run_with_timeout(
    func: Any,
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    timeout_seconds: float = DEFAULT_ALGO_TIME_BUDGET
) -> Tuple[bool, Any, str]:
    """
    Executes a function with a strict timeout using ThreadPoolExecutor.
    Returns (success: bool, return_value_or_exception, error_message: str)
    """
    if kwargs is None:
        kwargs = {}
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return True, result, ""
        except concurrent.futures.TimeoutError:
            return False, None, f"Too Slow: exceeded time budget of {timeout_seconds}s"
        except Exception as e:
            return False, None, f"Execution Error: {e}"
