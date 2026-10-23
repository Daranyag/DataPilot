import pandas as pd
from typing import Tuple

def detect_task_type(df: pd.DataFrame, target_col: str = None) -> Tuple[str, str]:
    """
    Detects the machine learning task type (regression, classification, clustering).
    Returns a tuple of (task_type, target_col).
    If target_col is None, it defaults to clustering.
    """
    if target_col is None:
        return "clustering", None
        
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    target_series = df[target_col]
    
    # Simple heuristic: if target is numeric and has many unique values, regression
    # otherwise, classification.
    if pd.api.types.is_numeric_dtype(target_series):
        unique_count = target_series.nunique()
        # If float, almost certainly regression
        if pd.api.types.is_float_dtype(target_series):
            return "regression", target_col
        # If int with many unique values relative to dataset size, likely regression
        # (Lowered the threshold to 5 to handle tiny sample datasets correctly)
        if unique_count > 5 or unique_count > (len(target_series) * 0.3):
            return "regression", target_col
            
    return "classification", target_col
