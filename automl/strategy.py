"""
Adaptive Execution Strategy Module for DataPilot AutoML.
Determines optimal evaluation parameters based on dataset size and characteristics.
"""

from typing import Dict, Any, Optional
import pandas as pd
from .config import (
    SMALL_DATASET_THRESHOLD,
    MEDIUM_DATASET_THRESHOLD,
    SMALL_DATASET_FOLDS,
    MEDIUM_DATASET_FOLDS,
    LARGE_DATASET_FOLDS,
    STAGE1_SAMPLE_SIZE,
    STAGE1_FOLDS,
    MEDIUM_TOP_K_MODELS,
    LARGE_TOP_K_MODELS,
)

def determine_strategy(df: pd.DataFrame, task_type: str, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes dataset properties and returns an adaptive evaluation plan.
    """
    n_rows = len(df)
    n_cols = len(df.columns) - (1 if target_col and target_col in df.columns else 0)
    
    # Categorical columns count
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    n_cat_cols = len(cat_cols)
    
    # Number of classes if classification
    n_classes = None
    if task_type == "classification" and target_col and target_col in df.columns:
        n_classes = int(df[target_col].nunique(dropna=True))
        
    if n_rows <= SMALL_DATASET_THRESHOLD:
        tier = "small"
        description = (
            f"Small dataset ({n_rows:,} rows, {n_cols} features): "
            f"Running full evaluation of every algorithm with {SMALL_DATASET_FOLDS}-fold cross-validation."
        )
        plan = {
            "tier": tier,
            "description": description,
            "use_screening": False,
            "stage1_sample_size": None,
            "stage1_folds": None,
            "stage2_folds": SMALL_DATASET_FOLDS,
            "stage2_top_k": None, # All algorithms run full evaluation
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_cat_cols": n_cat_cols,
            "n_classes": n_classes
        }
    elif n_rows <= MEDIUM_DATASET_THRESHOLD:
        tier = "medium"
        description = (
            f"Medium dataset ({n_rows:,} rows, {n_cols} features): "
            f"Stage 1 screening on a {min(STAGE1_SAMPLE_SIZE, n_rows):,} row sample ({STAGE1_FOLDS}-fold CV), "
            f"followed by Stage 2 full evaluation of top algorithms and fast models ({MEDIUM_DATASET_FOLDS}-fold CV)."
        )
        plan = {
            "tier": tier,
            "description": description,
            "use_screening": True,
            "stage1_sample_size": min(STAGE1_SAMPLE_SIZE, n_rows),
            "stage1_folds": STAGE1_FOLDS,
            "stage2_folds": MEDIUM_DATASET_FOLDS,
            "stage2_top_k": MEDIUM_TOP_K_MODELS,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_cat_cols": n_cat_cols,
            "n_classes": n_classes
        }
    else:
        tier = "large"
        class_str = f", {n_classes} classes" if n_classes else ""
        description = (
            f"Large dataset ({n_rows:,} rows, {n_cols} features{class_str}): "
            f"Stage 1 screening on a {STAGE1_SAMPLE_SIZE:,} row sample ({STAGE1_FOLDS}-fold CV), "
            f"followed by Stage 2 evaluation of top {LARGE_TOP_K_MODELS} algorithms ({LARGE_DATASET_FOLDS}-fold CV) with capped samples for slow models."
        )
        plan = {
            "tier": tier,
            "description": description,
            "use_screening": True,
            "stage1_sample_size": STAGE1_SAMPLE_SIZE,
            "stage1_folds": STAGE1_FOLDS,
            "stage2_folds": LARGE_DATASET_FOLDS,
            "stage2_top_k": LARGE_TOP_K_MODELS,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_cat_cols": n_cat_cols,
            "n_classes": n_classes
        }
        
    return plan
