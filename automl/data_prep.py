import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any

def optimize_and_clean_data(
    df: pd.DataFrame, target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    ONE shared data preparation function for every task type.
    Outputs a purely numeric matrix with no NaN, no infinity and no text.
    """
    df_clean = df.copy()
    
    prep_summary = {
        "dropped_cols": [],
        "encoded_onehot": [],
        "encoded_freq": [],
        "final_features": 0,
        "pca_used": False
    }
    
    n_rows = len(df_clean)
    
    # Exclude target from features
    feature_cols = [c for c in df_clean.columns if c != target_col]
    cols_to_drop = []
    
    for col in feature_cols:
        series = df_clean[col]
        n_null = series.isna().sum()
        n_null / n_rows if n_rows > 0 else 0
        
        # Data Pilot no longer drops columns silently!
        # The user must interactively drop them via Data Quality warnings.
        # We just compute n_unique for later.
        n_unique = series.nunique(dropna=True)
            
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        feature_cols = [c for c in feature_cols if c not in cols_to_drop]

    # Impute missing values
    for col in feature_cols:
        if df_clean[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
                else:
                    df_clean[col] = df_clean[col].fillna("Unknown")

    # Encode categorical columns
    cat_cols = df_clean[feature_cols].select_dtypes(include=['object', 'category', 'string', 'bool']).columns.tolist()
    for col in cat_cols:
        n_unique = df_clean[col].nunique()
        if n_unique < 20: # One-hot
            dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True, dtype=np.float32)
            df_clean = pd.concat([df_clean, dummies], axis=1)
            df_clean = df_clean.drop(columns=[col])
            prep_summary["encoded_onehot"].append(col)
        else: # Frequency encoding
            freq = df_clean[col].value_counts(normalize=True)
            df_clean[col] = df_clean[col].map(freq).astype(np.float32)
            prep_summary["encoded_freq"].append(col)

    # Standardize numeric columns (excluding target)
    feature_cols_after = [c for c in df_clean.columns if c != target_col]
    
    # 4. Remove perfectly correlated numerical columns
    if len(feature_cols_after) > 1:
        corr_matrix = df_clean[feature_cols_after].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.999)]
        if to_drop_corr:
            df_clean = df_clean.drop(columns=to_drop_corr)
            for c in to_drop_corr:
                prep_summary["dropped_cols"].append(f"'{c}' (perfectly correlated)")
            feature_cols_after = [c for c in feature_cols_after if c not in to_drop_corr]

    from sklearn.preprocessing import StandardScaler
    if len(feature_cols_after) > 0:
        scaler = StandardScaler()
        df_clean[feature_cols_after] = scaler.fit_transform(df_clean[feature_cols_after])
        
    prep_summary["final_features"] = len(feature_cols_after)
    
    if prep_summary["final_features"] == 0:
        raise ValueError("Data cannot be prepared: no usable columns left after cleaning.")

    # Ensure no NaN or infinity left
    if df_clean[feature_cols_after].isna().any().any():
         df_clean[feature_cols_after] = df_clean[feature_cols_after].fillna(0)
    df_clean.replace([np.inf, -np.inf], 0, inplace=True)


    # Handle Rare classes and imbalance
    if target_col and target_col in df_clean.columns:
        if df_clean[target_col].dtype == 'object' or df_clean[target_col].nunique() < 20:
            counts = df_clean[target_col].value_counts()
            if counts.max() / len(df_clean) > 0.7:
                if 'cleanup_messages' not in prep_summary: prep_summary['cleanup_messages'] = []
                prep_summary['cleanup_messages'].append(f"Data is highly imbalanced! Largest class has {counts.max()} rows, smallest has {counts.min()}. Ranking by F1 Macro.")
            counts = df_clean[target_col].value_counts()
            rare = counts[counts < 5].index
            if len(rare) > 0:
                df_clean = df_clean[~df_clean[target_col].isin(rare)]
                if 'cleanup_messages' not in prep_summary: prep_summary['cleanup_messages'] = []
                prep_summary['cleanup_messages'].append(f"Dropped {len(rare)} rare classes with <5 samples.")
                
    return df_clean, prep_summary
