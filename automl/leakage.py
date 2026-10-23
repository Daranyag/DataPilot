import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from automl.config import WARNING_EFFECT_THRESHOLDS

def get_effect_label(warn_type: str, val: float) -> str:
    if warn_type not in WARNING_EFFECT_THRESHOLDS:
        return "small"
    t = WARNING_EFFECT_THRESHOLDS[warn_type]
    if val >= t.get('large', 1.0): return "large"
    if val >= t.get('moderate', 1.0): return "moderate"
    return "small"

def detect_leakage_and_warnings(df: pd.DataFrame, target: str, task_type: str) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    warnings = []
    suspicious_cols = []
    silent_drops = []
    
    if target not in df.columns and task_type != "clustering":
        return warnings, suspicious_cols, silent_drops
        
    n_rows = len(df)
    if n_rows == 0: return warnings, suspicious_cols, silent_drops
    
    y = df[target] if target in df.columns else None
    
    # 1. Class imbalance
    if task_type == 'classification' and y is not None and y.nunique() > 1:
        val_counts = y.value_counts(normalize=True)
        max_class_pct = val_counts.max()
        if max_class_pct > 0.7:
            warnings.append({
                "type": "imbalance",
                "count": 1,
                "percentage": max_class_pct,
                "msg": f"Class imbalance: one class makes up {max_class_pct*100:.1f}% of the data.",
                "explanation": "Models may learn to just guess the majority class and ignore minority patterns.",
                "effect": get_effect_label("imbalance", max_class_pct),
                "action_type": "none",
                "cols": [target]
            })
            
    # 2. Duplicate rows
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        dupe_pct = n_dupes / n_rows
        warnings.append({
            "type": "duplicates",
            "count": n_dupes,
            "percentage": dupe_pct,
            "msg": f"Found {n_dupes} duplicate rows.",
            "explanation": "Duplicates artificially inflate scores by putting identical rows in both training and testing sets.",
            "effect": get_effect_label("duplicates", dupe_pct),
            "action_type": "drop_rows",
            "cols": []
        })
        
    X = df.drop(columns=[target]) if target in df.columns else df.copy()
    
    # Pre-flight feature checks
    for col in X.columns:
        series = X[col]
        n_unique = series.nunique(dropna=True)
        null_pct = series.isna().sum() / n_rows
        
        if null_pct > 0.90:
            warnings.append({
                "type": "empty_cols",
                "count": series.isna().sum(),
                "percentage": null_pct,
                "msg": f"Column '{col}' is >90% empty.",
                "explanation": "Empty columns contribute no information and can cause modeling errors.",
                "effect": get_effect_label("empty_cols", null_pct),
                "action_type": "drop_cols",
                "cols": [col]
            })
            suspicious_cols.append(col)
            continue
            
        if n_unique <= 1:
            warnings.append({
                "type": "constant",
                "count": 1,
                "percentage": 1.0,
                "msg": f"Column '{col}' is constant (only 1 unique value).",
                "explanation": "Constant columns add no predictive value and can break algorithms.",
                "effect": "moderate",
                "action_type": "drop_cols",
                "cols": [col]
            })
            suspicious_cols.append(col)
            continue
            
        is_id_name = any(kw in col.lower() for kw in ['id', 'uuid', 'guid', 'index', 'number'])
        if n_unique == n_rows and (str(series.dtype) in ['object', 'string'] or is_id_name):
            warnings.append({
                "type": "id_column",
                "count": n_rows,
                "percentage": 1.0,
                "msg": f"Column '{col}' looks like an ID (all unique values).",
                "explanation": "ID columns lead models to memorize specific rows rather than learning general patterns.",
                "effect": "large",
                "action_type": "drop_cols",
                "cols": [col]
            })
            suspicious_cols.append(col)
            continue
            
        # Target leakage check
        if y is not None and n_unique > 1 and n_rows <= 10000:
            if X[col].nunique() == y.nunique() and task_type == 'classification':
                if X[col].factorize()[0].tolist() == y.factorize()[0].tolist():
                    warnings.append({
                        "type": "leakage",
                        "count": 1,
                        "percentage": 1.0,
                        "msg": f"Column '{col}' is perfectly correlated with the target.",
                        "explanation": "This column is a direct leak of the target, making the model artificially perfect.",
                        "effect": "large",
                        "action_type": "drop_cols",
                        "cols": [col]
                    })
                    suspicious_cols.append(col)
                    continue

            try:
                x_single = series.copy()
                if str(x_single.dtype) in ['object', 'category', 'string', 'bool']:
                    x_single = LabelEncoder().fit_transform(x_single.astype(str))
                else:
                    x_single = x_single.fillna(x_single.median())
                    
                x_single_df = pd.DataFrame(x_single)
                
                if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(y):
                    corr = abs(series.corr(y))
                    if corr > 0.99:
                        warnings.append({
                            "type": "leakage",
                            "count": 1,
                            "percentage": corr,
                            "msg": f"Column '{col}' is almost a perfect copy of the target.",
                            "explanation": "Highly correlated columns act as a 'cheat sheet' for the model, ruining real-world performance.",
                            "effect": get_effect_label("leakage", corr),
                            "action_type": "drop_cols",
                            "cols": [col]
                        })
                        suspicious_cols.append(col)
                        continue
                
                if task_type == 'classification':
                    model = DecisionTreeClassifier(max_depth=3, random_state=42)
                    y_encoded = LabelEncoder().fit_transform(y.astype(str))
                    
                    min_class_count = pd.Series(y_encoded).value_counts().min()
                    cv_folds = max(2, min(3, min_class_count))
                    
                    if min_class_count < 2:
                        score = 0.0
                    else:
                        score = np.mean(cross_val_score(model, x_single_df, y_encoded, cv=cv_folds))
                        
                    if score > 0.95:
                        warnings.append({
                            "type": "leakage",
                            "count": 1,
                            "percentage": score,
                            "msg": f"Column '{col}' alone predicts the target with {score*100:.1f}% accuracy.",
                            "explanation": "This column contains future information not available during real prediction.",
                            "effect": get_effect_label("leakage", score),
                            "action_type": "drop_cols",
                            "cols": [col]
                        })
                        suspicious_cols.append(col)
                elif task_type == 'regression':
                    model = DecisionTreeRegressor(max_depth=3, random_state=42)
                    cv_folds = min(3, max(2, len(y) // 2))
                    if len(y) < 4:
                        score = 0.0
                    else:
                        score = np.mean(cross_val_score(model, x_single_df, y, cv=cv_folds, scoring='r2'))
                        
                    if score > 0.95:
                        warnings.append({
                            "type": "leakage",
                            "count": 1,
                            "percentage": score,
                            "msg": f"Column '{col}' alone predicts the target with R2 {score:.2f}.",
                            "explanation": "This column contains future information not available during real prediction.",
                            "effect": get_effect_label("leakage", score),
                            "action_type": "drop_cols",
                            "cols": [col]
                        })
                        suspicious_cols.append(col)
            except Exception:
                pass
                
    return warnings, suspicious_cols, silent_drops
