"""
Two-Stage (Successive Halving) AutoML Evaluation Engine.
Performs fast screening, full cross-validation, dynamic timing, and recommendation logic.
"""

import time
import tracemalloc
import platform
import psutil
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, List
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import silhouette_score

from .config import (
    DEFAULT_TOTAL_TIME_BUDGET,
    DEFAULT_ALGO_TIME_BUDGET
)
from .strategy import determine_strategy
from .data_prep import optimize_and_clean_data
from .models import get_regression_models, get_classification_models, get_clustering_models
from .safety import apply_row_cap, predict_execution_time, run_with_timeout
from .caching import compute_cache_key, get_cached_evaluation, set_cached_evaluation

def get_hardware_info() -> str:
    """Returns basic hardware system specs."""
    try:
        ram = f"{round(psutil.virtual_memory().total / (1024.**3))} GB"
        cpu = platform.processor() or "Unknown CPU"
        return f"CPU: {cpu}, RAM: {ram}"
    except Exception:
        return "Hardware info unavailable"

def measure_efficiency(
    pipeline: Any, 
    X: pd.DataFrame, 
    y: Optional[pd.Series], 
    task_type: str,
    n_repeats: int = 1
) -> Tuple[float, float, float]:
    """
    Measures mean training time, prediction time (per 1k rows), and peak memory (MB).
    """
    train_times = []
    pred_times = []
    peak_mems = []
    
    X_pred = X.sample(n=min(1000, len(X)), random_state=42)
    
    for _ in range(max(1, n_repeats)):
        tracemalloc.start()
        t0 = time.perf_counter()
        
        if task_type == "clustering":
            pipeline.fit_predict(X)
        else:
            pipeline.fit(X, y)
            
        t_train = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        train_times.append(t_train)
        peak_mems.append(peak_mem / (1024 * 1024))
        
        if task_type != "clustering":
            t0 = time.perf_counter()
            pipeline.predict(X_pred)
            t_pred = (time.perf_counter() - t0) * (1000.0 / len(X_pred))
            pred_times.append(t_pred)
        else:
            pred_times.append(0.0)
            
    return float(np.mean(train_times)), float(np.mean(pred_times)), float(np.mean(peak_mems))

def compute_recommendations(df_results: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    if df_results.empty:
        return df_results
        
    df_results['Is Most Accurate'] = False
    df_results['Is Fastest'] = False
    df_results['Is Best Balance'] = False
    
    df_results['Is Suspicious'] = False
    
    if 'Score Std Dev' in df_results.columns:
        suspicious_mask = (df_results['Mean Score'] >= 0.999) & (df_results['Score Std Dev'] < 0.01)
        df_results.loc[suspicious_mask, 'Status'] = 'Suspicious result'
        df_results.loc[suspicious_mask, 'Is Suspicious'] = True
    
    if 'Status' in df_results.columns:
        valid_mask = df_results['Mean Score'].notna() & (~df_results['Status'].isin(['Not valid', 'Failed', 'Too slow', 'Skipped', 'Suspicious result']))
        if not valid_mask.any():
            valid_mask = df_results['Mean Score'].notna() & (~df_results['Status'].isin(['Not valid', 'Failed', 'Too slow', 'Skipped']))
    else:
        valid_mask = df_results['Mean Score'].notna()
        
    if not valid_mask.any():
        return df_results
        
    valid_df = df_results[valid_mask]
    
    if higher_is_better:
        best_score = valid_df['Mean Score'].max()
        best_acc_idx = valid_df.loc[valid_df['Mean Score'] == best_score].index[0]
    else:
        best_score = valid_df['Mean Score'].min()
        best_acc_idx = valid_df.loc[valid_df['Mean Score'] == best_score].index[0]
        
    df_results.at[best_acc_idx, 'Is Most Accurate'] = True
    
    df_results['Is Fastest Predict'] = False
    
    fastest_train = valid_df['Mean Train Time (s)'].min()
    fastest_train_idx = valid_df[valid_df['Mean Train Time (s)'] == fastest_train].index
    df_results.loc[fastest_train_idx, 'Is Fastest'] = True
    
    if 'Pred Time per 1k (s)' in valid_df.columns:
        fastest_pred = valid_df['Pred Time per 1k (s)'].min()
        fastest_pred_idx = valid_df[valid_df['Pred Time per 1k (s)'] == fastest_pred].index
        df_results.loc[fastest_pred_idx, 'Is Fastest Predict'] = True
    
    std_dev = valid_df['Mean Score'].std()
    if pd.isna(std_dev) or std_dev == 0:
        std_dev = 0.01 
        
    if higher_is_better:
        top_tier = valid_df[valid_df['Mean Score'] >= (best_score - std_dev)]
    else:
        top_tier = valid_df[valid_df['Mean Score'] <= (best_score + std_dev)]
        
    df_results['Why Recommended'] = ""
    
    best_balance_idx = None
    for idx_a, row_a in top_tier.iterrows():
        dominated = False
        for idx_b, row_b in valid_df.iterrows():
            if idx_a == idx_b: continue
            
            if higher_is_better:
                score_ok = row_b['Mean Score'] >= row_a['Mean Score'] - std_dev
                score_better = row_b['Mean Score'] > row_a['Mean Score']
            else:
                score_ok = row_b['Mean Score'] <= row_a['Mean Score'] + std_dev
                score_better = row_b['Mean Score'] < row_a['Mean Score']
                
            train_ok = row_b['Mean Train Time (s)'] <= row_a['Mean Train Time (s)']
            train_better = row_b['Mean Train Time (s)'] < row_a['Mean Train Time (s)']
            
            pred_ok = row_b.get('Pred Time per 1k (s)', 0) <= row_a.get('Pred Time per 1k (s)', 0)
            pred_better = row_b.get('Pred Time per 1k (s)', 0) < row_a.get('Pred Time per 1k (s)', 0)
            
            mem_ok = row_b.get('Python-tracked Peak Memory (approximate) (MB)', 0) <= row_a.get('Python-tracked Peak Memory (approximate) (MB)', 0)
            mem_better = row_b.get('Python-tracked Peak Memory (approximate) (MB)', 0) < row_a.get('Python-tracked Peak Memory (approximate) (MB)', 0)
            
            if score_ok and train_ok and pred_ok and mem_ok:
                if score_better or train_better or pred_better or mem_better:
                    dominated = True
                    break
        if not dominated:
            best_balance_idx = idx_a
            break
            
    if best_balance_idx is None:
        best_balance_idx = top_tier.sort_values(by='Mean Train Time (s)', ascending=True).index[0]
        
    df_results.at[best_balance_idx, 'Is Best Balance'] = True
    row_bb = df_results.loc[best_balance_idx]
    df_results.at[best_balance_idx, 'Why Recommended'] = f"Score: {row_bb['Mean Score']:.4f} | Train: {row_bb['Mean Train Time (s)']:.4f}s | Pred: {row_bb.get('Pred Time per 1k (s)', 0):.4f}s | Mem: {row_bb.get('Python-tracked Peak Memory (approximate) (MB)', 0)}MB (Undominated)"
    
    return df_results

def evaluate_models(
    df: pd.DataFrame,
    task_type: str,
    target_col: Optional[str] = None,
    total_time_budget: float = DEFAULT_TOTAL_TIME_BUDGET,
    algo_time_budget: float = DEFAULT_ALGO_TIME_BUDGET,
    progress_callback: Optional[Callable[[str, int, int, str, pd.DataFrame], None]] = None,
    use_pca: Optional[bool] = None,
    columns_to_drop: Optional[List[str]] = None,
    selected_models: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    start_total_time = time.perf_counter()
    
    if columns_to_drop and len(columns_to_drop) > 0:
        df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
    
    # 1. Check Cache
    cache_key = compute_cache_key(df, task_type, target_col, {
        "total_budget": total_time_budget,
        "algo_budget": algo_time_budget
    })
    cached = get_cached_evaluation(cache_key)
    if cached:
        cached_df, cached_metric, cached_meta = cached
        cached_meta["is_cached"] = True
        return cached_df, cached_metric, cached_meta
        
    # 2. Clean & Preprocess Data (ONE shared preprocessor for everything)
    df_clean, prep_summary = optimize_and_clean_data(df, target_col)
    
    # 3. Determine Adaptive Strategy
    strategy = determine_strategy(df_clean, task_type, target_col)
    
    X = df_clean.drop(columns=[target_col]) if target_col and target_col in df_clean.columns else df_clean
    y = df_clean[target_col] if target_col and target_col in df_clean.columns else None
    
    if y is not None and y.isna().sum() > 0:
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
    # Auto PCA for clustering
    if use_pca is True or (use_pca is None and len(X.columns) > 20):
        prep_summary["pca_used"] = True
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(10, len(X.columns)))
        X_pca = pca.fit_transform(X)
        X = pd.DataFrame(X_pca, index=X.index)
            
    # Define tasks & models
    if task_type == "regression":
        all_models = get_regression_models(n_samples=len(X))
        primary_metric = 'r2'
        higher_is_better = True
        scoring = {'r2': 'r2', 'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error'}
    elif task_type == "classification":
        all_models = get_classification_models(n_samples=len(X))
        class_counts = y.value_counts(normalize=True)
        is_imbalanced = class_counts.max() > 0.7
        primary_metric = 'f1' if is_imbalanced else 'accuracy'
        higher_is_better = True
        scoring = {'accuracy': 'accuracy', 'f1': 'f1_macro', 'precision': 'precision_macro', 'recall': 'recall_macro'}
    else: # clustering
        all_models = get_clustering_models(n_samples=len(X))
        primary_metric = 'silhouette'
        higher_is_better = True
        scoring = {'silhouette': 'silhouette'}

    if selected_models:
        all_models = {k: v for k, v in all_models.items() if k in selected_models}

    n_repeats = 1
    
    # STAGE 1: SCREENING (If medium or large dataset)
    models_for_stage2 = list(all_models.keys())
    stage1_scores = {}
    
    if strategy["use_screening"] and not selected_models:
        sample_size = strategy["stage1_sample_size"]
        if y is not None and task_type == "classification":
            sample_idx = X.groupby(y, group_keys=False).apply(
                lambda s: s.sample(min(len(s), max(1, int(sample_size * len(s) / len(X)))), random_state=42)
            ).index
        else:
            sample_idx = X.sample(n=min(sample_size, len(X)), random_state=42).index
            
        X_s1 = X.loc[sample_idx]
        y_s1 = y.loc[sample_idx] if y is not None else None
        
        total_s1 = len(all_models)
        for idx, (m_name, model_inst) in enumerate(all_models.items(), 1):
            if progress_callback:
                progress_callback("Stage 1: Screening", idx, total_s1, m_name, pd.DataFrame())
                
            try:
                if task_type == "clustering":
                    import warnings
                    from sklearn.exceptions import ConvergenceWarning
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        preds = model_inst.fit_predict(X_s1)
                        if len(set(preds)) > 1:
                            score = silhouette_score(X_s1, preds)
                        else:
                            score = -999.0
                else:
                    cv = KFold(n_splits=strategy["stage1_folds"], shuffle=True, random_state=42)
                    import warnings
                    from sklearn.exceptions import ConvergenceWarning
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        res = cross_validate(model_inst, X_s1, y_s1, cv=cv, scoring=scoring)
                    score = float(np.mean(res[f'test_{primary_metric}']))
                    if primary_metric in ['rmse', 'mae']:
                        score = -score
                stage1_scores[m_name] = score
            except Exception:
                stage1_scores[m_name] = -999.0
                
        sorted_s1 = sorted(stage1_scores.items(), key=lambda x: x[1], reverse=higher_is_better)
        top_k = strategy["stage2_top_k"]
        models_for_stage2 = [name for name, _ in sorted_s1[:top_k]]
        
    # STAGE 2: FULL EVALUATION
    results = []
    total_s2 = len(models_for_stage2)
    
    for idx, m_name in enumerate(models_for_stage2, 1):
        if time.perf_counter() - start_total_time > total_time_budget:
            results.append({
                "Model": m_name,
                "Status": "Not run: total time budget reached",
                "Reason": "Budget exhausted",
                "Mean Score": np.nan,
                "Score Std Dev": np.nan,
                "Mean Train Time (s)": np.nan,
                "Pred Time per 1k (s)": np.nan,
                "Peak Memory (MB)": np.nan,
                "Rows Evaluated": "-",
                "Folds": "-",
                "Stage 1 Screening Score": round(stage1_scores.get(m_name, np.nan), 4) if m_name in stage1_scores else "N/A"
            })
            continue
            
        if progress_callback:
            progress_callback("Stage 2: Full Evaluation", idx, total_s2, m_name, pd.DataFrame(results))
            
        model_inst = all_models[m_name]
        
        # 1. Apply Row Capping if scaling challenged
        X_eval, y_eval, cap_label = apply_row_cap(m_name, X, y)
        
        # 2. Check Execution Time Predictor
        should_skip, pred_sec, skip_reason = predict_execution_time(
            m_name, model_inst, X_eval, y_eval, len(X_eval), algo_time_budget
        )
        if should_skip:
            if strategy.get("allow_slow_sample", False) and len(X_eval) > 5000:
                    cap_label = "Sampled to 5k rows (predicted slow)"
                    sample_indices = X_eval.sample(n=min(5000, len(X_eval)), random_state=42).index
                    X_eval = X_eval.loc[sample_indices]
                    if y_eval is not None:
                        y_eval = y_eval.loc[sample_indices]
            else:
                res_dict = {
                        "Model": m_name,
                        "Status": "Too slow",
                        "Reason": f"Predicted to exceed {algo_time_budget}s budget",
                        "Mean Score": np.nan,
                        "Score Std Dev": np.nan,
                        "Mean Train Time (s)": np.nan,
                        "Pred Time per 1k (s)": np.nan,
                        "Peak Memory (MB)": np.nan,
                        "Rows Evaluated": "-",
                        "Folds": "-",
                        "Stage 1 Screening Score": round(stage1_scores.get(m_name, np.nan), 4) if m_name in stage1_scores else "N/A"
                    }
                results.append(res_dict)
                continue
            
        # 3. Perform Cross-Validation & Timing
        try:
            n_folds = strategy["stage2_folds"]
            if task_type == "clustering":
                import warnings
                from sklearn.exceptions import ConvergenceWarning
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)
                    preds = model_inst.fit_predict(X_eval)
                    n_clusters = len(set(p for p in preds if p != -1))
                    
                    if n_clusters > 1 and len(set(preds)) > 1:
                        # use max 5000 for silhouette score as per instruction
                        if len(X_eval) > 5000:
                            sample_idx = np.random.choice(len(X_eval), 5000, replace=False)
                            score = silhouette_score(X_eval.iloc[sample_idx], preds[sample_idx])
                        else:
                            score = silhouette_score(X_eval, preds)
                    else:
                        score = np.nan
                        
                    std_dev = 0.0
                    train_time, pred_time, peak_mem = measure_efficiency(model_inst, X_eval, y_eval, task_type, n_repeats)
                    has_convergence_warning = any(issubclass(warn.category, ConvergenceWarning) for warn in w)
                    
                    cluster_sizes = dict(zip(*np.unique(preds, return_counts=True)))
                    noise_pts = cluster_sizes.get(-1, 0)
                    if -1 in cluster_sizes:
                        del cluster_sizes[-1]
                    
                    if np.isnan(score):
                        status_str = "Not valid"
                        reason_str = "Found only 1 cluster or everything is noise."
                    else:
                        status_str = "Evaluated on a sample" if cap_label else "Evaluated"
                        reason_str = "Success"
                        if has_convergence_warning:
                            reason_str = "did not fully converge (max iterations reached)"
                            
                    res_dict = {
                        "Model": m_name,
                        "Status": status_str,
                        "Reason": reason_str,
                        "Mean Score": round(score, 4) if not np.isnan(score) else np.nan,
                        "Score Std Dev": round(std_dev, 4) if not np.isnan(score) else np.nan,
                        "Mean Train Time (s)": round(train_time, 4),
                        "Pred Time per 1k (s)": round(pred_time, 4),
                        "Peak Memory (MB)": round(peak_mem, 2),
                        "Rows Evaluated": f"{len(X_eval):,}" + (f" ({cap_label})" if cap_label else ""),
                        "Folds": f"k={n_clusters}",
                        "Clusters Found": n_clusters,
                        "Noise Points": noise_pts,
                        "Cluster Sizes": str(cluster_sizes),
                        "Stage 1 Screening Score": round(stage1_scores.get(m_name, np.nan), 4) if m_name in stage1_scores else "N/A"
                    }
                    results.append(res_dict)
                    continue
            else:
                if task_type == "classification":
                    min_class = y_eval.value_counts().min()
                    if min_class < 2:
                        cv_folds = min(n_folds, len(y_eval))
                        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    else:
                        cv_folds = min(n_folds, min_class)
                        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                else:
                    cv_folds = min(n_folds, len(y_eval))
                    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    
                import warnings
                from sklearn.exceptions import ConvergenceWarning
                def _do_cv():
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", ConvergenceWarning)
                        cv_res = cross_validate(model_inst, X_eval, y_eval, cv=cv, scoring=scoring)
                        eff = measure_efficiency(model_inst, X_eval, y_eval, task_type, n_repeats)
                        has_convergence_warning = any(issubclass(warn.category, ConvergenceWarning) for warn in w)
                                
                    return cv_res, eff, has_convergence_warning
                    
                ok, output, err_msg = run_with_timeout(_do_cv, timeout_seconds=algo_time_budget)
                if not ok:
                    import logging
                    logging.error(f"Algorithm {m_name} failed: {err_msg}")
                    results.append({
                        "Model": m_name,
                        "Status": "Failed",
                        "Reason": err_msg if err_msg else "Execution error or timeout.",
                        "Mean Score": np.nan,
                        "Score Std Dev": np.nan,
                        "Mean Train Time (s)": np.nan,
                        "Pred Time per 1k (s)": np.nan,
                        "Peak Memory (MB)": np.nan,
                        "Rows Evaluated": "-",
                        "Folds": f"{cv_folds}-fold",
                        "Stage 1 Screening Score": round(stage1_scores.get(m_name, np.nan), 4) if m_name in stage1_scores else "N/A"
                    })
                    continue
                    
                cv_res, (train_time, pred_time, peak_mem), has_convergence_warning = output
                raw_scores = cv_res[f'test_{primary_metric}']
                if primary_metric in ['rmse', 'mae']:
                    raw_scores = -raw_scores
                score = float(np.mean(raw_scores))
                std_dev = float(np.std(raw_scores))

            rows_str = f"{len(X_eval):,}"
            if cap_label:
                rows_str += f" ({cap_label})"
                
            status_str = "Evaluated on a sample" if cap_label else "Evaluated"
            reason_str = "Success"
            if has_convergence_warning:
                reason_str = "did not fully converge (max iterations reached)"
                
            res_dict = {
                "Model": m_name,
                "Status": status_str,
                "Reason": reason_str,
                "Mean Score": round(score, 4),
                "Score Std Dev": round(std_dev, 4),
                "Mean Train Time (s)": round(train_time, 4),
                "Pred Time per 1k (s)": round(pred_time, 4),
                "Peak Memory (MB)": round(peak_mem, 2),
                "Rows Evaluated": rows_str,
                "Folds": f"{cv_folds}-fold",
                "Stage 1 Screening Score": round(stage1_scores.get(m_name, np.nan), 4) if m_name in stage1_scores else "N/A"
            }
            results.append(res_dict)
            
        except Exception as e:
            import logging
            logging.error(f"Algorithm {m_name} failed: {e}")
            results.append({
                "Model": m_name,
                "Status": "Failed",
                "Reason": f"Execution error: {str(e)[:80]}",
                "Mean Score": np.nan,
                "Score Std Dev": np.nan,
                "Mean Train Time (s)": np.nan,
                "Pred Time per 1k (s)": np.nan,
                "Peak Memory (MB)": np.nan,
                "Rows Evaluated": "-",
                "Folds": "-",
                "Stage 1 Screening Score": round(stage1_scores.get(m_name, np.nan), 4) if m_name in stage1_scores else "N/A"
            })
            
    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        # Check against baseline
        baseline_score = None
        baseline_std = 0.0
        for b_name in ['Baseline (Predicts Most Common)', 'Baseline (Predicts Mean)']:
            b_row = df_results[df_results['Model'] == b_name]
            if not b_row.empty and not pd.isna(b_row.iloc[0]['Mean Score']):
                baseline_score = b_row.iloc[0]['Mean Score']
                baseline_std = b_row.iloc[0].get('Score Std Dev', 0.0)
                break
                
        if baseline_score is not None:
            for idx, row in df_results.iterrows():
                if row['Model'].startswith('Baseline'):
                    continue
                if row['Status'] not in ['Failed', 'Too slow', 'Skipped', 'Not valid']:
                    if higher_is_better and row['Mean Score'] <= baseline_score + baseline_std:
                        df_results.at[idx, 'Status'] = 'Weak'
                        df_results.at[idx, 'Reason'] = 'no better than guessing'
                    elif not higher_is_better and row['Mean Score'] >= baseline_score - baseline_std:
                        df_results.at[idx, 'Status'] = 'Weak'
                        df_results.at[idx, 'Reason'] = 'no better than guessing'

        try:
            df_results = compute_recommendations(df_results, higher_is_better)
            df_results['Hardware Info'] = get_hardware_info()
            df_results = df_results.sort_values(by="Mean Score", ascending=not higher_is_better).reset_index(drop=True)
        except Exception as e:
            import logging
            logging.error(f"Failed during ranking/scoring: {e}", exc_info=True)
        
    total_elapsed = round(time.perf_counter() - start_total_time, 2)
    
    meta_info = {
        "strategy": strategy,
        "total_elapsed_seconds": total_elapsed,
        "prep_summary": prep_summary,
        "is_cached": False,
        "primary_metric": primary_metric,
        "hardware_info": get_hardware_info(),
        "X_clust": X if task_type == 'clustering' else None  # Save this for UI 2D PCA & table
    }
    
    # Store in Cache
    set_cached_evaluation(cache_key, df_results, primary_metric, meta_info)
    
    return df_results, primary_metric, meta_info
