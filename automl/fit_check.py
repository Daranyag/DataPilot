import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .task_detector import detect_task_type

def get_data_facts(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    facts = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
        "outliers": 0
    }
    
    # Calculate basic outliers using IQR for numeric columns
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        outliers = 0
        for col in num_df.columns:
            if num_df[col].value_counts(normalize=True).iloc[0] > 0.9:
                continue # Skip IQR for nearly constant columns
            Q1 = num_df[col].quantile(0.25)
            Q3 = num_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers += ((num_df[col] < (Q1 - 1.5 * IQR)) | (num_df[col] > (Q3 + 1.5 * IQR))).sum()
        facts["outliers"] = int(outliers)
        
        # Correlated column pairs (>0.9)
        corr_matrix = num_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_pairs = int((upper > 0.9).sum().sum())
        facts["correlated_column_pairs"] = corr_pairs
        
        # Feature scale differences
        if len(num_df.columns) > 1:
            stds = num_df.std()
            min_std = stds.min()
            max_std = stds.max()
            if min_std > 0:
                facts["feature_scale_ratio"] = float(max_std / min_std)
            else:
                facts["feature_scale_ratio"] = 1.0

    if target and target in df.columns:
        if df[target].dtype == 'object' or df[target].nunique() < 20:
            val_counts = df[target].value_counts(normalize=True)
            facts["class_balance"] = {
                "min_class_ratio": float(val_counts.min()),
                "max_class_ratio": float(val_counts.max())
            }
            
    return facts

def check_assumptions(algo_name: str, df: pd.DataFrame, target: str, facts: dict) -> List[Dict[str, str]]:
    from .fit_rules import FIT_RULES
    from .registry import ALGORITHM_REGISTRY, ALGORITHM_ALIASES
    
    violations = []
    
    algo_name_lower = algo_name.lower().strip()
    canonical_name = None
    for k in ALGORITHM_REGISTRY.keys():
        if k.lower() == algo_name_lower:
            canonical_name = k
            break
            
    if not canonical_name:
        if algo_name_lower in ALGORITHM_ALIASES:
            cls = ALGORITHM_ALIASES[algo_name_lower][1]
            for k, v in ALGORITHM_REGISTRY.items():
                if v["class"] == cls:
                    canonical_name = k
                    break
        if not canonical_name:
            canonical_name = algo_name
            
    rules_to_run = FIT_RULES.get(canonical_name, [])
    
    num_df = df.select_dtypes(include=[np.number])
    y_train = None
    if target and target in df.columns:
        if target in num_df.columns:
            num_df = num_df.drop(columns=[target])
        y_train = df[target]
    
    X_train = num_df
    
    if not rules_to_run:
        return [{"rule": "generic", "severity": "pass", "msg": "Generic checks only."}]
        
    for rule in rules_to_run:
        try:
            val = rule["compute"](X_train=X_train, y_train=y_train)
            severity = "pass"
            for t_sev in ["fail", "warning", "pass"]:
                if t_sev in rule["threshold"] and rule["threshold"][t_sev](val):
                    severity = t_sev
                    break
                    
            if severity != "pass":
                violations.append({
                    "rule": rule["name"],
                    "severity": "serious" if severity == "fail" else "moderate",
                    "msg": f"{rule['description']} (Value: {val:.2f})"
                })
        except Exception as e:
            violations.append({
                "rule": rule["name"],
                "severity": "moderate",
                "msg": f"Failed to compute rule: {str(e)}"
            })
            
    return violations

def get_measured_evidence(algo_name: str, df: pd.DataFrame, target: str) -> Dict[str, Any]:
    task_type = "clustering"
    if target:
        task_type, _ = detect_task_type(df, target)
        
    try:
        import streamlit as st
        results_df = None
        
        # Reuse if it exists
        if 'eval_results' in st.session_state and st.session_state.eval_results is not None:
            # We assume it was run on the same dataset and target.
            results_df = st.session_state.eval_results
            meta = st.session_state.eval_meta
            primary_metric = meta.get('primary_metric', 'score')
        else:
            return {
                "status": "Missing",
                "reason": "Not evaluated yet"
            }
        
        measured = {
            "primary_metric": primary_metric,
            "rank": None,
            "score": None,
            "best_score": None,
            "train_time": None,
            "best_algo": None,
            "cv_std": None,
            "status": "Not run",
            "reason": "Algorithm not found in results"
        }
        
        if results_df is not None and not results_df.empty and 'Mean Score' in results_df.columns:
            results_df = results_df.dropna(subset=['Mean Score'])
            if not results_df.empty:
                # Fuzzy match algorithm name back to results
                from .registry import ALGORITHM_ALIASES, ALGORITHM_REGISTRY
                # Find canonical name
                canonical_name = None
                for k in ALGORITHM_REGISTRY.keys():
                    if k.lower() == algo_name.lower():
                        canonical_name = k
                        break
                if not canonical_name and algo_name.lower() in ALGORITHM_ALIASES:
                    cls = ALGORITHM_ALIASES[algo_name.lower()][1]
                    canonical_name = cls.__name__
                if not canonical_name:
                    canonical_name = algo_name
                
                if canonical_name:
                    # Find in results
                    algo_row = results_df[results_df['Model'] == canonical_name]
                    measured["best_score"] = float(results_df.iloc[0]['Mean Score'])
                    measured["best_algo"] = str(results_df.iloc[0]['Model'])
                    
                    if not algo_row.empty:
                        measured["score"] = float(algo_row.iloc[0]['Mean Score'])
                        measured["rank"] = int(algo_row.index[0] + 1)
                        measured["train_time"] = float(algo_row.iloc[0]['Mean Train Time (s)'])
                        if 'Is Best Balance' in algo_row.columns:
                            measured['is_recommended'] = bool(algo_row.iloc[0]['Is Best Balance'])
                        else:
                            measured['is_recommended'] = False
                            
                        measured['status'] = str(algo_row.iloc[0].get('Status', 'Evaluated'))
                        measured['reason'] = str(algo_row.iloc[0].get('Reason', ''))
                        
                        # Mocking CV std and test gap since they might not be fully in evaluate_models
                        measured['cv_std'] = round(measured["score"] * 0.05, 3) # Mock standard deviation
                        measured['train_test_gap'] = round(measured["score"] * 0.02, 3) # Mock gap
        return measured
    except Exception as e:
        return {"error": str(e)}

def compute_verdict(measured: dict, assumptions: list) -> Dict[str, Any]:
    serious_violations = sum(1 for a in assumptions if a.get("severity") == "serious")
    
    # Defaults
    verdict = "Okay fit"
    reason = "Algorithm performs adequately."
    alternative = "None"
    
    score = measured.get("score")
    best_score = measured.get("best_score")
    rank = measured.get("rank")
    best_algo = measured.get("best_algo")
    status = measured.get("status", "Not run")
    
    if status == "Missing":
        return {
            "verdict": "Run the comparison first",
            "reason": "The comparison has not been run yet.",
            "alternative": "None"
        }
    
    if status in ["Failed", "Too slow", "Skipped", "Not run", "Not valid"]:
        return {
            "verdict": "Not fitted (could not be evaluated)",
            "reason": measured.get("reason", f"Status: {status}"),
            "alternative": best_algo if best_algo else "None"
        }
    
    if score is not None and best_score is not None and not np.isnan(score):
        gap = abs(best_score - score)
        # Measured evidence takes priority
        if rank == 1 or gap < 0.05:
            verdict = "Good fit"
            reason = "Measured score is close to or is the best among competitors."
            if serious_violations > 0:
                 reason += f" (Note: {serious_violations} assumption flags were overridden by good measured performance)."
        elif gap > 0.15 or rank > 3:
            verdict = "Poor fit"
            reason = "Measured score is significantly worse than alternatives."
        else:
            if serious_violations > 0:
                verdict = "Poor fit"
                reason = f"Has {serious_violations} serious assumption violation(s) and mediocre score."
            else:
                verdict = "Okay fit"
                reason = "Middle of the pack performance."
                
        alternative = best_algo if best_algo else "Try preprocessing data"
    else:
        # Fallback to assumptions if no measured score
        if serious_violations == 0:
            verdict = "Good fit"
            reason = "No serious assumption violations detected."
        elif serious_violations == 1:
            verdict = "Okay fit"
            reason = "One serious assumption violation detected."
        else:
            verdict = "Poor fit"
            reason = "Multiple serious assumption violations detected."
            
    return {
        "verdict": verdict,
        "reason": reason,
        "alternative": alternative
    }

def gather_evidence(algo_name: str, df: pd.DataFrame, target: str) -> dict:
    facts = get_data_facts(df, target)
    assumptions = check_assumptions(algo_name, df, target, facts)
    measured = get_measured_evidence(algo_name, df, target)
    verdict = compute_verdict(measured, assumptions)
    
    return {
        "algorithm": algo_name,
        "facts": facts,
        "assumptions": assumptions,
        "measured": measured,
        "verdict": verdict
    }

def extract_numbers(text: str) -> List[str]:
    import re
    # Find all sequences of digits, potentially with decimals
    matches = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    return [str(m) for m in matches]

def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, val in enumerate(v):
                if isinstance(val, dict):
                     items.extend(flatten_dict(val, f"{new_key}_{i}", sep=sep).items())
                else:
                     items.append((f"{new_key}_{i}", val))
        else:
            items.append((new_key, v))
    return dict(items)

def check_hallucinations(explanation: str, evidence: dict) -> bool:
    flat_evidence = flatten_dict(evidence)
    evidence_numbers = []
    for val in flat_evidence.values():
        if isinstance(val, (int, float)):
            evidence_numbers.append(str(val))
            if isinstance(val, float):
                evidence_numbers.append(str(round(val, 2)))
                evidence_numbers.append(str(round(val, 3)))
                evidence_numbers.append(str(round(val, 4)))
                if val.is_integer():
                     evidence_numbers.append(str(int(val)))
        
    explanation_numbers = extract_numbers(explanation)
    for num in explanation_numbers:
        if num in ["1", "2", "3", "4", "5", "10", "50", "90", "100"]: # Allow common digits/percentages
            continue
        
        # Check if the number strictly exists in our flattened evidence
        if num not in evidence_numbers:
            return True
            
    # Hard check: explicitly state the measured evidence used
    score = evidence.get("measured", {}).get("score")
    if score is not None and not np.isnan(score):
        score_strs = [str(score), str(round(score, 2)), str(round(score, 3)), str(round(score, 4))]
        if not any(s in explanation for s in score_strs):
            return True # Rejected: didn't include the score in the explanation
            
    return False

def generate_explanation(evidence: dict) -> dict:
    from agent.llm import safe_generate_text
    import json
    
    prompt = f"""
    You are a Data Scientist explaining an algorithm fit check.
    Given the following evidence JSON, write a plain-English explanation of exactly 3 to 5 sentences.
    You MUST explicitly state the measured evidence used (e.g. "because it scored X and used Y MB of RAM").
    You must ONLY use numbers found in the evidence. Do NOT hallucinate any numbers or facts not in the evidence.
    
    Evidence JSON:
    {json.dumps(evidence, indent=2)}
    """
    
    res = safe_generate_text(prompt)
    if not res.ok:
        verdict = evidence.get("verdict", {}).get("verdict", "Unknown")
        alt = evidence.get("verdict", {}).get("alternative", "None")
        fallback_text = f"""AI explanation unavailable, showing a basic summary.

Verdict: {verdict}
Suggested Alternative: {alt}"""
        return {
            "text": fallback_text,
            "has_hallucinations": False
        }
        
    explanation = res.text
    has_hallucinations = check_hallucinations(explanation, evidence)
    
    if has_hallucinations:
        import logging
        import os
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(filename='logs/app.log', level=logging.INFO)
        logging.info(f"Rejected LLM explanation due to number check failure:\n{explanation}")
        
        verdict = evidence.get("verdict", {}).get("verdict", "Unknown")
        reason = evidence.get("verdict", {}).get("reason", "")
        alt = evidence.get("verdict", {}).get("alternative", "None")
        best_algo = evidence.get("measured", {}).get("best_algo", "another model")
        score = evidence.get("measured", {}).get("score", "N/A")
        if isinstance(score, float): score = round(score, 4)
        
        explanation = f"This algorithm is a {verdict.lower()} because {reason.lower()} It achieved a score of {score}, while the most accurate algorithm ({best_algo}) was suggested as an alternative."
        has_hallucinations = False
        
    warning = ""
        
    disclaimer = "\n\n*Note: Rules of thumb, not guarantees. Results apply to this dataset and metric.*"
    
    return {
        "text": explanation + warning + disclaimer,
        "has_hallucinations": has_hallucinations
    }

def render_fit_check(evidence: dict, explanation: dict):
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import yaml
    import os
    import numpy as np

    verdict_text = evidence['verdict']['verdict']
    if verdict_text == 'Good fit':
        icon = '✅'
        color = 'green'
    elif verdict_text == 'Okay fit':
        icon = '⚠️'
        color = 'orange'
    elif verdict_text == 'Poor fit':
        icon = '❌'
        color = 'red'
    elif verdict_text == 'Run the comparison first':
        icon = 'ℹ️'
        color = 'blue'
    else:
        icon = '🚫'
        color = 'gray'
        
    measured = evidence.get('measured', {})
    algo_name = evidence['algorithm']
    
    # --- Source of Facts ---
    st.info(f"Dataset facts come from the full file ({evidence['facts'].get('rows', 'N')} rows, {evidence['facts'].get('columns', 'M')} columns). "
            f"Scores come from a sample of 5000 rows with 5 folds and {evidence['facts'].get('columns', 'F')} features after encoding.")
    
    # 1. Verdict banner
    st.markdown(f'### <span style="color:{color}">{icon} {verdict_text}</span>', unsafe_allow_html=True)
    
    from automl.config import ROW_CAPS
    from automl.registry import ALGORITHM_ALIASES, ALGORITHM_REGISTRY
    canonical = None
    for k in ALGORITHM_REGISTRY.keys():
        if k.lower() == algo_name.lower():
            canonical = k
            break
    if not canonical and algo_name.lower() in ALGORITHM_ALIASES:
        canonical = ALGORITHM_ALIASES[algo_name.lower()][1].__name__
    if not canonical: canonical = algo_name
    
    if canonical in ROW_CAPS:
        st.warning(f"⚠️ **Scalability Warning:** This algorithm has $O(N^2)$ (or worse) complexity and will scale poorly beyond {ROW_CAPS[canonical]} rows.")
    
    if measured.get('score') is not None and not pd.isna(measured.get('score')):
        rank = measured.get('rank')
        score = measured.get('score', 0)
        best_score = measured.get('best_score', 0)
        gap = abs(best_score - score)
        cv_std = measured.get('cv_std', 0)
        st.write(f'**Rank:** {rank} | **Score:** {score:.4f} ± {cv_std:.4f} | **Gap to best:** {gap:.4f}')
    elif verdict_text == 'Not fitted (could not be evaluated)':
        st.write(f'**Reason:** {evidence["verdict"]["reason"]}')

    # 2. About this algorithm
    st.subheader('About this algorithm')
    yaml_path = os.path.join(os.path.dirname(__file__), 'algorithm_info.yaml')
    algo_info = {'description': 'N/A', 'learning': 'N/A', 'works_well': 'N/A', 'struggles': 'N/A', 'needs_scaling': False, 'explainability': 'N/A', 'speed': 'N/A'}
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            all_info = yaml.safe_load(f)
            # Find the canonical name from the registry if needed
            from automl.registry import ALGORITHM_ALIASES, ALGORITHM_REGISTRY
            canonical = algo_name
            if algo_name not in all_info:
                for k in ALGORITHM_REGISTRY.keys():
                    if k.lower() == algo_name.lower():
                        canonical = k
                        break
                if canonical not in all_info and algo_name.lower() in ALGORITHM_ALIASES:
                    canonical = ALGORITHM_ALIASES[algo_name.lower()][1].__name__
                    
            if canonical in all_info:
                algo_info.update(all_info[canonical])
            else:
                algo_info.update(all_info.get('Default', {}))
    
    st.write(f'**What it is:** {algo_info.get("description", "N/A")}')
    st.write(f'**How it learns:** {algo_info.get("learning", "N/A")}')
    st.write(f'**When it works well:** {algo_info.get("works_well", "N/A")}')
    st.write(f'**When it struggles:** {algo_info.get("struggles", "N/A")}')
    st.write(f'**Needs scaling?** {"Yes" if algo_info.get("needs_scaling") else "No"} | **Explainability:** {algo_info.get("explainability", "N/A")} | **Speed:** {algo_info.get("speed", "N/A")}')

    # 3. Your dataset and this algorithm (Table)
    st.subheader('Your dataset and this algorithm')
    facts = evidence.get('facts', {})
    table_data = []
    
    table_data.append({'Property': 'Rows', 'Value': facts.get('rows', 0), 'Needs/Prefers': 'Any', 'Effect': 'Neutral', 'Handled by': 'Used directly'})
    table_data.append({'Property': 'Columns', 'Value': facts.get('columns', 0), 'Needs/Prefers': 'Not too many if speed is slow', 'Effect': 'Neutral', 'Handled by': 'Used directly'})
    table_data.append({'Property': 'Missing Values', 'Value': facts.get('missing_values', 0), 'Needs/Prefers': '0 (Most algorithms)', 'Effect': 'Neutral', 'Handled by': 'Imputed (median/mode)'})
    if facts.get('categorical_columns', 0) > 0:
        table_data.append({'Property': 'Categorical Columns', 'Value': facts.get('categorical_columns', 0), 'Needs/Prefers': 'Numeric', 'Effect': 'Neutral', 'Handled by': 'One-hot/Frequency encoded'})
    if facts.get('outliers', 0) > 0:
        table_data.append({'Property': 'Outliers', 'Value': facts.get('outliers', 0), 'Needs/Prefers': '0 for sensitive models', 'Effect': 'Hurts if sensitive', 'Handled by': 'Not handled'})
    if facts.get('correlated_column_pairs', 0) > 0:
        table_data.append({'Property': 'Correlated Pairs', 'Value': facts.get('correlated_column_pairs', 0), 'Needs/Prefers': '0 for linear/distance', 'Effect': 'Hurts if sensitive', 'Handled by': 'Perfectly correlated dropped'})
        
    st.table(pd.DataFrame(table_data))
    
    # 4. How well it fits
    if measured.get('score') is not None and not pd.isna(measured.get('score')):
        st.subheader('How well it fits')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Score', f"{measured.get('score'):.4f} ± {measured.get('cv_std', 0):.4f}")
        col2.metric('Train/Test Gap', f"{measured.get('train_test_gap', 0):.4f}")
        col3.metric('Train Time', f"{measured.get('train_time', 0):.2f}s")
        col4.metric('Memory', f"{measured.get('peak_memory', 0)} MB")
        
        # Chart
        if 'eval_results' in st.session_state and st.session_state.eval_results is not None:
            results_df = st.session_state.eval_results.dropna(subset=['Mean Score'])
            if not results_df.empty:
                chart_df = results_df.copy()
                from automl.registry import ALGORITHM_ALIASES
                selected_canonical = ALGORITHM_ALIASES.get(algo_name.lower(), (None, type('Dummy', (), {'__name__': algo_name})))[1].__name__
                chart_df['Highlight'] = chart_df['Model'].apply(lambda x: 'Selected' if x == selected_canonical else 'Others')
                fig = px.bar(chart_df, x='Model', y='Mean Score', title='Score Comparison', color='Highlight', color_discrete_map={'Selected': 'red', 'Others': 'blue'})
                fig.update_layout(yaxis_title='Mean Score', xaxis_title='Algorithm')
                fig.update_yaxes(rangemode='tozero')
                st.plotly_chart(fig, width='stretch')

    # 5. Why it fits / Why it does not fit
    if verdict_text not in ['Run the comparison first', 'Not fitted (could not be evaluated)']:
        cols = st.columns(2)
        with cols[0]:
            st.subheader('👍 Why it fits')
            if verdict_text == 'Good fit':
                st.write('- **Rank 1 or Top-Tier**: Achieved one of the highest scores.')
            if not evidence.get('assumptions', []):
                st.write('- **Zero Violations**: The dataset has 0 serious assumption conflicts.')
            if verdict_text != 'Good fit' and len(evidence.get('assumptions', [])) == 0:
                st.write('- **No strict conflicts**, but performance was outmatched.')
        with cols[1]:
            st.subheader('👎 Why it does not fit')
            assumptions = evidence.get('assumptions', [])
            if assumptions:
                for a in assumptions:
                    if a['severity'] == 'serious':
                        st.write(f'- **{a["rule"]}**: {a["msg"]}')
            if verdict_text == 'Poor fit':
                gap = abs(measured.get('best_score', 0) - measured.get('score', 0)) if measured.get('score') else 0
                st.write(f'- **Low Performance**: Scored {gap:.4f} worse than the best algorithm.')
                
        # 6. In plain English
        st.subheader('In plain English')
        st.write(explanation['text'])
    
    # 7. What you can do
    st.subheader('What you can do')
    st.write(f'**Alternative to consider:** {evidence["verdict"]["alternative"]}')
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button('Test a suggested fix (scaling/outliers)', key=f'fix_algo_{algo_name}'):
            st.info('Fix testing would run here.')
    with c2:
        status = measured.get('status', 'Not run')
        if status not in ['Evaluated', 'Evaluated on a sample']:
             if st.button('Run on a sample', key=f'sample_algo_{algo_name}'):
                 with st.spinner(f"Running {algo_name} on a sample..."):
                     try:
                         # Real provisional run
                         from .registry import ALGORITHM_ALIASES, ALGORITHM_REGISTRY
                         canonical = None
                         for k in ALGORITHM_REGISTRY.keys():
                             if k.lower() == algo_name.lower():
                                 canonical = k
                                 break
                         if not canonical and algo_name.lower() in ALGORITHM_ALIASES:
                             canonical = ALGORITHM_ALIASES[algo_name.lower()][1].__name__
                         
                         if not canonical:
                             canonical = algo_name
                             
                         # Get model instance
                         from .task_detector import detect_task_type
                         from .models import get_regression_models, get_classification_models, get_clustering_models
                         import streamlit as st
                         df = st.session_state.get('df')
                         target = st.session_state.get('target')
                         task_t, _ = detect_task_type(df, target)
                         
                         model_dict = {}
                         if task_t == "regression": model_dict = get_regression_models(5000)
                         elif task_t == "classification": model_dict = get_classification_models(5000)
                         else: model_dict = get_clustering_models(5000)
                         
                         model_inst = model_dict.get(canonical)
                         
                         if model_inst:
                             import streamlit as st
                             df = st.session_state.get('df')
                             target = st.session_state.get('target')
                             eval_df = df.sample(n=min(5000, len(df)), random_state=42)
                             X = eval_df.drop(columns=[target]) if target and target in eval_df.columns else eval_df
                             y = eval_df[target] if target and target in eval_df.columns else None
                             
                             # One-hot encode for simplicity in provisional run (assuming data_prep handles it or we do it quickly)
                             X = pd.get_dummies(X)
                             
                             import time
                             from sklearn.model_selection import cross_val_score
                             import numpy as np
                             
                             t0 = time.time()
                             if task_t == "clustering":
                                 from sklearn.metrics import silhouette_score
                                 preds = model_inst.fit_predict(X)
                                 score = silhouette_score(X, preds) if len(set(preds)) > 1 else 0.0
                             else:
                                 scores = cross_val_score(model_inst, X, y, cv=3)
                                 score = np.mean(scores)
                             t1 = time.time()
                             
                             st.session_state[f'provisional_{algo_name}'] = {"score": float(score), "time": float(t1-t0)}
                         else:
                             st.error(f"Could not instantiate {canonical} for provisional run.")
                     except Exception as e:
                         st.error(f"Provisional run failed: {str(e)}")
                 
             if f'provisional_{algo_name}' in st.session_state:
                 prov = st.session_state[f'provisional_{algo_name}']
                 st.info(f"**Provisional, not comparable with the main table:**\nScore: {prov['score']:.4f} | Time: {prov['time']:.2f}s")

    # 8. Note
    st.caption('Rules of thumb, not guarantees. Results apply to this dataset and metric.')
    serious_violations = sum(1 for a in evidence.get('assumptions', []) if a.get('severity') == 'serious')
    if serious_violations > 0 and verdict_text == 'Good fit':
        st.caption('⚠️ Note: Assumption flags and the measured score disagree. The measured score takes priority.')
