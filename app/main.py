import streamlit as st
import pandas as pd
import os
import tempfile
import plotly.express as px
from dotenv import load_dotenv

# Load API key
load_dotenv()
if "GEMINI_API_KEY" not in os.environ and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import agent and automl modules
from agent.profiler import profile_dataset
from agent.graph import build_graph
from automl.task_detector import detect_task_type
from automl.evaluation import evaluate_models
from automl.profiler import generate_profile
from automl.config import DEFAULT_TOTAL_TIME_BUDGET, DEFAULT_ALGO_TIME_BUDGET
from agent.llm import safe_generate_text
import json

st.set_page_config(page_title="DataPilot", layout="wide")

st.title("DataPilot: Agent for Choosing the Right ML Algorithm")

# Session state
if "df" not in st.session_state:
    st.session_state.df = None
if "df_path" not in st.session_state:
    st.session_state.df_path = None
if "agent_workflow" not in st.session_state:
    st.session_state.agent_workflow = build_graph()
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "eval_meta" not in st.session_state:
    st.session_state.eval_meta = None
if "ai_status" not in st.session_state:
    st.session_state.ai_status = "Unknown"

# AI Status Sidebar
st.sidebar.subheader("🤖 AI Analysis")
if st.sidebar.button("Test AI connection"):
    with st.spinner("Testing..."):
        res = safe_generate_text("Say hello")
        if res.ok:
            st.session_state.ai_status = "Connected"
        else:
            st.session_state.ai_status = f"Unavailable ({res.error_type})"

status_color = "green" if "Connected" in st.session_state.ai_status else "red" if "Unavailable" in st.session_state.ai_status else "gray"
st.sidebar.markdown(f"**Status:** <span style='color:{status_color}'>{st.session_state.ai_status}</span>", unsafe_allow_html=True)
st.sidebar.divider()

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    if st.session_state.df is None or st.sidebar.button("Reload Dataset"):
        try:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
                
            if df.empty:
                st.sidebar.error("The uploaded file is empty.")
            elif len(df) < 5:
                st.sidebar.error("The dataset has too few rows to be analyzed.")
            else:
                st.session_state.df = df
                
                # Save temp file for sandbox
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                df.to_csv(temp_path, index=False)
                st.session_state.df_path = temp_path
                st.sidebar.success("Data loaded!")
        except Exception as e:
            import logging
            logging.error(f"Error loading file: {e}")
            st.sidebar.error(f"Could not read the file. Please ensure it's a valid CSV. Error: {str(e)[:50]}")
            st.session_state.df = None

        # Compute Profile if not cached
        if st.session_state.df is not None and 'profile' not in st.session_state:
            st.session_state.profile = generate_profile(df)
            
            # Generate LLM Summary safely from hard numbers
            prompt = f"""
            You are a senior data scientist. Write a short, plain-English executive summary (2-3 paragraphs max) of this dataset.
            DO NOT invent or guess any numbers. ONLY use the data provided in this JSON profile.
            Mention the dataset size, potential data quality issues (missing/duplicates/outliers), and highlight any strong correlations if they exist.
            
            Profile JSON: {json.dumps(st.session_state.profile)}
            """
            res = safe_generate_text(prompt)
            if res.ok:
                st.session_state.profile_summary = res.text
            else:
                p = st.session_state.profile
                fallback = (
                    "**AI explanation unavailable, showing a basic summary.**\n\n"
                    f"This dataset has {p.get('rows', 0)} rows and {p.get('columns', 0)} columns. "
                    f"It contains {len(p.get('numeric_cols', []))} numeric and {len(p.get('categorical_cols', []))} categorical columns.\n\n"
                    f"**Missing values:** {sum(p.get('missing_values', {}).values())}\n"
                    f"**Duplicates:** {p.get('duplicates', 0)}"
                )
                st.session_state.profile_summary = fallback

if st.session_state.df is not None:
    df = st.session_state.df
    
    with st.expander("Raw Dataset Preview", expanded=False):
        st.dataframe(df.head())
        
    tab_report, tab1, tab_automl = st.tabs(["📊 Data Report", "💬 Ask your data", "🤖 Find best algorithm (AutoML)"])
    
    # === TAB: DATA REPORT ===
    with tab_report:
        try:
            st.subheader("Executive Summary")
            if 'profile_summary' in st.session_state:
                st.write(st.session_state.profile_summary)
            
            st.divider()
            st.subheader("Data Quality Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Missing Values Bar Chart
                profile = st.session_state.get('profile', {})
                missing = profile.get('missing_values', {})
                if missing:
                    df_missing = pd.DataFrame(list(missing.items()), columns=['Column', 'Missing Count'])
                    fig_missing = px.bar(df_missing, x='Column', y='Missing Count', title="Missing Values per Column", color='Missing Count', color_continuous_scale='Reds')
                    st.plotly_chart(fig_missing, width='stretch')
                else:
                    st.success("No missing values found! 🎉")
                    
            with col2:
                # Correlation Heatmap
                num_cols = profile.get('numeric_cols', [])
                if len(num_cols) > 1:
                    corr_matrix = st.session_state.df[num_cols].corr()
                    fig_corr = px.imshow(corr_matrix, title="Feature Correlation Heatmap", text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto", zmin=-1, zmax=1)
                    st.plotly_chart(fig_corr, width='stretch')
                else:
                    st.info("Not enough numeric columns for a correlation heatmap.")
                    
            st.divider()
            with st.expander("View Raw JSON Profile"):
                st.json(st.session_state.get('profile', {}))
        except Exception as e:
            import logging
            logging.error(f"Error in Data Report tab: {e}", exc_info=True)
            st.error("Something went wrong while generating the report. The details were saved to logs/app.log")
            if st.button("Reset Data Report"):
                st.rerun()

    # --- TAB 1: Data Analyst Agent ---
    with tab1:
        try:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.header("Chat with your data")
            with col2:
                if st.button("Clear Chat"):
                    st.session_state.agent_history = []
                    st.rerun()
                    
            example_clicked = None
            if not st.session_state.agent_history:
                st.write("**Here are some things you can ask:**")
                c1, c2 = st.columns(2)
                cols = list(df.columns)
                ex1 = "How many rows and columns are there?"
                ex2 = "Are there any missing values?"
                ex3 = f"Show the distribution of {cols[0]}." if len(cols) > 0 else "Show the data distribution."
                ex4 = "Can you summarize the numeric columns?"
                if c1.button(ex1): example_clicked = ex1
                if c1.button(ex2): example_clicked = ex2
                if c2.button(ex3): example_clicked = ex3
                if c2.button(ex4): example_clicked = ex4
            for msg in st.session_state.agent_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("output") and not msg.get("fit_evidence"):
                        st.text(msg["output"])
                    if msg.get("chart"):
                        st.image(msg["chart"])
                    if msg.get("code"):
                        with st.expander("Generated Code"):
                            st.code(msg["code"], language="python")
                    if msg.get("fit_evidence"):
                        fit_evidence = msg["fit_evidence"]
                        verdict = fit_evidence["verdict"]["verdict"]
                        color = "green" if verdict == "Good fit" else "orange" if verdict == "Okay" else "red"
                        st.markdown(f"### <span style='color:{color}'>{verdict}</span>", unsafe_allow_html=True)
                        st.write(f"**Alternative:** {fit_evidence['verdict']['alternative']}")
                        with st.expander("Assumption Violations"):
                            if fit_evidence["assumptions"]:
                                st.table(pd.DataFrame(fit_evidence["assumptions"]))
                            else:
                                st.write("No violations found.")
                    
            user_question = st.chat_input("Ask a question about your data (e.g., 'Draw a chart of average salary by department')")
            if example_clicked:
                user_question = example_clicked
        
            if user_question:
                st.session_state.agent_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)
                
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing and writing code..."):
                        initial_state = {
                            "dataset_summary": profile_dataset(df),
                            "chat_history": st.session_state.agent_history[:-1], # pass all but the current question
                            "question": user_question,
                            "dataset_path": st.session_state.df_path,
                            "generated_code": None,
                            "execution_output": None,
                            "execution_error": None,
                            "retries": 0,
                            "max_retries": 3,
                            "final_explanation": None
                        }
                    
                        try:
                            final_state = st.session_state.agent_workflow.invoke(initial_state)
                            explanation = final_state.get("final_explanation")
                            exec_output = final_state.get("execution_output")
                            if explanation:
                                st.markdown(explanation)
                            elif exec_output:
                                st.text(exec_output)
                            else:
                                st.write("Code executed successfully.")

                            with st.expander("Generated Code"):
                                st.code(final_state["generated_code"], language="python")
                            
                            # Handle FIT_CHECK rendering
                            fit_evidence = None
                            if exec_output:
                                    try:
                                        out_json = json.loads(exec_output)
                                        if "fit_check_evidence" in out_json:
                                             fit_evidence = out_json["fit_check_evidence"]
                                    except:
                                        pass
                                     
                                    if fit_evidence:
                                        verdict = fit_evidence["verdict"]["verdict"]
                                        color = "green" if verdict == "Good fit" else "orange" if verdict == "Okay" else "red"
                                        st.markdown(f"### <span style='color:{color}'>{verdict}</span>", unsafe_allow_html=True)
                                
                                        st.write(f"**Alternative:** {fit_evidence['verdict']['alternative']}")
                                
                                        with st.expander("Assumption Violations"):
                                            if fit_evidence["assumptions"]:
                                                st.table(pd.DataFrame(fit_evidence["assumptions"]))
                                            else:
                                                st.write("No violations found.")
                                         
                                        if fit_evidence["measured"].get("score"):
                                            st.write(f"Measured Score: {fit_evidence['measured']['score']:.4f}")
                                
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("Test the suggested fix", key=f"fix_{len(st.session_state.agent_history)}"):
                                            st.info("Fix testing initiated...")
                                    with col2:
                                        if st.button("Run this algorithm", key=f"run_{len(st.session_state.agent_history)}"):
                                            st.info(f"Running {fit_evidence['algorithm']}...")
                            
                                    chart_path = "sandbox_temp/chart.png"
                                    has_chart = os.path.exists(chart_path)
                                    if has_chart:
                                        st.image(chart_path)
                                    st.session_state.agent_history.append({
                                        "role": "assistant",
                                        "content": explanation or (f"```text\n{exec_output}\n```" if exec_output else "Executed."),
                                        "output": exec_output,
                                        "code": final_state["generated_code"],
                                        "chart": chart_path if has_chart else None,
                                        "fit_evidence": fit_evidence
                                    })
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Agent failed: {e}")
        except Exception as e:
            import logging
            logging.error(f"Error in Chat Agent tab: {e}", exc_info=True)
            st.error("Something went wrong while chatting. The details were saved to logs/app.log")

    # === TAB 3: AUTOML ===
    with tab_automl:
        try:
            st.header("Find the Best Machine Learning Algorithm (Two-Stage Engine)")
            
            # Controls Header
            ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
            with ctrl_col1:
                target_col = st.selectbox("Select Target Column", [None] + list(df.columns))
            with ctrl_col2:
                if target_col:
                    detected_task, _ = detect_task_type(df, target_col)
                    st.info(f"Task: **{detected_task.title()}**")
                else:
                    detected_task = "clustering"
                    st.info("Task: **Clustering**")
            with ctrl_col3:
                total_budget = st.number_input("Total Time Budget (sec)", min_value=30, max_value=600, value=DEFAULT_TOTAL_TIME_BUDGET, step=30)
            with ctrl_col4:
                algo_budget = st.number_input("Per-Algo Budget (sec)", min_value=5, max_value=120, value=DEFAULT_ALGO_TIME_BUDGET, step=5)

            quick_mode = st.checkbox("⚡ Quick Demo Mode (Cap data to 5,000 rows for fast execution)")
            use_pca = None
            if target_col is None:
                use_pca = st.checkbox("🔮 Use PCA for clustering (Default: Auto if >20 cols after encoding)", value=True)

            # Generate current settings key
            current_eval_key = f"eval_{id(df)}_{target_col}_{quick_mode}_{total_budget}_{algo_budget}_{use_pca if detected_task == 'clustering' else False}"
            if st.session_state.get("eval_cache_key") != current_eval_key:
                st.session_state.eval_results = None
                st.session_state.eval_meta = None
                st.session_state.eval_results_baseline = None
                
            # --- Pre-flight Checks (Leakage, Imbalance, Constants) ---
            try:
                warnings_list = []
                suspicious_cols = []
                
                # Initialize overrides if not present
                if "df_cleaned_overrides" not in st.session_state or st.session_state.get("override_id") != id(df):
                    st.session_state.df_cleaned_overrides = df.copy()
                    st.session_state.override_id = id(df)
                    
                current_df = st.session_state.df_cleaned_overrides
                
                if target_col and target_col in current_df.columns:
                    from automl.leakage import detect_leakage_and_warnings
                    # Recompute warnings on the current modified dataframe
                    cache_leak_key = f"leakage_{id(current_df)}_{target_col}"
                    if cache_leak_key not in st.session_state:
                        with st.spinner("Running pre-flight data quality checks..."):
                            w_list, s_cols, _ = detect_leakage_and_warnings(current_df, target_col, detected_task)
                            st.session_state[cache_leak_key] = (w_list, s_cols)
                    warnings_list, suspicious_cols = st.session_state[cache_leak_key]
                    
                rerun_triggered = False
                
                if warnings_list:
                    st.subheader("⚠️ Data Quality Warnings")
                    for i, w in enumerate(warnings_list):
                        with st.container(border=True):
                            c1, c2, c3 = st.columns([3, 1, 1.5])
                            effect_color = "red" if w['effect'] == "large" else "orange" if w['effect'] == "moderate" else "green"
                            with c1:
                                st.markdown(f"**{w['msg']}**")
                                st.caption(w['explanation'])
                            with c2:
                                st.markdown(f"*{w['percentage']*100:.1f}% affected*<br>Effect: :{effect_color}[{w['effect'].title()}]", unsafe_allow_html=True)
                            with c3:
                                action_key = f"action_{i}_{w['type']}"
                                if w['action_type'] == 'drop_rows':
                                    if st.button("Fix & Compare", key=action_key+"_comp"):
                                        st.session_state.df_cleaned_overrides = current_df.drop_duplicates()
                                        if st.session_state.eval_results is not None:
                                            st.session_state.eval_results_baseline = st.session_state.eval_results
                                            st.session_state.eval_meta_baseline = st.session_state.eval_meta
                                            st.session_state.eval_cache_key = None
                                            rerun_triggered = True
                                        st.rerun()
                                elif w['action_type'] == 'drop_cols':
                                    cols_to_drop = [c for c in w['cols'] if c in current_df.columns]
                                    if cols_to_drop:
                                        if st.button("Drop & Compare", key=action_key+"_comp"):
                                            st.session_state.df_cleaned_overrides = current_df.drop(columns=cols_to_drop)
                                            if st.session_state.eval_results is not None:
                                                st.session_state.eval_results_baseline = st.session_state.eval_results
                                                st.session_state.eval_meta_baseline = st.session_state.eval_meta
                                                st.session_state.eval_cache_key = None
                                                rerun_triggered = True
                                            st.rerun()
                                    else:
                                        st.success("Fixed")
                                else:
                                    st.write("No auto-fix")
                                    
                    if st.button("Undo All Fixes", type="tertiary"):
                        st.session_state.df_cleaned_overrides = df.copy()
                        st.session_state.eval_results_baseline = None
                        st.session_state.eval_cache_key = None
                        st.rerun()

            except Exception as e:
                st.error(f"Could not load data warnings: {e}")
                import logging
                logging.error(f"Data warnings error: {e}", exc_info=True)
                rerun_triggered = False
                        
            run_col, cancel_col = st.columns([1, 1])
            with run_col:
                start_btn = st.button("🚀 Run Algorithm Comparison", type="primary") or rerun_triggered
            with cancel_col:
                if st.button("Clear Results"):
                    st.session_state.eval_results = None
                    st.session_state.eval_meta = None
                    st.session_state.eval_results_baseline = None
                    st.session_state.eval_cache_key = None
                    st.rerun()

            if start_btn:
                # Validation checks
                if target_col:
                    valid_df = df.dropna(subset=[target_col])
                    if len(valid_df) < 5:
                        st.error("Data Problem: Not enough rows with a valid target value. Please clean the data or choose another target.")
                        st.stop()
                    if detected_task == "classification" and valid_df[target_col].nunique() < 2:
                        st.error("Data Problem: Target column has only one class. A model cannot learn from a single outcome.")
                        st.stop()
                        
                if len(df) < 20:
                    st.error(f"Data Problem: Dataset has only {len(df)} rows. DataPilot requires at least 20 rows to perform meaningful AutoML comparison.")
                    st.stop()
                        
                # Placeholders for live feedback
                status_box = st.empty()
                progress_bar = st.progress(0)
                table_placeholder = st.empty()

                if quick_mode and len(df) <= 5000:
                    st.info(f"Dataset has {len(df)} rows; Quick Demo Mode is not needed, using all rows.")
                    eval_df = df
                elif quick_mode or len(df) > 5000:
                    if len(df) > 5000 and not quick_mode:
                        st.warning(f"⚠️ Dataset is very large ({len(df):,} rows). Capping to 5,000 rows for faster algorithm comparison. You can adjust the time limits above.")
                    eval_df = df.sample(n=min(5000, len(df)), random_state=42)
                else:
                    eval_df = df

                def live_callback(stage_name: str, step: int, total_steps: int, model_name: str, current_df: pd.DataFrame):
                    pct = int((step / total_steps) * 100)
                    progress_bar.progress(pct)
                    status_box.info(f"⏳ **{stage_name}** ({step}/{total_steps}): Currently evaluating `{model_name}`...")
                    if not current_df.empty:
                        table_placeholder.dataframe(current_df)

                cols_to_drop = []
                with st.spinner("Executing Two-Stage AutoML Evaluation..."):
                    try:
                        results_df, primary_metric, meta_info = evaluate_models(
                            eval_df, 
                            detected_task, 
                            target_col,
                            total_time_budget=float(total_budget),
                            algo_time_budget=float(algo_budget),
                            progress_callback=live_callback,
                            use_pca=use_pca if detected_task == 'clustering' else False,
                            columns_to_drop=cols_to_drop
                        )
                        
                        # Store by dataset/target/settings
                        cache_key = current_eval_key
                        st.session_state.eval_cache_key = cache_key
                        st.session_state.eval_results = results_df
                        st.session_state.eval_meta = meta_info
                        progress_bar.progress(100)
                        status_box.success(f"✅ Evaluation complete in {meta_info['total_elapsed_seconds']}s! (Cached: {meta_info['is_cached']})")
                    except Exception as e:
                        import traceback
                        import logging
                        tb_str = traceback.format_exc()
                        logging.error(f"AutoML Execution Crash:\n{tb_str}")
                        status_box.error(f"AutoML Error: {e}")
                        with st.expander("Show technical details"):
                            st.code(tb_str, language="python")

            # Render Results if present
            if st.session_state.eval_results is not None:
                results_df = st.session_state.eval_results
                meta = st.session_state.eval_meta
                primary_metric = meta['primary_metric']

                st.divider()
            
                # --- 1. STRATEGY BANNERS ---
                try:
                    st.subheader("📋 Adaptive Execution Strategy & Metadata")
                    strat = meta['strategy']
            
                    stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
                    stat_c1.metric("Strategy Tier", strat['tier'].upper())
                    stat_c2.metric("Total Elapsed Time", f"{meta['total_elapsed_seconds']}s")
                    stat_c3.metric("Evaluated Features", strat['n_cols'])
                    stat_c4.metric("Primary Metric", primary_metric.title())

                    st.markdown(f"**Strategy Description:** {strat['description']}")
                
                    from automl.config import RECOMMENDATION_RULE
                    st.info(f"**Recommendation Rule:** {RECOMMENDATION_RULE}")

                    if meta.get('prep_summary'):
                        with st.expander("🛠️ Data Preparation Summary", expanded=True):
                            prep = meta['prep_summary']
                            if prep.get('dropped_cols'):
                                st.write("**Columns Dropped:** " + ", ".join(prep['dropped_cols']))
                            if prep.get('encoded_onehot'):
                                st.write("**One-Hot Encoded:** " + ", ".join(prep['encoded_onehot']))
                            if prep.get('encoded_freq'):
                                st.write("**Frequency Encoded:** " + ", ".join(prep['encoded_freq']))
                            st.write(f"**Final Features Count:** {prep.get('final_features', 0)}")
                            st.write(f"**PCA Used:** {'Yes' if prep.get('pca_used') else 'No'}")
                    elif meta.get('cleanup_messages'):
                        with st.expander("⚠️ Data Pre-cleaning Notifications"):
                            for msg in meta['cleanup_messages']:
                                st.write(f"- {msg}")

                    if strat['tier'] in ['medium', 'large'] or quick_mode:
                        st.warning("⚠️ **Sampling Warning:** Results are based on a sample strategy for speed. Ranking may vary slightly on 100% full dataset.")
                except Exception as e:
                    st.error(f"Could not build Strategy Banners: {e}")
                    import logging
                    logging.error(f"Strategy Banners error: {e}", exc_info=True)

                st.divider()
                st.subheader(f"🏆 Results Table (Ranked by {primary_metric.title()})")

                def highlight_winners(row):
                    styles = [''] * len(row)
                    if row.get('Status', '') == 'Suspicious result':
                        return ['background-color: #ffb3b3; color: #721c24; font-weight: bold;'] * len(row) # Bright Red
                    if row.get('Is Best Balance', False):
                        return ['background-color: #cce5ff'] * len(row) # Blue
                    if row.get('Is Most Accurate', False):
                        return ['background-color: #d4edda'] * len(row) # Green
                    if row.get('Is Fastest', False):
                        return ['background-color: #fff3cd'] * len(row) # Yellow
                    if row.get('Status', '') in ['Failed', 'Too slow', 'Skipped']:
                        return ['color: #721c24; background-color: #f8d7da'] * len(row) # Red text on light red
                    return styles

                # Format numeric columns to 4 decimal places
                format_dict = {
                    'Mean Score': '{:.4f}',
                    'Score Std Dev': '{:.4f}',
                    'Mean Train Time (s)': '{:.4f}',
                    'Pred Time per 1k (s)': '{:.4f}'
                }
                
                # Rearrange columns to show Status and Reason early
                cols = results_df.columns.tolist()
                if 'Status' in cols:
                    # Move Status and Reason to front after Model
                    cols.insert(1, cols.pop(cols.index('Status')))
                    cols.insert(2, cols.pop(cols.index('Reason')))
                    results_df = results_df[cols]

                styled_df = results_df.style.apply(highlight_winners, axis=1).format(format_dict)
                
                if st.session_state.get('eval_results_baseline') is not None:
                    st.subheader("Side-by-Side Comparison")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Before (Baseline):**")
                        b_df = st.session_state.eval_results_baseline[cols].style.apply(highlight_winners, axis=1).format(format_dict)
                        st.dataframe(b_df, width='stretch')
                    with c2:
                        st.markdown("**After (Excluded Flagged Columns):**")
                        st.dataframe(styled_df, width='stretch')
                else:
                    st.dataframe(styled_df, width='stretch')

                # Winner Badges
                st.divider()
                st.subheader("🏅 Algorithm Awards")
                badge_cols = st.columns(4)
                
                # 1. Most Accurate
                ma = results_df[results_df['Is Most Accurate'] == True]
                if not ma.empty:
                    m = ma.iloc[0]
                    badge_cols[0].markdown(f"<div style='background:#d4edda;padding:10px;border-radius:8px;height:100%'><b>🟢 Most Accurate</b><br>{m['Model']}<br><small>Score: {m['Mean Score']:.4f}</small></div>", unsafe_allow_html=True)
                
                # 2. Fastest Train
                ft = results_df[results_df['Is Fastest'] == True]
                if not ft.empty:
                    m = ft.iloc[0]
                    badge_cols[1].markdown(f"<div style='background:#fff3cd;padding:10px;border-radius:8px;height:100%'><b>🟡 Fastest to train</b><br>{m['Model']}<br><small>Train: {m['Mean Train Time (s)']:.4f}s | Pred: {m.get('Pred Time per 1k (s)', 0):.4f}s</small></div>", unsafe_allow_html=True)
                    
                # 3. Fastest Predict
                fp = results_df[results_df.get('Is Fastest Predict', False) == True]
                if not fp.empty:
                    m = fp.iloc[0]
                    badge_cols[2].markdown(f"<div style='background:#e2e3e5;padding:10px;border-radius:8px;height:100%'><b>⚡ Fastest to predict</b><br>{m['Model']}<br><small>Pred: {m.get('Pred Time per 1k (s)', 0):.4f}s | Score: {m['Mean Score']:.4f}</small></div>", unsafe_allow_html=True)
                    
                # 4. Best Balance
                bb = results_df[results_df['Is Best Balance'] == True]
                if not bb.empty:
                    m = bb.iloc[0]
                    from automl.config import ROW_CAPS
                    warn = ""
                    if m['Model'] in ROW_CAPS:
                        warn = "<br><span style='color:#d9534f;'>⚠️ $O(N^2)$ scaling risk</span>"
                    badge_cols[3].markdown(f"<div style='background:#cce5ff;padding:10px;border-radius:8px;height:100%'><b>🔵 Best Balance (Recommended)</b><br>{m['Model']}<br><small>{m.get('Why Recommended', '')}{warn}</small></div>", unsafe_allow_html=True)

                # Suspicious Card
                if 'Status' in results_df.columns:
                    suspicious = results_df[results_df['Status'] == 'Suspicious result']
                    if not suspicious.empty:
                        models_str = ", ".join(suspicious['Model'].tolist())
                        st.markdown(f"<div style='background:#ffb3b3;padding:10px;border-radius:8px;margin-top:10px;margin-bottom:10px;'>"
                                    f"<b>🚨 Suspicious Result Warning:</b> {models_str} achieved scores >=0.999 with std ~0. "
                                    f"This strongly indicates Data Leakage. They have been excluded from recommendations.</div>",
                                    unsafe_allow_html=True)
                                    
                st.divider()
                st.subheader("📉 Non-Recommended & Failures")
                fail_cols = st.columns(2)
                with fail_cols[0]:
                    st.markdown("#### Could not be evaluated (fit unknown)")
                    failed_df = results_df[results_df['Status'].isin(['Failed', 'Too slow', 'Skipped', 'Not run: total time budget reached'])]
                    if failed_df.empty:
                        st.write("None.")
                    for _, row in failed_df.iterrows():
                        st.markdown(f"**{row['Model']}**: {row['Status']} - {row.get('Reason', 'Unknown')}")
                        if row['Status'] in ['Too slow', 'Skipped', 'Failed']:
                            if st.button("Run on a sample", key=f"run_sample_list_{row['Model']}"):
                                st.info(f"To run this on a sample, go to the Fit Check below and select **{row['Model']}**.")
                                
                with fail_cols[1]:
                    st.markdown("#### Evaluated but weak")
                    # Lowest scores based on primary metric
                    higher_is_better = True if primary_metric in ['accuracy', 'f1', 'precision', 'recall', 'r2', 'silhouette'] else False
                    valid_df = results_df[results_df['Status'].isin(['Evaluated', 'Evaluated on a sample']) & (results_df['Is Best Balance'] == False) & (results_df['Is Most Accurate'] == False)]
                    if valid_df.empty:
                        st.write("None.")
                    else:
                        valid_df = valid_df.sort_values(by='Mean Score', ascending=higher_is_better)
                        for _, row in valid_df.head(5).iterrows():
                            st.markdown(f"**{row['Model']}**: Score: {row['Mean Score']:.4f} | Train: {row['Mean Train Time (s)']:.4f}s")

                # --- 2. VERIFY TOP 2 BUTTON ---
                try:
                    st.divider()
                    st.subheader("🔍 Verification & Integrity Check")
                    if st.button("🔎 Verify Top 2 Models on 100% Full Dataset"):
                        with st.spinner("Re-evaluating top 2 candidates on 100% full dataset with 5-fold CV..."):
                            top2_names = results_df.dropna(subset=['Mean Score'])['Model'].head(2).tolist()
                            if len(top2_names) >= 2:
                                st.info(f"Verifying candidates: `{top2_names[0]}` vs `{top2_names[1]}`")
                                # Run full eval on full dataset for top 2
                                v_res, _, _ = evaluate_models(df, detected_task, target_col, total_time_budget=300)
                                v_top1 = v_res['Model'].iloc[0]
                                orig_top1 = results_df['Model'].iloc[0]
                                if v_top1 == orig_top1:
                                    st.success(f"✅ **Verification Passed!** Winner remained identical: `{v_top1}`.")
                                else:
                                    st.warning(f"🔄 **Verification Updated Winner!** Full dataset winner is `{v_top1}` (sampled winner was `{orig_top1}`).")
                                st.dataframe(v_res[v_res['Model'].isin(top2_names)])
                            else:
                                st.warning("Not enough valid models to compare.")
                except Exception as e:
                    st.error(f"Could not build Verification check: {e}")
                    import logging
                    logging.error(f"Verification error: {e}", exc_info=True)

                # --- 3. CHARTS ---
                try:
                    chart_df = results_df.dropna(subset=['Mean Score']).copy()
                    if not chart_df.empty:
                        st.divider()
                        st.subheader("📊 Visual Performance Analysis")
                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        chart_df['Highlight'] = chart_df.apply(lambda r: 'Selected' if r.get('Is Best Balance') else 'Others', axis=1)
                        fig_acc = px.bar(chart_df, x='Model', y='Mean Score', title=f"Score ({primary_metric.title()}) by Model", color='Highlight', color_discrete_map={'Selected':'red', 'Others':'blue'}, text_auto='.3f')
                        fig_acc.update_layout(yaxis_title="Mean R2 score" if primary_metric == "r2" else f"Mean {primary_metric.title()} score")
                        fig_acc.update_yaxes(rangemode="tozero")
                        fig_acc.update_traces(textposition='outside')
                        fig_acc.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig_acc, width='stretch')
                    with chart_col2:
                        if 'Pred Time per 1k (s)' in chart_df.columns:
                            fig_time = px.bar(chart_df, x='Model', y='Pred Time per 1k (s)', title="Prediction Time per 1k Rows (Seconds)", color='Model', text_auto='.3f')
                            fig_time.update_layout(yaxis_title="Mean prediction time (seconds)")
                            fig_time.update_yaxes(rangemode="tozero")
                            fig_time.update_traces(textposition='outside')
                            fig_time.update_xaxes(tickangle=-45)
                            st.plotly_chart(fig_time, width='stretch')
                        elif 'Mean Train Time (s)' in chart_df.columns:
                            fig_time = px.bar(chart_df, x='Model', y='Mean Train Time (s)', title="Training Time (Seconds)", color='Model', text_auto='.3f')
                            fig_time.update_layout(yaxis_title="Mean training time (seconds)")
                            fig_time.update_yaxes(rangemode="tozero")
                            fig_time.update_traces(textposition='outside')
                            fig_time.update_xaxes(tickangle=-45)
                            st.plotly_chart(fig_time, width='stretch')
                except Exception as e:
                    st.error(f"Could not build charts: {e}")
                    import logging
                    logging.error(f"Charts error: {e}", exc_info=True)
                            
                # --- CLUSTERING CONTEXT ---
                try:
                    if target_col is None and not chart_df.empty and 'X_clust' in meta:
                        st.divider()
                        st.subheader("🧩 Clustering Insights (Best Model)")
                        best_model_name = chart_df.iloc[0]['Model']
                        
                        from automl.models import get_clustering_models
                        models = get_clustering_models(len(meta['X_clust']))
                        
                        if best_model_name in models:
                            best_model = models[best_model_name]
                            X_clust = meta['X_clust']
                            
                            with st.spinner("Generating cluster insights..."):
                                import warnings
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    preds = best_model.fit_predict(X_clust)
                                
                                c1, c2 = st.columns(2)
                                
                                # PCA Scatter
                                from sklearn.decomposition import PCA
                                pca = PCA(n_components=2)
                                X_2d = pca.fit_transform(X_clust)
                                
                                scatter_df = pd.DataFrame(X_2d, columns=['PCA1', 'PCA2'])
                                scatter_df['Cluster'] = preds.astype(str)
                                
                                fig_pca = px.scatter(scatter_df, x='PCA1', y='PCA2', color='Cluster', title="2D Cluster Projection")
                                c1.plotly_chart(fig_pca, width='stretch')
                                
                                # Cluster Differences
                                c2.write("**Cluster Averages (Top Features)**")
                                
                                X_with_clusters = X_clust.copy()
                                X_with_clusters['Cluster'] = preds
                                
                                cluster_means = X_with_clusters.groupby('Cluster').mean()
                                if len(cluster_means.columns) > 5:
                                    # Show top 5 features by variance across clusters
                                    variances = cluster_means.var()
                                    top_features = variances.nlargest(5).index
                                    cluster_means = cluster_means[top_features]
                                    
                                c2.dataframe(cluster_means.style.format("{:.2f}"))
                except Exception as e:
                    st.error(f"Could not build Clustering Insights: {e}")
                    import logging
                    logging.error(f"Clustering Insights error: {e}", exc_info=True)

            # --- 4. ALGORITHM FIT CHECK (Always Visible) ---
            st.divider()
            
            @st.fragment
            def render_fit_check_fragment():
                st.subheader("🕵️ Algorithm Fit Check")
                st.write("Select an algorithm to see if it fits this dataset based on assumptions and measured evidence.")
            
                from automl.task_detector import detect_task_type
                
                fit_task = "clustering"
                if target_col:
                    fit_task, _ = detect_task_type(df, target_col)
                    
                # Group algorithm options
                from automl.registry import ALGORITHM_REGISTRY
                syllabus_opts = []
                extended_opts = []
                for name, meta in ALGORITHM_REGISTRY.items():
                    if meta["task"] == fit_task:
                        if meta.get("category") == "Syllabus":
                            syllabus_opts.append(name)
                        else:
                            extended_opts.append(name)
                            
                syllabus_opts.sort()
                extended_opts.sort()
                algo_options = [""] + ["--- Syllabus ---"] + syllabus_opts + ["--- Extended ---"] + extended_opts
                
                col_algo, col_btn = st.columns([3, 1])
                with col_algo:
                    selected_algo = st.selectbox("Select Algorithm", algo_options, key="fit_check_select")
                    if selected_algo.startswith("---"):
                        selected_algo = ""
                
                with col_btn:
                    st.write("") # spacing
                    st.write("") 
                    st.button("Check fit", type="primary", width='stretch')
            
                # If they selected an algorithm, evaluate/show it. 
                # (Fragment automatically reruns when selectbox changes).
                if selected_algo:
                    cache_key = f"fit_check_{id(df)}_{target_col}_{selected_algo}"
                    
                    with st.spinner(f"Checking fit for {selected_algo}..."):
                        try:
                            from automl.fit_check import gather_evidence, generate_explanation, render_fit_check
                            if cache_key not in st.session_state:
                                evidence = gather_evidence(selected_algo, df, target_col)
                                explanation = generate_explanation(evidence)
                                st.session_state[cache_key] = {"evidence": evidence, "explanation": explanation}
                                
                        except Exception as fe:
                            import logging
                            logging.error(f"Fit Check Error: {fe}", exc_info=True)
                            st.error(f"An error occurred while checking this algorithm: {fe}")
                            
                        # Render cached result if exists for current selection
                        if cache_key in st.session_state:
                            from automl.fit_check import render_fit_check
                            cached_result = st.session_state[cache_key]
                            render_fit_check(cached_result["evidence"], cached_result["explanation"])
            render_fit_check_fragment()



        except Exception as e:
            import logging
            logging.error(f"Error in AutoML tab: {e}", exc_info=True)
            st.error("Something went wrong while running AutoML. The details were saved to logs/app.log")
            if st.button("Reset AutoML"):
                st.rerun()

else:
    st.info("Please upload a CSV file from the sidebar to begin.")
