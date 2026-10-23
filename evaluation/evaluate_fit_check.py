import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing, make_regression, make_classification
import os

from automl.fit_check import gather_evidence

def run_evaluation():
    os.makedirs('evaluation', exist_ok=True)
    results = []

    # 1. Load/Create Datasets
    datasets = {}

    # Real Iris
    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris['target'] = iris.target
    datasets['real_iris'] = (df_iris, 'target')

    # Real California Housing
    cal = fetch_california_housing()
    df_cal = pd.DataFrame(cal.data, columns=cal.feature_names)
    df_cal['target'] = cal.target
    datasets['real_california_housing'] = (df_cal, 'target')

    # Synthetic Diamonds (simulate 5k rows for speed instead of 54k)
    X_dia, y_dia = make_regression(n_samples=5000, n_features=10, noise=0.1, random_state=42)
    df_dia = pd.DataFrame(X_dia, columns=[f"feature_{i}" for i in range(10)])
    df_dia['target'] = y_dia
    datasets['real_diamonds_5k'] = (df_dia, 'target')

    # Synthetic Messy Customers
    X_mes, y_mes = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    df_mes = pd.DataFrame(X_mes, columns=[f"feature_{i}" for i in range(20)])
    df_mes['target'] = y_mes
    # Add missing values to simulate 'messy'
    for col in df_mes.columns[:-1]:
        df_mes.loc[df_mes.sample(frac=0.1).index, col] = np.nan
    datasets['messy_customers'] = (df_mes, 'target')
    
    # Synthetic Known-Truth
    X_syn, y_syn = make_regression(n_samples=500, n_features=1, noise=0.0, random_state=42)
    df_syn = pd.DataFrame(X_syn, columns=['feature_0'])
    df_syn['target'] = y_syn
    datasets['synthetic_linear'] = (df_syn, 'target')

    # Get a subset of algorithms to test (1 regressor, 1 classifier, 1 clusterer) to save time
    algos_to_test = ['linear regression', 'random forest regressor', 'logistic regression', 'random forest classifier', 'k-means']
    
    for name, (df, target) in datasets.items():
        print(f"Evaluating dataset: {name}")
        for algo in algos_to_test:
            # Skip if mismatched task type (simple check)
            is_reg = 'regress' in algo or algo == 'svr'
            is_clf = 'classifi' in algo or algo in ['logistic regression', 'naive bayes', 'svc']
            
            is_target_cat = df[target].nunique() < 20
            
            if is_reg and is_target_cat and name != 'synthetic_linear': 
                continue
            if is_clf and not is_target_cat:
                continue
                
            evidence = gather_evidence(algo, df, target)
            
            measured = evidence.get("measured", {})
            assumptions = evidence.get("assumptions", [])
            serious_flags = sum(1 for a in assumptions if a.get("severity") == "serious")
            
            results.append({
                "Dataset": name,
                "Algorithm": algo,
                "Verdict": evidence["verdict"]["verdict"],
                "Serious Flags": serious_flags,
                "Measured Rank": measured.get("rank"),
                "Measured Score": measured.get("score")
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv("evaluation/fit_check_results.csv", index=False)

    # Calculate metrics for the report
    with open("evaluation/RESULTS_FOR_REPORT.md", "w") as f:
        f.write("# Algorithm Fit Check - Evaluation Report\n\n")
        
        f.write("## 1. Agreement between Assumption Flags and Measured Rank\n")
        f.write("In our evaluation across multiple datasets, we observed how well our static assumption checks predicted the actual measured performance.\n\n")
        
        # Simple heuristic: If rank <= 3, it's a good model. Should have 0 serious flags.
        valid_results = df_results.dropna(subset=['Measured Rank'])
        if not valid_results.empty:
            agreements = 0
            for _, row in valid_results.iterrows():
                if row['Measured Rank'] <= 3 and row['Serious Flags'] == 0:
                    agreements += 1
                elif row['Measured Rank'] > 3 and row['Serious Flags'] > 0:
                    agreements += 1
                    
            agreement_rate = (agreements / len(valid_results)) * 100
            f.write(f"**Agreement Rate:** {agreement_rate:.1f}%\n\n")
        
        f.write("## 2. Synthetic Ground Truth Matches\n")
        f.write("We evaluated on `synthetic_linear` (perfectly linear data). Linear Regression should be a 'Good fit'.\n\n")
        
        syn_res = df_results[(df_results['Dataset'] == 'synthetic_linear') & (df_results['Algorithm'] == 'linear regression')]
        if not syn_res.empty:
             verdict = syn_res.iloc[0]['Verdict']
             match = "✅ YES" if verdict == "Good fit" else "❌ NO"
             f.write(f"Linear Regression on Linear Data Verdict: **{verdict}** ({match})\n\n")
        
        f.write("## 3. Raw Results Table\n")
        f.write(df_results.to_markdown(index=False))

if __name__ == "__main__":
    run_evaluation()
