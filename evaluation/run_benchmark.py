"""
Benchmarking Script for DataPilot Fast AutoML System.
Runs evaluation across all 9 benchmark datasets in testingg/test_data/
and saves results to evaluation/speed_benchmark.csv.
"""

import os
import sys
import time
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath('.'))

from automl.evaluation import evaluate_models

BENCHMARK_FILES = [
    ("testingg/test_data/real_iris.csv", "target", "classification"),
    ("testingg/test_data/real_breast_cancer.csv", "target", "classification"),
    ("testingg/test_data/real_california_housing.csv", "MedHouseVal", "regression"),
    ("testingg/test_data/real_titanic.csv", "survived", "classification"),
    ("testingg/test_data/real_diamonds_54k.csv", "price", "regression"),
    ("testingg/test_data/messy_customers.csv", "churn", "classification"),
    ("testingg/test_data/big_classification.csv", "target", "classification"),
    ("testingg/test_data/big_regression.csv", "target", "regression"),
    ("testingg/test_data/big_clusters.csv", None, "clustering"),
]

def run_benchmarks():
    print("[+] Starting DataPilot Fast AutoML Benchmark...\n")
    results = []
    
    for file_path, target_col, forced_task in BENCHMARK_FILES:
        rel_path = os.path.normpath(file_path)
        filename = os.path.basename(rel_path)
        
        if not os.path.exists(rel_path):
            print(f"[!] Warning: File {rel_path} not found. Skipping...")
            continue
            
        print(f"Testing {filename}...")
        df = pd.read_csv(rel_path)
        n_rows, n_cols = df.shape
        
        t0 = time.perf_counter()
        try:
            res_df, primary_metric, meta = evaluate_models(
                df, forced_task, target_col, total_time_budget=180, algo_time_budget=45
            )
            elapsed = round(time.perf_counter() - t0, 2)
            
            top_model = res_df['Model'].iloc[0] if not res_df.empty else "N/A"
            best_bal_winner = res_df[res_df.get('Is Best Balance', False)]['Model'].values
            best_bal_model = best_bal_winner[0] if len(best_bal_winner) > 0 else top_model
            
            strat_tier = meta['strategy']['tier'].upper()
            strat_desc = meta['strategy']['description']
            
            results.append({
                "Dataset": filename,
                "Rows": n_rows,
                "Columns": n_cols,
                "Task": forced_task,
                "Strategy Tier": strat_tier,
                "Fast Elapsed Time (s)": elapsed,
                "Most Accurate Model": top_model,
                "Recommended (Best Balance)": best_bal_model,
                "Primary Metric": primary_metric,
                "Strategy Description": strat_desc
            })
            print(f"  [OK] Completed in {elapsed}s | Winner: {best_bal_model} ({strat_tier} strategy)\n")
        except Exception as e:
            print(f"  [FAIL] Failed: {e}\n")
            results.append({
                "Dataset": filename,
                "Rows": n_rows,
                "Columns": n_cols,
                "Task": forced_task,
                "Strategy Tier": "FAILED",
                "Fast Elapsed Time (s)": -1,
                "Most Accurate Model": "Error",
                "Recommended (Best Balance)": f"Error: {e}",
                "Primary Metric": "-",
                "Strategy Description": str(e)
            })
            
    res_table = pd.DataFrame(results)
    out_dir = "evaluation"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "speed_benchmark.csv")
    res_table.to_csv(out_csv, index=False)
    print(f"[+] Benchmark finished! Saved summary to {out_csv}")
    print("\n" + res_table.to_string())

if __name__ == "__main__":
    run_benchmarks()
