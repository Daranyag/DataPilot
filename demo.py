import pandas as pd
from agent.profiler import profile_dataset
from agent.graph import build_graph
import os

def run_demo():
    print("--- DataPilot: Data Analyst Agent Demo ---")
    
    # 1. Load data
    df_path = os.path.abspath("sample.csv")
    df = pd.read_csv(df_path)
    
    # 2. Profile data
    print("Profiling dataset...")
    summary = profile_dataset(df)
    print("\n[Dataset Summary]")
    print(summary)
    
    # 3. Setup LangGraph
    print("\nBuilding agent workflow...")
    workflow = build_graph()
    
    question = "What is the average salary by department? Please also draw a bar chart."
    print(f"\nUser Question: {question}")
    
    # 4. Run Agent
    initial_state = {
        "dataset_summary": summary,
        "question": question,
        "dataset_path": df_path,
        "generated_code": None,
        "execution_output": None,
        "execution_error": None,
        "retries": 0,
        "max_retries": 3,
        "final_explanation": None
    }
    
    print("\nRunning agent loop...")
    final_state = workflow.invoke(initial_state)
    
    print("\n[Final Explanation]")
    print(final_state["final_explanation"])
    
    print("\n[Generated Code]")
    print(final_state["generated_code"])
    
    if os.path.exists("sandbox_temp/chart.png"):
        print("\n[Success] A chart was generated at sandbox_temp/chart.png")

if __name__ == "__main__":
    # Ensure key exists
    if "GEMINI_API_KEY" not in os.environ:
        # Load from .env for demo
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        os.environ["GEMINI_API_KEY"] = line.strip().split("=", 1)[1]
        except:
            print("Please set GEMINI_API_KEY in .env or environment")
            exit(1)
            
    run_demo()
