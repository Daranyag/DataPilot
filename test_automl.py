from sklearn.datasets import load_iris, fetch_california_housing, load_breast_cancer
from automl.task_detector import detect_task_type
from automl.evaluation import evaluate_models

def test_automl():
    print("--- AutoML Tests ---")
    
    # 1. Iris (Classification)
    print("\n--- Dataset: Iris (Classification) ---")
    iris = load_iris(as_frame=True)
    df_iris = iris.frame
    task, target = detect_task_type(df_iris, "target")
    print(f"Detected Task: {task}")
    res, metric, _ = evaluate_models(df_iris, task, target)
    print(f"Primary Metric: {metric}")
    print(res.to_string())

    # 2. California Housing (Regression)
    print("\n--- Dataset: California Housing (Regression) ---")
    housing = fetch_california_housing(as_frame=True)
    df_housing = housing.frame
    # Sampling for speed during demo
    df_housing = df_housing.sample(1000, random_state=42) 
    task, target = detect_task_type(df_housing, "MedHouseVal")
    print(f"Detected Task: {task}")
    res, metric, _ = evaluate_models(df_housing, task, target)
    print(f"Primary Metric: {metric}")
    print(res.to_string())

    # 3. Breast Cancer (Classification with slight imbalance)
    print("\n--- Dataset: Breast Cancer (Classification) ---")
    cancer = load_breast_cancer(as_frame=True)
    df_cancer = cancer.frame
    task, target = detect_task_type(df_cancer, "target")
    print(f"Detected Task: {task}")
    res, metric, _ = evaluate_models(df_cancer, task, target)
    print(f"Primary Metric: {metric}")
    print(res.to_string())

if __name__ == "__main__":
    test_automl()
