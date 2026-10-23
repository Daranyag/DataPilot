"""
Central Configuration for DataPilot AutoML Engine
All thresholds, time budgets, and sampling rules are configured here.
"""

# Dataset Size Thresholds (Row Counts)
SMALL_DATASET_THRESHOLD = 5000
MEDIUM_DATASET_THRESHOLD = 50000

# Time Budgets (Seconds)
DEFAULT_TOTAL_TIME_BUDGET = 180      # 3 minutes total
DEFAULT_ALGO_TIME_BUDGET = 45         # 45 seconds per algorithm

# Stage 1 (Screening) Parameters
STAGE1_SAMPLE_SIZE = 5000
STAGE1_FOLDS = 3
STAGE1_TIME_LIMIT = 15.0

# Stage 2 (Full Evaluation) Parameters
SMALL_DATASET_FOLDS = 5
MEDIUM_DATASET_FOLDS = 5
LARGE_DATASET_FOLDS = 3

MEDIUM_TOP_K_MODELS = 4
LARGE_TOP_K_MODELS = 3

# Fast Timing Sample Size for Time Prediction
PREDICTION_SAMPLE_SIZE = 500

# Row Caps for computationally expensive algorithms (O(N^2) or heavy prediction)
ROW_CAPS = {
    'Support Vector Machine': 10000,
    'Support Vector Regressor': 10000,
    'K-Nearest Neighbors': 20000,
    'KNN Regressor': 20000,
    'Hierarchical Clustering': 20000,
    'DBSCAN': 20000,
}

# Row Threshold for substituting faster algorithm variants
SUBSTITUTION_ROW_THRESHOLD = 20000

RECOMMENDATION_RULE = "No algorithm may be recommended while another is at least as good on accuracy (within one standard deviation), training time, prediction time, and memory usage."

# Warning Effect Thresholds
WARNING_EFFECT_THRESHOLDS = {
    'imbalance': {'large': 0.90, 'moderate': 0.70},
    'duplicates': {'large': 0.10, 'moderate': 0.05},
    'empty_cols': {'large': 0.90, 'moderate': 0.50},
    'leakage': {'large': 0.95, 'moderate': 0.90}
}
