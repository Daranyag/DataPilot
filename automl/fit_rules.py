import numpy as np
import pandas as pd

def check_multicollinearity(X_train: pd.DataFrame, target_col: str = None) -> float:
    # returns maximum correlation between features
    # (simplification of VIF for speed)
    if X_train.empty or X_train.shape[1] < 2:
        return 0.0
    corr = X_train.corr().abs()
    np.fill_diagonal(corr.values, 0)
    return float(corr.max().max())

def check_feature_to_row_ratio(X_train: pd.DataFrame, **kwargs) -> float:
    if len(X_train) == 0:
        return float('inf')
    return X_train.shape[1] / len(X_train)

def check_lasso_zeros(X_train: pd.DataFrame, y_train: pd.Series = None, **kwargs) -> float:
    if y_train is None or len(X_train) == 0:
        return 0.0
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=1.0)
    try:
        model.fit(X_train, y_train)
        return float(np.mean(model.coef_ == 0))
    except Exception:
        return 0.0

def check_linear_separability(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> float:
    # A proxy for linear separability without running full SVM.
    # For speed, we just return 0 (not easily measurable without model comparison).
    # The actual requirement says: "compare with RBF SVM". This requires running both models.
    # In `fit_check.py`, we can have access to the results DataFrame.
    # For now, let's return -1 and handle it in fit_check.
    return -1.0

def check_scaling_differences(X_train: pd.DataFrame, **kwargs) -> float:
    stds = X_train.std()
    if len(stds) < 2:
        return 0.0
    return float(stds.max() / (stds.min() + 1e-6))

def check_class_balance(y_train: pd.Series, **kwargs) -> float:
    if y_train is None:
        return 1.0
    counts = y_train.value_counts(normalize=True)
    return float(counts.min())

def check_rows_per_class(y_train: pd.Series, **kwargs) -> float:
    if y_train is None:
        return 0.0
    return float(y_train.value_counts().min())

def check_feature_normality(X_train: pd.DataFrame, **kwargs) -> float:
    # Skewness as proxy for non-normality
    return float(X_train.skew().abs().max())

def check_noise_sensitivity(X_train: pd.DataFrame, **kwargs) -> float:
    # Proxy: max number of outliers (e.g. > 3 std devs)
    z_scores = ((X_train - X_train.mean()) / (X_train.std() + 1e-6)).abs()
    outlier_ratio = (z_scores > 3).sum().sum() / X_train.size if X_train.size > 0 else 0
    return float(outlier_ratio)

def check_size_vs_time(X_train: pd.DataFrame, **kwargs) -> float:
    return len(X_train)

# Dictionary defining the rules for each algorithm
# Each rule has:
# - name: str
# - description: str
# - compute: function
# - threshold: dict mapping outcome to conditions
FIT_RULES = {
    "Linear Regression": [
        {
            "name": "Multicollinearity",
            "description": "Checks if features are highly correlated (>0.9). Linear Regression is highly sensitive to multicollinearity.",
            "compute": check_multicollinearity,
            "threshold": {"fail": lambda x: x > 0.95, "warning": lambda x: x > 0.9, "pass": lambda x: x <= 0.9}
        },
        {
            "name": "Feature to Row Ratio",
            "description": "Checks if there are many features compared to rows.",
            "compute": check_feature_to_row_ratio,
            "threshold": {"fail": lambda x: x > 0.9, "warning": lambda x: x > 0.5, "pass": lambda x: x <= 0.5}
        }
    ],
    "Ridge Regression": [
        {
            "name": "Multicollinearity",
            "description": "Checks if features are highly correlated (correlation > 0.9). Ridge handles this well.",
            "compute": check_multicollinearity,
            "threshold": {"pass": lambda x: True} # Ridge handles it well
        },
        {
            "name": "Feature to Row Ratio",
            "description": "Checks if there are many features compared to rows.",
            "compute": check_feature_to_row_ratio,
            "threshold": {"warning": lambda x: x > 0.5, "pass": lambda x: x <= 0.5}
        }
    ],
    "Lasso Regression": [
        {
            "name": "Multicollinearity",
            "description": "Checks if features are highly correlated (correlation > 0.9). Lasso might arbitrarily drop one of the correlated features.",
            "compute": check_multicollinearity,
            "threshold": {"warning": lambda x: x > 0.9, "pass": lambda x: x <= 0.9}
        },
        {
            "name": "Feature to Row Ratio",
            "description": "Checks if there are many features compared to rows.",
            "compute": check_feature_to_row_ratio,
            "threshold": {"warning": lambda x: x > 0.5, "pass": lambda x: x <= 0.5}
        },
        {
            "name": "Zeroed Coefficients",
            "description": "Proportion of coefficients Lasso set exactly to zero (feature selection).",
            "compute": check_lasso_zeros,
            "threshold": {"pass": lambda x: True}
        }
    ],
    "ElasticNet Regression": [
        {
            "name": "Multicollinearity",
            "description": "Checks if features are highly correlated.",
            "compute": check_multicollinearity,
            "threshold": {"pass": lambda x: True}
        },
        {
            "name": "Feature to Row Ratio",
            "description": "Checks if there are many features compared to rows.",
            "compute": check_feature_to_row_ratio,
            "threshold": {"warning": lambda x: x > 0.5, "pass": lambda x: x <= 0.5}
        }
    ],
    "Polynomial Regression": [
        # (Requires cross-validated score comparison, handled in fit_check)
    ],
    "Perceptron": [
        {
            "name": "Scaling Differences",
            "description": "Checks if features have vastly different scales.",
            "compute": check_scaling_differences,
            "threshold": {"fail": lambda x: x > 100, "warning": lambda x: x > 10, "pass": lambda x: x <= 10}
        }
    ],
    "SVM (Linear Kernel)": [
        {
            "name": "Scaling Differences",
            "description": "Checks if features have vastly different scales.",
            "compute": check_scaling_differences,
            "threshold": {"fail": lambda x: x > 100, "warning": lambda x: x > 10, "pass": lambda x: x <= 10}
        }
    ],
    "SVM (RBF Kernel)": [
        {
            "name": "Row Count Speed Check",
            "description": "RBF SVM scales poorly (O(N^2)) with dataset size.",
            "compute": check_size_vs_time,
            "threshold": {"fail": lambda x: x > 20000, "warning": lambda x: x > 5000, "pass": lambda x: x <= 5000}
        },
        {
            "name": "Scaling Differences",
            "description": "Checks if features have vastly different scales.",
            "compute": check_scaling_differences,
            "threshold": {"fail": lambda x: x > 100, "warning": lambda x: x > 10, "pass": lambda x: x <= 10}
        }
    ],
    "Linear Discriminant Analysis (LDA)": [
        {
            "name": "Class Balance",
            "description": "LDA can be sensitive to severe class imbalance.",
            "compute": check_class_balance,
            "threshold": {"warning": lambda x: x < 0.05, "pass": lambda x: x >= 0.05}
        },
        {
            "name": "Feature Normality",
            "description": "LDA assumes features are normally distributed (proxy: skewness).",
            "compute": check_feature_normality,
            "threshold": {"warning": lambda x: x > 2.0, "pass": lambda x: x <= 2.0}
        }
    ],
    "Quadratic Discriminant Analysis (QDA)": [
        {
            "name": "Rows Per Class",
            "description": "QDA needs enough rows per class to estimate covariance matrices robustly.",
            "compute": check_rows_per_class,
            "threshold": {"fail": lambda x: x < 10, "warning": lambda x: x < 30, "pass": lambda x: x >= 30}
        },
        {
            "name": "Feature Normality",
            "description": "QDA assumes features are normally distributed (proxy: skewness).",
            "compute": check_feature_normality,
            "threshold": {"warning": lambda x: x > 2.0, "pass": lambda x: x <= 2.0}
        }
    ],
    "Gaussian Naive Bayes": [
        {
            "name": "Feature Correlations",
            "description": "Naive Bayes assumes features are independent.",
            "compute": check_multicollinearity,
            "threshold": {"warning": lambda x: x > 0.5, "fail": lambda x: x > 0.8, "pass": lambda x: x <= 0.5}
        },
        {
            "name": "Feature Normality",
            "description": "Gaussian NB assumes features are normally distributed.",
            "compute": check_feature_normality,
            "threshold": {"warning": lambda x: x > 2.0, "pass": lambda x: x <= 2.0}
        }
    ],
    "Bernoulli Naive Bayes": [
        {
            "name": "Feature Correlations",
            "description": "Naive Bayes assumes features are independent.",
            "compute": check_multicollinearity,
            "threshold": {"warning": lambda x: x > 0.5, "fail": lambda x: x > 0.8, "pass": lambda x: x <= 0.5}
        }
    ],
    "AdaBoost Classifier": [
        {
            "name": "Noise Sensitivity",
            "description": "AdaBoost can overfit if there are many outliers.",
            "compute": check_noise_sensitivity,
            "threshold": {"warning": lambda x: x > 0.05, "pass": lambda x: x <= 0.05}
        }
    ],
    "AdaBoost Regressor": [
        {
            "name": "Noise Sensitivity",
            "description": "AdaBoost can overfit if there are many outliers.",
            "compute": check_noise_sensitivity,
            "threshold": {"warning": lambda x: x > 0.05, "pass": lambda x: x <= 0.05}
        }
    ],
    "MLP Classifier": [
        {
            "name": "Feature to Row Ratio",
            "description": "MLPs need lots of data compared to parameters.",
            "compute": check_feature_to_row_ratio,
            "threshold": {"warning": lambda x: x > 0.1, "pass": lambda x: x <= 0.1}
        },
        {
            "name": "Scaling Differences",
            "description": "MLPs require well-scaled data to converge.",
            "compute": check_scaling_differences,
            "threshold": {"fail": lambda x: x > 100, "warning": lambda x: x > 10, "pass": lambda x: x <= 10}
        }
    ],
    "MLP Regressor": [
        {
            "name": "Feature to Row Ratio",
            "description": "MLPs need lots of data compared to parameters.",
            "compute": check_feature_to_row_ratio,
            "threshold": {"warning": lambda x: x > 0.1, "pass": lambda x: x <= 0.1}
        },
        {
            "name": "Scaling Differences",
            "description": "MLPs require well-scaled data to converge.",
            "compute": check_scaling_differences,
            "threshold": {"fail": lambda x: x > 100, "warning": lambda x: x > 10, "pass": lambda x: x <= 10}
        }
    ]
}
