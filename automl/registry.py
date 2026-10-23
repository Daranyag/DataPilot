
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, Perceptron
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, 
    GradientBoostingRegressor, GradientBoostingClassifier, 
    BaggingClassifier, BaggingRegressor,
    ExtraTreesRegressor, ExtraTreesClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN, Birch, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.dummy import DummyClassifier, DummyRegressor

# Custom wrapper for Polynomial Regression
def get_polynomial_regression():
    return Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])

# Custom K-Medoids (simple implementation)
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class SimpleKMedoids(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_indices_ = None
        
    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        if n_samples < self.n_clusters:
            self.n_clusters = n_samples
        
        self.cluster_centers_indices_ = rng.choice(n_samples, self.n_clusters, replace=False)
        distances = pairwise_distances(X, metric='euclidean')
        
        for _ in range(self.max_iter):
            old_indices = self.cluster_centers_indices_.copy()
            self.labels_ = np.argmin(distances[:, self.cluster_centers_indices_], axis=1)
            
            for i in range(self.n_clusters):
                cluster_points = np.where(self.labels_ == i)[0]
                if len(cluster_points) == 0:
                    continue
                # Find the point that minimizes sum of distances to all other points in cluster
                cluster_distances = distances[np.ix_(cluster_points, cluster_points)]
                best_medoid_idx = np.argmin(np.sum(cluster_distances, axis=1))
                self.cluster_centers_indices_[i] = cluster_points[best_medoid_idx]
                
            if np.array_equal(old_indices, self.cluster_centers_indices_):
                break
        return self
        
    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

ALGORITHM_REGISTRY = {
    # REGRESSION
    "Baseline (Predicts Mean)": {
        "task": "regression",
        "category": "Syllabus",
        "class": DummyRegressor,
        "kwargs": {"strategy": "mean"},
        "needs_scaling": False,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {}
    },
    "Linear Regression": {
        "task": "regression",
        "category": "Syllabus",
        "class": LinearRegression,
        "needs_scaling": False,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {}
    },
    "Ridge Regression": {
        "task": "regression",
        "category": "Syllabus",
        "class": Ridge,
        "needs_scaling": True,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"alpha": {"type": float, "min": 0.001, "max": 1000.0}}
    },
    "Lasso Regression": {
        "task": "regression",
        "category": "Syllabus",
        "class": Lasso,
        "needs_scaling": True,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"alpha": {"type": float, "min": 0.001, "max": 1000.0}}
    },
    "ElasticNet Regression": {
        "task": "regression",
        "category": "Syllabus",
        "class": ElasticNet,
        "needs_scaling": True,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"alpha": {"type": float, "min": 0.001, "max": 1000.0}, "l1_ratio": {"type": float, "min": 0.0, "max": 1.0}}
    },
    "Polynomial Regression": {
        "task": "regression",
        "category": "Extended",
        "class": get_polynomial_regression,
        "needs_scaling": True,
        "speed": "medium",
        "row_cap": 20000,
        "skip_reason": None,
        "hyperparameters": {}
    },
    "Decision Tree Regressor": {
        "task": "regression",
        "category": "Syllabus",
        "class": DecisionTreeRegressor,
        "needs_scaling": False,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"max_depth": {"type": int, "min": 1, "max": 100}}
    },
    "Random Forest Regressor": {
        "task": "regression",
        "category": "Syllabus",
        "class": RandomForestRegressor,
        "needs_scaling": False,
        "speed": "medium",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"n_estimators": {"type": int, "min": 1, "max": 2000}, "max_depth": {"type": int, "min": 1, "max": 100}}
    },
    "Extra Trees Regressor": {
        "task": "regression",
        "category": "Extended",
        "class": ExtraTreesRegressor,
        "needs_scaling": False,
        "speed": "medium",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"n_estimators": {"type": int, "min": 1, "max": 2000}}
    },
    "AdaBoost Regressor": {
        "task": "regression",
        "category": "Extended",
        "class": AdaBoostRegressor,
        "needs_scaling": False,
        "speed": "medium",
        "row_cap": 50000,
        "skip_reason": None,
        "hyperparameters": {"n_estimators": {"type": int, "min": 1, "max": 1000}, "learning_rate": {"type": float, "min": 0.01, "max": 10.0}}
    },
    "Gradient Boosting Regressor": {
        "task": "regression",
        "category": "Syllabus",
        "class": GradientBoostingRegressor,
        "needs_scaling": False,
        "speed": "slow",
        "row_cap": 20000,
        "skip_reason": None,
        "hyperparameters": {"n_estimators": {"type": int, "min": 1, "max": 1000}, "learning_rate": {"type": float, "min": 0.01, "max": 1.0}}
    },
    "HistGradientBoosting Regressor": {
        "task": "regression",
        "category": "Extended",
        "class": HistGradientBoostingRegressor,
        "needs_scaling": False,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"learning_rate": {"type": float, "min": 0.01, "max": 1.0}}
    },
    "Bagging Regressor": {
        "task": "regression",
        "category": "Extended",
        "class": BaggingRegressor,
        "needs_scaling": False,
        "speed": "medium",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"n_estimators": {"type": int, "min": 1, "max": 500}}
    },
    "Support Vector Regressor": {
        "task": "regression",
        "category": "Syllabus",
        "class": SVR,
        "needs_scaling": True,
        "speed": "slow",
        "row_cap": 10000,
        "skip_reason": None,
        "hyperparameters": {"C": {"type": float, "min": 0.01, "max": 1000.0}}
    },
    "KNN Regressor": {
        "task": "regression",
        "category": "Syllabus",
        "class": KNeighborsRegressor,
        "needs_scaling": True,
        "speed": "medium",
        "row_cap": 50000,
        "skip_reason": None,
        "hyperparameters": {"n_neighbors": {"type": int, "min": 1, "max": 100}}
    },
    "MLP Regressor": {
        "task": "regression",
        "category": "Extended",
        "class": MLPRegressor,
        "kwargs": {"early_stopping": True, "hidden_layer_sizes": (50,)},
        "needs_scaling": True,
        "speed": "slow",
        "row_cap": 20000,
        "skip_reason": None,
        "hyperparameters": {"learning_rate_init": {"type": float, "min": 0.0001, "max": 0.1}}
    },

    # CLASSIFICATION
    "Baseline (Predicts Most Common)": {
        "task": "classification",
        "category": "Syllabus",
        "class": DummyClassifier,
        "kwargs": {"strategy": "most_frequent"},
        "needs_scaling": False,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {}
    },
    "Logistic Regression": {
        "task": "classification",
        "category": "Syllabus",
        "class": LogisticRegression,
        "needs_scaling": True,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"C": {"type": float, "min": 0.001, "max": 1000.0}}
    },
    "Perceptron": {
        "task": "classification",
        "category": "Extended",
        "class": Perceptron,
        "needs_scaling": True,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"alpha": {"type": float, "min": 0.0001, "max": 1.0}}
    },
    "SVM (Linear Kernel)": {
        "task": "classification",
        "category": "Extended",
        "class": SVC,
        "kwargs": {"kernel": "linear"},
        "needs_scaling": True,
        "speed": "slow",
        "row_cap": 10000,
        "skip_reason": None,
        "hyperparameters": {"C": {"type": float, "min": 0.01, "max": 1000.0}}
    },
    "SVM (RBF Kernel)": {
        "task": "classification",
        "category": "Syllabus",
        "class": SVC,
        "kwargs": {"kernel": "rbf"},
        "needs_scaling": True,
        "speed": "slow",
        "row_cap": 10000,
        "skip_reason": None,
        "hyperparameters": {"C": {"type": float, "min": 0.01, "max": 1000.0}}
    },
    "Linear Discriminant Analysis (LDA)": {
        "task": "classification",
        "category": "Extended",
        "class": LinearDiscriminantAnalysis,
        "kwargs": {"solver": "eigen", "shrinkage": "auto"},
        "needs_scaling": False,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {}
    },
    "Quadratic Discriminant Analysis (QDA)": {
        "task": "classification",
        "category": "Extended",
        "class": QuadraticDiscriminantAnalysis,
        "kwargs": {"reg_param": 0.1},
        "needs_scaling": False,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {}
    },
    "Gaussian Naive Bayes": {
        "task": "classification",
        "category": "Syllabus",
        "class": GaussianNB,
        "needs_scaling": False,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {}
    },
    "Bernoulli Naive Bayes": {
        "task": "classification",
        "category": "Extended",
        "class": BernoulliNB,
        "needs_scaling": False,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"alpha": {"type": float, "min": 0.0, "max": 1.0}}
    },
    "Decision Tree Classifier": {
        "task": "classification",
        "category": "Syllabus",
        "class": DecisionTreeClassifier,
        "needs_scaling": False,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"max_depth": {"type": int, "min": 1, "max": 100}}
    },
    "Random Forest Classifier": {
        "task": "classification",
        "category": "Syllabus",
        "class": RandomForestClassifier,
        "needs_scaling": False,
        "speed": "medium",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"n_estimators": {"type": int, "min": 1, "max": 2000}, "max_depth": {"type": int, "min": 1, "max": 100}}
    },
    "Extra Trees Classifier": {
        "task": "classification",
        "category": "Extended",
        "class": ExtraTreesClassifier,
        "needs_scaling": False,
        "speed": "medium",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"n_estimators": {"type": int, "min": 1, "max": 2000}}
    },
    "AdaBoost Classifier": {
        "task": "classification",
        "category": "Extended",
        "class": AdaBoostClassifier,
        "needs_scaling": False,
        "speed": "medium",
        "row_cap": 50000,
        "skip_reason": None,
        "hyperparameters": {"n_estimators": {"type": int, "min": 1, "max": 1000}, "learning_rate": {"type": float, "min": 0.01, "max": 10.0}}
    },
    "Gradient Boosting Classifier": {
        "task": "classification",
        "category": "Syllabus",
        "class": GradientBoostingClassifier,
        "needs_scaling": False,
        "speed": "slow",
        "row_cap": 20000,
        "skip_reason": None,
        "hyperparameters": {"n_estimators": {"type": int, "min": 1, "max": 1000}, "learning_rate": {"type": float, "min": 0.01, "max": 1.0}}
    },
    "HistGradientBoosting Classifier": {
        "task": "classification",
        "category": "Extended",
        "class": HistGradientBoostingClassifier,
        "needs_scaling": False,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"learning_rate": {"type": float, "min": 0.01, "max": 1.0}}
    },
    "Bagging Classifier": {
        "task": "classification",
        "category": "Extended",
        "class": BaggingClassifier,
        "needs_scaling": False,
        "speed": "medium",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"n_estimators": {"type": int, "min": 1, "max": 500}}
    },
    "KNN Classifier": {
        "task": "classification",
        "category": "Syllabus",
        "class": KNeighborsClassifier,
        "needs_scaling": True,
        "speed": "medium",
        "row_cap": 50000,
        "skip_reason": None,
        "hyperparameters": {"n_neighbors": {"type": int, "min": 1, "max": 100}}
    },
    "MLP Classifier": {
        "task": "classification",
        "category": "Extended",
        "class": MLPClassifier,
        "kwargs": {"early_stopping": True, "hidden_layer_sizes": (50,)},
        "needs_scaling": True,
        "speed": "slow",
        "row_cap": 20000,
        "skip_reason": None,
        "hyperparameters": {"learning_rate_init": {"type": float, "min": 0.0001, "max": 0.1}}
    },

    # CLUSTERING
    "K-Means": {
        "task": "clustering",
        "category": "Syllabus",
        "class": KMeans,
        "needs_scaling": True,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"n_clusters": {"type": int, "min": 2, "max": 100}}
    },
    "MiniBatch K-Means": {
        "task": "clustering",
        "category": "Extended",
        "class": MiniBatchKMeans,
        "needs_scaling": True,
        "speed": "fast",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"n_clusters": {"type": int, "min": 2, "max": 100}}
    },
    "Gaussian Mixture": {
        "task": "clustering",
        "category": "Extended",
        "class": GaussianMixture,
        "needs_scaling": True,
        "speed": "medium",
        "row_cap": 50000,
        "skip_reason": None,
        "hyperparameters": {"n_components": {"type": int, "min": 2, "max": 100}}
    },
    "Agglomerative (Ward)": {
        "task": "clustering",
        "category": "Extended",
        "class": AgglomerativeClustering,
        "kwargs": {"linkage": "ward"},
        "needs_scaling": True,
        "speed": "slow",
        "row_cap": 20000,
        "skip_reason": None,
        "hyperparameters": {"n_clusters": {"type": int, "min": 2, "max": 100}}
    },
    "Agglomerative (Average)": {
        "task": "clustering",
        "category": "Extended",
        "class": AgglomerativeClustering,
        "kwargs": {"linkage": "average"},
        "needs_scaling": True,
        "speed": "slow",
        "row_cap": 20000,
        "skip_reason": None,
        "hyperparameters": {"n_clusters": {"type": int, "min": 2, "max": 100}}
    },
    "Agglomerative (Complete)": {
        "task": "clustering",
        "category": "Extended",
        "class": AgglomerativeClustering,
        "kwargs": {"linkage": "complete"},
        "needs_scaling": True,
        "speed": "slow",
        "row_cap": 20000,
        "skip_reason": None,
        "hyperparameters": {"n_clusters": {"type": int, "min": 2, "max": 100}}
    },
    "DBSCAN": {
        "task": "clustering",
        "category": "Syllabus",
        "class": DBSCAN,
        "needs_scaling": True,
        "speed": "slow",
        "row_cap": 20000,
        "skip_reason": None,
        "hyperparameters": {"eps": {"type": float, "min": 0.001, "max": 100.0}, "min_samples": {"type": int, "min": 1, "max": 1000}}
    },
    "OPTICS": {
        "task": "clustering",
        "category": "Extended",
        "class": OPTICS,
        "needs_scaling": True,
        "speed": "slow",
        "row_cap": 10000,
        "skip_reason": None,
        "hyperparameters": {"min_samples": {"type": int, "min": 2, "max": 1000}}
    },
    "BIRCH": {
        "task": "clustering",
        "category": "Syllabus",
        "class": Birch,
        "needs_scaling": True,
        "speed": "medium",
        "row_cap": None,
        "skip_reason": None,
        "hyperparameters": {"n_clusters": {"type": int, "min": 2, "max": 100}}
    },
    "K-Medoids": {
        "task": "clustering",
        "category": "Extended",
        "class": SimpleKMedoids,
        "needs_scaling": True,
        "speed": "slow",
        "row_cap": 5000,
        "skip_reason": None,
        "hyperparameters": {"n_clusters": {"type": int, "min": 2, "max": 100}}
    }
}

ALGORITHM_ALIASES = {}
for name, meta in ALGORITHM_REGISTRY.items():
    ALGORITHM_ALIASES[name.lower()] = (meta["task"], meta["class"])

def get_estimator(algo_name: str, task_type: str = None, kwargs: dict = None):
    algo_name_lower = algo_name.lower().strip()
    kwargs = kwargs or {}
    
    # Try finding exact match in registry first (case-insensitive)
    found_key = None
    for k in ALGORITHM_REGISTRY.keys():
        if k.lower() == algo_name_lower:
            found_key = k
            break
            
    if not found_key:
        raise ValueError(f"Algorithm '{algo_name}' is not supported.")
        
    meta = ALGORITHM_REGISTRY[found_key]
    
    # Merge predefined kwargs (like kernel='linear' for SVM)
    base_kwargs = meta.get("kwargs", {}).copy()
    
    # Validate dynamic hyperparameters
    for key, value in kwargs.items():
        if key not in meta["hyperparameters"]:
            raise ValueError(f"Hyperparameter '{key}' is not allowed for {found_key}.")
            
        rules = meta["hyperparameters"][key]
        try:
            val = rules["type"](value)
        except ValueError:
            raise ValueError(f"Hyperparameter '{key}' must be of type {rules['type'].__name__}.")
            
        if "min" in rules and val < rules["min"]:
            raise ValueError(f"Hyperparameter '{key}' must be >= {rules['min']}.")
        if "max" in rules and val > rules["max"]:
            raise ValueError(f"Hyperparameter '{key}' must be <= {rules['max']}.")
        if "allowed" in rules and val not in rules["allowed"]:
            raise ValueError(f"Hyperparameter '{key}' must be one of {rules['allowed']}.")
            
        base_kwargs[key] = val
        
    return meta["class"](**base_kwargs)
