from typing import Dict, Any
from .registry import ALGORITHM_REGISTRY, get_estimator

def get_regression_models(n_samples: int = 100) -> Dict[str, Any]:
    safe_k = max(1, min(5, n_samples - 1))
    models = {}
    for name, meta in ALGORITHM_REGISTRY.items():
        if meta["task"] == "regression":
            kwargs = {}
            if "n_neighbors" in meta["hyperparameters"]:
                kwargs["n_neighbors"] = safe_k
            models[name] = get_estimator(name, kwargs=kwargs)
    return models

def get_classification_models(n_samples: int = 100) -> Dict[str, Any]:
    safe_k = max(1, min(5, n_samples - 1))
    models = {}
    for name, meta in ALGORITHM_REGISTRY.items():
        if meta["task"] == "classification":
            kwargs = {}
            if "n_neighbors" in meta["hyperparameters"]:
                kwargs["n_neighbors"] = safe_k
            models[name] = get_estimator(name, kwargs=kwargs)
    return models

def get_clustering_models(n_samples: int = 100, n_clusters: int = 3) -> Dict[str, Any]:
    models = {}
    for name, meta in ALGORITHM_REGISTRY.items():
        if meta["task"] == "clustering":
            kwargs = {}
            if "n_clusters" in meta["hyperparameters"]:
                kwargs["n_clusters"] = n_clusters
            models[name] = get_estimator(name, kwargs=kwargs)
    return models
