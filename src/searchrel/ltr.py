"""Learning to rank. LightGBM lambdarank when it can be imported (pip install searchrel[lightgbm], plus libomp on
macOS), otherwise scikit-learn gradient boosting trained pointwise on the grade. Pointwise regression optimizes
squared error on labels rather than NDCG directly, but on this data it ranks almost as well and has no native
dependency, so it is what CI and the default install use. SEARCHREL_LTR=sklearn forces the fallback.
"""

from __future__ import annotations

import os

import numpy as np
from threadpoolctl import threadpool_limits


def lightgbm_available() -> bool:
    if os.environ.get("SEARCHREL_LTR", "").lower() == "sklearn":
        return False
    try:
        import lightgbm  # noqa: F401
    except Exception:  # ImportError, or OSError when libomp is missing
        return False
    return True


class Ranker:
    def __init__(self, backend: str | None = None, seed: int = 0, n_estimators: int = 150):
        self.backend = backend or ("lightgbm" if lightgbm_available() else "sklearn")
        self.seed = seed
        self.n_estimators = n_estimators
        self.model = None
        self.medians: np.ndarray | None = None
        self.train_rows = 0

    def fit(self, X: np.ndarray, y: np.ndarray, groups: list[int]) -> Ranker:
        """Rows must be grouped by query in order; groups gives the number of rows per query."""
        self.medians = np.median(X, axis=0)
        self.train_rows = len(y)
        if self.backend == "lightgbm":
            import lightgbm

            self.model = lightgbm.LGBMRanker(objective="lambdarank", n_estimators=self.n_estimators,
                                             learning_rate=0.05, num_leaves=31, min_child_samples=20,
                                             random_state=self.seed, verbose=-1)
            self.model.fit(X, np.rint(y).astype(int), group=groups)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.model = HistGradientBoostingRegressor(max_iter=self.n_estimators, learning_rate=0.08,
                                                       max_leaf_nodes=31, min_samples_leaf=20,
                                                       l2_regularization=1.0, early_stopping=False,
                                                       random_state=self.seed)
            with threadpool_limits(limits=1):  # tens of thousands of rows: one thread beats OpenMP overhead
                self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        with threadpool_limits(limits=1):
            return np.asarray(self.model.predict(X), dtype=np.float64)

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Per-feature contribution to each score. LightGBM gives tree SHAP values. For scikit-learn this walks each
        tree and credits every split's change in node value to the split feature (the Saabas method), so the
        contributions plus a constant add up exactly to the score."""
        if len(X) == 0:
            return np.zeros_like(X)
        if self.backend == "lightgbm":
            return np.asarray(self.model.predict(X, pred_contrib=True))[:, :-1]
        out = np.zeros_like(X, dtype=np.float64)
        lr = self.model.learning_rate
        for (tree,) in self.model._predictors:
            nodes = tree.nodes
            # scikit-learn applies the learning rate to leaf values only, so scale split nodes to match
            value = np.where(nodes["is_leaf"] == 1, nodes["value"], nodes["value"] * lr)
            for r in range(len(X)):
                i = 0
                while not nodes["is_leaf"][i]:
                    f = nodes["feature_idx"][i]
                    nxt = nodes["left"][i] if X[r, f] <= nodes["num_threshold"][i] else nodes["right"][i]
                    out[r, f] += value[nxt] - value[i]
                    i = nxt
        return out

    def bias(self) -> float:
        """Score of a document with no contributions (the sum of the trees' root values plus the baseline)."""
        if self.backend == "lightgbm":
            return 0.0
        base = float(np.ravel(self.model._baseline_prediction)[0])
        lr = self.model.learning_rate
        return base + sum(float(t.nodes["value"][0]) * lr for (t,) in self.model._predictors)


def rank_order(scores: np.ndarray, resolution: float = 0.01) -> np.ndarray:
    """Indices that sort scores high to low. Scores within the resolution count as ties and keep the incoming
    (BM25) order, so tree-model noise in the third decimal does not shuffle equally good results."""
    return np.argsort(-np.round(scores / resolution), kind="stable")
