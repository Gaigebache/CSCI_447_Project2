
from __future__ import annotations

import copy
import sys

import numpy as np

sys.setrecursionlimit(20000)

class Node:
    __slots__ = ("feature", "threshold", "children", "prediction", "n_samples",
                 "class_counts", "score")

    def __init__(self, prediction, n_samples, class_counts=None):
        self.feature = None       
        self.threshold = None     
        self.children = None       
        self.prediction = prediction
        self.n_samples = n_samples
        self.class_counts = class_counts
        self.score = None          

    @property
    def is_leaf(self):
        return self.children is None

    def make_leaf(self):
        self.feature = self.threshold = self.children = None

def _entropy_from_counts(counts):
    """Entropy (base 2) for each row of a (m, k) count matrix."""
    counts = np.atleast_2d(counts).astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0, counts / totals, 0.0)
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=1)


def _split_info(sizes, n):
    """Intrinsic value IV = -sum |D_j|/|D| lg(|D_j|/|D|)."""
    p = np.asarray(sizes, dtype=float) / n
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

class DecisionTree:

    def __init__(self, task, feature_types, feature_names=None, categories=None,
                 class_names=None):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.feature_types = list(feature_types)
        self.feature_names = feature_names or [f"x{j}" for j in range(len(feature_types))]
        self.categories = categories or {}
        self.class_names = class_names
        self.root = None
        self.n_classes = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task == "classification":
            y = y.astype(int)
            self.n_classes = int(y.max()) + 1 if self.n_classes is None else self.n_classes
        else:
            y = y.astype(float)
        self.root = self._grow(X, y, used_cat=frozenset())

