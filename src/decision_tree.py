
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



