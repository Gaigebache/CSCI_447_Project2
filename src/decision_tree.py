
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


