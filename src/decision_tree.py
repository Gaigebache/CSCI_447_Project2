
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
        return self

    def _leaf_value(self, y):
        if self.task == "classification":
            counts = np.bincount(y, minlength=self.n_classes)
            return int(np.argmax(counts)), counts
        return float(y.mean()), None

    def _grow(self, X, y, used_cat):
        pred, counts = self._leaf_value(y)
        node = Node(pred, len(y), counts)
        # Stop only when the node cannot be split any further.
        if self.task == "classification":
            if np.count_nonzero(counts) <= 1:
                return node
        elif np.all(y == y[0]):
            return node

        best = self._best_split(X, y, used_cat)
        if best is None:           
            return node

        j, threshold, score = best
        node.feature, node.threshold, node.score = j, threshold, score
        node.children = {}
        if threshold is None:      
            col = X[:, j]
            child_used = used_cat | {j}
            for v in np.unique(col):
                mask = col == v
                node.children[int(v)] = self._grow(X[mask], y[mask], child_used)
        else:                      
            mask = X[:, j] <= threshold
            node.children["<="] = self._grow(X[mask], y[mask], used_cat)
            node.children[">"] = self._grow(X[~mask], y[~mask], used_cat)
        return node

    def _best_split(self, X, y, used_cat):
        """Return (feature, threshold or None, score) of the best split."""
        best, best_key = None, None
        if self.task == "classification":
            parent_H = _entropy_from_counts(np.bincount(y, minlength=self.n_classes))[0]
        for j, ftype in enumerate(self.feature_types):
            if ftype == "cat":
                if j in used_cat:
                    continue
                res = (self._cat_split_cls(X[:, j], y, parent_H)
                       if self.task == "classification" else self._cat_split_reg(X[:, j], y))
            else:
                res = (self._num_split_cls(X[:, j], y, parent_H)
                       if self.task == "classification" else self._num_split_reg(X[:, j], y))
            if res is None:
                continue
            threshold, score, tie = res
            
            key = (score, tie) if self.task == "classification" else (-score, 0.0)
            if best_key is None or key > best_key:
                best_key, best = key, (j, threshold, score)
        return best

    def _cat_split_cls(self, col, y, parent_H):
        values, inv = np.unique(col, return_inverse=True)
        if len(values) < 2:
            return None
        counts = np.zeros((len(values), self.n_classes))
        np.add.at(counts, (inv, y), 1)
        sizes = counts.sum(axis=1)
        n = len(y)
        expected = float((sizes / n * _entropy_from_counts(counts)).sum())
        gain = parent_H - expected
        iv = _split_info(sizes, n)
        return None, gain / iv, gain

    def _num_split_cls(self, col, y, parent_H):
        order = np.argsort(col, kind="mergesort")
        xs, ys = col[order], y[order]
        n = len(ys)
        cut = np.nonzero(xs[1:] != xs[:-1])[0]          
        if len(cut) == 0:
            return None
        onehot = np.zeros((n, self.n_classes))
        onehot[np.arange(n), ys] = 1
        cum = np.cumsum(onehot, axis=0)
        left = cum[cut]
        right = cum[-1] - left
        nl = cut + 1.0
        nr = n - nl
        expected = (nl * _entropy_from_counts(left) + nr * _entropy_from_counts(right)) / n
        gains = parent_H - expected
        i = int(np.argmax(gains))
        threshold = (xs[cut[i]] + xs[cut[i] + 1]) / 2.0
        iv = _split_info([nl[i], nr[i]], n)
        return threshold, gains[i] / iv, gains[i]

    @staticmethod
    def _cat_split_reg(col, y):
        values, inv = np.unique(col, return_inverse=True)
        if len(values) < 2:
            return None
        sizes = np.bincount(inv)
        sums = np.bincount(inv, weights=y)
        sq = np.bincount(inv, weights=y * y)
        sse = float((sq - sums ** 2 / sizes).sum())
        return None, max(sse, 0.0) / len(y), 0.0

    @staticmethod
    def _num_split_reg(col, y):
        order = np.argsort(col, kind="mergesort")
        xs, ys = col[order], y[order]
        n = len(ys)
        cut = np.nonzero(xs[1:] != xs[:-1])[0]
        if len(cut) == 0:
            return None
        cs, cs2 = np.cumsum(ys), np.cumsum(ys * ys)
        nl = cut + 1.0
        nr = n - nl
        sl, sl2 = cs[cut], cs2[cut]
        sr, sr2 = cs[-1] - sl, cs2[-1] - sl2
        sse = (sl2 - sl ** 2 / nl) + (sr2 - sr ** 2 / nr)
        i = int(np.argmin(sse))
        threshold = (xs[cut[i]] + xs[cut[i] + 1]) / 2.0
        return threshold, max(float(sse[i]), 0.0) / n, 0.0

    def _route(self, node, x):
        """Branch key for example x at an internal node (None if unseen value)."""
        v = x[node.feature]
        if node.threshold is None:
            k = int(v)
            return k if k in node.children else None
        return "<=" if v <= node.threshold else ">"

    def predict_one(self, x):
        node = self.root
        while not node.is_leaf:
            key = self._route(node, x)
            if key is None:        
                break
            node = node.children[key]
        return node.prediction

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = [self.predict_one(x) for x in X]
        return np.array(out, dtype=int if self.task == "classification" else float)

    def _error(self, prediction, y):
        if len(y) == 0:
            return 0.0
        if self.task == "classification":
            return float(np.sum(y != prediction))            
        return float(np.sum((y - prediction) ** 2))           

    def prune(self, X_prune, y_prune):
        """Reduced-error pruning using a held-out validation (pruning) set."""
        X_prune = np.asarray(X_prune, dtype=float)
        y_prune = np.asarray(y_prune)
        self._prune_node(self.root, X_prune, y_prune)
        return self

    def _prune_node(self, node, X, y):
        """Prune bottom-up; return the error of the (possibly pruned) subtree."""
        leaf_error = self._error(node.prediction, y)
        if node.is_leaf:
            return leaf_error
        subtree_error = 0.0
        keys = [self._route(node, x) for x in X]
        keys_arr = np.array(keys, dtype=object)
        stop = np.array([k is None for k in keys], dtype=bool)
        subtree_error += self._error(node.prediction, y[stop])
        for key, child in node.children.items():
            mask = np.array([k == key for k in keys_arr], dtype=bool) if len(keys) else \
                np.zeros(0, dtype=bool)
            subtree_error += self._prune_node(child, X[mask], y[mask])
        if leaf_error <= subtree_error:
            node.make_leaf()
            return leaf_error
        return subtree_error

    def copy(self):
        return copy.deepcopy(self)

    def stats(self):
        n_nodes = n_leaves = depth = 0
        stack = [(self.root, 0)]
        while stack:
            node, d = stack.pop()
            n_nodes += 1
            depth = max(depth, d)
            if node.is_leaf:
                n_leaves += 1
            else:
                stack.extend((c, d + 1) for c in node.children.values())
        return {"nodes": n_nodes, "leaves": n_leaves, "depth": depth}

    def _label(self, value):
        if self.task == "classification" and self.class_names:
            return self.class_names[value]
        return f"{value:.3g}" if self.task == "regression" else str(value)

    def to_text(self, max_depth=None):
        """Human-readable rules, one line per branch."""
        lines = []

        def walk(node, depth, prefix):
            if node.is_leaf or (max_depth is not None and depth >= max_depth):
                tag = "" if node.is_leaf else " ..."
                lines.append(f"{prefix}-> {self._label(node.prediction)} "
                             f"(n={node.n_samples}){tag}")
                return
            name = self.feature_names[node.feature]
            for key, child in node.children.items():
                if node.threshold is None:
                    labels = self.categories.get(node.feature)
                    cond = f"{name} = {labels[key] if labels else key}"
                else:
                    cond = f"{name} {key} {node.threshold:.4g}"
                lines.append(f"{prefix}{cond}")
                walk(child, depth + 1, prefix + "|   ")

        walk(self.root, 0, "")
        return "\n".join(lines)


