import os
import numpy as np
import pandas as pd

from data_loader import load_all
from decision_tree import DecisionTree

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")


def split_indices(y, frac, rng, stratify):
    """Return (rest, part) index arrays; `part` is ~frac of the data.
    Stratified by class label when stratify=True (classification)."""
    if not stratify:
        perm = rng.permutation(len(y))
        k = int(round(len(y) * frac))
        return perm[k:], perm[:k]
    rest, part = [], []
    for c in np.unique(y):
        ids = rng.permutation(np.where(y == c)[0])
        k = int(round(len(ids) * frac))
        part.extend(ids[:k])
        rest.extend(ids[k:])
    print(f"splitting finished")
    return rng.permutation(np.array(rest)), rng.permutation(np.array(part))


def score(task, y_true, y_pred):
    if task == "classification":
        return float(np.mean(y_true != y_pred))           # 0/1 error rate
    return float(np.mean((y_true - y_pred) ** 2))          # MSE


def run_dataset(ds, repeats=5, prune_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    clf = ds.task == "classification"

    # 1) pull out the 20% pruning set ONCE; it stays fixed for every fold
    rest, prune_idx = split_indices(ds.y, prune_frac, rng, clf)
    X_rest, y_rest = ds.X[rest], ds.y[rest]
    X_prune, y_prune = ds.X[prune_idx], ds.y[prune_idx]

    rows = []
    for rep in range(repeats):
        # 2) 5x2 CV on the remaining 80%: shuffle, split in half, swap
        a, b = split_indices(y_rest, 0.5, rng, clf)
        for fold, (tr, te) in enumerate([(a, b), (b, a)]):
            tree = DecisionTree(ds.task, ds.feature_types, ds.feature_names,
                                ds.categories, ds.class_names)
            if clf:
                tree.n_classes = len(ds.class_names)   # in case a fold misses a class
            tree.fit(X_rest[tr], y_rest[tr])
            pruned = tree.copy().prune(X_prune, y_prune)

            # 3) test BOTH trees on the same held-out fold
            for method, model in (("unpruned", tree), ("pruned", pruned)):
                pred = model.predict(X_rest[te])
                rows.append(dict(dataset=ds.name, task=ds.task, rep=rep, fold=fold,
                                 method=method,
                                 score=score(ds.task, y_rest[te], pred),
                                 **model.stats()))
    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for ds in load_all():
        print(f"running {ds.name} ...")
        rows += run_dataset(ds)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "cv_results.csv"), index=False)
    print(df.groupby(["dataset", "method"])[["score", "nodes", "leaves", "depth"]]
            .mean().round(4))


if __name__ == "__main__":
    main()