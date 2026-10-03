from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


@dataclass
class Dataset:
    name: str
    task: str                      
    X: np.ndarray                  
    y: np.ndarray                  
    feature_names: list
    feature_types: list            
    categories: dict = field(default_factory=dict)
    class_names: list | None = None
    target_note: str = ""


def _path(*names):
    """Return the first file in DATA_DIR that exists among ``names``."""
    for n in names:
        p = os.path.join(DATA_DIR, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of {names} found in {DATA_DIR}")

def _encode(df: pd.DataFrame, feature_types: list, orders: dict | None = None):
    """Turn a DataFrame into a float matrix, coding categorical columns."""
    orders = orders or {}
    X = np.empty(df.shape, dtype=float)
    categories = {}
    for j, col in enumerate(df.columns):
        if feature_types[j] == "cat":
            vals = df[col].astype(str)
            labels = orders.get(col) or sorted(vals.unique())
            lookup = {v: i for i, v in enumerate(labels)}
            X[:, j] = vals.map(lookup).astype(float).values
            categories[j] = labels
        else:
            X[:, j] = pd.to_numeric(df[col]).astype(float).values
    return X, categories


def _encode_target(y_raw):
    labels = sorted(pd.Series(y_raw).astype(str).unique())
    lookup = {v: i for i, v in enumerate(labels)}
    return np.array([lookup[str(v)] for v in y_raw], dtype=int), labels


month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_order = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

# Machine: drop identifiers and erp (the authors' own regression estimate)
machine_cols = ['vendor_name', 'model_name', 'myct', 'mmin', 'mmax',
                'cach', 'chmin', 'chmax', 'prp', 'erp']
machine = pd.read_csv(_path("machine.data"), header=None, names=machine_cols)
machine = machine.drop(columns=['vendor_name', 'model_name', 'erp'])

# Abalone
abalone_cols = ['sex', 'length', 'diameter', 'height', 'whole_weight',
                'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
abalone = pd.read_csv(_path("abalone.data"), header=None, names=abalone_cols)

# Car
car_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
car = pd.read_csv(_path("car.data"), header=None, names=car_cols)

# Breast Cancer: drop id, drop the 16 rows with '?' in bare_nuclei
cancer_cols = ['id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',
               'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
               'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
cancer = pd.read_csv(_path("breast-cancer.data"), header=None, names=cancer_cols)
cancer = cancer.drop(columns=['id'])
cancer = cancer[cancer['bare_nuclei'] != '?'].reset_index(drop=True)
cancer['bare_nuclei'] = cancer['bare_nuclei'].astype(int)

# Forest Fires: log-transform the skewed target
fires = pd.read_csv(_path("forestfires.csv"))
fires['area'] = np.log1p(fires['area'])

# Congressional Votes: '?' means abstain, kept as its own category
house_cols = ['party', 'handicapped-infants', 'water-project-cost-sharing',
              'adoption-of-the-budget-resolution', 'physician-fee-freeze',
              'el-salvador-aid', 'religious-groups-in-schools',
              'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
              'mx-missile', 'immigration', 'synfuels-corporation-cut',
              'education-spending', 'superfund-right-to-sue', 'crime',
              'duty-free-exports', 'export-administration-act-south-africa']
house_df = pd.read_csv(_path("HouseVotes84.csv"), header=0, names=house_cols,
                       dtype=str, keep_default_na=False)
house_df = house_df.replace({'?': 'abstain', 'NA': 'abstain', '': 'abstain'})


# --- Machine (regression, target = prp) ---
machine_y = machine['prp'].to_numpy(dtype=float)
machine_feats = machine.drop(columns=['prp'])
machine_types = ['num'] * machine_feats.shape[1]
machine_X, machine_cats = _encode(machine_feats, machine_types)
machine_ds = Dataset("machine", "regression", machine_X, machine_y,
                     list(machine_feats.columns), machine_types, machine_cats)

# --- Abalone (regression, target = rings; sex stays categorical) ---
abalone_y = abalone['rings'].to_numpy(dtype=float)
abalone_feats = abalone.drop(columns=['rings'])
abalone_types = ['cat' if c == 'sex' else 'num' for c in abalone_feats.columns]
abalone_X, abalone_cats = _encode(abalone_feats, abalone_types)
abalone_ds = Dataset("abalone", "regression", abalone_X, abalone_y,
                     list(abalone_feats.columns), abalone_types, abalone_cats)

# --- Car (classification, all features categorical) ---
car_y, car_class_names = _encode_target(car['class'])
car_feats = car.drop(columns=['class'])
car_types = ['cat'] * car_feats.shape[1]
car_X, car_cats = _encode(car_feats, car_types)
car_ds = Dataset("car", "classification", car_X, car_y,
                 list(car_feats.columns), car_types, car_cats, car_class_names)

# --- Breast Cancer (classification, all features numeric) ---
cancer_y, _ = _encode_target(cancer['class'])          # '2' -> 0, '4' -> 1
cancer_class_names = ["benign", "malignant"]
cancer_feats = cancer.drop(columns=['class'])
cancer_types = ['num'] * cancer_feats.shape[1]
cancer_X, cancer_cats = _encode(cancer_feats, cancer_types)
cancer_ds = Dataset("breast_cancer", "classification", cancer_X, cancer_y,
                    list(cancer_feats.columns), cancer_types, cancer_cats,
                    cancer_class_names)


# --- Congressional Votes (classification, all features categorical) ---
house_y, house_class_names = _encode_target(house_df['party'])
house_feats = house_df.drop(columns=['party'])
house_types = ['cat'] * house_feats.shape[1]
house_X, house_cats = _encode(house_feats, house_types)
house_ds = Dataset("congressional_votes", "classification", house_X, house_y,
                   list(house_feats.columns), house_types, house_cats,
                   house_class_names)

# --- Forest Fires (regression, target = log1p(area); month/day categorical) ---
fires_y = fires['area'].to_numpy(dtype=float)           # already log1p-transformed
fires_feats = fires.drop(columns=['area'])
fires_types = ['cat' if c in ('month', 'day') else 'num' for c in fires_feats.columns]
fires_X, fires_cats = _encode(fires_feats, fires_types,
                              orders={'month': month_order, 'day': day_order})
fires_ds = Dataset("forest_fires", "regression", fires_X, fires_y,
                   list(fires_feats.columns), fires_types, fires_cats,
                   target_note="log1p(area); MSE is on the log scale")

def load_all():
    return [car_ds, cancer_ds, house_ds, abalone_ds, machine_ds, fires_ds]


if __name__ == "__main__":
    for ds in load_all():
        print(f"{ds.name:22s} {ds.task:15s} X={ds.X.shape} y={ds.y.shape}")