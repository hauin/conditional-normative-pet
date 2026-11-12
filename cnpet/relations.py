"""Inter-organ relationships in controls (Section 2.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from . import config as C
from .data import Cohort


def control_matrix(d: Cohort):
    ctrl = d.first[d.is_control.loc[d.first].values]
    X = d.P.loc[ctrl]
    return X.fillna(X.median()), d.covariates.loc[ctrl, ["age", "sex"]].values


def partial_correlations(d: Cohort) -> pd.DataFrame:
    """Pearson correlations of log clean median SUV after regressing out age and sex."""
    X, cov = control_matrix(d)
    res = X - LinearRegression().fit(cov, X).predict(cov)
    return res.corr()


def cv_r2(X, y, n_rep=20, n_folds=5):
    """Out-of-fold R2 of a ridge model, 5-fold CV repeated n_rep times (mean over repetitions)."""
    out = []
    for rep in range(n_rep):
        pred = np.zeros_like(y)
        for tr, te in KFold(n_folds, shuffle=True, random_state=rep).split(X):
            pred[te] = RidgeCV(alphas=C.ALPHAS).fit(X[tr], y[tr]).predict(X[te])
        out.append(r2_score(y, pred))
    return float(np.mean(out))


def predictor_sets(d: Cohort, n_rep=20) -> pd.DataFrame:
    """R2 for predicting each organ from
       M0  age and sex
       M1  M0 + global uptake (mean of the other organs)
       M2c M0 + organs outside the target's tissue compartment
       M2  M0 + all other organs"""
    X, cov = control_matrix(d)
    rows = []
    for o in X.columns:
        y = X[o].values
        others = X.drop(columns=o)
        hide = set(d.systems[d.organ_system[o]]) if o in d.organ_system else set()
        outside = others[[c for c in others.columns if c not in hide]]
        rows.append({"organ": o, "compartment": d.organ_system.get(o, "none"),
                     "M0": cv_r2(cov, y, n_rep),
                     "M1": cv_r2(np.hstack([cov, others.mean(axis=1).values[:, None]]), y, n_rep),
                     "M2c": cv_r2(np.hstack([cov, outside.values]), y, n_rep),
                     "M2": cv_r2(np.hstack([cov, others.values]), y, n_rep)})
    return pd.DataFrame(rows)
