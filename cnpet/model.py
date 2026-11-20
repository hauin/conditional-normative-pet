"""Deviation scores (Section 2.4).

Every method turns the log target y_io of organ o in patient i into a standardized deviation

    z_io = (y_io - mu_o(x_i,-o)) / (k_o * sigma_o(x_i,-o))                       (1)

  raw          mu = training-control mean, sigma = SD                      (SUVmax)
  liver_ref    y - log(liver mean), standardized on controls                (slope fixed at 1)
  blood_ref    y - log(aortic mean), standardized on controls
  marg_lin     ridge regression on age and sex                              (marginal normative)
  glob_ref     y - mean log uptake of the other organs, standardized        (global uptake, slope 1)
  glob_lin     ridge regression on age, sex and that global uptake          (global uptake, M1)
  cond_lin     ridge regression on age, sex and the other organs, own compartment hidden (M2c)
  cond_M2      cond_lin without compartment masking                         (M2)
  cond_lin_het cond_lin divided by a patient-specific aleatoric SD           (variance model)
  cond_gbm / cond_het   the same with histogram gradient boosting

Ridge regression is the primary model: the predictors (other organs) are many and strongly
correlated, and the penalty stabilizes their weights while keeping a linear model whose
coefficients generalize the single fixed weight of reference-region normalization. Gradient
boosting is the non-linear alternative.

All scores are out-of-fold: patients are split into 5 folds and models are fitted on the
control organs of the training folds only. An inner 5-fold split of the training data gives
out-of-fold residuals for every training control; these set the scale k_o and train the
variance model, so neither is fitted on in-sample residuals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data import Cohort, global_uptake

REF_METHODS = ("raw", "liver_ref", "blood_ref")
LINEAR = ("marg_lin", "glob_lin", "cond_lin", "cond_M2", "cond_lin_het")
HETERO = ("cond_lin_het", "cond_het")
METHODS = ["raw", "liver_ref", "blood_ref", "marg_lin", "glob_ref", "glob_lin", "cond_lin", "cond_M2",
           "cond_lin_het", "cond_gbm", "cond_het"]


def ridge():
    """Predictors standardized on the training data; penalty by efficient leave-one-out CV."""
    return make_pipeline(StandardScaler(), RidgeCV(alphas=C.ALPHAS))


def gbm(seed: int = 0):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=20, l2_regularization=1.0, random_state=seed)


def _mean_model(method, seed):
    return ridge() if method in LINEAR else gbm(seed)


def fit_predict(Xtr, ytr, Xte, method, *, seed=0, var_shrink=C.VAR_SHRINK):
    """Return (mu_test, sigma_test or None, inner out-of-fold residuals, sigma_train or None)."""
    mu_te = _mean_model(method, seed).fit(Xtr, ytr).predict(Xte)
    res = np.zeros_like(ytr)
    for itr, ite in KFold(5, shuffle=True, random_state=seed).split(Xtr):
        res[ite] = ytr[ite] - _mean_model(method, seed).fit(Xtr[itr], ytr[itr]).predict(Xtr[ite])
    if method not in HETERO:
        return mu_te, None, res, None
    # aleatoric SD (Eq. 2): ridge regression g_o of log r^2 on the same predictors, shrunk by
    # lambda toward the mean m_o; +1.27 removes the bias E[log chi2_1] = -1.27
    lr2 = np.log(res ** 2 + 1e-6)
    g = ridge().fit(Xtr, lr2)
    shrink = lambda p: var_shrink * p + (1 - var_shrink) * lr2.mean()
    sd_tr = np.sqrt(np.exp(shrink(g.predict(Xtr)) - C.LOG_CHI2_1_MEAN))
    sd_te = np.sqrt(np.exp(shrink(g.predict(Xte)) - C.LOG_CHI2_1_MEAN))
    return mu_te, sd_te, res, sd_tr


def design(o: str, method: str, d: Cohort, train_ctrl, compartment_mask: bool = True) -> pd.DataFrame:
    """Predictor matrix for organ o: covariates (+ other organs). Missing values are imputed with
    the medians of the training controls of the fold."""
    cov = d.covariates.copy()
    extra = [c for c in cov.columns if c not in ("age", "sex")]
    if extra:                                   # optional covariates: winsorize and impute within fold
        c = cov.loc[train_ctrl, extra]
        cov[extra] = cov[extra].clip(lower=c.quantile(0.01), upper=c.quantile(0.99), axis=1).fillna(c.median())
    if method == "marg_lin":
        return cov
    if method == "glob_lin":
        X = pd.concat([cov, global_uptake(d, o).rename("global")], axis=1)
        return X.fillna(X.loc[train_ctrl].median())
    hide = {o}
    if compartment_mask and method != "cond_M2" and o in d.organ_system:
        hide |= set(d.systems[d.organ_system[o]])
    P = d.P[[c for c in d.P.columns if c not in hide]]
    P = P.fillna(P.loc[train_ctrl].median())
    return pd.concat([cov, P], axis=1)


def score_organ(o, method, d: Cohort, train_idx, test_idx, *, train="controls", compartment_mask=True,
                seed=0, T=None) -> tuple[pd.Series, pd.Series]:
    """Score organ o for test_idx with a model fitted on train_idx. Returns (z, aleatoric SD)."""
    y = (d.T if T is None else T)[o]
    z = pd.Series(np.nan, index=pd.Index(test_idx))
    alea = z.copy()
    tr = pd.Index(train_idx)
    ctrl = d.is_control.loc[tr].values
    # training rows: lesion-free organs of controls (or, with train='patients', of all patients)
    base = (ctrl if train == "controls" else np.ones(len(tr), bool)) & d.lesion_free.loc[tr, o].values \
        & y.loc[tr].notna().values
    if base.sum() < C.MIN_TRAIN:
        return z, alea
    if method in REF_METHODS or method == "glob_ref":
        r = global_uptake(d, o) if method == "glob_ref" else d.refs[method].get(o)
        if r is None:
            return z, alea
        s = y - r
        ok = base & ctrl & s.loc[tr].notna().values
        st = s.loc[tr][ok]
        z[:] = ((s.loc[test_idx] - st.mean()) / st.std()).values
        return z, alea
    X = design(o, method, d, tr[ctrl], compartment_mask)
    tri = tr[base]
    mu, sd_te, res_tr, sd_tr = fit_predict(X.loc[tri].values, y.loc[tri].values, X.loc[test_idx].values,
                                           method, seed=seed)
    ctrl_tri = d.is_control.loc[tri].values           # the scale k_o is always set on controls
    if sd_te is None:
        sd_te = res_tr.std()
        k = np.std(res_tr[ctrl_tri] / sd_te)
    else:
        k = np.std((res_tr / sd_tr)[ctrl_tri])
        alea[:] = sd_te
    z[:] = (y.loc[test_idx].values - mu) / (sd_te * k)
    return z, alea


def patient_folds(d: Cohort, seed: int = 0, n: int = 5):
    """Folds over patients: (training first studies, all studies of the held-out patients)."""
    pats = d.patient_id.loc[d.first].values
    for tr, te in KFold(n, shuffle=True, random_state=seed).split(d.first):
        yield d.first[tr], d.studies[d.patient_id.isin(set(pats[te])).values]


def _oof(o, m, d, kw):
    zs, als = [], []
    for tr, te in patient_folds(d):
        z, a = score_organ(o, m, d, tr, te, **kw)
        zs.append(z)
        als.append(a)
    z, a = pd.concat(zs), pd.concat(als)
    return pd.DataFrame({"study_id": z.index, "organ": o, "method": m, "z": z.values, "alea_sd": a.values})


def oof_scores(d: Cohort, methods=METHODS, n_jobs: int = 1, **kw) -> pd.DataFrame:
    """Out-of-fold deviation scores for every organ and method (long format)."""
    from joblib import Parallel, delayed
    jobs = [(o, m) for o in d.organs for m in methods if not (m in REF_METHODS and d.refs[m].get(o) is None)]
    Z = pd.concat(Parallel(n_jobs=n_jobs)(delayed(_oof)(o, m, d, kw) for o, m in jobs), ignore_index=True)
    Z["diagnosis"] = d.diagnosis.loc[Z.study_id].values
    lm = d.lesion_ml.stack().rename("lesion_ml").reset_index()
    lm.columns = ["study_id", "organ", "lesion_ml"]
    return Z.merge(lm, on=["study_id", "organ"], how="left")


def two_pass(d: Cohort, flag_z: float = C.FLAG_Z, n_jobs: int = 1) -> tuple[Cohort, pd.DataFrame]:
    """Label-free two-pass procedure (Section 2.6): score each organ's input median from the other
    organs (conditional linear, out of fold), hide predictor values with z > flag_z, and return a
    cohort whose predictors are imputed in the second pass like any missing value."""
    import copy
    Zp = oof_scores(d, ["cond_lin"], n_jobs=n_jobs, T=d.P)
    zp = Zp.pivot_table(index="study_id", columns="organ", values="z").reindex(index=d.studies, columns=d.P.columns)
    d2 = copy.copy(d)
    d2.P = d.P.mask(zp > flag_z)
    return d2, zp
