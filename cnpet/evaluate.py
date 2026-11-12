"""Evaluation (Sections 2.5 and 2.8): detection, calibration, repeat-scan behavior."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import config as C


def auc(lab, s):
    """Mann-Whitney estimate of the area under the ROC curve."""
    r = rankdata(s)
    n1 = lab.sum()
    n0 = len(lab) - n1
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fpr_at_sensitivity(lab, s, sens=0.9):
    thr = np.quantile(s[lab == 1], 1 - sens)
    return float((s[lab == 0] >= thr).mean())


def _members(groups):
    codes, ug = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.cumsum(np.bincount(codes))[:-1]), len(ug)


def bootstrap_diff(lab, a, b, groups, n_boot=1000, seed=0, stat=auc):
    """Percentile 95% CI of stat(a) - stat(b), resampling patients."""
    rng = np.random.default_rng(seed)
    mem, n = _members(groups)
    out = []
    for _ in range(n_boot):
        ii = np.concatenate([mem[k] for k in rng.integers(0, n, n)])
        if lab[ii].min() != lab[ii].max():
            out.append(stat(lab[ii], a[ii]) - stat(lab[ii], b[ii]))
    return tuple(np.percentile(out, [2.5, 97.5])) if out else (np.nan, np.nan)


def wide(Z: pd.DataFrame, methods) -> pd.DataFrame:
    """One row per study and organ; only organs scored by all methods (paired comparison)."""
    W = Z.pivot_table(index=["study_id", "organ", "diagnosis", "lesion_ml"], columns="method",
                      values="z").reset_index()
    return W.dropna(subset=[m for m in methods if m in W.columns])


def comparisons(W: pd.DataFrame, min_ml: float = C.MIN_LESION_ML) -> dict:
    """Positives: organs with >= min_ml lesion. Negatives: all organs of controls, or the
    organs of patients without any lesion voxel (within patients)."""
    pos = W.lesion_ml >= min_ml
    ctrl = W.diagnosis == "NEGATIVE"
    return {"against_control_organs": pos | ctrl,
            "within_patients": pos | (~ctrl & (W.lesion_ml == 0))}


def detection(Z: pd.DataFrame, methods, *, min_ml=C.MIN_LESION_ML, base="liver_ref", n_boot=1000):
    """Pooled AUC and FPR at 90% sensitivity, and AUC differences from `base` with patient-level
    bootstrap CIs."""
    W = wide(Z, methods)
    rows, diffs = [], []
    for cname, keep in comparisons(W, min_ml).items():
        w = W[keep]
        lab = (w.lesion_ml >= min_ml).values.astype(int)
        for m in methods:
            rows.append({"comparison": cname, "method": m, "n_pos": int(lab.sum()), "n_neg": int((lab == 0).sum()),
                         "AUC": auc(lab, w[m].values), "FPR_at_90sens": fpr_at_sensitivity(lab, w[m].values)})
            if m != base:
                lo, hi = bootstrap_diff(lab, w[m].values, w[base].values, w.study_id.values, n_boot)
                diffs.append({"comparison": cname, "method": m, "vs": base,
                              "dAUC": auc(lab, w[m].values) - auc(lab, w[base].values), "lo": lo, "hi": hi})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def per_organ_auc(Z, methods, *, min_ml=C.MIN_LESION_ML, comparison="within_patients", min_n=10):
    W = wide(Z, methods)
    W = W[comparisons(W, min_ml)[comparison]]
    rows = []
    for o, e in W.groupby("organ"):
        lab = (e.lesion_ml >= min_ml).values.astype(int)
        if lab.sum() >= min_n and (lab == 0).sum() >= min_n:
            rows.append({"organ": o, **{m: auc(lab, e[m].values) for m in methods}})
    return pd.DataFrame(rows)


def calibration(Z: pd.DataFrame, methods, het_method="cond_lin_het"):
    """Share of control organs with |z| > 1.96 (nominal 5%), overall and by within-organ
    quintile of the aleatoric SD predicted by the variance model."""
    neg = Z[(Z.diagnosis == "NEGATIVE") & Z.z.notna()]
    cal = neg.groupby("method").z.apply(lambda s: (s.abs() > 1.96).mean()).rename("overall")
    if het_method not in methods:
        return cal.to_frame()
    sd = Z[(Z.method == het_method) & Z.alea_sd.notna()][["study_id", "organ", "alea_sd"]].copy()
    sd["q"] = sd.groupby("organ").alea_sd.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    byq = neg.merge(sd[["study_id", "organ", "q"]], on=["study_id", "organ"]) \
        .groupby(["method", "q"]).z.apply(lambda s: (s.abs() > 1.96).mean()).unstack()
    byq.columns = [f"Q{int(c) + 1}" for c in byq.columns]
    return pd.concat([cal, byq], axis=1).reindex([m for m in methods if m in cal.index])


# ------------------------------------------------------------------ repeat scans (Section 2.8)
def icc_c1(x, y):
    """ICC(C,1): two-way, consistency, single measurement."""
    a = np.c_[x, y]
    n = len(a)
    gm, rm, cm = a.mean(), a.mean(1), a.mean(0)
    msr = 2 * ((rm - gm) ** 2).sum() / (n - 1)
    mse = ((a - rm[:, None] - cm[None, :] + gm) ** 2).sum() / (n - 1)
    return (msr - mse) / (msr + mse)


def repeat_stability(pairs: pd.DataFrame) -> dict:
    """pairs: one row per scan pair and lesion-free organ with columns pair, z1, z2.

    shared fraction = var(pair-mean dz) / var(dz); about 1/(number of organs) if organs change
    independently. Flags persisting = share of organs with z > 1.96 at the first scan that
    exceed 1.96 again at follow-up."""
    dz = pairs.z2 - pairs.z1
    f1 = pairs.z1 > 1.96
    return {"n_pairs": pairs.pair.nunique(), "ICC_C1": icc_c1(pairs.z1.values, pairs.z2.values),
            "SD_dz": dz.std(), "shared_fraction": dz.groupby(pairs.pair.values).transform("mean").var() / dz.var(),
            "flags_persisting": (pairs.z2[f1] > 1.96).mean() if f1.any() else np.nan}
