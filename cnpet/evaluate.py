"""Evaluation (Section 2.5 and the secondary analyses): detection, false-positive organs per
patient, calibration (overall, by predicted SD, upper tail, empirical), repeat-scan behavior.

The FPR is read at 90% sensitivity for the pooled comparisons; false positives per patient are
summarized at 80% sensitivity (and 50-90%), with thresholds cross-fitted over patient folds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
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


def fold_of(d) -> pd.Series:
    """Patient-level fold of every study (the folds of model.patient_folds)."""
    from .model import patient_folds
    f = pd.Series(-1, index=d.studies)
    for k, (_, te) in enumerate(patient_folds(d)):
        f.loc[te] = k
    return f


def fp_per_patient(Z: pd.DataFrame, methods, fold: pd.Series, *, sens_levels=(0.5, 0.7, 0.8, 0.9),
                   min_ml=C.MIN_LESION_ML):
    """Lesion-free organs flagged per patient at fixed sensitivity. Each fold's threshold is set on
    the lesion-bearing organs of the other folds (cross-fitted). Returns the summary table and the
    per-patient counts {(method, sensitivity): Series}."""
    W = wide(Z, methods)
    pat = W[W.diagnosis != "NEGATIVE"].copy()
    pat = pat[(pat.lesion_ml >= min_ml) | (pat.lesion_ml == 0)]
    pat["fold"] = fold.loc[pat.study_id].values
    is_pos = (pat.lesion_ml >= min_ml).values
    ids = pat.study_id.unique()
    rows, per_pt = [], {}
    for m in methods:
        for sens in sens_levels:
            thr = pd.Series(np.nan, index=pat.index)
            for k in pat.fold.unique():
                thr[pat.fold == k] = np.quantile(pat[m].values[(pat.fold != k).values & is_pos], 1 - sens)
            hit = pat[m] >= thr
            fp = hit[~is_pos].groupby(pat.study_id[~is_pos]).sum().reindex(ids).fillna(0)
            per_pt[(m, sens)] = fp
            rows.append({"method": m, "target_sens": sens, "achieved_sens": hit[is_pos].mean(),
                         "FP_organs_per_patient": fp.mean(), "pct_patients_0FP": 100 * (fp == 0).mean()})
    return pd.DataFrame(rows), per_pt


def paired_fp(per_pt: dict, a: str, b: str, n_boot=1000, seed=0) -> pd.DataFrame:
    """Per-patient difference in false-positive organs (b - a), with a patient bootstrap CI and the
    share of patients with fewer, the same or more false positives."""
    rng = np.random.default_rng(seed)
    rows = []
    for (m, sens), x_b in per_pt.items():
        if m != b or (a, sens) not in per_pt:
            continue
        x = (x_b - per_pt[(a, sens)]).dropna()
        bs = [x.values[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
        rows.append({"target_sens": sens, "method": b, "vs": a, "mean_diff": x.mean(),
                     "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "pct_fewer": 100 * (x < 0).mean(),
                     "pct_same": 100 * (x == 0).mean(), "pct_more": 100 * (x > 0).mean()})
    return pd.DataFrame(rows)


def tail_calibration(Z: pd.DataFrame, methods, tails=(1.645, 1.96, 2.326, 2.576)) -> pd.DataFrame:
    """One-sided exceedance in control organs (nominal 5, 2.5, 1, 0.5%). Log SUVmax is a maximum, so
    its deviations are right-skewed and the upper tail can exceed the nominal rate even when the
    two-sided rate is close to 5%."""
    neg = Z[(Z.diagnosis == "NEGATIVE") & Z.z.notna() & Z.method.isin(methods)]
    rows = []
    for m, g in neg.groupby("method"):
        r = {"method": m, "abs_z_gt_1.96_pct": 100 * (g.z.abs() > 1.96).mean(),
             "skew_median_by_organ": g.groupby("organ").z.apply(stats.skew).median()}
        for t in tails:
            r[f"z_gt_{t}_pct (nominal {100 * stats.norm.sf(t):.1f})"] = 100 * (g.z > t).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def empirical_calibration(Z: pd.DataFrame, fold: pd.Series) -> pd.DataFrame:
    """Cross-fitted empirical calibration: each z is converted to an upper-tail p value against the
    out-of-fold control scores of the same organ in the other folds, p = (1 + n_ge) / (n + 2), and
    back to a standard-normal quantile. Methods are renamed <method>_emp."""
    Z = Z.copy()
    Z["fold"] = fold.loc[Z.study_id].values
    out = []
    for (m, o), g in Z[Z.z.notna()].groupby(["method", "organ"]):
        ctrl = g[g.diagnosis == "NEGATIVE"]
        ze = pd.Series(np.nan, index=g.index)
        for k in g.fold.unique():
            ref = np.sort(ctrl.z[ctrl.fold != k].values)
            x = g.z[g.fold == k].values
            n_ge = len(ref) - np.searchsorted(ref, x, side="left")
            ze[g.fold == k] = stats.norm.isf((1 + n_ge) / (len(ref) + 2))
        out.append(g.assign(z=ze.values, method=m + "_emp"))
    return pd.concat(out).drop(columns="fold")


# ------------------------------------------------------------------ repeat scans
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
