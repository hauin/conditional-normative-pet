"""Cohort assembly: study x organ matrices used by every analysis."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Cohort:
    studies: pd.Index            # analyzed studies (first study per patient unless follow-ups requested)
    first: pd.Index              # first studies
    is_control: pd.Series        # no malignant lesion (diagnosis NEGATIVE)
    diagnosis: pd.Series
    patient_id: pd.Series
    covariates: pd.DataFrame     # age, sex (+ optional extra covariates)
    P: pd.DataFrame              # log lesion-cleaned median SUV (predictors)
    T: pd.DataFrame              # log target (SUVmax on the full mask)
    lesion_ml: pd.DataFrame      # lesion volume per organ
    refs: dict = field(default_factory=dict)   # reference-region log uptake per method and organ
    organs: list = field(default_factory=list)
    systems: dict = field(default_factory=lambda: C.SYSTEMS)
    organ_system: dict = field(default_factory=lambda: C.ORGAN_SYSTEM)

    @property
    def lesion_free(self) -> pd.DataFrame:
        return self.lesion_ml.eq(0)


def _pivot(tab, studies, feature, n_col, cols=None):
    """Study x organ matrix of log(feature); truncated or tiny masks are missing."""
    d = tab[tab.study_id.isin(studies) & ~tab.organ.isin(C.EXCLUDE)].copy()
    d.loc[d.truncated.astype(bool) | (d[n_col].fillna(0) < C.MIN_VOX), feature] = np.nan
    X = d.pivot_table(index="study_id", columns="organ", values=feature).reindex(studies)
    if cols is not None:
        X = X.reindex(columns=cols)
    return np.log(X.clip(lower=C.LOG_FLOOR))


def load_cohort(manifest: pd.DataFrame, organ_stats: pd.DataFrame, *, target: str = C.TARGET,
                extra_covariates: pd.DataFrame | None = None, followups: bool = False) -> Cohort:
    """Build the analysis matrices.

    manifest     one row per study: study_id, patient_id, diagnosis, age, sex (0/1), first_study
    organ_stats  one row per study and organ, as written by features.organ_features
    """
    man = manifest.set_index("study_id")
    have = man.index.isin(organ_stats.study_id.unique()) & man[["age", "sex"]].notna().all(axis=1).values
    first_flag = man.first_study.astype(bool).values
    studies = man.index[have if followups else have & first_flag]
    first = studies[man.loc[studies, "first_study"].astype(bool).values]
    dx = man.loc[studies, "diagnosis"].astype(str)
    is_control = dx.eq("NEGATIVE")
    cov = man.loc[studies, ["age", "sex"]].astype(float)
    if extra_covariates is not None:
        cov = pd.concat([cov, extra_covariates.reindex(studies).astype(float)], axis=1)

    P = _pivot(organ_stats, studies, C.PRED_FEATURE, "clean_n")
    controls_first = first[is_control.loc[first].values]
    P = P.loc[:, P.loc[controls_first].notna().mean() >= C.MIN_PRESENT]
    organs = list(P.columns)
    T = _pivot(organ_stats, studies, target, "full_n", organs)
    les = organ_stats[organ_stats.study_id.isin(studies)].pivot_table(
        index="study_id", columns="organ", values="lesion_ml").reindex(index=studies, columns=organs).fillna(0)

    # reference regions (Section 2.4): liver (liver itself -> aorta) and aortic blood pool
    # (blood-pool organs and heart -> liver); reference uptake is the lesion-cleaned mean
    M = _pivot(organ_stats, studies, C.REF_FEATURE, "clean_n")
    liver, aorta = M.get("liver"), M.get("aorta")
    blood = set(C.SYSTEMS["blood"]) | {"heart"}
    refs = {"raw": {o: 0.0 for o in organs},
            "liver_ref": {o: (aorta if o == "liver" else liver) for o in organs} if liver is not None else {},
            "blood_ref": {o: (liver if o in blood else aorta) for o in organs} if aorta is not None else {}}
    return Cohort(studies=studies, first=first, is_control=is_control, diagnosis=dx,
                  patient_id=man.loc[studies, "patient_id"], covariates=cov, P=P, T=T, lesion_ml=les,
                  refs=refs, organs=organs)
