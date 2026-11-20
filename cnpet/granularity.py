"""Finer organ units (Sections 2.11 and 3.12 of the paper).

Schemes
  base  the organ groups of config.GROUP_RULES
  mid   + left/right split of paired structures, lung lobes, SVC/IVC, sternum vs costal cartilages
  fine  mid + single vertebrae, ribs by side and level band (1-4, 5-8, 9-12), single gluteal
        muscles, single arteries

Workflow
  1. unit_map(class_map, scheme) -> TotalSegmentator class -> unit -> parent organ -> compartment
  2. unit_label_map(ts_labels, m) relabels a TotalSegmentator output into units; organ statistics are
     then computed with features.organ_features(..., names=unit_names(m))
  3. load_cohort(...) on the unit table, then set cohort.systems / organ_system with
     unit_systems(m), and score with model.oof_scores as for the organs
  4. parent_scores(...) combines unit z scores into parent-organ scores for a common evaluation
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C

PAIRED = {"adrenals", "scapulae", "clavicles", "humeri", "pelvis", "femurs", "autochthon", "iliopsoas",
          "gluteal", "ribs"}


def _group(name):
    return next((g for g, rx in C.GROUP_RULES if re.match(rx, name)), None)


def _side(name):
    return "left" if name.endswith("_left") or "_left_" in name else \
        "right" if name.endswith("_right") or "_right_" in name else ""


def unit_of(name: str, scheme: str):
    """(unit, parent organ) of a TotalSegmentator class, or None if the class is not used."""
    g = _group(name)
    if g is None or scheme == "base":
        return None if g is None else (g, g)
    u = g
    if g in ("lung_left", "lung_right") or g == "sternum":
        u = name                                        # lung_upper_lobe_left, costal_cartilages, ...
    elif g == "vena_cava":
        u = "svc" if name.startswith("superior") else "ivc"
    elif g in PAIRED and _side(name):
        u = f"{g}_{_side(name)}"
    if scheme == "fine":
        if g.startswith("vert_") or g in ("gluteal", "arteries_other"):
            u = name                                    # vertebrae_T5, gluteus_medius_left, ...
        elif g == "ribs":
            k = int(re.search(r"_(\d+)$", name).group(1))
            u = f"ribs_{_side(name)}_{'upper' if k <= 4 else 'mid' if k <= 8 else 'lower'}"
    return u, g


def unit_map(class_map: dict[int, str], scheme: str) -> pd.DataFrame:
    rows = [{"ts_label": lab, "ts_class": n, "unit": r[0], "parent": r[1], "compartment": C.ORGAN_SYSTEM.get(r[1], "")}
            for lab, n in sorted(class_map.items()) if (r := unit_of(n, scheme))]
    m = pd.DataFrame(rows)
    order = sorted(m.unit.unique(), key=lambda u: (C.GROUP_NAMES.index(m[m.unit == u].parent.iloc[0]), u))
    m["unit_id"] = m.unit.map({u: i + 1 for i, u in enumerate(order)})
    return m


def unit_names(m: pd.DataFrame) -> list[str]:
    return m.drop_duplicates("unit").sort_values("unit_id").unit.tolist()


def unit_label_map(ts_labels: np.ndarray, m: pd.DataFrame) -> np.ndarray:
    lut = np.zeros(m.ts_label.max() + 1, dtype=np.int16)
    lut[m.ts_label.values] = m.unit_id.values
    return lut[np.clip(ts_labels.astype(np.int64), 0, len(lut) - 1)]


def unit_systems(m: pd.DataFrame) -> tuple[dict, dict]:
    """Compartments over units. Units of a compartment-free parent with several units form a
    sibling group, hidden from each other like a compartment."""
    um = m.drop_duplicates("unit")
    sy: dict[str, list[str]] = {}
    for r in um.itertuples():
        key = r.compartment or (f"sib_{r.parent}" if (um.parent == r.parent).sum() > 1 else "")
        if key:
            sy.setdefault(key, []).append(r.unit)
    return sy, {u: s for s, us in sy.items() for u in us}


def sidak_score(zmax, k):
    """-log of the Sidak-corrected one-sided p of the maximum of k unit z scores."""
    logp = norm.logsf(zmax)
    p = np.exp(logp)
    with np.errstate(divide="ignore"):
        lp = np.where(p > 1e-8, np.log(-np.expm1(k * np.log1p(-p))), np.log(k) + logp)
    return -lp


def parent_scores(Z: pd.DataFrame, parent_of: dict, methods, how: str = "sidak") -> pd.DataFrame:
    """Combine unit z scores (long format of model.oof_scores) into parent-organ scores.
    how='max' keeps the maximum z; how='sidak' corrects it for the number of scored units."""
    W = Z.pivot_table(index=["study_id", "organ", "diagnosis"], columns="method", values="z").reset_index()
    W = W.dropna(subset=list(methods))
    W["parent"] = W.organ.map(parent_of)
    g = W.groupby(["study_id", "parent", "diagnosis"])
    P = g[list(methods)].max().reset_index()
    P["k"] = g.size().values
    if how == "sidak":
        for mth in methods:
            P[mth] = sidak_score(P[mth].values, P.k.values)
    return P.rename(columns={"parent": "organ"})
