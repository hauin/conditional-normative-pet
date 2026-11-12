"""Image processing (Sections 2.1-2.2): organ masks and organ-level SUV statistics.

Inputs per autoPET study folder (NIfTI release of FDG-PET-CT-Lesions):
  SUV.nii.gz    PET in SUV units
  CTres.nii.gz  CT resampled to the PET grid
  SEG.nii.gz    manual lesion segmentation released with the dataset (binary)

TotalSegmentator is run on CTres, so its labels are on the PET grid and no further
resampling is needed. Its 117 classes are merged into the organ groups of config.GROUP_RULES.
For each group, SUV statistics are computed on three masks:
  full    the organ mask
  eroded  the mask eroded by 4 mm (partial-volume and misregistration margin)
  clean   the eroded mask minus lesions dilated by 4 mm
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import config as C


# ------------------------------------------------------------------ helpers
def load_nii(path):
    import nibabel as nib
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), img


def superior_axis(affine: np.ndarray) -> tuple[int, int]:
    """(array axis along head-foot, +1 if increasing index is superior else -1)."""
    axis = int(np.argmax(np.abs(affine[2, :3])))
    return axis, int(np.sign(affine[2, axis]) or 1)


def ball(mm: float, zooms) -> np.ndarray:
    """Ellipsoidal structuring element with radius `mm` in each voxel direction."""
    r = [max(1, int(round(mm / z))) for z in zooms]
    g = np.ogrid[tuple(slice(-k, k + 1) for k in r)]
    return sum((gi / ri) ** 2 for gi, ri in zip(g, r)) <= 1.0


def suv_stats(v: np.ndarray, prefix: str) -> dict:
    if v.size == 0:
        return {f"{prefix}_n": 0}
    return {f"{prefix}_n": int(v.size), f"{prefix}_mean": float(v.mean()), f"{prefix}_max": float(v.max()),
            f"{prefix}_median": float(np.median(v)), f"{prefix}_p90": float(np.percentile(v, 90))}


def group_lookup(class_map: dict[int, str]) -> np.ndarray:
    """Lookup array: TotalSegmentator label -> organ group id (0 = not used)."""
    lut = np.zeros(max(class_map) + 1, dtype=np.uint8)
    for lab, name in class_map.items():
        for gid, (_, rx) in enumerate(C.GROUP_RULES, start=1):
            if re.match(rx, name):
                lut[lab] = gid
                break
    return lut


# ------------------------------------------------------------------ segmentation
def segment_study(study_dir: Path, out_dir: Path, device: str = "gpu") -> Path:
    """TotalSegmentator v2 ('total' task, 1.5 mm model, body crop, split inference) on CTres,
    then merge classes into organ groups. Returns the path of the group label map."""
    import nibabel as nib
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.python_api import totalsegmentator

    out_dir.mkdir(parents=True, exist_ok=True)
    sid = f"{study_dir.parent.name}__{study_dir.name}"
    raw, grp = out_dir / f"{sid}_ts.nii.gz", out_dir / f"{sid}_groups.nii.gz"
    if not raw.exists():
        totalsegmentator(str(study_dir / "CTres.nii.gz"), str(raw), ml=True, body_seg=True,
                         force_split=True, device=device, quiet=True)
    if not grp.exists():
        ts, img = load_nii(raw)
        lut = group_lookup(dict(class_map["total"]))
        groups = lut[np.clip(ts.astype(np.int64), 0, len(lut) - 1)]
        nib.save(nib.Nifti1Image(groups.astype(np.uint8), img.affine), str(grp))
    return grp


# ------------------------------------------------------------------ organ statistics
def organ_features(study_dir: Path, groups_path: Path) -> tuple[list[dict], dict]:
    """Organ-level SUV statistics and study QC for one study."""
    groups, img = load_nii(groups_path)
    groups = groups.astype(np.int16)
    suv = load_nii(study_dir / "SUV.nii.gz")[0].astype(np.float32)
    seg = load_nii(study_dir / "SEG.nii.gz")[0] > 0
    zooms = img.header.get_zooms()[:3]
    vml = float(np.prod(zooms)) / 1000.0
    st = ball(C.ERODE_MM, zooms)
    lesion_dil = ndi.binary_dilation(seg, st) if seg.any() else seg
    sup_ax, sup_sign = superior_axis(img.affine)
    sid = f"{study_dir.parent.name}__{study_dir.name}"
    rows, qc = [], {"study_id": sid, "lesion_ml": float(seg.sum() * vml)}
    in_any = np.zeros(groups.shape, bool)
    for gid, sl in enumerate(ndi.find_objects(groups), start=1):
        if sl is None or gid > len(C.GROUP_NAMES):
            continue
        name = C.GROUP_NAMES[gid - 1]
        pad = tuple(slice(max(s.start - 3, 0), min(s.stop + 3, n)) for s, n in zip(sl, groups.shape))
        m = groups[pad] == gid
        in_any[pad] |= m
        v = suv[pad]
        er = ndi.binary_erosion(m, st)
        clean = er & ~lesion_dil[pad]
        # organs cut by the scan range are flagged and later set to missing
        truncated = sl[sup_ax].start == 0 or sl[sup_ax].stop == groups.shape[sup_ax]
        rec = {"study_id": sid, "organ": name, "vol_ml": float(m.sum() * vml), "truncated": bool(truncated),
               "lesion_ml": float(seg[pad][m].sum() * vml)}
        rec.update(suv_stats(v[m], "full"))
        rec.update(suv_stats(v[er], "eroded"))
        rec.update(suv_stats(v[clean], "clean"))
        rows.append(rec)
        if name == "liver" and er.any():
            # respiratory misregistration index: upper 15 mm of the liver vs eroded liver
            idx = np.where(m.any(axis=tuple(a for a in range(3) if a != sup_ax)))[0]
            n_sl = max(1, int(round(C.DOME_MM / zooms[sup_ax])))
            dome_idx = idx[-n_sl:] if sup_sign > 0 else idx[:n_sl]
            sel = [slice(None)] * 3
            sel[sup_ax] = dome_idx
            dome = np.zeros_like(m)
            dome[tuple(sel)] = m[tuple(sel)]
            qc["liver_dome_index"] = float(v[dome].mean() / max(v[er].mean(), 1e-6))
    qc["lesion_outside_organs_ml"] = float((seg & ~in_any).sum() * vml)
    return rows, qc


def build_manifest(root: Path, meta_csv: Path) -> pd.DataFrame:
    """Study list (<root>/<patient>/<study>) joined with the TCIA clinical metadata."""
    rows = []
    for s in sorted(p for p in root.glob("*/*") if p.is_dir()):
        d = re.match(r"(\d{2})-(\d{2})-(\d{4})", s.name)
        rows.append({"study_id": f"{s.parent.name}__{s.name}", "patient_id": s.parent.name, "study_folder": s.name,
                     "study_date": f"{d.group(3)}-{d.group(1)}-{d.group(2)}" if d else "", "dir": str(s)})
    man = pd.DataFrame(rows)
    meta = pd.read_csv(meta_csv)

    def study_of(r):      # File Location: ./FDG-PET-CT-Lesions/<subject>/<study folder>/<series>
        parts = str(r["File Location"]).replace("\\", "/").split("/")
        subj = str(r["Subject ID"])
        return parts[parts.index(subj) + 1] if subj in parts and parts.index(subj) + 1 < len(parts) else ""

    meta["study_folder"] = meta.apply(study_of, axis=1)
    meta["patient_id"] = meta["Subject ID"]
    per = meta.groupby(["patient_id", "study_folder"]).agg(diagnosis=("diagnosis", "first"), age=("age", "first"),
                                                          sex=("sex", "first")).reset_index()
    man = man.merge(per, on=["patient_id", "study_folder"], how="left")
    man["age"] = pd.to_numeric(man["age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    man["sex"] = man["sex"].astype(str).str.upper().str[0].map({"M": 0, "F": 1})
    man["diagnosis"] = man["diagnosis"].astype(str).str.upper()
    man = man.sort_values(["patient_id", "study_date"])
    man["first_study"] = ~man.duplicated("patient_id")
    return man
