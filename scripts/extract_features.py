"""Step 1: organ masks and organ-level SUV statistics for the autoPET dataset.

Usage
  python scripts/extract_features.py --root /data/FDG-PET-CT-Lesions \
      --meta /data/Clinical-Metadata-FDG-PET_CT-Lesions.csv --out results/ [--device gpu]

Writes results/manifest.csv, results/organ_stats.csv, results/study_qc.csv and the
TotalSegmentator outputs in results/totalseg/. Segmentation is the slow part (GPU recommended);
finished studies are skipped when the script is run again.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnpet.features import build_manifest, organ_features, segment_study  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="NIfTI root: <root>/<patient>/<study>/")
    ap.add_argument("--meta", type=Path, required=True, help="TCIA clinical metadata CSV")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="gpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    man = build_manifest(a.root, a.meta)
    man.to_csv(a.out / "manifest.csv", index=False)
    print(f"{len(man)} studies, {man.patient_id.nunique()} patients")

    rows, qcs = [], []
    for i, r in enumerate(man.itertuples(), 1):
        grp = segment_study(Path(r.dir), a.out / "totalseg", device=a.device)
        org, qc = organ_features(Path(r.dir), grp)
        rows += org
        qcs.append(qc)
        print(f"[{i}/{len(man)}] {r.study_id}", flush=True)
    pd.DataFrame(rows).to_csv(a.out / "organ_stats.csv", index=False)
    q = pd.DataFrame(qcs)
    q.to_csv(a.out / "study_qc.csv", index=False)
    x = q.liver_dome_index.dropna()
    print(f"liver-dome index < 0.7 (respiratory misregistration): {(x < 0.7).sum()} studies ({(x < 0.7).mean():.1%})")


if __name__ == "__main__":
    main()
