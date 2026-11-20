"""Step 2: inter-organ relationships, deviation scores, detection, false positives per patient
and calibration.

Usage
  python scripts/run_analysis.py --data results/ [--n_jobs 8] [--min_lesion_ml 2]
                                 [--inputs eroded|clean] [--two_pass]
                                 [--methods raw,liver_ref,blood_ref,marg_lin,glob_ref,glob_lin,cond_lin,cond_lin_het]
                                 [--no_compartment_mask] [--train controls|patients] [--target full_max]

Reads manifest.csv and organ_stats.csv (from extract_features.py, or any tables in the same
format). The default inputs ('eroded') need no lesion labels; lesion labels define only the
positives and negatives of the evaluation. Writes
  relations_r2.csv        out-of-fold R2 of M0, M1, M2c, M2 in controls
  deviation_scores.csv    out-of-fold z per study, organ and method (+ empirically calibrated z)
  detection.csv, detection_diff.csv   pooled AUC, FPR at 90% sensitivity, dAUC vs liver referencing
  per_organ_auc.csv
  fp_per_patient.csv, fp_per_patient_diff.csv   lesion-free organs flagged per patient (50-90% sensitivity)
  calibration.csv, calibration_tails.csv        |z| > 1.96 by predicted SD; one-sided upper tail
Options reproduce the sensitivity analyses of Section 2.6 (lesion threshold, compartment masking,
training population, target, lesion-cleaned inputs, two-pass procedure).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnpet import evaluate as E  # noqa: E402
from cnpet.data import load_cohort  # noqa: E402
from cnpet.model import oof_scores, two_pass  # noqa: E402
from cnpet.relations import predictor_sets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--inputs", choices=["eroded", "clean"], default="eroded",
                    help="eroded: label-free (main); clean: lesion-cleaned, needs lesion labels")
    ap.add_argument("--two_pass", action="store_true", help="hide predictor values flagged by a first pass")
    ap.add_argument("--min_lesion_ml", type=float, default=2.0)
    ap.add_argument("--methods", default="raw,liver_ref,blood_ref,marg_lin,glob_ref,glob_lin,cond_lin,cond_lin_het,"
                                         "cond_gbm,cond_het")
    ap.add_argument("--no_compartment_mask", action="store_true")
    ap.add_argument("--train", choices=["controls", "patients"], default="controls")
    ap.add_argument("--target", default="full_max", help="e.g. full_p90 for the 90th-percentile target")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_rep", type=int, default=20, help="repetitions of 5-fold CV for R2")
    ap.add_argument("--skip_relations", action="store_true")
    a = ap.parse_args()
    out = a.out or a.data
    out.mkdir(parents=True, exist_ok=True)
    methods = a.methods.split(",")

    d = load_cohort(pd.read_csv(a.data / "manifest.csv"), pd.read_csv(a.data / "organ_stats.csv"),
                    target=a.target, inputs=a.inputs)
    n_ctrl = int(d.is_control.loc[d.first].sum())
    print(f"cohort: {len(d.first)} patients ({n_ctrl} controls), {len(d.organs)} organs; inputs {a.inputs}")

    if not a.skip_relations:
        r = predictor_sets(d, n_rep=a.n_rep)
        r.round(4).to_csv(out / "relations_r2.csv", index=False)
        print("median R2:", r[["M0", "M1", "M2c", "M2"]].median().round(3).to_dict())
    if a.two_pass:
        d, zp = two_pass(d, n_jobs=a.n_jobs)
        print(f"two-pass: {100 * (zp > 2.5).values.mean():.1f}% of predictor values hidden")

    Z = oof_scores(d, methods, n_jobs=a.n_jobs, train=a.train, compartment_mask=not a.no_compartment_mask)
    fold = E.fold_of(d)
    emp = [m for m in ("liver_ref", "cond_lin") if m in methods]
    Z = pd.concat([Z, E.empirical_calibration(Z[Z.method.isin(emp)], fold)], ignore_index=True)
    Z.to_csv(out / "deviation_scores.csv", index=False)
    methods_all = methods + [m + "_emp" for m in emp]

    A, Dd = E.detection(Z, methods_all, min_ml=a.min_lesion_ml, n_boot=a.n_boot)
    A.round(4).to_csv(out / "detection.csv", index=False)
    Dd.round(4).to_csv(out / "detection_diff.csv", index=False)
    E.per_organ_auc(Z, methods, min_ml=a.min_lesion_ml).round(4).to_csv(out / "per_organ_auc.csv", index=False)
    fp_methods = [m for m in ("raw", "liver_ref", "glob_lin", "cond_lin") if m in methods]
    F, per_pt = E.fp_per_patient(Z, fp_methods, fold, min_ml=a.min_lesion_ml)
    F.round(3).to_csv(out / "fp_per_patient.csv", index=False)
    Fd = E.paired_fp(per_pt, "liver_ref", "cond_lin", a.n_boot) if {"liver_ref", "cond_lin"} <= set(methods) else pd.DataFrame()
    Fd.round(3).to_csv(out / "fp_per_patient_diff.csv", index=False)
    cal = E.calibration(Z, methods)
    cal.round(4).to_csv(out / "calibration.csv")
    T = E.tail_calibration(Z, methods_all)
    T.round(3).to_csv(out / "calibration_tails.csv", index=False)

    print(A.pivot(index="method", columns="comparison", values=["AUC", "FPR_at_90sens"]).round(3).to_string())
    print(Dd.round(4).to_string(index=False))
    print(F.pivot(index="method", columns="target_sens", values="FP_organs_per_patient").round(2).to_string())
    print(Fd.round(3).to_string(index=False))
    print(cal.round(3).to_string())
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
