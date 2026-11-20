# Conditional normative modeling of whole-body FDG PET/CT

Code for the paper

> Park C-h, Park H-J. *Conditional normative modeling of whole-body FDG PET/CT: inter-organ relationships outperform reference-region normalization for label-free detection of lesion-bearing organs.* medRxiv (2026). doi: (to be added)

Clinical reading criteria judge an organ's FDG uptake against a single reference region, such as the liver or the blood pool. This repository implements a normative model that predicts each organ's uptake from the patient's age, sex and the uptake of the patient's other organs. The model is trained on control patients, and each organ is scored as a calibrated deviation from that expectation. Liver and blood-pool referencing are special cases: a regression on one organ with the slope fixed at one.

The scores are label-free. Predictors and reference regions come from organ masks eroded by 4 mm, so no lesion segmentation is needed to compute them. The manual lesion labels serve only as the reference standard for evaluation.

This is a concise implementation of the core analysis. The modeling and scoring functions give the same deviation scores as the full analysis pipeline used for the paper. Scripts that produce the paper's figures and tables are not included.

## What the code does

| Step | Paper | Module |
| --- | --- | --- |
| Organ masks: TotalSegmentator on CT resampled to the PET grid, merged into organ groups | 2.2 | `cnpet/features.py` |
| Organ SUV statistics on full, eroded (4 mm) and lesion-cleaned masks; truncation and liver-dome QC | 2.2 | `cnpet/features.py` |
| Cohort matrices: first study per patient, organ selection, reference regions; eroded (label-free, default) or lesion-cleaned inputs | 2.1, 2.2, 2.4 | `cnpet/data.py` |
| Inter-organ relationships: partial correlations; out-of-fold R² of predictor sets M0, M1, M2c, M2 | 2.3 | `cnpet/relations.py` |
| Deviation scores: SUVmax, liver- and blood-pool-referenced, marginal, global uptake (slope fixed at one or learned), conditional (ridge regression or gradient boosting), with or without the aleatoric variance model; compartment masking; out-of-fold scaling; label-free two-pass procedure | 2.4, 2.6 | `cnpet/model.py` |
| Detection (AUC, FPR at 90% sensitivity, patient-level bootstrap CI of ΔAUC), per-organ AUC, false-positive organs per patient (cross-fitted thresholds), calibration overall and by predicted SD, upper-tail and empirical calibration, repeat-scan metrics | 2.5, 2.9 | `cnpet/evaluate.py` |
| Finer organ units: unit maps (mid, fine), compartment and sibling masking, Šidák-corrected parent-organ scores | 2.11 | `cnpet/granularity.py` |

Command-line options of `scripts/run_analysis.py` cover the sensitivity analyses of Section 2.6: lesion threshold, compartment masking, training population, target, lesion-cleaned inputs (`--inputs clean`) and the label-free two-pass procedure (`--two_pass`). The other secondary analyses use the same scoring function with different inputs:
- residual soft-tissue regions are extra rows in the organ table, scored as targets (drop them from the predictors so that, for example, mediastinal nodes do not contaminate the predictors of other organs);
- CT covariates are `extra_covariates` in `load_cohort`;
- repeat scans use `followups=True` and `evaluate.repeat_stability`;
- training-set size varies the training controls passed to `score_organ`.

The analysis code used for the paper (including cancer-type, subgroup and hotspot analyses and the scripts that draw the figures and tables) is a larger pipeline; this repository keeps the concepts and the scoring functions that give the same deviation scores.

## Installation

```bash
git clone https://github.com/hauin/conditional-normative-pet.git
cd conditional-normative-pet
pip install -r requirements.txt
```

Python 3.10 or later. TotalSegmentator and a GPU are needed only for step 1.

## Data

The autoPET FDG PET/CT dataset (FDG-PET-CT-Lesions) is available from The Cancer Imaging Archive, DOI [10.7937/gkr0-xv29](https://doi.org/10.7937/gkr0-xv29), under the TCIA data usage policy. The code expects the NIfTI version of the dataset, with one folder per study (`<root>/<patient>/<study>/`) containing:
- `SUV.nii.gz`;
- `CTres.nii.gz` (CT resampled to the PET grid);
- `SEG.nii.gz` (manual lesion segmentation released with the dataset).

The NIfTI files are distributed by the autoPET challenge organizers, or can be converted from the TCIA DICOM files with their conversion scripts. The clinical metadata CSV from TCIA provides diagnosis, age and sex.

No data are included in this repository.

## Usage

```bash
# 1. organ masks and organ statistics (TotalSegmentator; a few minutes per study on a GPU, resumable)
python scripts/extract_features.py --root /data/FDG-PET-CT-Lesions \
    --meta /data/Clinical-Metadata-FDG-PET_CT-Lesions.csv --out results/

# 2. relationships, deviation scores, detection and calibration
python scripts/run_analysis.py --data results/ --n_jobs 8

# sensitivity analyses (Section 2.6)
python scripts/run_analysis.py --data results/ --out results/clean --inputs clean --skip_relations
python scripts/run_analysis.py --data results/ --out results/twopass --two_pass --skip_relations
python scripts/run_analysis.py --data results/ --out results/les5 --min_lesion_ml 5 --skip_relations
python scripts/run_analysis.py --data results/ --out results/nomask --no_compartment_mask --skip_relations
python scripts/run_analysis.py --data results/ --out results/trainpat --train patients --skip_relations
python scripts/run_analysis.py --data results/ --out results/p90 --target full_p90 --skip_relations
```

Gradient-boosting methods (`cond_gbm`, `cond_het`) take most of the run time. Omit them with `--methods raw,liver_ref,blood_ref,marg_lin,glob_ref,glob_lin,cond_lin,cond_lin_het` for a quick run.

### Try it without imaging data

```bash
python examples/synthetic_demo.py
```

The demo simulates organ tables with inter-organ structure and lesions, then runs step 2 on them. It finishes in under a minute.

## Outputs of step 2

| File | Content |
| --- | --- |
| `relations_r2.csv` | out-of-fold R² per organ for M0, M1, M2c and M2 |
| `deviation_scores.csv` | out-of-fold z per study, organ and method (with the predicted aleatoric SD), plus empirically calibrated z (`*_emp`) |
| `detection.csv` | pooled AUC and FPR at 90% sensitivity, against control organs and within patients |
| `detection_diff.csv` | ΔAUC from liver referencing with patient-level bootstrap 95% CI |
| `per_organ_auc.csv` | per-organ AUC (organs with at least 10 positives and 10 negatives) |
| `fp_per_patient.csv` | lesion-free organs flagged per patient at 50, 70, 80 and 90% sensitivity (thresholds cross-fitted over patient folds) |
| `fp_per_patient_diff.csv` | conditional minus liver-referenced false positives per patient, with bootstrap CI and the share of patients with fewer, the same or more |
| `calibration.csv` | share of control organs with \|z\| > 1.96, overall and by quintile of the predicted aleatoric SD |
| `calibration_tails.csv` | one-sided exceedance in control organs at nominal 5, 2.5, 1 and 0.5% |

Two operating points are used. The pooled false-positive rate is read at 90% sensitivity, because an aid that flags organs for review should miss few lesion-bearing organs. The number of falsely flagged organs per patient is summarized at 80% sensitivity, a point where a reader can review the flags patient by patient.

## Key settings

Defaults are in `cnpet/config.py`:
- organs with a usable input value in fewer than 90% of controls are excluded;
- the target is the log SUVmax on the full mask, and the predictors are the log median SUV of the other organs on the eroded masks (`INPUTS = "eroded"`; `"clean"` uses the lesion-cleaned masks instead);
- ridge regression is the primary model: the other organs are many, strongly correlated predictors, and the penalty stabilizes their weights while keeping the coefficients interpretable as a generalization of reference-region normalization;
- ridge penalties are chosen from 13 log-spaced values between 10⁻³ and 10³ by efficient leave-one-out cross-validation (`RidgeCV`), with standardized predictors;
- gradient boosting uses 300 iterations, learning rate 0.05, 15 leaves, at least 20 samples per leaf and L2 penalty 1;
- the variance model's log-variance prediction is shrunk halfway (λ = 0.5) toward its mean;
- the two-pass procedure hides predictor values whose first-pass z exceeds 2.5;
- all scores are out of fold over 5 patient-level folds, with an inner 5-fold split for scaling and the variance model.

## Citation

If you use this code, please cite the paper above (see `CITATION.cff`).

## License

MIT License (see `LICENSE`). The autoPET data are subject to the TCIA data usage policy.
