"""Run the analysis on simulated organ tables (no imaging data needed).

Organ uptake is generated from a shared systemic factor, compartment factors and an
organ-specific term, so that the patient's other organs carry information about the expected
uptake of each organ. Lesions raise SUVmax in some organs of patients. The demo writes tables in
the same format as scripts/extract_features.py and runs scripts/run_analysis.py on them.

  python examples/synthetic_demo.py            # writes and analyzes examples/synthetic/
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ORGANS = {   # organ: (compartment factor, loading on the systemic factor)
    "liver": (None, 0.6), "spleen": (None, 0.5), "heart": (None, 0.3), "thyroid": (None, 0.2),
    "aorta": ("blood", 0.7), "vena_cava": ("blood", 0.7), "pulmonary_vessels": ("blood", 0.6),
    "lung_left": ("lung", 0.4), "lung_right": ("lung", 0.4), "kidney_left": ("kidney", 0.3),
    "kidney_right": ("kidney", 0.3), "vert_thoracic": ("bone", 0.3), "vert_lumbar": ("bone", 0.3),
    "pelvis": ("bone", 0.3), "ribs": ("bone", 0.3), "autochthon": ("muscle", 0.4), "gluteal": ("muscle", 0.4),
    "stomach": ("gi", 0.2), "colon": ("gi", 0.2),
}


def simulate(n_patients=800, frac_controls=0.45, seed=1):
    rng = np.random.default_rng(seed)
    age = rng.uniform(25, 85, n_patients)
    sex = rng.integers(0, 2, n_patients)
    control = rng.random(n_patients) < frac_controls
    systemic = rng.normal(0, 0.25, n_patients)
    comp = {c: rng.normal(0, 0.15, n_patients) for c in {v[0] for v in ORGANS.values() if v[0]}}
    man = pd.DataFrame({"study_id": [f"P{i:04d}__S1" for i in range(n_patients)],
                        "patient_id": [f"P{i:04d}" for i in range(n_patients)],
                        "diagnosis": np.where(control, "NEGATIVE", "LYMPHOMA"),
                        "age": age.round(), "sex": sex, "first_study": True})
    rows = []
    for o, (c, load) in ORGANS.items():
        base = np.log(2.0) + 0.002 * (age - 55) + 0.03 * sex
        clean = base + load * systemic + (comp[c] if c else 0) + rng.normal(0, 0.08, n_patients)
        full_max = clean + np.log(1.6) + rng.normal(0, 0.10, n_patients)
        lesion = np.where(~control & (rng.random(n_patients) < 0.12), rng.gamma(2, 3, n_patients), 0.0)
        full_max = full_max + np.where(lesion >= 2, rng.gamma(2, 0.25, n_patients), 0)
        for i in range(n_patients):
            rows.append({"study_id": man.study_id[i], "organ": o, "truncated": False, "lesion_ml": lesion[i],
                         "full_n": 500, "clean_n": 300, "full_max": np.exp(full_max[i]),
                         "full_p90": np.exp(clean[i] + 0.2), "clean_median": np.exp(clean[i]),
                         "clean_mean": np.exp(clean[i] + 0.02)})
    return man, pd.DataFrame(rows)


if __name__ == "__main__":
    out = HERE / "synthetic"
    out.mkdir(exist_ok=True)
    man, stats = simulate()
    man.to_csv(out / "manifest.csv", index=False)
    stats.to_csv(out / "organ_stats.csv", index=False)
    subprocess.run([sys.executable, str(HERE.parent / "scripts" / "run_analysis.py"), "--data", str(out),
                    "--methods", "raw,liver_ref,marg_lin,cond_lin,cond_lin_het", "--n_boot", "200",
                    "--n_rep", "3"], check=True)
