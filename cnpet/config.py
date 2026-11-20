"""Organ definitions and analysis constants (Sections 2.2-2.5 of the paper)."""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ analysis constants
# Inputs of the models. 'eroded' (main analysis): median / mean SUV of the organ masks eroded by
# 4 mm, which needs no lesion labels. 'clean': the eroded masks minus the dilated manual lesions
# (needs lesion labels; used to show how little the label-free inputs lose).
INPUTS = "eroded"
TARGET = "full_max"             # scored feature: SUVmax on the full mask (lesion voxels included)
EXCLUDE = {"bladder", "prostate", "gallbladder"}   # urinary excretion, present in men only, often removed
MIN_PRESENT = 0.90              # organ kept if it has a usable input value in >= 90% of controls
MIN_VOX = 20                    # masks with fewer voxels are treated as missing
MIN_LESION_ML = 2.0             # lesion-bearing organ: >= 2 ml of manually segmented lesion (evaluation only)
MIN_TRAIN = 30                  # minimum training rows per organ and fold
ALPHAS = np.logspace(-3, 3, 13) # ridge penalty grid, selected by efficient leave-one-out CV (RidgeCV)
LOG_FLOOR = 0.05                # SUV floor before taking the log
ERODE_MM = 4.0                  # erosion of organ masks and dilation of lesions
DOME_MM = 15.0                  # liver-dome misregistration index
FLAG_Z = 2.5                    # two-pass procedure: predictor values with z above this are hidden
VAR_SHRINK = 0.5                # lambda: shrinkage of the log-variance prediction toward its mean
LOG_CHI2_1_MEAN = -1.2704       # E[log chi2_1], bias of log squared residuals

# ------------------------------------------------------------------ organ groups
# (group name, regex on TotalSegmentator 'total' class names); order defines group id 1..N
GROUP_RULES: list[tuple[str, str]] = [
    ("brain", r"^brain$"),
    ("thyroid", r"^thyroid_gland$"),
    ("spinal_cord", r"^spinal_cord$"),
    ("lung_left", r"^lung_.*_left$"),
    ("lung_right", r"^lung_.*_right$"),
    ("trachea", r"^trachea$"),
    ("esophagus", r"^esophagus$"),
    ("heart", r"^heart"),
    ("aorta", r"^aorta$"),
    ("pulmonary_vessels", r"^pulmonary_(artery|vein)$"),
    ("vena_cava", r"^(superior|inferior)_vena_cava$"),
    ("arteries_other", r"^(brachiocephalic_trunk|subclavian_artery_.*|common_carotid_artery_.*|iliac_artery_.*)$"),
    ("portal_splenic_vein", r"^portal_vein_and_splenic_vein$"),
    ("liver", r"^liver$"),
    ("spleen", r"^spleen$"),
    ("pancreas", r"^pancreas$"),
    ("stomach", r"^stomach$"),
    ("duodenum", r"^duodenum$"),
    ("small_bowel", r"^small_bowel$"),
    ("colon", r"^colon$"),
    ("gallbladder", r"^gallbladder$"),
    ("adrenals", r"^adrenal_gland_(left|right)$"),
    ("kidney_left", r"^kidney_left$"),
    ("kidney_right", r"^kidney_right$"),
    ("bladder", r"^urinary_bladder$"),
    ("prostate", r"^prostate$"),
    ("skull", r"^skull$"),
    ("vert_cervical", r"^vertebrae_C\d+$"),
    ("vert_thoracic", r"^vertebrae_T\d+$"),
    ("vert_lumbar", r"^vertebrae_L\d+$"),
    ("sacrum", r"^(sacrum|vertebrae_S1)$"),
    ("ribs", r"^rib_(left|right)_\d+$"),
    ("sternum", r"^(sternum|costal_cartilages)$"),
    ("scapulae", r"^scapula_(left|right)$"),
    ("clavicles", r"^clavicula_(left|right)$"),
    ("humeri", r"^humerus_(left|right)$"),
    ("pelvis", r"^hip_(left|right)$"),
    ("femurs", r"^femur_(left|right)$"),
    ("autochthon", r"^autochthon_(left|right)$"),
    ("iliopsoas", r"^iliopsoas_(left|right)$"),
    ("gluteal", r"^gluteus_(maximus|medius|minimus)_(left|right)$"),
]
GROUP_NAMES = [g for g, _ in GROUP_RULES]

# Tissue compartments. When an organ is the prediction target, all organs of its compartment
# are hidden from the predictors (compartment masking), so a diffuse process confined to one
# compartment (e.g. marrow activation) is not explained away. Unlisted organs (liver, spleen,
# heart, thyroid, ...) belong to no compartment.
SYSTEMS: dict[str, list[str]] = {
    "bone": ["clavicles", "femurs", "humeri", "pelvis", "ribs", "sacrum", "scapulae", "skull", "sternum",
             "vert_cervical", "vert_lumbar", "vert_thoracic"],
    "blood": ["aorta", "vena_cava", "arteries_other", "portal_splenic_vein", "pulmonary_vessels"],
    "muscle": ["autochthon", "gluteal", "iliopsoas"],
    "lung": ["lung_left", "lung_right", "trachea"],
    "gi": ["esophagus", "stomach", "duodenum", "small_bowel", "colon"],
    "kidney": ["kidney_left", "kidney_right"],
}
ORGAN_SYSTEM = {o: s for s, members in SYSTEMS.items() for o in members}
