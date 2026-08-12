"""Paths, switches and the label registry for the Merlin pancreas pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

REPORTS = ROOT / "reports_final.csv"
ZERO_SHOT = ROOT / "zero_shot_findings_disease_cls.csv"
METADATA = ROOT / "metadata.csv"
FIVE_YEARS = ROOT / "five_years_disease_task.csv"

# ---------------------------------------------------------------- switches --
# The 29-finding labels are HYBRID by user decision: the original 0/1 wins,
# and only -1 cells are filled from text extraction.  Flip this to True to let
# the text extractor override original cells it strongly disagrees with.
OVERRIDE_CONFLICTS = False

# Hedged mentions ("possible pancreatitis") flatten to 0 in the binary column.
# The full four-state assertion is always kept in <finding>_assertion.
UNCERTAIN_AS_POSITIVE = False

# Impression text is used as supporting evidence for pancreas labels. It holds
# the radiologist's conclusion, so it improves recall -- but it is also a
# leakage source for any downstream report-generation task.
USE_IMPRESSION_FOR_PANCREAS = True

# ------------------------------------------------------------------ splits --
SPLIT_COL = "Split"
EVAL_SPLITS = ("val", "test")  # side to drop when a text collision crosses splits

# ------------------------------------------------- known duplicate study_ids --
# Root cause: two report versions of one exam (preliminary ED read + final
# report). Dedupe keeps the LONGEST text = the structured final report.
KNOWN_DUPLICATE_IDS = ("AC4214d72", "AC423c4f0", "AC4242d57")

# ------------------------------------------------------------ label registry --
# The 29 non-pancreas findings. pancreatic_atrophy is deliberately excluded --
# it moves to the pancreas head.
OTHER_FINDINGS = [
    "submucosal_edema", "renal_hypodensities", "aortic_valve_calcification",
    "coronary_calcification", "thrombosis", "metastatic_disease", "renal_cyst",
    "osteopenia", "surgically_absent_gallbladder", "atelectasis",
    "abdominal_aortic_aneurysm", "anasarca", "hiatal_hernia", "lymphadenopathy",
    "prostatomegaly", "biliary_ductal_dilation", "cardiomegaly", "splenomegaly",
    "hepatomegaly", "atherosclerosis", "ascites", "pleural_effusion",
    "hepatic_steatosis", "appendicitis", "gallstones", "hydronephrosis",
    "bowel_obstruction", "free_air", "fracture",
]

PANCREAS_LABELS = [
    "panc_atrophy", "panc_fatty_replacement", "panc_pancreatitis",
    "panc_pancreatitis_acute", "panc_pancreatitis_chronic", "panc_necrosis",
    "panc_pseudocyst", "panc_cystic_lesion", "panc_ipmn", "panc_solid_mass",
    "panc_malignancy", "panc_duct_dilation", "panc_calcification",
    "panc_peripancreatic_inflam", "panc_divisum_annular", "panc_stent",
    "panc_transplant", "panc_postop_resection", "panc_explicit_normal",
    "panc_no_abnormality", "panc_not_evaluated",
]

# Labels that make the pancreas ABNORMAL. panc_postop_resection is NOT here --
# a post-Whipple pancreas is altered anatomy, not disease, and gets its own class.
PANCREAS_ABNORMAL = [
    "panc_atrophy", "panc_fatty_replacement", "panc_pancreatitis",
    "panc_pancreatitis_acute", "panc_pancreatitis_chronic", "panc_necrosis",
    "panc_pseudocyst", "panc_cystic_lesion", "panc_ipmn", "panc_solid_mass",
    "panc_malignancy", "panc_duct_dilation", "panc_calcification",
    "panc_peripancreatic_inflam", "panc_divisum_annular", "panc_stent",
    "panc_transplant",
]

PANCREAS_STATUS = ("NORMAL", "ABNORMAL", "POSTOPERATIVE", "REVIEW_REQUIRED")

# --------------------------------------------------- 5-class primary target --
# Names are taken verbatim from Merlin_Analysis.ipynb (cell 74) so downstream
# code that already expects these strings keeps working.
CLASS_5 = [
    "REVIEW_REQUIRED", "PANCREAS_AND_OTHER", "PANCREAS_ONLY",
    "OTHER_ONLY", "ALL_NORMAL",
]

# How many positive findings make the "other organs" axis ABNORMAL.
OTHER_DISEASE_MIN_FINDINGS = 1

# When True, the incidental/degenerative findings below stop counting towards
# the "other organs" axis. They are near-universal in an elderly ED cohort, so
# including them collapses the axis: PANCREAS_ONLY is 113 studies with them and
# 1,016 without.
OTHER_DISEASE_EXCLUDE_INCIDENTAL = False

INCIDENTAL_FINDINGS = [
    "atherosclerosis", "osteopenia", "renal_cyst", "hepatic_steatosis",
    "hiatal_hernia", "prostatomegaly", "coronary_calcification",
    "aortic_valve_calcification", "surgically_absent_gallbladder",
    "atelectasis", "anasarca", "renal_hypodensities",
]

# A post-Whipple pancreas counts as "pancreas abnormal" for the 5-class target.
# pancreas_status keeps POSTOPERATIVE as its own value regardless.
POSTOP_IS_PANCREAS_ABNORMAL = True


def other_finding_cols() -> list[str]:
    """The findings that feed the 'other organs' axis, per the switches above."""
    if OTHER_DISEASE_EXCLUDE_INCIDENTAL:
        return [f for f in OTHER_FINDINGS if f not in INCIDENTAL_FINDINGS]
    return list(OTHER_FINDINGS)

# ------------------------------------------------------- regression guards --
# Headline numbers established during the data audit. validate.py recomputes
# them; a large drift means an upstream assumption broke.
GUARDS = {
    "report_rows": 25494,
    "study_rows": 25486,
    "orig_zs_rows": 25275,
    "metadata_rows_dedup": 25412,
    "five_years_rows_dedup": 12353,
    "reports_without_metadata": 77,
    "reports_without_zero_shot": 214,
    "cross_split_text_collisions": 3,
}
