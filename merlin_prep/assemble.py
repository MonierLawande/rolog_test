"""Stage 4/5 -- hybrid label merge, metadata join, QC flags."""
import numpy as np
import pandas as pd

from . import assertion as A
from . import config as C


def merge_findings(df: pd.DataFrame, zs: pd.DataFrame) -> pd.DataFrame:
    """Hybrid rule (user decision): the original 0/1 wins, -1 cells are filled
    from text extraction.

    This preserves comparability with the Merlin paper but inherits the known
    errors in the original 0s (1,381 in surgically_absent_gallbladder, 342 cells
    where label=1 while the report explicitly denies the finding). Every such
    cell is flagged in ``<finding>_conflict`` and can be flipped wholesale with
    ``config.OVERRIDE_CONFLICTS``.
    """
    df = df.merge(zs, on="study_id", how="left")
    df["has_zero_shot"] = df["orig_pancreatic_atrophy"].notna().astype(int)

    for f in C.OTHER_FINDINGS:
        orig = df.get(f"orig_{f}")
        if orig is None:  # finding absent from the original file
            orig = pd.Series(np.nan, index=df.index)
        state = df[f"ext_{f}"]
        ext_bin = state.map(lambda s: A.to_binary(s, C.UNCERTAIN_AS_POSITIVE))

        labelled = orig.isin([0, 1])
        conflict = (
            (labelled & orig.eq(0) & state.eq(A.PRESENT))
            | (labelled & orig.eq(1) & state.eq(A.ABSENT))
        )

        final = np.where(labelled, orig, ext_bin)
        if C.OVERRIDE_CONFLICTS:
            final = np.where(conflict, ext_bin, final)

        df[f] = pd.Series(final, index=df.index).astype("int8")
        df[f"{f}_source"] = np.where(labelled, "original", "extracted")
        df[f"{f}_assertion"] = state
        df[f"{f}_conflict"] = conflict.astype("int8")

    return df


def join_metadata(df: pd.DataFrame, meta: pd.DataFrame, five: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(meta, on="study_id", how="left")
    df["has_metadata"] = df["Age"].notna().astype(int)
    df = df.merge(five, on="study_id", how="left")
    df["has_five_years"] = df["fy_cvd"].notna().astype(int)
    return df


def qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Advisory flags, not filters -- the cohort decision stays with the user."""
    df["slice_ok"] = (df["slicethickness"] <= 2.5).fillna(False).astype("int8")
    df["phase_portal"] = df["phase"].eq("portal_venous").astype("int8")
    # 1,519 of the 1,536 non-contrast scans are still tagged portal_venous, so
    # one of the two fields is unreliable. Flag, do not silently pick a winner.
    df["metadata_contradiction"] = (
        df["contrast"].eq(False) & df["phase"].eq("portal_venous")
    ).astype("int8")
    df["age_implausible"] = (
        df["Age"].lt(18) | df["Age"].gt(100)
    ).fillna(False).astype("int8")
    return df


CORE_COLS = [
    # class_5 is the primary training target; the axis columns follow it so the
    # label can be audited without leaving the first screen of the file.
    "study_id", "Split", "Few_Shot",
    "class_5", "class_5_reason",
    "other_organs_status", "n_other_present", "n_other_uncertain",
    "pancreas_status", "pancreas_status_reason", "pancreas_deciding_concept",
    "pancreas_any_abnormality",
    "pancreas_evidence", "pancreas_context_source", "has_pancreas_section",
    "duplicate_conflict", "zero_shot_source_ambiguous", "text_collision_group",
    "has_metadata", "has_zero_shot", "has_five_years",
    "Age", "Gender", "Race", "contrast", "manufacturer", "manufacturermodelname",
    "kvp", "slicethickness", "xraytubecurrent", "phase",
    "slice_ok", "phase_portal", "metadata_contradiction", "age_implausible",
]


def final_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in CORE_COLS if c in df.columns]
    cols += C.PANCREAS_LABELS
    for f in C.OTHER_FINDINGS:
        cols += [f, f"{f}_source", f"{f}_assertion", f"{f}_conflict"]
    cols += ["orig_pancreatic_atrophy"]
    cols += [c for c in df.columns if c.startswith("fy_")]
    return [c for c in cols if c in df.columns]
