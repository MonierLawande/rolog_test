"""Build a 300-study manual validation sheet, weighted towards where the rules fail.

Random sampling would spend most of its budget on "Pancreas: Normal." templates,
which the rules already get right. Every slice here targets a known or suspected
failure mode instead, so 300 annotations buy far more information.

    python -m merlin_prep.make_validation_sample
    # annotate output/manual_validation_sample.csv, then:
    python -m merlin_prep.score_validation
"""
import pandas as pd

from . import assertion as A
from . import config as C
from . import lexicon as L

SEED = 20260811
GENERIC = ["panc_solid_mass", "panc_cystic_lesion", "panc_stent", "panc_calcification"]


def _take(pool: pd.DataFrame, n: int, used: set, slice_name: str) -> pd.DataFrame:
    pool = pool[~pool.study_id.isin(used)]
    if pool.empty:
        return pool.assign(slice=slice_name)
    out = pool.sample(min(n, len(pool)), random_state=SEED).copy()
    out["slice"] = slice_name
    used |= set(out.study_id)
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    used: set = set()
    parts = []

    generic_pos = df[df[GENERIC].eq(A.PRESENT).any(axis=1)]
    parts.append(_take(generic_pos, 90, used, "generic_concept_present"))

    parts.append(_take(df[df.pancreas_context_source.eq("freetext")], 50, used,
                       "freetext_context"))
    parts.append(_take(df[df.class_5.eq("PANCREAS_ONLY")], 40, used, "class_PANCREAS_ONLY"))

    conflict_cols = [c for c in df.columns if c.endswith("_conflict")]
    parts.append(_take(df[df[conflict_cols].eq(1).any(axis=1)], 40, used,
                       "conflicts_with_original"))

    hedged = df[df.pancreas_status_reason.str.startswith("hedged_only")
                | df[C.PANCREAS_ABNORMAL].eq(A.UNCERTAIN).any(axis=1)]
    parts.append(_take(hedged, 30, used, "hedged_boundary"))

    parts.append(_take(df[df.pancreas_status.eq("POSTOPERATIVE")], 25, used, "postoperative"))
    parts.append(_take(df[df.pancreas_status.eq("NORMAL")], 25, used, "random_normal"))

    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)

    keep = ["study_id", "slice", "Split", "class_5", "pancreas_status",
            "pancreas_status_reason", "pancreas_deciding_concept",
            "pancreas_evidence", "pancreas_context_source", "pancreas_context",
            "other_organs_status", "n_other_present"]
    out = out[[c for c in keep if c in out.columns]].copy()

    # Blank columns for the annotator.
    out["human_pancreas_status"] = ""      # NORMAL / ABNORMAL / POSTOPERATIVE / REVIEW_REQUIRED
    out["human_class_5"] = ""              # one of C.CLASS_5
    out["human_notes"] = ""
    return out


def main() -> None:
    df = pd.read_csv(C.OUT / "merlin_pancreas_dataset.csv")
    ctx_path = C.OUT / "pancreas_context.csv"
    if ctx_path.exists() and "pancreas_context" not in df.columns:
        # Only the text column -- pancreas_context_source already lives in df.
        ctx = pd.read_csv(ctx_path)[["study_id", "pancreas_context"]]
        df = df.merge(ctx, on="study_id", how="left")
    out = build(df)
    path = C.OUT / "manual_validation_sample.csv"
    out.to_csv(path, index=False)
    print(f"wrote {len(out)} rows -> {path}")
    print(out.slice.value_counts().to_string())


if __name__ == "__main__":
    main()
