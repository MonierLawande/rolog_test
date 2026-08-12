"""Stage 0 -- load the four CSVs, resolve duplicates, protect split integrity.

Duplicate root cause (established by audit):

* ``reports_final.csv`` has exactly 25,494 rows == the documented scan count, so
  no rows were accidentally added. The de-identified ``study_id`` simply is not
  unique: three exams carry a colliding id because the same accession has two
  report versions -- a short preliminary ED read and the structured final
  report.
* Merging on that non-unique key multiplied rows downstream. ``AC4242d57``
  appears 64 times in ``five_years_disease_task.csv`` -- 2**6, one doubling per
  disease table joined.

Dedupe therefore keeps the LONGEST text (the final report). Using
``keep='first'`` picks the preliminary read by accident, which is what earlier
passes did: for ``AC4242d57`` that read says "no acute intraabdominal findings"
while the final report describes hepatic steatosis and a hepatic lesion.
"""
import html
import re

import pandas as pd

from . import config as C


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_reports() -> pd.DataFrame:
    """Report-level table -- 25,494 rows, nothing dropped. Audit trail."""
    rep = pd.read_csv(C.REPORTS)
    rep["report_idx"] = rep.index
    rep["Findings"] = rep["Findings"].fillna("")
    # PHI placeholders are inconsistently escaped: <DELETED> appears raw in
    # 3,956 reports and as &lt;DELETED&gt; in 4,436. Normalise both to one form.
    rep["text"] = rep["Findings"].map(html.unescape)
    rep["text_len"] = rep["text"].str.len()
    return rep


def dedupe_reports(rep: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse to one row per study_id. Returns (study_level, dropped_rows)."""
    grp = rep.groupby("study_id")["text"]
    rep = rep.assign(
        report_row_count=grp.transform("size"),
        report_text_variants=grp.transform("nunique"),
    )
    rep["duplicate_conflict"] = (rep["report_text_variants"] > 1).astype(int)
    # Genuinely ambiguous: the original zero-shot labels for these ids were
    # computed from one of the two texts and we cannot tell which.
    rep["zero_shot_source_ambiguous"] = rep["duplicate_conflict"]

    ordered = rep.sort_values(["study_id", "text_len"], ascending=[True, False])
    study = ordered.drop_duplicates("study_id", keep="first").copy()
    dropped = ordered[~ordered["report_idx"].isin(study["report_idx"])].copy()
    dropped["drop_reason"] = "duplicate_study_id__shorter_text"
    return study.reset_index(drop=True), dropped


def resolve_text_collisions(study: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Different study_ids sharing byte-identical report text.

    Four such groups exist; three straddle train/val, which leaks. Drop the
    evaluation-side copy so val/test stay clean and training data is preserved.
    """
    key = study["text"].map(_norm_ws)
    study = study.assign(_key=key)
    dup = study[key.duplicated(keep=False)]
    n_splits = dup.groupby("_key")[C.SPLIT_COL].transform("nunique")
    crossing = dup[n_splits > 1]

    study["text_collision_group"] = -1
    for gid, (_, grp) in enumerate(dup.groupby("_key")):
        study.loc[grp.index, "text_collision_group"] = gid

    to_drop = crossing[crossing[C.SPLIT_COL].isin(C.EVAL_SPLITS)]
    dropped = study.loc[to_drop.index].copy()
    dropped["drop_reason"] = "cross_split_text_collision"
    out = study.drop(index=to_drop.index).drop(columns=["_key"])
    return out.reset_index(drop=True), dropped.drop(columns=["_key"])


def load_zero_shot() -> pd.DataFrame:
    zs = pd.read_csv(C.ZERO_SHOT).drop_duplicates("study_id")
    return zs.rename(columns={c: f"orig_{c}" for c in zs.columns if c != "study_id"})


def load_metadata() -> pd.DataFrame:
    """2 byte-identical duplicate rows exist; drop them."""
    return pd.read_csv(C.METADATA).drop_duplicates("study_id", keep="first")


def load_five_years() -> pd.DataFrame:
    """63 duplicate rows of AC4242d57 (the 2**6 merge blow-up); labels agree."""
    five = pd.read_csv(C.FIVE_YEARS).drop_duplicates("study_id", keep="first")
    return five.rename(columns={c: f"fy_{c}" for c in five.columns if c != "study_id"})


def run() -> dict:
    rep = load_reports()
    study, dropped_dupes = dedupe_reports(rep)
    study, dropped_collisions = resolve_text_collisions(study)
    dropped = pd.concat([dropped_dupes, dropped_collisions], ignore_index=True)
    return {
        "reports_raw": rep,
        "study": study,
        "dropped": dropped,
        "zero_shot": load_zero_shot(),
        "metadata": load_metadata(),
        "five_years": load_five_years(),
    }
