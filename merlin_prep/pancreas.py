"""Stage 3 -- collapse the pancreas multi-label into a 4-class status.

Why four classes and not the earlier 2x2 (pancreas x other organs): the
"other organs" axis is degenerate in this cohort. These are emergency-department
abdominal CTs, and 87.7% of studies carry at least one positive finding in some
other organ (mean 2.96 findings per study). A 2x2 therefore produces a cell with
10 studies -- untrainable and unevaluable. Dropping that axis leaves four
pancreas classes whose smallest member is ~1,000 studies.

POSTOPERATIVE is its own class rather than a flavour of ABNORMAL: a post-Whipple
pancreas is altered anatomy, not disease, and folding it into ABNORMAL teaches
the model that resection artefacts are pathology.
"""
import pandas as pd

from . import assertion as A
from . import config as C

NORMAL, ABNORMAL, POSTOP, REVIEW = C.PANCREAS_STATUS


def classify_row(row) -> tuple[str, str, str]:
    """Returns (status, reason, deciding_concept). First matching rule wins."""
    if row["panc_postop_resection"] == A.PRESENT:
        return POSTOP, "postop_resection", "panc_postop_resection"

    if not row["pancreas_context"]:
        return REVIEW, "no_pancreas_mention", ""

    if row["panc_not_evaluated"] == A.PRESENT:
        return REVIEW, "not_evaluated", "panc_not_evaluated"

    present = [k for k in C.PANCREAS_ABNORMAL if row[k] == A.PRESENT]
    if present:
        return ABNORMAL, "abnormal:" + ",".join(present[:4]), present[0]

    if row["panc_explicit_normal"] == A.PRESENT:
        return NORMAL, "explicit_normal", "panc_explicit_normal"
    if row["panc_no_abnormality"] == A.PRESENT:
        return NORMAL, "explicit_normal", "panc_no_abnormality"

    uncertain = [k for k in C.PANCREAS_ABNORMAL if row[k] == A.UNCERTAIN]
    if uncertain:
        return REVIEW, "hedged_only:" + ",".join(uncertain[:4]), uncertain[0]

    # Everything denied or unmentioned, with a real pancreas context present:
    # radiologists routinely write "Pancreas: No mass. No ductal dilation."
    if row["has_pancreas_section"]:
        return NORMAL, "all_denied_in_section", ""

    return REVIEW, "insufficient_evidence", ""


def run(df: pd.DataFrame) -> pd.DataFrame:
    res = df.apply(classify_row, axis=1, result_type="expand")
    df = df.copy()
    df["pancreas_status"] = res[0]
    df["pancreas_status_reason"] = res[1]
    df["pancreas_deciding_concept"] = res[2]
    df["pancreas_any_abnormality"] = (
        df[C.PANCREAS_ABNORMAL].eq(A.PRESENT).any(axis=1).astype(int)
    )
    # Evidence for the rule that actually decided the class -- not merely the
    # first non-empty span, which would show an unrelated finding.
    df["pancreas_evidence"] = [
        (r[f"ev_{k}"] if k and isinstance(r.get(f"ev_{k}"), str) else "")
        for k, (_, r) in zip(res[2], df.iterrows())
    ]
    return df
