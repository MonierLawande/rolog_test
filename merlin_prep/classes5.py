"""Stage 5b -- the 5-class primary target.

Class names match ``Merlin_Analysis.ipynb`` (cell 74) exactly. The *axes* do not,
and deliberately so. The notebook drives the pancreas axis off
``pancreatic_atrophy`` alone and maps its ``-1`` straight to REVIEW_REQUIRED,
which has two consequences measured on this data:

* A pancreas with a mass, pancreatitis, IPMN or ductal dilation but no atrophy
  reads as a normal pancreas.
* ``-1`` means "not mentioned", not "ambiguous", and it covers 33% of the file.
  4,378 of the 5,218 genuinely abnormal pancreases (84%) end up hidden inside
  REVIEW_REQUIRED.

Here the pancreas axis comes from ``pancreas_status`` (20 pancreas concepts) and
the other-organ axis from the 29 findings at 100% coverage.
"""
import pandas as pd

from . import assertion as A
from . import config as C

ABNORMAL, UNCERTAIN, NORMAL = "ABNORMAL", "UNCERTAIN", "NORMAL"


def other_organs_status(df: pd.DataFrame) -> pd.DataFrame:
    """ABNORMAL / UNCERTAIN / NORMAL for everything that is not the pancreas.

    Reads the ``<finding>_assertion`` columns rather than the binary labels, so
    a hedged finding routes to REVIEW instead of being silently flattened to 0.
    """
    cols = C.other_finding_cols()
    states = df[[f"{f}_assertion" for f in cols]]
    n_present = states.eq(A.PRESENT).sum(axis=1)
    n_uncertain = states.eq(A.UNCERTAIN).sum(axis=1)

    status = pd.Series(NORMAL, index=df.index, dtype=object)
    status[n_uncertain > 0] = UNCERTAIN
    status[n_present >= C.OTHER_DISEASE_MIN_FINDINGS] = ABNORMAL

    out = df.copy()
    out["other_organs_status"] = status
    out["n_other_present"] = n_present.astype("int16")
    out["n_other_uncertain"] = n_uncertain.astype("int16")
    return out


def run(df: pd.DataFrame) -> pd.DataFrame:
    df = other_organs_status(df)

    panc_abn = df["pancreas_status"].eq("ABNORMAL")
    if C.POSTOP_IS_PANCREAS_ABNORMAL:
        panc_abn = panc_abn | df["pancreas_status"].eq("POSTOPERATIVE")
    other_abn = df["other_organs_status"].eq(ABNORMAL)

    cls = pd.Series("", index=df.index, dtype=object)
    reason = pd.Series("", index=df.index, dtype=object)

    # Ordered cascade -- first match wins.
    review_panc = df["pancreas_status"].eq("REVIEW_REQUIRED")
    review_other = df["other_organs_status"].eq(UNCERTAIN) & ~review_panc

    cls[review_panc] = "REVIEW_REQUIRED"
    reason[review_panc] = "pancreas_" + df.loc[review_panc, "pancreas_status_reason"].str.split(":").str[0]
    cls[review_other] = "REVIEW_REQUIRED"
    reason[review_other] = "other_organs_hedged_only"

    rest = cls.eq("")
    for mask, name in (
        (rest & panc_abn & other_abn, "PANCREAS_AND_OTHER"),
        (rest & panc_abn & ~other_abn, "PANCREAS_ONLY"),
        (rest & ~panc_abn & other_abn, "OTHER_ONLY"),
        (rest & ~panc_abn & ~other_abn, "ALL_NORMAL"),
    ):
        cls[mask] = name
        reason[mask] = name.lower()

    df["class_5"] = cls
    df["class_5_reason"] = reason
    return df
