"""Score the annotated validation sheet.

    python -m merlin_prep.score_validation

Reads ``output/manual_validation_sample.csv`` once ``human_pancreas_status`` and/or
``human_class_5`` are filled in, and reports per-class precision/recall plus a
per-slice error breakdown. Because the sample is risk-weighted rather than random,
the per-slice numbers are the useful ones -- the pooled figure is pessimistic by
construction and must not be quoted as the corpus accuracy.
"""
import pandas as pd

from . import config as C


def _score(df: pd.DataFrame, pred_col: str, true_col: str, labels) -> pd.DataFrame:
    rows = []
    for lab in labels:
        tp = int(((df[pred_col] == lab) & (df[true_col] == lab)).sum())
        fp = int(((df[pred_col] == lab) & (df[true_col] != lab)).sum())
        fn = int(((df[pred_col] != lab) & (df[true_col] == lab)).sum())
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
        rows.append({"label": lab, "n_true": tp + fn, "TP": tp, "FP": fp, "FN": fn,
                     "precision": round(prec, 3) if prec is not None else None,
                     "recall": round(rec, 3) if rec is not None else None,
                     "f1": round(f1, 3) if f1 is not None else None})
    return pd.DataFrame(rows)


def main() -> None:
    path = C.OUT / "manual_validation_sample.csv"
    df = pd.read_csv(path)
    for pred, true, labels, title in (
        ("pancreas_status", "human_pancreas_status", C.PANCREAS_STATUS, "pancreas_status"),
        ("class_5", "human_class_5", C.CLASS_5, "class_5"),
    ):
        sub = df[df[true].notna() & df[true].astype(str).str.strip().ne("")]
        if sub.empty:
            print(f"\n### {title}: no annotations yet in `{true}` -- skipped")
            continue
        sub = sub.assign(**{true: sub[true].astype(str).str.strip().str.upper()})
        print(f"\n### {title}  ({len(sub)} annotated of {len(df)})")
        print(f"exact agreement: {100*(sub[pred] == sub[true]).mean():.1f}%")
        print(_score(sub, pred, true, labels).to_string(index=False))
        print("\nerrors by slice:")
        err = sub.assign(wrong=(sub[pred] != sub[true]))
        print(err.groupby("slice").wrong.agg(["sum", "count"])
                 .assign(rate=lambda x: (100 * x["sum"] / x["count"]).round(1))
                 .to_string())
        print("\nconfusion (rows = ours, cols = human):")
        print(pd.crosstab(sub[pred], sub[true]).to_string())


if __name__ == "__main__":
    main()
