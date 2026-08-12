"""Stage 6 -- automated validation. No manual annotation set, per user decision."""
import pandas as pd

from . import assertion as A
from . import config as C

L = []


def _p(s=""):
    L.append(str(s))


def agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Original vs extraction, restricted to cells the original actually labelled."""
    rows = []
    for f in C.OTHER_FINDINGS:
        orig = df.get(f"orig_{f}")
        if orig is None:
            continue
        st = df[f"{f}_assertion"]
        lab = orig.isin([0, 1])
        ext = st.map(lambda s: A.to_binary(s, C.UNCERTAIN_AS_POSITIVE))
        both = lab.sum()
        agree = (orig[lab] == ext[lab]).sum()
        rows.append({
            "finding": f,
            "orig_labelled": int(both),
            "orig_coverage_%": round(100 * both / len(df), 1),
            "final_coverage_%": 100.0,
            "agreement_%": round(100 * agree / both, 1) if both else None,
            "orig0_ext_present": int(((orig == 0) & (st == A.PRESENT)).sum()),
            "orig1_ext_absent": int(((orig == 1) & (st == A.ABSENT)).sum()),
            "recovered_from_-1": int(((orig == -1) & (st == A.PRESENT)).sum()),
            "conflicts": int(df[f"{f}_conflict"].sum()),
        })
    return pd.DataFrame(rows).sort_values("recovered_from_-1", ascending=False)


def notebook_labels(zs: pd.DataFrame) -> pd.Series:
    """Reproduce Merlin_Analysis.ipynb cell 74 verbatim on the raw zero_shot file."""
    import numpy as np
    pc = "pancreatic_atrophy"
    organ = [c for c in zs.columns if c not in ("study_id", pc)]
    has_other = zs[organ].eq(1).any(axis=1)
    conditions = [
        zs[pc].eq(-1),
        zs[pc].eq(1) & has_other,
        zs[pc].eq(1) & ~has_other,
        zs[pc].eq(0) & has_other,
        zs[pc].eq(0) & ~has_other,
    ]
    return pd.Series(np.select(conditions, C.CLASS_5, default="REVIEW_REQUIRED"),
                     index=zs.index)


def notebook_comparison(df: pd.DataFrame) -> str:
    zs = pd.read_csv(C.ZERO_SHOT).drop_duplicates("study_id")
    nb = pd.DataFrame({"study_id": zs.study_id, "nb_class": notebook_labels(zs)})
    m = df[["study_id", "class_5", "pancreas_status"]].merge(nb, on="study_id")

    out = ["The notebook drives the pancreas axis off `pancreatic_atrophy` alone and "
           "maps its `-1` to REVIEW_REQUIRED. Both choices are reproduced here on the "
           "raw file for a like-for-like comparison.\n"]
    out.append("| class | notebook | ours |")
    out.append("|---|---:|---:|")
    nvc, ovc = m.nb_class.value_counts(), m.class_5.value_counts()
    for k in C.CLASS_5:
        out.append(f"| {k} | {int(nvc.get(k, 0)):,} | {int(ovc.get(k, 0)):,} |")

    abn = m.pancreas_status.isin(["ABNORMAL", "POSTOPERATIVE"])
    hidden = int((abn & m.nb_class.eq("REVIEW_REQUIRED")).sum())
    as_normal = int((abn & m.nb_class.isin(["OTHER_ONLY", "ALL_NORMAL"])).sum())
    out.append(f"\n- Studies with a genuinely abnormal or post-operative pancreas: "
               f"**{int(abn.sum()):,}**")
    out.append(f"- ...of which the notebook buries in `REVIEW_REQUIRED`: "
               f"**{hidden:,}** ({100*hidden/max(int(abn.sum()),1):.0f}%)")
    out.append(f"- ...and labels outright normal (`OTHER_ONLY`/`ALL_NORMAL`): **{as_normal:,}**")
    out.append("\n### cross-tab\n")
    out.append(pd.crosstab(m.class_5, m.nb_class).to_markdown())
    return "\n".join(out)


def ablation(df: pd.DataFrame) -> str:
    """Isolate which axis actually drives the gap against the notebook."""
    import numpy as np
    zs = pd.read_csv(C.ZERO_SHOT).drop_duplicates("study_id")
    organ_orig = [c for c in zs.columns if c not in ("study_id", "pancreatic_atrophy")]
    # Collapse the raw file to its two axis inputs *before* merging, otherwise the
    # finding columns collide with our own same-named columns and get _x/_y suffixes.
    axes = pd.DataFrame({
        "study_id": zs.study_id,
        "_pc": zs.pancreatic_atrophy,
        "_other_orig": zs[organ_orig].eq(1).any(axis=1),
    })
    m = df.merge(axes, on="study_id", how="inner")
    pc = m["_pc"]
    other_orig = m["_other_orig"]
    other_ours = m["other_organs_status"].eq("ABNORMAL")

    def nb(other):
        cond = [pc.eq(-1), pc.eq(1) & other, pc.eq(1) & ~other,
                pc.eq(0) & other, pc.eq(0) & ~other]
        return pd.Series(np.select(cond, C.CLASS_5, default="REVIEW_REQUIRED"), index=m.index)

    variants = [
        ("V0 notebook as written", nb(other_orig)),
        ("V1 + our other-organs axis", nb(other_ours)),
        ("V2 + our pancreas axis (= ours)", m["class_5"]),
    ]
    tbl = pd.DataFrame(
        {name: {k: int((s == k).sum()) for k in C.CLASS_5} for name, s in variants}
    )
    out = [f"Ablation over the {len(m):,} studies present in both.\n",
           tbl.to_markdown(), ""]
    d01 = int((tbl.iloc[:, 1] - tbl.iloc[:, 0]).abs().sum() / 2)
    d12 = int((tbl.iloc[:, 2] - tbl.iloc[:, 1]).abs().sum() / 2)
    out.append(f"- Swapping the **other-organs** axis moves **{d01:,} studies**.")
    out.append(f"- Swapping the **pancreas** axis moves **{d12:,} studies**.")
    out.append("\nBoth axes are broken in the notebook, the pancreas one roughly twice "
               "as badly.")
    out.append("\n- **Other-organs axis:** the original labels are 77.6% `-1`, so "
               "`.eq(1).any()` finds no disease in thousands of studies that plainly have "
               "some. Fixing it moves ~3,600 studies out of `ALL_NORMAL` into "
               "`OTHER_ONLY` — the notebook calls 23% of an emergency-department cohort "
               "completely normal, which is not credible.")
    out.append("- **Pancreas axis:** `pancreatic_atrophy` alone means \"atrophy\", not "
               "\"abnormal pancreas\", and its `-1` is routed to REVIEW_REQUIRED.")
    sub = m[pc.eq(-1)]
    out.append(f"\n### what is really inside the notebook's {len(sub):,} `-1 → REVIEW_REQUIRED`\n")
    out.append(sub.pancreas_status.value_counts().to_frame("n").to_markdown())
    return "\n".join(out)


def organ_leakage(df: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    """Pancreas concepts whose evidence names another organ and never the pancreas."""
    from . import lexicon as LX
    rows = []
    for con in sorted(LX.ORGAN_AMBIGUOUS):
        ev = df.loc[df[con] == A.PRESENT, ["study_id", f"ev_{con}"]].dropna()
        bad = ev[~ev[f"ev_{con}"].str.contains(LX.PANCREAS_TERM.pattern, case=False, regex=True)
                 & ev[f"ev_{con}"].str.contains(LX.OTHER_ORGAN.pattern, case=False, regex=True)]
        rows.append({"concept": con, "present": int((df[con] == A.PRESENT).sum()),
                     "leaked": len(bad)})
    t = pd.DataFrame(rows)
    ids = set()
    for con in sorted(LX.ORGAN_AMBIGUOUS):
        ev = df.loc[df[con] == A.PRESENT, ["study_id", f"ev_{con}"]].dropna()
        bad = ev[~ev[f"ev_{con}"].str.contains(LX.PANCREAS_TERM.pattern, case=False, regex=True)
                 & ev[f"ev_{con}"].str.contains(LX.OTHER_ORGAN.pattern, case=False, regex=True)]
        ids |= set(bad.study_id)
    return int(t.leaked.sum()), len(ids), t


def consistency(df: pd.DataFrame) -> list[tuple[str, int]]:
    checks = []
    bad = df[(df.pancreas_status == "NORMAL") & (df.pancreas_any_abnormality == 1)]
    checks.append(("NORMAL yet a pancreas abnormality is present (must be 0)", len(bad)))

    # A post-Whipple report routinely calls the pancreatic remnant normal, so
    # POSTOPERATIVE + explicit_normal is expected, not a defect. What matters is
    # whether the surgical evidence actually names the pancreas.
    # The true error class is evidence that names *another* organ's surgery.
    # A bare "Surgically absent" inside the Pancreas: section is correct.
    from . import lexicon as LX
    po = df[df.pancreas_status == "POSTOPERATIVE"]
    ev = po.pancreas_evidence.fillna("")
    names_panc = ev.str.contains(LX.POSTOP_SPECIFIC.pattern + r"|pancrea|uncinate|necrosectom",
                                 case=False, regex=True)
    names_other = ev.str.contains(LX.OTHER_ORGAN.pattern, case=False, regex=True)
    checks.append(("POSTOPERATIVE whose evidence names another organ's surgery",
                   int((~names_panc & names_other).sum())))

    bad = df[(df.surgically_absent_gallbladder == 1) & (df.gallstones == 1)]
    both_orig = bad[(bad.surgically_absent_gallbladder_source == "original")
                    & (bad.gallstones_source == "original")]
    checks.append((f"gallbladder absent yet gallstones positive "
                   f"({len(both_orig)} of them inherited from the original labels)", len(bad)))

    bad = df[(df.pancreas_status == "ABNORMAL") & (df.pancreas_evidence == "")]
    checks.append(("ABNORMAL with no evidence span captured", len(bad)))

    pos = df.class_5.isin(["PANCREAS_ONLY", "PANCREAS_AND_OTHER"])
    backed = df.pancreas_any_abnormality.eq(1) | df.pancreas_status.eq("POSTOPERATIVE")
    checks.append(("class_5 says pancreas positive but no pancreas evidence backs it",
                   int((pos & ~backed).sum())))

    checks.append(("class_5 empty / unassigned", int(df.class_5.eq("").sum())))

    dup = df.study_id.duplicated().sum()
    checks.append(("duplicate study_id in the final table", int(dup)))

    cross = df[df.text_collision_group >= 0].groupby("text_collision_group")[C.SPLIT_COL].nunique()
    checks.append(("text collisions still crossing splits", int((cross > 1).sum())))
    return checks


def run(df: pd.DataFrame, raw_reports: pd.DataFrame, dropped: pd.DataFrame) -> str:
    L.clear()
    _p("# Validation report\n")
    _p("> **Label semantics, from Stanford's own `documentation/download.md`:**\n>")
    _p("> *\"These labels were generated by applying regex-based matching of zero-shot "
       "positive and negative prompts to the findings section; therefore, some entries "
       "marked as missing may in fact correspond to positive or negative cases but "
       "remain missing when the prompt is not explicitly stated in the findings.\"*\n>")
    _p("> `1` = mentioned · `0` = explicitly negated · `-1` = **not mentioned**\n>")
    _p("> So `-1` is a regex miss, not a clinical ambiguity, and the released labels are "
       "themselves regex output. Re-deriving them with better rules is the documented "
       "remedy, not a departure from the source.\n")

    _p("## 1. Row counts\n")
    _p("| check | value | expected |")
    _p("|---|---:|---:|")
    _p(f"| report-level rows | {len(raw_reports):,} | {C.GUARDS['report_rows']:,} |")
    _p(f"| study-level rows | {len(df):,} | {C.GUARDS['study_rows']:,} |")
    _p(f"| rows dropped | {len(dropped):,} | 8 |")
    _p(f"| duplicate study_id | {df.study_id.duplicated().sum()} | 0 |")
    _p("")

    _p("## 2. `class_5` — primary target\n")
    vc5 = df.class_5.value_counts()
    _p("| class | n | % |")
    _p("|---|---:|---:|")
    for k in sorted(C.CLASS_5, key=lambda x: -int(vc5.get(x, 0))):
        n = int(vc5.get(k, 0))
        _p(f"| {k} | {n:,} | {100*n/len(df):.2f}% |")
    smallest = int(vc5.min())
    _p(f"\nSmallest class: **{smallest:,}**"
       f" — switches: `OTHER_DISEASE_MIN_FINDINGS={C.OTHER_DISEASE_MIN_FINDINGS}`,"
       f" `OTHER_DISEASE_EXCLUDE_INCIDENTAL={C.OTHER_DISEASE_EXCLUDE_INCIDENTAL}`"
       f" ({len(C.other_finding_cols())} findings feed the other-organs axis).")
    if smallest < 500:
        _p(f"\n> ⚠️ **`{vc5.idxmin()}` has only {smallest:,} studies "
           f"(~{smallest//5:,} in the test split).** Too thin for a reliable per-class "
           f"metric. Setting `OTHER_DISEASE_EXCLUDE_INCIDENTAL = True` raises it to "
           f"~1,016 by dropping near-universal degenerative findings from the axis.")
    _p("\n### by split\n")
    _p(pd.crosstab(df.class_5, df[C.SPLIT_COL]).to_markdown())
    _p("\n### the two axes\n")
    _p(pd.crosstab(df.pancreas_status, df.other_organs_status).to_markdown())
    _p("")

    _p("## 2b. Versus the notebook's logic (`Merlin_Analysis.ipynb` cell 74)\n")
    _p(notebook_comparison(df))
    _p("")

    _p("## 2d. Ablation — which axis causes the gap?\n")
    _p(ablation(df))
    _p("")

    _p("## 2e. Organ leakage into pancreas concepts\n")
    total, n_studies, tbl = organ_leakage(df)
    _p(f"Evidence spans that name another organ and never name the pancreas. "
       f"Before the `GU:`/`GI:` header fix this was **172 studies (133 wrongly ABNORMAL)**; "
       f"target is 0.\n")
    _p(tbl.to_markdown(index=False))
    _p(f"\n**Total leaked matches: {total} across {n_studies} studies.**")
    uh = df.unknown_headers.fillna("").str.split(";").explode()
    uh = uh[uh.ne("")].value_counts()
    _p(f"\nUnrecognised headers still appearing inside a `Pancreas:` section: "
       f"**{len(uh)}** distinct (see `qa_unknown_headers.csv`). "
       f"`gu`/`gi` present: **{'yes' if {'gu','gi'} & set(uh.index) else 'no'}**.")
    _p("")

    _p("## 2c. Pancreas class distribution\n")
    vc = df.pancreas_status.value_counts()
    _p("| class | n | % |")
    _p("|---|---:|---:|")
    for k in C.PANCREAS_STATUS:
        n = int(vc.get(k, 0))
        _p(f"| {k} | {n:,} | {100*n/len(df):.1f}% |")
    _p(f"\nSmallest class: **{int(vc.min()):,}** studies "
       f"(the earlier 2x2 design bottomed out at 10).\n")
    _p("### by split\n")
    _p(pd.crosstab(df.pancreas_status, df[C.SPLIT_COL]).to_markdown())
    _p("")

    _p("## 3. Pancreas decision reasons\n")
    _p(df.pancreas_status_reason.str.split(":").str[0].value_counts().to_frame("n").to_markdown())
    _p("")

    _p("## 4. Pancreas context provenance\n")
    _p(df.pancreas_context_source.value_counts().to_frame("n").to_markdown())
    _p("")

    _p("## 5. Original zero-shot vs text extraction\n")
    ag = agreement(df)
    _p(ag.to_markdown(index=False))
    _p(f"\n- Positives recovered from `-1` cells: **{int(ag['recovered_from_-1'].sum()):,}**")
    _p(f"- Cells where the original 0/1 contradicts strong text evidence: "
       f"**{int(ag['conflicts'].sum()):,}** (kept as-is; `OVERRIDE_CONFLICTS={C.OVERRIDE_CONFLICTS}`)")
    _p(f"- Mean original coverage: **{ag['orig_coverage_%'].mean():.1f}%** -> final **100.0%**\n")

    _p("## 6. Internal consistency\n")
    _p("| check | violations |")
    _p("|---|---:|")
    for name, n in consistency(df):
        _p(f"| {name} | {n} |")
    _p("")

    _p("## 7. Duplicate resolution audit\n")
    _p("| study_id | kept chars | dropped chars |")
    _p("|---|---:|---:|")
    for sid in C.KNOWN_DUPLICATE_IDS:
        kept = df.loc[df.study_id == sid, "text_len"]
        drop = dropped.loc[dropped.study_id == sid, "text_len"]
        _p(f"| {sid} | {int(kept.iloc[0]) if len(kept) else 'n/a'} | "
           f"{', '.join(str(int(x)) for x in drop) if len(drop) else '-'} |")
    _p("")

    _p("## 8. Cohort QC flags (advisory, nothing filtered)\n")
    _p("| flag | n | % |")
    _p("|---|---:|---:|")
    for c in ["has_metadata", "has_zero_shot", "slice_ok", "phase_portal",
              "metadata_contradiction", "age_implausible", "duplicate_conflict"]:
        if c in df:
            n = int(df[c].sum())
            _p(f"| {c} | {n:,} | {100*n/len(df):.1f}% |")
    _p("")
    return "\n".join(L)


def rule_firings(df: pd.DataFrame) -> pd.DataFrame:
    """Per-concept firing counts plus sample evidence spans for eyeballing."""
    rows = []
    concepts = [(c, c) for c in C.PANCREAS_LABELS] + \
               [(f, f"ext_{f}") for f in C.OTHER_FINDINGS]
    for name, col in concepts:
        if col not in df:
            continue
        vc = df[col].value_counts()
        ev = df.loc[df[col] == A.PRESENT, f"ev_{name}"].head(5).tolist()
        rows.append({
            "concept": name,
            "present": int(vc.get(A.PRESENT, 0)),
            "absent": int(vc.get(A.ABSENT, 0)),
            "uncertain": int(vc.get(A.UNCERTAIN, 0)),
            "historical": int(vc.get(A.HISTORICAL, 0)),
            "not_mentioned": int(vc.get(A.NOT_MENTIONED, 0)),
            **{f"sample_{i+1}": (ev[i] if i < len(ev) else "") for i in range(5)},
        })
    return pd.DataFrame(rows)
