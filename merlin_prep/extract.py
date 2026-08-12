"""Stage 3/4 -- run the lexicon over each report and emit assertion states."""
import pandas as pd

from . import assertion as A
from . import config as C
from . import lexicon as L
from . import segment as S


def _organ_scoped_state(ctx: str, pattern) -> tuple[str, str]:
    """Like ``A.classify_concept`` but drops matches that belong to another organ.

    A clause is rejected only when it names some other organ and never names the
    pancreas. "Mass abutting the pancreatic body and the stomach" survives, while
    the renal lesion in a GU sentence does not.
    """
    best, evidence = A.NOT_MENTIONED, ""
    for m in pattern.finditer(ctx):
        left, right = A.clause_bounds(ctx, m.start(), m.end())
        clause = ctx[left:right]
        if not L.PANCREAS_TERM.search(clause) and L.OTHER_ORGAN.search(clause):
            continue
        state = A.classify_match(ctx, m.start(), m.end())
        if A.RANK[state] > A.RANK[best]:
            best, evidence = state, clause.strip()[:220]
            if best == A.PRESENT:
                break
    return best, evidence


def _postop_state(ctx: str) -> tuple[str, str]:
    """Post-operative pancreas, with an organ-proximity guard on generic terms.

    "Whipple" is unambiguous. "surgically absent" / "resection" are not: without
    the guard they fire on an appendectomy or low anterior resection mentioned in
    a sentence that happens to also name the pancreas, which mislabelled ~10
    studies as POSTOPERATIVE in the first run.
    """
    best, evidence = A.NOT_MENTIONED, ""
    for pattern, generic in ((L.POSTOP_SPECIFIC, False), (L.POSTOP_GENERIC, True)):
        for m in pattern.finditer(ctx):
            left, right = A.clause_bounds(ctx, m.start(), m.end())
            clause = ctx[left:right]
            if generic and not L.PANCREAS_TERM.search(clause) and L.OTHER_ORGAN.search(clause):
                continue  # the surgery belongs to another organ
            state = A.classify_match(ctx, m.start(), m.end())
            if A.RANK[state] > A.RANK[best]:
                best, evidence = state, clause.strip()[:220]
    return best, evidence


def extract_row(text: str) -> dict:
    sections = S.split_sections(text)
    panc_ctx, prov = S.pancreas_context(sections, C.USE_IMPRESSION_FOR_PANCREAS)

    row: dict = {
        "pancreas_context": panc_ctx,
        "pancreas_context_source": prov,
        "has_pancreas_section": int(prov == "section"),
        "n_sections": len(sections),
        "has_impression": int(bool(S.impression_text(sections))),
        # Alarm for a pancreas section that is absorbing a neighbouring organ.
        "unknown_headers": ";".join(sorted(set(
            S.unknown_headers(sections.get("pancreas", ""))))),
    }

    # 29 findings -- self-identifying terms, whole-report scope.
    for name in C.OTHER_FINDINGS:
        state, ev = A.classify_concept(text, L.OTHER_PATTERNS[name])
        # Fallback for the silent negative: the finding term never appears, but
        # the section that would have to contain it is declared normal. Fires
        # ONLY on not_mentioned, so any positive, hedged or historical evidence
        # found in the report -- including in the impression, outside the
        # section -- keeps precedence over the section header.
        if state == A.NOT_MENTIONED and name in L.NORMAL_SECTION:
            m = L.NORMAL_SECTION[name].search(text)
            if m:
                state, ev = A.ABSENT, m.group(0).strip(".\n; ")[:220]
        row[f"ext_{name}"] = state
        row[f"ev_{name}"] = ev

    # Pancreas concepts -- organ-agnostic terms, pancreas scope only.
    for name in C.PANCREAS_LABELS:
        if not panc_ctx:
            state, ev = A.NOT_MENTIONED, ""
        elif name == "panc_postop_resection":
            state, ev = _postop_state(panc_ctx)
        elif name in L.ORGAN_AMBIGUOUS:
            state, ev = _organ_scoped_state(panc_ctx, L.PANCREAS_PATTERNS[name])
        else:
            state, ev = A.classify_concept(
                panc_ctx, L.PANCREAS_PATTERNS[name], name in L.NEGATION_EXEMPT
            )
        row[name] = state
        row[f"ev_{name}"] = ev

    return row


def run(study: pd.DataFrame) -> pd.DataFrame:
    recs = [extract_row(t) for t in study["text"]]
    ext = pd.DataFrame.from_records(recs, index=study.index)
    return pd.concat([study, ext], axis=1)
