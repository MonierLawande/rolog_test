"""Stage 2 -- decide whether a matched concept is asserted, denied, hedged or historical.

This is where the earlier passes lost the most accuracy:

* They scoped negation by a fixed character window. A window straddles sentence
  boundaries, so "Normal enhancement of the pancreas although mildly atrophic
  appearance" was read as negated. Scope here is the *clause*, split on
  terminators and on contrastive conjunctions (although / but / however).
* They had no notion of resolved or historical findings. 1,787 reports contain
  "resolution"/"resolved"; "interval resolution of pancreatitis" is not current
  pancreatitis.
* Hedging was ignored, yet it is everywhere: 13,423 reports say may/might/could,
  6,770 say possible, 4,091 say suspicious/concerning for, 5,456 say
  "too small to characterize".
"""
import re

PRESENT = "present"
ABSENT = "absent"
UNCERTAIN = "uncertain"
HISTORICAL = "historical"
NOT_MENTIONED = "not_mentioned"

# Strength order used when one concept matches several times in a report.
RANK = {PRESENT: 4, UNCERTAIN: 3, HISTORICAL: 2, ABSENT: 1, NOT_MENTIONED: 0}

CLAUSE_BOUNDARY = re.compile(
    r"[.;:!?]|\b(?:although|though|however|whereas|but|while|otherwise|except)\b",
    re.IGNORECASE,
)

# Hedges that contain a negation token but mean "uncertain", not "absent".
# Checked before negation so "cannot exclude a mass" is not read as denial.
UNC_OVERRIDE = re.compile(
    r"cannot\s+(?:be\s+)?(?:exclude|excluded|rule\s*out|ruled\s*out)|"
    r"not\s+(?:entirely\s+|completely\s+)?excluded|"
    r"difficult\s+to\s+(?:exclude|characterize)|"
    r"not\s+(?:well\s+)?characteriz",
    re.IGNORECASE,
)

NEG_PRE = re.compile(
    r"\b(?:no|not|non|without|absent|nor|neither|denies|devoid)\b|"
    r"\bnon-?(?:dilat|obstruct|enlarg|distend|occlus)\w*|"
    r"negative\s+for|free\s+of|lack\s+of|rather\s+than|unremarkable\s+for|"
    r"no\s+evidence|no\s+definite|no\s+significant|no\s+acute|no\s+focal|no\s+residual",
    re.IGNORECASE,
)

# Negation tokens that may sit *inside* a match, e.g. the pattern for ductal
# dilation spans "pancreatic duct is not dilated". 'absent' is deliberately
# excluded -- it is part of the concept in surgically_absent_gallbladder.
NEG_INNER = re.compile(
    r"\b(?:not|without)\b|\bno\b(?!\w)|"
    # Closed-form negations: "the main pancreatic duct is nondilated" reads as
    # dilation present because \bno\b cannot match a glued "non" prefix.
    # Deliberately excludes nonspecific / nonenhancing / nonaggressive -- those
    # describe a finding that IS present, they do not deny it.
    r"\bnon-?(?:dilat|obstruct|enlarg|distend|occlus)\w*",
    re.IGNORECASE,
)

NEG_POST = re.compile(
    r"^\W{0,4}(?:is|are|was|were|has|have)?\s*(?:not\s+(?:identified|seen|present|"
    r"visualized|appreciated|noted|dilated|enlarged)|absent|unremarkable|negative)",
    re.IGNORECASE,
)

UNC = re.compile(
    r"\b(?:may|might|could|possible|possibly|probable|questionable|equivocal|"
    r"indeterminate|presumed|presumably|apparent|suggests?|suggestive|consider)\b|"
    r"suspicious\s+for|concerning\s+for|worrisome\s+for|favor(?:ed|s)?\b|"
    r"\bversus\b|\bvs\.?\b|too\s+small\s+to\s+characteriz",
    re.IGNORECASE,
)

HIST = re.compile(
    r"history\s+of|\bhx\b|resolution\s+of|\bresolved\b|no\s+longer|"
    r"previously\s+(?:noted|seen|described)[^.;]{0,30}\b(?:resolved|no longer)\b",
    re.IGNORECASE,
)


def clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    for m in CLAUSE_BOUNDARY.finditer(text, 0, start):
        left = m.end()
    right = len(text)
    m = CLAUSE_BOUNDARY.search(text, end)
    if m:
        right = m.start()
    return left, right


def classify_match(text: str, start: int, end: int, negation_exempt: bool = False) -> str:
    """Assertion state for one concept occurrence."""
    left, right = clause_bounds(text, start, end)
    clause = text[left:right]
    pre = text[left:start]
    inner = text[start:end]
    post = text[end:right]

    if UNC_OVERRIDE.search(clause):
        return UNCERTAIN
    if HIST.search(pre) or HIST.search(inner):
        return HISTORICAL
    if not negation_exempt:
        if NEG_PRE.search(pre) or NEG_INNER.search(inner) or NEG_POST.search(post):
            return ABSENT
    if UNC.search(pre) or UNC.search(inner):
        return UNCERTAIN
    return PRESENT


def classify_concept(text: str, pattern, negation_exempt: bool = False) -> tuple[str, str]:
    """Aggregate every occurrence of one concept. Returns (state, evidence)."""
    best, evidence = NOT_MENTIONED, ""
    for m in pattern.finditer(text):
        state = classify_match(text, m.start(), m.end(), negation_exempt)
        if RANK[state] > RANK[best]:
            left, right = clause_bounds(text, m.start(), m.end())
            best, evidence = state, text[left:right].strip()[:220]
            if best == PRESENT:
                break
    return best, evidence


def to_binary(state: str, uncertain_as_positive: bool = False) -> int:
    if state == PRESENT:
        return 1
    if state == UNCERTAIN:
        return 1 if uncertain_as_positive else 0
    return 0
