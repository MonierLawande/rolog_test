"""Stage 1 -- split a report into organ sections, impression, and a pancreas context.

Coverage measured on the real data:

* 22,210 reports (87.1%) carry an explicit ``Pancreas:`` header.
* 3,279 reports are free prose with no organ headers; 3,230 of those still
  mention the pancreas in a sentence.
* 49 reports never mention the pancreas at all -> REVIEW_REQUIRED downstream.
* 2,204 reports mention the pancreas inside IMPRESSION as well.

Headers are matched against a closed vocabulary taken from a survey of the
corpus rather than a generic ``Word:`` pattern, which would fire on ordinary
prose (e.g. "measuring 3 cm: ...").
"""
import re

SECTION_HEADERS = [
    "findings", "technique", "indication", "clinical history", "history",
    "comparison", "impression", "summary", "conclusion",
    "lower thorax", "lower chest", "lung bases", "chest", "thorax", "heart",
    "liver and biliary tree", "liver, biliary tree", "liver and biliary system",
    "liver", "biliary tree", "biliary system", "biliary",
    "gallbladder and biliary tree", "gallbladder/bile ducts", "gallbladder",
    "spleen", "pancreas", "pancreas and peritoneal cavity",
    "adrenal glands", "adrenals", "adrenal",
    "kidneys and ureters", "kidneys/ureters", "kidneys, ureters", "kidneys",
    "kidney", "urinary tract", "genitourinary",
    "gastrointestinal tract", "gastrointestinal", "bowel", "small bowel",
    "colon", "stomach", "stomach and duodenum", "esophagus", "appendix",
    # Abbreviated headers. Their absence was a real defect: the Pancreas section
    # ran on and swallowed the following organ, so renal and bowel findings were
    # scored as pancreatic in 172 studies. "GU" appears 262 times, "GI" 258.
    "gu", "gi", "gu tract", "gi tract", "genitourinary tract", "gyn",
    "peritoneal cavity", "peritoneal space", "peritoneum", "abdomen",
    "abdomen and pelvis", "retroperitoneum",
    "bladder", "urinary bladder",
    "prostate and seminal vesicles", "prostate", "seminal vesicles",
    "uterus and ovaries", "uterus and adnexa", "uterus", "ovaries", "adnexa",
    "pelvic organs", "reproductive organs", "pelvis",
    "vasculature", "vascular", "vessels", "aorta",
    "lymph nodes", "lymphatics", "nodes",
    "abdominal wall", "soft tissues", "body wall",
    "musculoskeletal", "bones", "bones/soft tissues", "osseous structures",
    "osseous", "skeleton", "other", "additional findings", "miscellaneous",
]
# Longest first so "liver and biliary tree" wins over "liver".
_HEAD_ALT = "|".join(re.escape(h) for h in sorted(SECTION_HEADERS, key=len, reverse=True))
HEADER_RE = re.compile(rf"(?:(?<=^)|(?<=[.\s;])) *({_HEAD_ALT}) *:", re.IGNORECASE)

PANCREAS_TERMS = re.compile(r"pancrea|uncinate", re.IGNORECASE)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Sections whose prose may legitimately describe the pancreas.
PANCREAS_SECTIONS = {"pancreas", "pancreas and peritoneal cavity"}


def split_sections(text: str) -> dict[str, str]:
    """Return {canonical_header: body}. Unheaded lead text lands under ''."""
    out: dict[str, str] = {}
    marks = list(HEADER_RE.finditer(text))
    if not marks:
        return {"": text}
    if marks[0].start() > 0:
        out[""] = text[: marks[0].start()].strip()
    for i, m in enumerate(marks):
        name = m.group(1).lower().strip()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end(): end].strip()
        # A header may repeat; concatenate rather than overwrite.
        out[name] = (out.get(name, "") + " " + body).strip() if name in out else body
    return out


def pancreas_context(sections: dict[str, str], use_impression: bool = True) -> tuple[str, str]:
    """Build the text the pancreas lexicon is allowed to match against.

    Returns (context_text, provenance) where provenance is one of
    ``section`` / ``freetext`` / ``impression_only`` / ``none``.
    """
    parts, prov = [], "none"

    for key in PANCREAS_SECTIONS:
        if sections.get(key):
            parts.append(sections[key])
            prov = "section"

    if not parts:
        # Free-prose fallback: keep only sentences that talk about the pancreas,
        # so organ-agnostic words ("mass", "cyst") stay scoped.
        body = " ".join(v for k, v in sections.items() if k not in {"impression", "summary", "conclusion"})
        sents = [s for s in SENT_SPLIT.split(body) if PANCREAS_TERMS.search(s)]
        if sents:
            parts.extend(sents)
            prov = "freetext"

    if use_impression:
        imp = " ".join(sections.get(k, "") for k in ("impression", "summary", "conclusion"))
        imp_sents = [s for s in SENT_SPLIT.split(imp) if PANCREAS_TERMS.search(s)]
        if imp_sents:
            parts.extend(imp_sents)
            if prov == "none":
                prov = "impression_only"

    return " ".join(p for p in parts if p).strip(), prov


# Sub-headers that legitimately live *inside* a pancreas-protocol CT section.
# They must NOT become top-level sections, and they must not raise the alarm below.
PANCREAS_SUBHEADERS = {
    "parenchyma", "pancreatic duct", "masses", "mass", "fluid collections",
    "morphologic evaluation", "arterial evaluation", "venous evaluation",
    "celiac axis", "sma", "cha", "mpv", "smv", "gda", "arterial variant",
    "variant vessel contact", "appearance of mass", "cm location",
}
_KNOWN = {h.lower() for h in SECTION_HEADERS} | PANCREAS_SUBHEADERS
_INNER_HEADER = re.compile(r"(?:(?<=^)|(?<=[.\s;])) *([A-Za-z][A-Za-z /&'-]{0,40}?) *:")


def unknown_headers(body: str) -> list[str]:
    """Header-looking tokens inside a pancreas section that we do not recognise.

    A hit means the section is probably absorbing a neighbouring organ, which is
    exactly how renal findings ended up labelled as pancreatic. Surfaced in
    ``qa_unknown_headers.csv`` so the failure cannot recur silently.
    """
    out = []
    for m in _INNER_HEADER.finditer(body):
        h = m.group(1).strip().lower()
        if len(h) > 1 and h not in _KNOWN:
            out.append(h)
    return out


def impression_text(sections: dict[str, str]) -> str:
    return " ".join(sections.get(k, "") for k in ("impression", "summary", "conclusion")).strip()
