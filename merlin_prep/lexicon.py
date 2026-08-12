"""Concept patterns.

Two scopes exist, and the distinction is the whole point of this module:

* ``OTHER_PATTERNS`` -- the 29 non-pancreas findings. Every term here is
  *self-identifying* ("atelectasis", "hydronephrosis", "cholecystectomy"), so
  matching over the whole report is safe.

* ``PANCREAS_PATTERNS`` -- terms like "mass", "cyst", "atrophy",
  "calcification" are organ-agnostic. Matching these over the whole report is
  what produced the bogus counts in the earlier passes (renal and muscular
  atrophy scored as pancreatic). They are therefore matched ONLY inside the
  pancreas context built by ``segment.pancreas_context``.
"""
import re

_F = re.IGNORECASE


def _c(p):
    return re.compile(p, _F)


# --------------------------------------------------------------------------
# 29 non-pancreas findings -- whole-report scope
# --------------------------------------------------------------------------
OTHER_PATTERNS = {
    "atelectasis": _c(r"\batelecta\w*"),
    "pleural_effusion": _c(r"pleural effusions?|effusions? (?:in|of) the pleural"),
    "cardiomegaly": _c(r"cardiomegal\w*|(?:heart|cardiac silhouette)\s+(?:is|are|size is)?\s*"
                       r"(?:mildly |moderately |markedly )?enlarged|enlarged (?:heart|cardiac silhouette)"),
    "coronary_calcification": _c(r"coronary[^.;]{0,40}calcifi\w*|calcifi\w*[^.;]{0,30}coronary|"
                                 r"coronary (?:artery )?(?:atherosclerosis|athero\w*)"),
    "aortic_valve_calcification": _c(r"aortic valv\w*[^.;]{0,30}calcifi\w*|"
                                     r"calcifi\w*[^.;]{0,25}aortic valv\w*"),
    "atherosclerosis": _c(r"atheroscleros\w*|atheromatous|calcific plaque|"
                          r"vascular calcifi\w*|calcifi\w*\s+atheroscl\w*"),
    "abdominal_aortic_aneurysm": _c(r"(?:abdominal )?aortic aneurysm\w*|\baaa\b|"
                                    r"aneurysmal[^.;]{0,25}aorta|aorta[^.;]{0,25}aneurysmal"),
    "thrombosis": _c(r"\bthromb\w*|embol\w*\s+(?:in|within|of) the"),
    "hepatic_steatosis": _c(r"hepatic steatosis|steatosis|fatty (?:liver|infiltration)|"
                            r"(?:liver|hepatic)[^.;]{0,30}fatty (?:change|infiltration)|hepatosteatosis"),
    "hepatomegaly": _c(r"hepatomegal\w*|(?:liver|hepatic)[^.;]{0,20}\benlarged\b|enlarged liver"),
    "splenomegaly": _c(r"splenomegal\w*|spleen[^.;]{0,20}\benlarged\b|enlarged spleen"),
    "biliary_ductal_dilation": _c(r"(?:biliary|bile duct\w*|common bile duct|\bcbd\b|intrahepatic duct\w*|"
                                  r"extrahepatic duct\w*)[^.;]{0,45}dilat\w*|"
                                  r"dilat\w*[^.;]{0,35}(?:biliary|bile duct\w*|\bcbd\b)"),
    "gallstones": _c(r"gallstone\w*|cholelithiasis|(?:stone|calculus|calculi)[^.;]{0,35}gallbladder|"
                     r"gallbladder[^.;]{0,35}(?:stone|calculus|calculi)"),
    "surgically_absent_gallbladder": _c(r"cholecystectom\w*|gallbladder[^.;]{0,30}(?:surgically )?absent|"
                                        r"(?:surgically )?absent[^.;]{0,20}gallbladder|"
                                        r"gallbladder (?:is |has been )?(?:been )?(?:surgically )?removed"),
    "renal_cyst": _c(r"renal cyst\w*|cyst\w*[^.;]{0,30}(?:kidney|renal)|"
                     r"(?:kidney|renal|kidneys)[^.;]{0,30}cyst\w*"),
    "renal_hypodensities": _c(r"(?:renal|kidney|kidneys)[^.;]{0,35}(?:hypodens\w*|hypoattenuat\w*|low.attenuation)|"
                              r"(?:hypodens\w*|hypoattenuat\w*)[^.;]{0,30}(?:renal|kidney|kidneys)"),
    "hydronephrosis": _c(r"hydronephro\w*|hydroureteronephro\w*|pelvicaliectas\w*|caliectas\w*"),
    "prostatomegaly": _c(r"prostatomegal\w*|prostat\w*[^.;]{0,20}\benlarged\b|enlarged prostat\w*|"
                         r"prostatic (?:enlargement|hypertrophy)"),
    "ascites": _c(r"\bascites\b|\bascitic\b"),
    "free_air": _c(r"pneumoperitoneum|free (?:intraperitoneal |intra-abdominal )?(?:air|gas)|"
                   r"extraluminal (?:air|gas)"),
    "bowel_obstruction": _c(r"bowel obstruction|\bsbo\b|\blbo\b|transition point|"
                            r"obstructive pattern|obstructed bowel"),
    "appendicitis": _c(r"appendicitis"),
    "submucosal_edema": _c(r"submucosal edema|(?:bowel|colonic|mural|small bowel|colon)[^.;]{0,20}"
                           r"wall (?:edema|thickening)|mural (?:edema|thickening|hyperenhancement)|"
                           r"wall thickening (?:of|in) the (?:bowel|colon)"),
    "hiatal_hernia": _c(r"hiatal hernia\w*|hiatus hernia\w*"),
    "lymphadenopathy": _c(r"lymphadenopathy|adenopathy|(?:enlarged|prominent) lymph node\w*|"
                          r"lymph node\w*[^.;]{0,20}\benlarged\b"),
    "metastatic_disease": _c(r"metasta\w*"),
    "anasarca": _c(r"\banasarca\b|body wall edema|(?:subcutaneous|soft tissue)[^.;]{0,15}edema|"
                   r"diffuse[^.;]{0,20}edema"),
    "osteopenia": _c(r"osteopeni\w*|osteoporo\w*|(?:osseous |bone )?demineraliz\w*"),
    "fracture": _c(r"fractur\w*"),
}

# --------------------------------------------------------------------------
# Pancreas concepts -- matched ONLY inside the pancreas context
# --------------------------------------------------------------------------
PANCREAS_PATTERNS = {
    "panc_atrophy": _c(r"atroph\w*"),
    "panc_fatty_replacement": _c(r"fatty (?:replacement|infiltrat\w*|change\w*|atroph\w*)|"
                                 r"lipomatos\w*|steatos\w*|fat(?:ty)? (?:deposition|involution)"),
    "panc_pancreatitis": _c(r"pancreatitis"),
    "panc_pancreatitis_acute": _c(r"acute (?:necrotizing |interstitial |edematous )?pancreatitis"),
    "panc_pancreatitis_chronic": _c(r"chronic\w*\s+pancreatitis|pancreatitis[,\s]+(?:which is )?chronic"),
    "panc_necrosis": _c(r"necrosis|necrotic|necrotizing"),
    "panc_pseudocyst": _c(r"pseudocyst\w*"),
    "panc_cystic_lesion": _c(r"\bcyst\w*"),
    "panc_ipmn": _c(r"\bipmn\b|intraductal papillary|side.branch|mucinous"),
    "panc_solid_mass": _c(r"\bmass\w*|\bnodul\w*|(?:focal|solid|hypodense|hypoattenuating|hypoenhancing|"
                          r"enhancing|discrete)\s+lesion\w*|\blesion\w*"),
    "panc_malignancy": _c(r"carcinoma|adenocarcinoma|malignan\w*|neoplas\w*|\bcancer\w*|\btumor\w*|"
                          r"\bmetasta\w*"),
    "panc_duct_dilation": _c(r"(?:pancreatic duct|main duct|main pancreatic duct|ductal?)[^.;]{0,40}"
                             r"(?:dilat\w*|prominen\w*|ectasi\w*)|"
                             r"(?:dilat\w*|prominen\w*)[^.;]{0,30}(?:pancreatic duct|ductal?)"),
    "panc_calcification": _c(r"calcifi\w*|calculus|calculi|\bstone\w*"),
    "panc_peripancreatic_inflam": _c(r"peripancreatic[^.;]{0,40}(?:strand\w*|fluid|inflamm\w*|edema|fat)|"
                                     r"(?:strand\w*|inflamm\w*|fluid)[^.;]{0,30}peripancreatic|"
                                     r"fat stranding"),
    "panc_divisum_annular": _c(r"pancreas divisum|annular pancreas"),
    # Bare "catheter"/"drain" fired on Foley catheters in the bladder, so the
    # generic device words now require a pancreatic or ductal qualifier.
    "panc_stent": _c(r"\bstent\w*|"
                     r"(?:pancrea\w*|ductal?|biliary|pseudocyst|collection)[^.;]{0,30}"
                     r"(?:drain\w*|pigtail|catheter)|"
                     r"(?:drain\w*|pigtail|catheter)[^.;]{0,30}(?:pancrea\w*|pseudocyst)"),
    "panc_transplant": _c(r"transplant\w*|allograft"),
    # Split into specific + generic below; this entry is the union and is only
    # used for pattern bookkeeping. extract.py applies the proximity rule.
    "panc_postop_resection": _c(r"whipple|appleby|p[eu]+stow|pancre\w*(?:ectom|ostom)\w*|"
                                r"necrosectom\w*|necresectom\w*|necrostom\w*|"
                                r"cyst\w?o?[- ]?gastrostom\w*|cystojejunostom\w*|"
                                r"cystoduodenostom\w*|(?:surgically )?absent|resect\w*|"
                                r"post.?(?:operative|surgical)\b|\bremnant\b|enucleat\w*|"
                                r"debride\w*|(?:resection|pancreatectomy|operative|surgical)\s+bed"),
    "panc_explicit_normal": _c(r"\bnormal\w*|unremarkable|within normal limits|\bwnl\b|"
                               r"enhances? homogeneous\w*|homogeneous(?:ly)? enhanc\w*"),
    # Polarity is baked into these patterns, so the negation logic must not run
    # on them -- "no significant abnormality" means normal, not "not normal".
    "panc_no_abnormality": _c(r"no (?:significant |focal |acute |definite )?abnormalit\w*|"
                              r"no acute (?:finding\w*|process\w*)|"
                              r"no evidence of (?:acute )?abnormalit\w*"),
    "panc_not_evaluated": _c(r"not (?:well |adequately |fully )?(?:visuali\w*|evaluat\w*|assess\w*|"
                             r"seen)|obscur\w*|suboptimal\w*|"
                             r"limited (?:evaluation|assessment|visualization)|"
                             r"degraded|streak artifact|beam.hardening"),
}

# Concepts whose regex already encodes polarity. ``assertion.classify_concept``
# skips negation handling for these; without this, the leading "no"/"not" in the
# pattern itself would flip the result to ABSENT.
NEGATION_EXEMPT = {"panc_no_abnormality", "panc_not_evaluated"}

# -- post-operative status: specific vs generic -----------------------------
# "Whipple" can only mean the pancreas. "Surgically absent" / "resection" cannot:
# they leak in from a colectomy or appendectomy mentioned in the same sentence.
# Generic terms are therefore accepted only when the clause names the pancreas,
# or names no competing organ at all (covers the bare "Pancreas: Surgically
# absent." template).
# Concepts whose wording is organ-agnostic ("mass", "cyst", "atrophy",
# "calcification"). Matching them anywhere in the pancreas context lets a
# neighbouring organ's finding score as pancreatic, so extract.py requires the
# clause to name the pancreas whenever it names some other organ.
ORGAN_AMBIGUOUS = {
    "panc_solid_mass", "panc_cystic_lesion", "panc_calcification", "panc_stent",
    "panc_necrosis", "panc_atrophy", "panc_fatty_replacement",
    "panc_transplant", "panc_malignancy", "panc_pseudocyst",
}

# Vocabulary mined from the corpus rather than guessed: every -ectomy/-ostomy
# token appearing in a pancreas context was listed and split by organ. The
# `pancre\w*` form generalises the whole family in one pattern — it covers
# pancreatectomy, pancreaticojejunostomy, pancreaticoduodenectomy,
# pancreaticogastrostomy, pancreaticoenterostomy, pancreaticoduodenostomy,
# pancreatojejunostomy, and the misspelling `pancreectomy` seen 6 times.
# Bare `gastrostomy` is deliberately excluded (35 hits — feeding tubes, not
# pancreatic); only the cyst-drainage compounds are matched.
POSTOP_SPECIFIC = _c(r"whipple|appleby|p[eu]+stow|"
                     r"pancre\w*(?:ectom|ostom)\w*|"
                     r"necrosectom\w*|necresectom\w*|necrostom\w*|"
                     r"cyst\w?o?[- ]?gastrostom\w*|cystojejunostom\w*|cystoduodenostom\w*")

# Organ-agnostic surgical wording — accepted only when the clause names the
# pancreas or names no competing organ (see extract._postop_state).
POSTOP_GENERIC = _c(r"(?:surgically )?absent|resect\w*|post.?(?:operative|surgical)\b|"
                    r"\bremnant\b|enucleat\w*|debride\w*|"
                    r"(?:resection|pancreatectomy|operative|surgical)\s+bed")

PANCREAS_TERM = _c(r"pancrea\w*|uncinate")
OTHER_ORGAN = _c(r"append\w*|colon\w*|colect\w*|colostom\w*|hemicolect\w*|"
                 r"(?:low )?anterior resection|\blar\b|proctect\w*|sigmoid|rectum|rectal|"
                 r"gallbladder|cholecystect\w*|spleen|splenect\w*|kidney|renal|"
                 r"nephrect\w*|uter\w*|hysterect\w*|ovar\w*|prostat\w*|bladder|"
                 r"stomach|gastrect\w*|gastric|small bowel|ileum|ileal|jejun\w*|"
                 r"liver|hepat\w*|lung|thyroid|adrenal|hernia\w*|aort\w*")
