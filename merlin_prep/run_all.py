"""Orchestrator:  python -m merlin_prep.run_all"""
import json
import time

import pandas as pd

from . import assemble, classes5, config as C, extract, ingest, pancreas, regex_zero_shot, validate


def main() -> None:
    t0 = time.time()
    C.OUT.mkdir(parents=True, exist_ok=True)

    print("[0/6] ingest + dedupe ...")
    d = ingest.run()
    study, raw, dropped = d["study"], d["reports_raw"], d["dropped"]
    print(f"      report-level {len(raw):,}  ->  study-level {len(study):,} "
          f"(dropped {len(dropped)})")

    print("[1/6] segment + extract ...")
    study = extract.run(study)

    print("[2/6] pancreas status ...")
    study = pancreas.run(study)

    print("[3/6] hybrid label merge ...")
    study = assemble.merge_findings(study, d["zero_shot"])

    print("[4/6] metadata join + QC flags + 5-class target ...")
    study = assemble.join_metadata(study, d["metadata"], d["five_years"])
    study = assemble.qc_flags(study)
    study = classes5.run(study)

    print("[5/6] validate ...")
    report = validate.run(study, raw, dropped)
    (C.OUT / "validation_report.md").write_text(report, encoding="utf-8")
    validate.rule_firings(study).to_csv(C.OUT / "qa_rule_firings.csv", index=False)

    print("[6/6] write outputs ...")
    cols = assemble.final_columns(study)
    study[cols].to_csv(C.OUT / "merlin_pancreas_dataset.csv", index=False)

    # Drop-in replacement for Stanford's zero_shot file, built from our rules.
    zs_flat, zs_full = regex_zero_shot.build(study)
    zs_flat.to_csv(C.OUT / "zero_shot_findings_regex.csv", index=False)
    zs_full.to_csv(C.OUT / "zero_shot_findings_regex_full.csv", index=False)
    regex_zero_shot.compare_to_original(zs_flat).to_csv(
        C.OUT / "zero_shot_regex_vs_original.csv", index=False)

    # Kept out of the main table (long free text) but needed for manual review.
    study[["study_id", "pancreas_context", "pancreas_context_source"]].to_csv(
        C.OUT / "pancreas_context.csv", index=False)

    uh = study.unknown_headers.fillna("").str.split(";").explode()
    uh = uh[uh.ne("")].value_counts().rename_axis("header").reset_index(name="n")
    uh.to_csv(C.OUT / "qa_unknown_headers.csv", index=False)

    raw_out = raw[["study_id", "report_idx", "Split", "Few_Shot", "text_len", "text"]]
    raw_out.to_csv(C.OUT / "merlin_reports_clean.csv", index=False)
    dropped[["study_id", "report_idx", "Split", "text_len", "drop_reason", "text"]] \
        .to_csv(C.OUT / "dropped_rows_audit.csv", index=False)

    ev_cols = [c for c in study.columns if c.startswith("ev_")]
    with (C.OUT / "label_evidence.jsonl").open("w", encoding="utf-8") as fh:
        for _, r in study.iterrows():
            ev = {c[3:]: r[c] for c in ev_cols if isinstance(r[c], str) and r[c]}
            if ev:
                fh.write(json.dumps({"study_id": r.study_id, "evidence": ev},
                                    ensure_ascii=False) + "\n")

    print(f"\ndone in {time.time()-t0:.0f}s -> {C.OUT}")
    print(study.class_5.value_counts().to_string())


if __name__ == "__main__":
    main()
