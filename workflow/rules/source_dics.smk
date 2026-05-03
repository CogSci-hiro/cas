from datetime import UTC, datetime
import json
from pathlib import Path

from cas.source_dics.io import discover_epoch_records, discover_preprocessed_records

SOURCE_DICS_FPP_SPP_ALPHA_BETA_CONFIG_PATH = (
    f"{CONFIG_DIR}/source_dics_fpp_spp_alpha_beta.yaml"
)

with open(SOURCE_DICS_FPP_SPP_ALPHA_BETA_CONFIG_PATH, encoding="utf-8") as _source_dics_handle:
    _SOURCE_DICS_FPP_SPP_ALPHA_BETA_CONFIG = yaml.safe_load(_source_dics_handle) or {}

SOURCE_DICS_FPP_SPP_ALPHA_BETA_DERIVATIVES_DIR = Path(
    _SOURCE_DICS_FPP_SPP_ALPHA_BETA_CONFIG["paths"]["derivatives_dir"]
)
SOURCE_DICS_FPP_SPP_ALPHA_BETA_QC_DIR = SOURCE_DICS_FPP_SPP_ALPHA_BETA_DERIVATIVES_DIR / "qc"
SOURCE_DICS_FPP_SPP_ALPHA_BETA_SUMMARY_OUTPUT = str(
    SOURCE_DICS_FPP_SPP_ALPHA_BETA_QC_DIR / "run_summary.json"
)
SOURCE_DICS_FPP_SPP_ALPHA_BETA_FIGURES_INDEX = str(
    SOURCE_DICS_FPP_SPP_ALPHA_BETA_DERIVATIVES_DIR / "figures" / "index" / "figure_index.csv"
)


def _discover_source_dics_records() -> list[dict[str, str]]:
    paths_cfg = dict(_SOURCE_DICS_FPP_SPP_ALPHA_BETA_CONFIG.get("paths") or {})
    preprocessed_root = paths_cfg.get("preprocessed_eeg_root")
    bids_root = paths_cfg.get("bids_root")
    if preprocessed_root and bids_root:
        records = discover_preprocessed_records(
            Path(preprocessed_root),
            bids_root=Path(bids_root),
        )
    else:
        records = discover_epoch_records(Path(paths_cfg["epochs_dir"]))
    return [
        {"subject": str(record.subject_id), "run": str(record.run_id)}
        for record in records
    ]


SOURCE_DICS_FPP_SPP_ALPHA_BETA_RECORDS = _discover_source_dics_records()
SOURCE_DICS_FPP_SPP_ALPHA_BETA_RECORD_SUMMARIES = [
    str(
        SOURCE_DICS_FPP_SPP_ALPHA_BETA_QC_DIR
        / "by_record"
        / f"{record['subject']}_run-{record['run']}"
        / "run_summary.json"
    )
    for record in SOURCE_DICS_FPP_SPP_ALPHA_BETA_RECORDS
]


rule run_source_dics_fpp_spp_alpha_beta_record:
    input:
        config=SOURCE_DICS_FPP_SPP_ALPHA_BETA_CONFIG_PATH,
    output:
        summary=(
            str(
                SOURCE_DICS_FPP_SPP_ALPHA_BETA_QC_DIR
                / "by_record"
                / "{subject}_run-{run}"
                / "run_summary.json"
            )
        ),
    params:
        qc_subdir=lambda wildcards: f"by_record/{wildcards.subject}_run-{wildcards.run}",
    threads: 1
    shell:
        (
            "PYTHONPATH={SRC_DIR} "
            "{PYTHON_BIN} -m cas.cli.main source-dics-fpp-spp "
            "--config {input.config} "
            "--subjects {wildcards.subject} "
            "--runs {wildcards.run} "
            "--qc-subdir {params.qc_subdir}"
        )


rule aggregate_source_dics_fpp_spp_alpha_beta:
    input:
        summaries=SOURCE_DICS_FPP_SPP_ALPHA_BETA_RECORD_SUMMARIES,
    output:
        summary=SOURCE_DICS_FPP_SPP_ALPHA_BETA_SUMMARY_OUTPUT,
    run:
        summary_rows = []
        status_values = []
        failures = []
        bands = None
        anchor_types = None
        config_path = None
        common_filter_guardrail = None
        post_onset_guardrail = None
        lmeeeg_model_spec = None
        forward_templates: list[str] = []

        for path_text in input.summaries:
            payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
            status_values.append(str(payload.get("status", "ok")))
            summary_rows.extend(list(payload.get("subject_runs") or []))
            failures.extend(list(payload.get("failures") or []))
            if bands is None:
                bands = list(payload.get("bands") or [])
            if anchor_types is None:
                anchor_types = list(payload.get("anchor_types") or [])
            if config_path is None:
                config_path = payload.get("config_path")
            if common_filter_guardrail is None:
                common_filter_guardrail = payload.get("common_filter_guardrail")
            if post_onset_guardrail is None:
                post_onset_guardrail = payload.get("post_onset_guardrail")
            if lmeeeg_model_spec is None:
                lmeeeg_model_spec = payload.get("lmeeeg_model_spec")
            forward_template = payload.get("forward_template")
            if forward_template not in {None, ""}:
                forward_templates.append(str(forward_template))

        aggregate_summary = {
            "status": "failed" if any(value != "ok" for value in status_values) else "ok",
            "config_path": config_path,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "n_subject_runs_discovered": int(len(input.summaries)),
            "n_subject_runs_processed": int(len(summary_rows)),
            "bands": bands or [],
            "anchor_types": anchor_types or [],
            "common_filter_guardrail": common_filter_guardrail,
            "post_onset_guardrail": post_onset_guardrail,
            "lmeeeg_model_spec": lmeeeg_model_spec,
            "forward_template": sorted(set(forward_templates))[0] if forward_templates else None,
            "subject_runs": summary_rows,
            "failures": failures,
            "per_record_summaries": list(input.summaries),
        }
        output_path = Path(output.summary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(aggregate_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


rule plot_source_dics_fpp_spp_alpha_beta:
    input:
        summary=SOURCE_DICS_FPP_SPP_ALPHA_BETA_SUMMARY_OUTPUT,
        config=SOURCE_DICS_FPP_SPP_ALPHA_BETA_CONFIG_PATH,
    output:
        index=SOURCE_DICS_FPP_SPP_ALPHA_BETA_FIGURES_INDEX,
    threads: 1
    shell:
        (
            "PYTHONPATH={SRC_DIR} "
            "{PYTHON_BIN} -m cas.cli.main plot-source-dics-fpp-spp "
            "--config {input.config}"
        )


rule source_dics_fpp_spp_alpha_beta_figures:
    input:
        SOURCE_DICS_FPP_SPP_ALPHA_BETA_FIGURES_INDEX
