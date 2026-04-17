GLHMM_OUTPUT_DIR = f"{OUT_DIR}/models/glhmm"
GLHMM_INPUT_MANIFEST = f"{GLHMM_OUTPUT_DIR}/input_manifest.csv"
GLHMM_SPEECH_INTERVALS_DIR = f"{GLHMM_OUTPUT_DIR}/speech_intervals"
GLHMM_MODEL_SELECTION_OUTPUT = f"{GLHMM_OUTPUT_DIR}/model_selection.csv"
GLHMM_CHUNKS_OUTPUT = f"{GLHMM_OUTPUT_DIR}/chunks.csv"
GLHMM_FIT_SUMMARY_OUTPUT = f"{GLHMM_OUTPUT_DIR}/fit_summary.json"
HMM_SETTINGS = dict(HMM_CONFIG.get("hmm", HMM_CONFIG))


def _fit_tde_hmm_cli_args() -> str:
    candidate_k = HMM_SETTINGS.get("candidate_k", [4, 5, 6, 7])
    if not isinstance(candidate_k, list) or not candidate_k:
        raise ValueError("`hmm.candidate_k` in config/hmm.yaml must be a non-empty list.")

    args = [
        f'--sampling-rate-hz {float(HMM_SETTINGS.get("sampling_rate_hz", 128.0))}',
        f'--causal-history-ms {float(HMM_SETTINGS.get("causal_history_ms", 100.0))}',
        f'--speech-guard-pre-s {float(HMM_SETTINGS.get("speech_guard_pre_s", 0.2))}',
        f'--speech-guard-post-s {float(HMM_SETTINGS.get("speech_guard_post_s", 0.2))}',
        f'--minimum-chunk-duration-s {float(HMM_SETTINGS.get("minimum_chunk_duration_s", 0.5))}',
        "--candidate-k " + " ".join(str(int(value)) for value in candidate_k),
        f'--n-initializations {int(HMM_SETTINGS.get("n_initializations", 5))}',
        f'--random-seed {int(HMM_SETTINGS.get("random_seed", 123))}',
        f'--covariance-type "{str(HMM_SETTINGS.get("covariance_type", "full"))}"',
    ]

    pca_n_components = HMM_SETTINGS.get("pca_n_components")
    if pca_n_components is not None:
        args.append(f"--pca-n-components {int(pca_n_components)}")

    if not bool(HMM_SETTINGS.get("standardize_per_run", True)):
        args.append("--no-standardize-per-run")

    if not bool(HMM_SETTINGS.get("verbose", True)):
        args.append("--quiet")

    return " ".join(args)


rule build_tde_hmm_manifest:
    output:
        manifest=GLHMM_INPUT_MANIFEST,
        speech_dir=directory(GLHMM_SPEECH_INTERVALS_DIR),
    run:
        from pathlib import Path

        import pandas as pd

        from cas.annotations.io import load_textgrid

        def _speaker_tier_name(subject_value: str) -> str:
            return f"ipu-{'A' if int(subject_value) % 2 == 1 else 'B'}"

        def _write_speech_intervals(annotation_path: str, subject_value: str, task_value: str, run_value: str) -> str:
            textgrid = load_textgrid(Path(annotation_path))
            tier_name = _speaker_tier_name(subject_value)
            tier = next((current_tier for current_tier in textgrid.tiers if current_tier.name == tier_name), None)
            if tier is None:
                raise ValueError(f"Tier `{tier_name}` not found in annotation file: {annotation_path}")

            speech_rows = []
            for interval in tier.intervals:
                label = str(interval.text).strip()
                if label == "#":
                    continue
                if float(interval.xmax) <= float(interval.xmin):
                    continue
                speech_rows.append(
                    {
                        "onset_s": float(interval.xmin),
                        "offset_s": float(interval.xmax),
                    }
                )

            speech_output_path = (
                Path(output.speech_dir)
                / f"sub-{subject_value}_task-{task_value}_run-{run_value}_speech.csv"
            )
            speech_output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(speech_rows, columns=["onset_s", "offset_s"]).to_csv(speech_output_path, index=False)
            return str(speech_output_path)

        manifest_rows = []
        Path(output.speech_dir).mkdir(parents=True, exist_ok=True)

        for record in EPOCH_RECORDS:
            subject = str(record["subject"])
            task = str(record["task"])
            run = str(record["run"])
            annotation_path = _epoch_annotation_path(subject, run)
            if annotation_path is None:
                continue

            source_path = PREPROCESSED_EEG_OUTPUT_PATTERN.format(subject=subject, task=task, run=run)
            if not os.path.exists(source_path):
                continue

            speech_path = _write_speech_intervals(annotation_path, subject, task, run)
            manifest_rows.append(
                {
                    "subject": f"sub-{subject}",
                    "run": run,
                    "source_path": source_path,
                    "speech_path": speech_path,
                }
            )

        if not manifest_rows:
            raise ValueError(
                "No preprocessed EEG runs were found for TDE-HMM manifest generation. "
                "Run the preprocessing workflow first so the expected FIF files exist."
            )

        manifest_frame = pd.DataFrame(manifest_rows).sort_values(["subject", "run"]).reset_index(drop=True)
        Path(output.manifest).parent.mkdir(parents=True, exist_ok=True)
        manifest_frame.to_csv(output.manifest, index=False)


rule fit_tde_hmm:
    input:
        manifest=GLHMM_INPUT_MANIFEST,
        speech_dir=GLHMM_SPEECH_INTERVALS_DIR,
    output:
        model_selection=GLHMM_MODEL_SELECTION_OUTPUT,
        chunks=GLHMM_CHUNKS_OUTPUT,
        fit_summary=GLHMM_FIT_SUMMARY_OUTPUT,
    params:
        output_dir=GLHMM_OUTPUT_DIR,
        cli_args=_fit_tde_hmm_cli_args(),
    shell:
        """
        PYTHONPATH="{SRC_DIR}" python -m cas.cli.main fit-tde-hmm \
            --input-manifest "{input.manifest}" \
            --output-dir "{params.output_dir}" \
            {params.cli_args}
        """
