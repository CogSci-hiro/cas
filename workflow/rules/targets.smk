INDUCED_EPOCH_SUBJECTS = sorted({record["subject"] for record in EPOCH_RECORDS})
INDUCED_EPOCH_SUMMARY_OUTPUTS = expand(
    f"{OUT_DIR}/induced_epochs/sub-{{subject}}/summary.json",
    subject=INDUCED_EPOCH_SUBJECTS,
)


def induced_epoch_source_inputs(wildcards):
    subject_records = [
        record
        for record in EPOCH_RECORDS
        if record["subject"] == wildcards.subject
    ]
    if not subject_records:
        raise ValueError(f"No epoch records found for subject {wildcards.subject}.")
    return [
        INDUCED_SOURCE_EPOCHS_OUTPUT_PATTERN.format(
            subject=record["subject"],
            task=record["task"],
            run=record["run"],
        )
        for record in subject_records
    ]


rule preprocess_all:
    input:
        PREPROCESSED_EEG_OUTPUTS

rule envelope_all:
    input:
        ENVELOPE_OUTPUTS


rule f0_all:
    input:
        F0_OUTPUTS


rule acoustic_all:
    input:
        ENVELOPE_OUTPUTS,
        ENVELOPE_SUMMARY_OUTPUTS,
        F0_OUTPUTS,
        F0_SUMMARY_OUTPUTS


rule trf_all:
    input:
        TRF_SCORE_OUTPUTS,
        TRF_COEF_OUTPUTS


rule events_all:
    input:
        EVENTS_CSV_OUTPUT,
        PAIRING_ISSUES_CSV_OUTPUT


rule epochs_all:
    input:
        EPOCH_OUTPUTS


rule make_induced_epochs_subject:
    input:
        epochs=induced_epoch_source_inputs,
    output:
        summary=f"{OUT_DIR}/induced_epochs/sub-{{subject}}/summary.json",
    run:
        import json
        from pathlib import Path

        import mne
        import numpy as np
        import pandas as pd

        from cas.epochs.io import (
            write_epoch_events_array,
            write_epoch_metadata,
            write_epoch_summary,
            write_epochs,
        )
        from cas.induced_epochs.transform import (
            build_induced_epochs,
            resolve_induced_band_limits_hz,
            resolve_induced_band_names,
        )

        source_epochs = [
            mne.read_epochs(path, preload=True, verbose="ERROR")
            for path in input.epochs
        ]
        concatenated = mne.concatenate_epochs(
            source_epochs,
            add_offset=True,
            on_mismatch="raise",
            verbose="ERROR",
        )

        if concatenated.metadata is None:
            metadata_df = pd.DataFrame(index=np.arange(len(concatenated)))
        else:
            metadata_df = concatenated.metadata.copy().reset_index(drop=True)

        band_names = resolve_induced_band_names(EPOCHS_CONFIG)
        written_bands = []
        for band_name in band_names:
            low_hz, high_hz = resolve_induced_band_limits_hz(band_name, EPOCHS_CONFIG)
            induced_epochs = build_induced_epochs(concatenated, band_name=band_name, config=EPOCHS_CONFIG)

            band_dir = Path(OUT_DIR) / "induced_epochs" / band_name / f"sub-{wildcards.subject}"
            epochs_output = band_dir / "epochs-time_s.fif"
            metadata_output = band_dir / "metadata-time_s.csv"
            events_array_output = band_dir / "events-time_s.npy"
            band_summary_output = band_dir / "epoching_summary-time_s.json"

            band_summary = {
                "status": "ok",
                "band_name": band_name,
                "band_limits_hz": [low_hz, high_hz],
                "subject_id": f"sub-{wildcards.subject}",
                "source_epochs_paths": [str(path) for path in input.epochs],
                "n_source_files": len(input.epochs),
                "n_epochs": int(len(induced_epochs)),
                "n_channels": int(len(induced_epochs.ch_names)),
                "n_times": int(len(induced_epochs.times)),
                "tmin_s": float(induced_epochs.times[0]) if len(induced_epochs.times) else 0.0,
                "tmax_s": float(induced_epochs.times[-1]) if len(induced_epochs.times) else 0.0,
                "sampling_frequency_hz": float(induced_epochs.info["sfreq"]),
                "method": "bandpass_hilbert_envelope",
            }

            write_epochs(induced_epochs, epochs_output)
            write_epoch_metadata(metadata_df, metadata_output)
            write_epoch_events_array(induced_epochs.events.copy(), events_array_output)
            write_epoch_summary(band_summary, band_summary_output)
            written_bands.append(
                {
                    "band_name": band_name,
                    "metadata_output": str(metadata_output),
                    "epochs_output": str(epochs_output),
                    "events_array_output": str(events_array_output),
                    "summary_output": str(band_summary_output),
                }
            )

        subject_summary = {
            "status": "ok",
            "subject_id": f"sub-{wildcards.subject}",
            "bands": written_bands,
        }
        output_path = Path(output.summary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(subject_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


rule induced_epochs_all:
    input:
        INDUCED_EPOCH_SUMMARY_OUTPUTS


rule lmeeeg_all:
    input:
        LMEEEG_SUMMARY_OUTPUT


rule induced_lmeeeg_all:
    input:
        LMEEEG_INDUCED_SUMMARY_OUTPUT


rule lme_eeg_fpp_spp_cycle_position:
    input:
        FPP_SPP_CYCLE_POSITION_LMEEEG_SUMMARY_OUTPUT,
        *FPP_SPP_CYCLE_POSITION_LMEEEG_CONTRAST_OUTPUTS


rule tde_hmm_all:
    input:
        GLHMM_FIT_SUMMARY_OUTPUT


rule hazard_behavior_all:
    input:
        HAZARD_BEHAVIOR_RISKSET_OUTPUT,
        *HAZARD_BEHAVIOR_TIMING_CONTROL_OUTPUTS,
        *HAZARD_BEHAVIOR_GLMM_EXPORT_OUTPUTS,
        *HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS,
        *HAZARD_BEHAVIOR_GLMM_FIGURES


rule hazard_behavior_timing_control_all:
    input:
        *HAZARD_BEHAVIOR_TIMING_CONTROL_OUTPUTS


rule hazard_behavior_glmm_all:
    input:
        *HAZARD_BEHAVIOR_GLMM_EXPORT_OUTPUTS,
        *HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS,
        *HAZARD_BEHAVIOR_GLMM_FIGURES


rule hazard_behavior_latency_regime_all:
    input:
        *HAZARD_BEHAVIOR_LATENCY_REGIME_EXPORT_OUTPUTS,
        *HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS,
        *HAZARD_BEHAVIOR_LATENCY_REGIME_FIGURES,
        *HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES,
        *HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_DIAGNOSTICS


rule hazard_behavior_final_all:
    input:
        rules.behavior_final_all.input


rule hazard_neural_all:
    input:
        *HAZARD_NEURAL_ALL_OUTPUTS


rule hazard_neural_riskset_all:
    input:
        *HAZARD_NEURAL_RISKSET_OUTPUTS


rule hazard_neural_models_all:
    input:
        *HAZARD_NEURAL_MODEL_OUTPUTS


rule hazard_neural_lag_selection_all:
    # Aggregates the lag-selection-specific neural outputs so we can request one target
    # instead of remembering the individual model, JSON, null-summary, and figure paths.
    input:
        *HAZARD_NEURAL_LAG_SELECTION_OUTPUTS


rule hazard_neural_permutation_null_all:
    input:
        *HAZARD_NEURAL_PERMUTATION_NULL_OUTPUTS


rule hazard_neural_figures_all:
    input:
        *HAZARD_NEURAL_FIGURES


rule neural_hazard_fpp_spp:
    input:
        *NEURAL_HAZARD_FPP_SPP_OUTPUTS


rule source_dics_fpp_spp_alpha_beta_all:
    input:
        SOURCE_DICS_FPP_SPP_ALPHA_BETA_SUMMARY_OUTPUT


rule source_dics_all:
    input:
        SOURCE_DICS_FPP_SPP_ALPHA_BETA_SUMMARY_OUTPUT


rule figures_lmeeeg_all:
    input:
        LMEEEG_FIGURE_MANIFEST,
        LMEEEG_INFERENCE_FIGURE_MANIFEST


rule figures_lme_eeg_fpp_spp_cycle_position:
    input:
        FPP_SPP_CYCLE_POSITION_LMEEEG_FIGURE_MANIFEST,
        FPP_SPP_CYCLE_POSITION_LMEEEG_INFERENCE_FIGURE_MANIFEST


rule neural_hazard_fpp_spp_renyi_alpha:
    input:
        *NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS
