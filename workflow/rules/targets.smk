rule preprocess_all:
    input:
        PREPROCESSED_EEG_OUTPUTS

rule envelope_all:
    input:
        ENVELOPE_OUTPUTS


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


rule lmeeeg_all:
    input:
        LMEEEG_SUMMARY_OUTPUT


rule figures_lmeeeg_all:
    input:
        LMEEEG_FIGURE_MANIFEST,
        LMEEEG_INFERENCE_FIGURE_MANIFEST
