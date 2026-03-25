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
