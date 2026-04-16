GLHMM_OUTPUT_DIR = f"{OUT_DIR}/models/glhmm"
GLHMM_INPUT_MANIFEST = f"{GLHMM_OUTPUT_DIR}/input_manifest.csv"
GLHMM_MODEL_SELECTION_OUTPUT = f"{GLHMM_OUTPUT_DIR}/model_selection.csv"
GLHMM_CHUNKS_OUTPUT = f"{GLHMM_OUTPUT_DIR}/chunks.csv"
GLHMM_FIT_SUMMARY_OUTPUT = f"{GLHMM_OUTPUT_DIR}/fit_summary.json"


rule fit_tde_hmm:
    input:
        manifest=GLHMM_INPUT_MANIFEST,
    output:
        model_selection=GLHMM_MODEL_SELECTION_OUTPUT,
        chunks=GLHMM_CHUNKS_OUTPUT,
        fit_summary=GLHMM_FIT_SUMMARY_OUTPUT,
    params:
        output_dir=GLHMM_OUTPUT_DIR,
    shell:
        """
        PYTHONPATH="{SRC_DIR}" python -m cas.cli.main fit-tde-hmm \
            --input-manifest "{input.manifest}" \
            --output-dir "{params.output_dir}"
        """
