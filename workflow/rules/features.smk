rule extract_acoustic_envelope:
    input:
        audio=AUDIO_PATTERN,
        config=ACOUSTIC_CONFIG_PATH,
    output:
        envelope=ENVELOPE_OUTPUT_PATTERN,
        envelope_summary=ENVELOPE_SUMMARY_OUTPUT_PATTERN,
    shell:
        """
        PYTHONPATH="{SRC_DIR}" python -m cas.cli.main acoustic-envelope \
            --input "{input.audio}" \
            --config "{input.config}" \
            --output "{output.envelope}" \
            --summary-json "{output.envelope_summary}"
        """


rule extract_acoustic_f0:
    input:
        audio=AUDIO_PATTERN,
        config=ACOUSTIC_CONFIG_PATH,
    output:
        f0=F0_OUTPUT_PATTERN,
        f0_summary=F0_SUMMARY_OUTPUT_PATTERN,
    shell:
        """
        PYTHONPATH="{SRC_DIR}" python -m cas.cli.main acoustic-f0 \
            --input "{input.audio}" \
            --config "{input.config}" \
            --output "{output.f0}" \
            --summary-json "{output.f0_summary}"
        """
