rule extract_acoustic_features:
    input:
        audio=AUDIO_PATTERN,
        config=ACOUSTIC_CONFIG_PATH,
    output:
        envelope=ENVELOPE_OUTPUT_PATTERN,
        envelope_summary=ENVELOPE_SUMMARY_OUTPUT_PATTERN,
        f0=F0_OUTPUT_PATTERN,
        f0_summary=F0_SUMMARY_OUTPUT_PATTERN,
    shell:
        """
        PYTHONPATH="{SRC_DIR}" python -m cas.cli.main acoustic-features \
            --input "{input.audio}" \
            --config "{input.config}" \
            --envelope-output "{output.envelope}" \
            --f0-output "{output.f0}" \
            --envelope-summary-json "{output.envelope_summary}" \
            --f0-summary-json "{output.f0_summary}"
        """
