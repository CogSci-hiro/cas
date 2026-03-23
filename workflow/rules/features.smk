rule extract_envelope:
    input:
        AUDIO_PATTERN
    output:
        ENVELOPE_OUTPUT_PATTERN
    params:
        src_dir=SRC_DIR,
        lowpass_hz=ENVELOPE_LOWPASS_HZ,
        filter_order=ENVELOPE_FILTER_ORDER
    shell:
        """
        PYTHONPATH="{params.src_dir}" python -m cas.cli.main envelope \
            --input "{input}" \
            --output "{output}" \
            --lowpass-hz {params.lowpass_hz} \
            --filter-order {params.filter_order}
        """
