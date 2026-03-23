rule extract_envelope:
    input:
        AUDIO_PATTERN,
    output:
        f"{FEATURES_ROOT}/envelope/sub-{{subject}}/sub-{{subject}}_task-{{task}}_run-{{run}}_envelope.npy",
    run:
        shell(
            'PYTHONPATH="{src_dir}" python -m cas.cli.main envelope '
            '--input "{input_file}" '
            '--output "{output_file}" '
            '--lowpass-hz {lowpass_hz} '
            '--filter-order {filter_order}'.format(
                src_dir=SRC_DIR,
                input_file=input[0],
                output_file=output[0],
                lowpass_hz=ENVELOPE_LOWPASS_HZ,
                filter_order=ENVELOPE_FILTER_ORDER,
            )
        )
