rule extract_all_envelopes:
    input:
        expand(
            AUDIO_PATTERN,
            zip,
            subject=AUDIO_RECORD_SUBJECTS,
            task=AUDIO_RECORD_TASKS,
            run=AUDIO_RECORD_RUNS,
        )
    output:
        "/Volumes/work-4T/speech-rate-testing/features/envelope/.extract_all_complete"
    script:
        "../scripts/extract_all_envelopes.py"
