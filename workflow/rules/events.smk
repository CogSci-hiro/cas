EVENTS_OUTPUT_DIR = f"{OUT_DIR}/events"
EVENTS_CSV_OUTPUT = f"{EVENTS_OUTPUT_DIR}/events.csv"
PAIRING_ISSUES_CSV_OUTPUT = f"{EVENTS_OUTPUT_DIR}/pairing_issues.csv"


rule extract_events_csv:
    input:
        annotations_dir=ANNOTATIONS_DIR,
    output:
        events_csv=EVENTS_CSV_OUTPUT,
        pairing_issues_csv=PAIRING_ISSUES_CSV_OUTPUT,
    run:
        shell(
            'PYTHONPATH="{src_dir}" python -m cas.cli.main annotations extract-events '
            '"{annotations_dir}" '
            '"{events_csv}" '
            '--output-pairing-issues-csv "{pairing_issues_csv}" '
            '--pairing-margin-s {pairing_margin_s} '
            '--recursive'.format(
                src_dir=SRC_DIR,
                annotations_dir=input.annotations_dir,
                events_csv=output.events_csv,
                pairing_issues_csv=output.pairing_issues_csv,
                pairing_margin_s=EVENTS_MATCHING_MARGIN_S,
            )
        )


rule events_all:
    input:
        EVENTS_CSV_OUTPUT,
        PAIRING_ISSUES_CSV_OUTPUT
