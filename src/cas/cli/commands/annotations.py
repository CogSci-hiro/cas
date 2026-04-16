"""CLI commands for annotation validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.annotations.io import discover_textgrid_files, write_textgrid
from cas.annotations.models import ValidationConfig
from cas.annotations.report import collect_issues, summarize_results, write_csv_report
from cas.annotations.validation import validate_textgrids
from cas.events import ExtractionConfig, extract_events_from_paths, write_events_csv, write_pairing_issues_csv
from cas.events.report import summarize_extraction


def add_annotations_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the `annotations` command tree."""

    annotations_parser = subparsers.add_parser(
        "annotations",
        help="Validate conversational annotation TextGrid files.",
    )
    annotation_subparsers = annotations_parser.add_subparsers(dest="annotations_command", required=True)

    validate_parser = annotation_subparsers.add_parser(
        "validate",
        help="Validate TextGrid tiers, labels, and timing consistency.",
    )
    validate_parser.add_argument("input_dir", help="Directory containing TextGrid files to scan.")
    validate_parser.add_argument("output_dir", help="Directory where CSV logs will be written.")
    validate_parser.add_argument(
        "--write-corrected",
        action="store_true",
        help="Write corrected TextGrids under <out_dir>/corrected without modifying originals.",
    )
    validate_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories for TextGrid files.",
    )
    validate_parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return a non-zero exit code if any warning is found.",
    )
    validate_parser.add_argument(
        "--snap-tolerance-ms",
        type=float,
        default=0.0,
        help="Optional tolerance in milliseconds for tiny boundary snapping.",
    )
    validate_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file validation progress.",
    )

    extract_parser = annotation_subparsers.add_parser(
        "extract-events",
        help="Extract deterministic FPP-SPP event pairs into a canonical CSV.",
    )
    extract_parser.add_argument("input_path", help="TextGrid file or directory to scan.")
    extract_parser.add_argument("events_csv", help="Output CSV path for canonical paired events.")
    extract_parser.add_argument(
        "--output-pairing-issues-csv",
        default=None,
        help="Optional CSV path for extraction issues and skipped candidates.",
    )
    extract_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories for TextGrid files.",
    )
    extract_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code if any extraction issue is recorded.",
    )
    extract_parser.add_argument(
        "--allow-same-speaker-spp",
        action="store_true",
        help="Allow pairing an FPP with an SPP from the same speaker.",
    )
    extract_parser.add_argument(
        "--pairing-margin-s",
        type=float,
        default=1.0,
        help="Symmetric search window in seconds around the FPP offset for SPP onset matching.",
    )
    extract_parser.add_argument(
        "--metadata-regex",
        action="append",
        dest="metadata_regexes",
        default=None,
        help="Regex with named groups `recording_id` and optional `run` used for filename metadata extraction.",
    )
    extract_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file extraction progress.",
    )


def run_annotations_command(args: argparse.Namespace) -> int:
    """Dispatch annotation-related subcommands."""

    if args.annotations_command == "validate":
        return run_annotations_validate(args)
    if args.annotations_command == "extract-events":
        return run_annotations_extract_events(args)
    raise ValueError(f"Unknown annotations subcommand: {args.annotations_command}")


def run_annotations_validate(args: argparse.Namespace) -> int:
    """Validate one or many TextGrid files and write reports."""

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    corrected_output_dir = output_dir / "corrected" if args.write_corrected else None

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ValidationConfig(
        write_corrected=bool(args.write_corrected),
        corrected_output_dir=corrected_output_dir,
        fail_on_warning=bool(args.fail_on_warning),
        snap_tolerance_ms=float(args.snap_tolerance_ms),
    )

    textgrid_paths = discover_textgrid_files(input_dir, recursive=bool(args.recursive))
    if not textgrid_paths:
        raise FileNotFoundError(f"No TextGrid files found under: {input_dir}")

    if args.verbose:
        for path in textgrid_paths:
            print(f"Checking {path}")

    results = validate_textgrids(textgrid_paths, config)
    combined_report_path = output_dir / "validation_summary.csv"
    write_csv_report(collect_issues(results), combined_report_path)

    for result in results:
        file_report_path = _file_report_destination(
            input_dir=input_dir,
            file_path=result.file_path,
            output_dir=output_dir,
        )
        write_csv_report(result.issues, file_report_path)

    if config.write_corrected and config.corrected_output_dir is not None:
        for result in results:
            destination_path = _corrected_destination(
                input_dir=input_dir,
                file_path=result.file_path,
                corrected_output_dir=config.corrected_output_dir,
            )
            write_textgrid(result.corrected_textgrid, destination_path)

    summary = summarize_results(results)
    print(
        "Checked {files_checked} file(s): {errors} error(s), {warnings} warning(s), "
        "{auto_corrections} auto-correction(s).".format(**summary)
    )
    print(f"CSV logs directory: {output_dir}")
    print(f"Combined summary CSV: {combined_report_path}")
    if config.write_corrected and corrected_output_dir is not None:
        print(f"Corrected TextGrids: {corrected_output_dir}")

    if summary["errors"] > 0:
        return 1
    if config.fail_on_warning and summary["warnings"] > 0:
        return 1
    return 0


def run_annotations_extract_events(args: argparse.Namespace) -> int:
    """Extract deterministic FPP-SPP events from one or more TextGrids."""

    input_path = Path(args.input_path).resolve()
    events_csv_path = Path(args.events_csv).resolve()
    pairing_issues_csv_path = (
        Path(args.output_pairing_issues_csv).resolve()
        if args.output_pairing_issues_csv is not None
        else None
    )

    textgrid_paths = discover_textgrid_files(input_path, recursive=bool(args.recursive))
    if not textgrid_paths:
        raise FileNotFoundError(f"No TextGrid files found under: {input_path}")

    if args.verbose:
        for path in textgrid_paths:
            print(f"Extracting {path}")

    config = ExtractionConfig(
        allow_same_speaker_spp=bool(args.allow_same_speaker_spp),
        pairing_margin_s=float(args.pairing_margin_s),
        metadata_regexes=tuple(args.metadata_regexes) if args.metadata_regexes else ExtractionConfig().metadata_regexes,
        strict=bool(args.strict),
    )

    result = extract_events_from_paths(textgrid_paths, config)
    write_events_csv(result.events, events_csv_path)
    if pairing_issues_csv_path is not None:
        write_pairing_issues_csv(result.issues, pairing_issues_csv_path)

    summary = summarize_extraction(result)
    print(
        "Processed {files_processed} file(s): {fpp_count} FPP(s), {paired_events} paired event(s), "
        "{unpaired_fpp} unpaired FPP(s), {unused_spp} unused SPP(s).".format(**summary)
    )
    print(f"Events CSV: {events_csv_path}")
    if pairing_issues_csv_path is not None:
        print(f"Pairing issues CSV: {pairing_issues_csv_path}")

    if config.strict and result.issues:
        return 1
    return 0


def _corrected_destination(input_dir: Path, file_path: Path, corrected_output_dir: Path) -> Path:
    """Resolve the output path for a corrected TextGrid."""

    relative_path = file_path.relative_to(input_dir)
    return corrected_output_dir / relative_path


def _file_report_destination(input_dir: Path, file_path: Path, output_dir: Path) -> Path:
    """Resolve the CSV log path for a validated TextGrid."""

    relative_path = file_path.relative_to(input_dir)
    csv_name = f"{relative_path.name}.csv"
    return output_dir / relative_path.parent / csv_name
