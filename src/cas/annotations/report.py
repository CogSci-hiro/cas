"""CSV reporting helpers for annotation validation results."""

from __future__ import annotations

import csv
from pathlib import Path

from cas.annotations.constants import CSV_COLUMNS
from cas.annotations.models import FileValidationResult, ValidationIssue


def collect_issues(results: list[FileValidationResult]) -> list[ValidationIssue]:
    """Collect issues across validation results."""

    return [issue for result in results for issue in result.issues]


def summarize_results(results: list[FileValidationResult]) -> dict[str, int]:
    """Compute aggregate summary counts."""

    return {
        "files_checked": len(results),
        "errors": sum(result.error_count for result in results),
        "warnings": sum(result.warning_count for result in results),
        "auto_corrections": sum(result.correction_count for result in results),
    }


def write_csv_report(issues: list[ValidationIssue], output_path: Path) -> None:
    """Write validation issues to a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue.to_csv_row())
