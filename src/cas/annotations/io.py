"""TextGrid discovery, parsing, and writing utilities."""

from __future__ import annotations

from pathlib import Path
import re

from cas.annotations.constants import TEXTGRID_SUFFIXES
from cas.annotations.models import Interval, TextGrid, Tier

_XMIN_RE = re.compile(r"^\s*xmin\s*=\s*([-+0-9.eE]+)\s*$")
_XMAX_RE = re.compile(r"^\s*xmax\s*=\s*([-+0-9.eE]+)\s*$")
_SIZE_RE = re.compile(r"^\s*(?:size|intervals:\s*size)\s*=\s*(\d+)\s*$")
_CLASS_RE = re.compile(r'^\s*class\s*=\s*"([^"]+)"\s*$')
_NAME_RE = re.compile(r'^\s*name\s*=\s*"((?:[^"]|"")*)"\s*$')
_TEXT_RE = re.compile(r'^\s*text\s*=\s*"((?:[^"]|"")*)"\s*$')
_ITEM_RE = re.compile(r"^\s*item\s*\[\d+\]:\s*$")
_INTERVAL_RE = re.compile(r"^\s*intervals\s*\[\d+\]:\s*$")


def discover_textgrid_files(input_path: Path, *, recursive: bool) -> list[Path]:
    """Discover TextGrid files from a file or directory."""

    resolved_path = input_path.resolve()
    if resolved_path.is_file():
        if resolved_path.suffix.lower() not in TEXTGRID_SUFFIXES:
            return []
        return [resolved_path]

    if not resolved_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved_path}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        path.resolve()
        for path in resolved_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in TEXTGRID_SUFFIXES
    )


def load_textgrid(path: Path) -> TextGrid:
    """Load a TextGrid from disk."""

    return parse_textgrid(_read_textgrid_text(path))


def parse_textgrid(content: str) -> TextGrid:
    """Parse a standard long-text Praat TextGrid with interval tiers."""

    lines = content.splitlines()
    xmin, xmax = _parse_global_bounds(lines)
    tiers: list[Tier] = []

    line_index = 0
    while line_index < len(lines):
        if not _ITEM_RE.match(lines[line_index]):
            line_index += 1
            continue

        tier, line_index = _parse_tier(lines, line_index + 1)
        tiers.append(tier)

    return TextGrid(xmin=xmin, xmax=xmax, tiers=tiers)


def write_textgrid(textgrid: TextGrid, path: Path) -> None:
    """Write a TextGrid in standard long-text Praat format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {textgrid.xmin:.15g}",
        f"xmax = {textgrid.xmax:.15g}",
        "tiers? <exists>",
        f"size = {len(textgrid.tiers)}",
        "item []:",
    ]

    for tier_index, tier in enumerate(textgrid.tiers, start=1):
        lines.extend(
            [
                f"    item [{tier_index}]:",
                f'        class = "{_escape_text(tier.class_name)}"',
                f'        name = "{_escape_text(tier.name)}"',
                f"        xmin = {tier.xmin:.15g}",
                f"        xmax = {tier.xmax:.15g}",
                f"        intervals: size = {len(tier.intervals)}",
            ]
        )
        for interval_index, interval in enumerate(tier.intervals, start=1):
            lines.extend(
                [
                    f"        intervals [{interval_index}]:",
                    f"            xmin = {interval.xmin:.15g}",
                    f"            xmax = {interval.xmax:.15g}",
                    f'            text = "{_escape_text(interval.text)}"',
                ]
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_global_bounds(lines: list[str]) -> tuple[float, float]:
    """Parse TextGrid-level xmin/xmax from the header section."""

    xmin: float | None = None
    xmax: float | None = None
    seen_item = False

    for line in lines:
        if _ITEM_RE.match(line):
            seen_item = True
        if seen_item:
            break
        xmin_match = _XMIN_RE.match(line)
        if xmin_match and xmin is None:
            xmin = float(xmin_match.group(1))
            continue
        xmax_match = _XMAX_RE.match(line)
        if xmax_match and xmax is None:
            xmax = float(xmax_match.group(1))

    if xmin is None or xmax is None:
        raise ValueError("Could not parse TextGrid xmin/xmax header.")
    return xmin, xmax


def _parse_tier(lines: list[str], start_index: int) -> tuple[Tier, int]:
    """Parse a single tier block."""

    class_name = "IntervalTier"
    tier_name = ""
    xmin: float | None = None
    xmax: float | None = None
    intervals: list[Interval] = []

    index = start_index
    while index < len(lines):
        line = lines[index]
        if _ITEM_RE.match(line):
            break
        class_match = _CLASS_RE.match(line)
        if class_match:
            class_name = _unescape_text(class_match.group(1))
            index += 1
            continue
        name_match = _NAME_RE.match(line)
        if name_match:
            tier_name = _unescape_text(name_match.group(1))
            index += 1
            continue
        xmin_match = _XMIN_RE.match(line)
        if xmin_match and xmin is None:
            xmin = float(xmin_match.group(1))
            index += 1
            continue
        xmax_match = _XMAX_RE.match(line)
        if xmax_match and xmax is None:
            xmax = float(xmax_match.group(1))
            index += 1
            continue
        if _INTERVAL_RE.match(line):
            interval, index = _parse_interval(lines, index + 1)
            intervals.append(interval)
            continue
        index += 1

    if xmin is None or xmax is None:
        raise ValueError(f"Tier is missing xmin/xmax: {tier_name or '<unnamed>'}")
    return Tier(name=tier_name, xmin=xmin, xmax=xmax, intervals=intervals, class_name=class_name), index


def _parse_interval(lines: list[str], start_index: int) -> tuple[Interval, int]:
    """Parse a single interval block."""

    xmin: float | None = None
    xmax: float | None = None
    text = ""

    index = start_index
    while index < len(lines):
        line = lines[index]
        if _INTERVAL_RE.match(line) or _ITEM_RE.match(line):
            break
        xmin_match = _XMIN_RE.match(line)
        if xmin_match and xmin is None:
            xmin = float(xmin_match.group(1))
            index += 1
            continue
        xmax_match = _XMAX_RE.match(line)
        if xmax_match and xmax is None:
            xmax = float(xmax_match.group(1))
            index += 1
            continue
        text_match = _TEXT_RE.match(line)
        if text_match:
            text = _unescape_text(text_match.group(1))
            index += 1
            continue
        index += 1

    if xmin is None or xmax is None:
        raise ValueError("Interval is missing xmin/xmax.")
    return Interval(xmin=xmin, xmax=xmax, text=text), index


def _escape_text(value: str) -> str:
    """Escape text for TextGrid output."""

    return value.replace('"', '""')


def _unescape_text(value: str) -> str:
    """Unescape doubled quotes from TextGrid text."""

    return value.replace('""', '"')


def _read_textgrid_text(path: Path) -> str:
    """Read a TextGrid using a conservative encoding fallback strategy."""

    encodings = ("utf-8", "utf-16", "utf-16-le", "utf-16-be")
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    if last_error is not None:
        raise last_error
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Could not decode TextGrid file: {path}")
