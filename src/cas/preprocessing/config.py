"""Configuration and path helpers for preprocessing outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

_TEMPLATE_PATTERN = re.compile(r"\{([^{}]+)\}")


@dataclass(frozen=True, slots=True)
class PreprocessingOutputLayout:
    """Resolved preprocessing output roots."""

    output_dir: Path
    preprocessing_root: Path
    final_dir: Path
    intermediate_dir: Path
    tables_dir: Path
    qc_dir: Path
    save_intermediates: bool


@dataclass(frozen=True, slots=True)
class PreprocessingRunPaths:
    """Resolved output paths for one preprocessing run."""

    eeg_path: Path
    emg_path: Path
    events_path: Path
    summary_path: Path
    intermediates_dir: Path | None


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return payload


def _expand_template(value: str, context: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        return context.get(token, match.group(0))

    expanded = value
    for _ in range(8):
        updated = _TEMPLATE_PATTERN.sub(replace, expanded)
        if updated == expanded:
            return updated
        expanded = updated
    return expanded


def _coerce_path(path_text: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def resolve_output_dir(paths_config: Mapping[str, Any], *, base_dir: Path) -> Path:
    configured = paths_config.get("output_dir", paths_config.get("derivatives_root"))
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError("`output_dir` or `derivatives_root` must be configured in paths.yaml.")
    return _coerce_path(configured, base_dir=base_dir)


def resolve_paths_map(paths_config: Mapping[str, Any], *, base_dir: Path) -> dict[str, Path]:
    output_dir = resolve_output_dir(paths_config, base_dir=base_dir)
    raw_paths = dict(paths_config.get("paths") or {})
    resolved: dict[str, str] = {}
    context = {"output_dir": str(output_dir)}

    for _ in range(max(len(raw_paths), 1) + 2):
        changed = False
        current_context = dict(context)
        current_context.update({f"paths.{key}": value for key, value in resolved.items()})
        for key, raw_value in raw_paths.items():
            if not isinstance(raw_value, str):
                continue
            expanded = _expand_template(raw_value, current_context)
            normalized = str(_coerce_path(expanded, base_dir=base_dir))
            if resolved.get(key) != normalized:
                resolved[key] = normalized
                changed = True
        if not changed:
            break

    return {key: Path(value) for key, value in resolved.items()}


def resolve_preprocessing_output_layout(
    paths_config: Mapping[str, Any],
    preprocessing_config: Mapping[str, Any],
    *,
    base_dir: Path,
) -> PreprocessingOutputLayout:
    output_dir = resolve_output_dir(paths_config, base_dir=base_dir)
    resolved_paths = resolve_paths_map(paths_config, base_dir=base_dir)
    preprocessing_root = resolved_paths.get("preprocessing", output_dir / "preprocessing")
    preprocessing_settings = dict(preprocessing_config.get("preprocessing") or {})

    context = {
        "output_dir": str(output_dir),
        "paths.preprocessing": str(preprocessing_root),
    }

    final_dir_value = str(preprocessing_settings.get("final_dir", "{paths.preprocessing}/eeg"))
    intermediate_dir_value = str(
        preprocessing_settings.get("intermediate_dir", "{paths.preprocessing}/eeg_verbose")
    )
    tables_dir_value = str(preprocessing_settings.get("tables_dir", "{paths.preprocessing}/tables"))
    qc_dir_value = str(preprocessing_settings.get("qc_dir", "{paths.preprocessing}/qc"))

    return PreprocessingOutputLayout(
        output_dir=output_dir,
        preprocessing_root=preprocessing_root,
        final_dir=_coerce_path(_expand_template(final_dir_value, context), base_dir=base_dir),
        intermediate_dir=_coerce_path(_expand_template(intermediate_dir_value, context), base_dir=base_dir),
        tables_dir=_coerce_path(_expand_template(tables_dir_value, context), base_dir=base_dir),
        qc_dir=_coerce_path(_expand_template(qc_dir_value, context), base_dir=base_dir),
        save_intermediates=bool(preprocessing_settings.get("save_intermediates", False)),
    )


def load_preprocessing_output_layout(
    *,
    preprocessing_config_path: str | Path,
    paths_config_path: str | Path,
) -> PreprocessingOutputLayout:
    resolved_preprocessing_config_path = Path(preprocessing_config_path).resolve()
    resolved_paths_config_path = Path(paths_config_path).resolve()
    base_dir = resolved_paths_config_path.parent.parent
    return resolve_preprocessing_output_layout(
        _load_yaml_mapping(resolved_paths_config_path),
        _load_yaml_mapping(resolved_preprocessing_config_path),
        base_dir=base_dir,
    )


def build_preprocessing_run_paths(
    *,
    layout: PreprocessingOutputLayout,
    subject: str,
    task: str,
    run: str,
    dyad_id: str | None = None,
) -> PreprocessingRunPaths:
    subject_token = subject if str(subject).startswith("sub-") else f"sub-{subject}"
    dyad_token = dyad_id or "dyad-unknown"
    run_token = str(run)
    task_token = str(task)
    run_dir = layout.final_dir / subject_token / f"task-{task_token}" / f"run-{run_token}"
    run_stem = f"{subject_token}_task-{task_token}_run-{run_token}"
    intermediates_dir = (
        layout.intermediate_dir / dyad_token / subject_token / f"run-{run_token}"
        if layout.save_intermediates
        else None
    )
    return PreprocessingRunPaths(
        eeg_path=run_dir / "preprocessed_eeg.fif",
        emg_path=run_dir / "emg.npz",
        events_path=layout.tables_dir / "by_run" / f"{run_stem}_events.tsv",
        summary_path=layout.qc_dir / "by_run" / f"{run_stem}_summary.json",
        intermediates_dir=intermediates_dir,
    )
