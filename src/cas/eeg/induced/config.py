from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return payload


def discover_config_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "paths.yaml").exists():
            return candidate
    return start


def load_paths_config(config_root: Path) -> dict[str, Any]:
    resolved_root = discover_config_root(config_root)
    paths_path = resolved_root / "paths.yaml"
    if not paths_path.exists():
        raise FileNotFoundError(f"Paths config not found: {paths_path}")
    return _load_yaml(paths_path)


def resolve_out_dir(config_root: Path) -> Path:
    paths_config = load_paths_config(config_root)
    candidates = (
        paths_config.get("output_dir"),
        ((paths_config.get("io") or {}).get("out_dir") if isinstance(paths_config.get("io"), dict) else None),
        ((paths_config.get("paths") or {}).get("out_dir") if isinstance(paths_config.get("paths"), dict) else None),
        paths_config.get("derivatives_root"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return Path(candidate).resolve()
    raise ValueError(
        f"`output_dir`, `io.out_dir`, `paths.out_dir`, or `derivatives_root` must be configured in "
        f"{discover_config_root(config_root) / 'paths.yaml'}."
    )


def resolve_induced_lmeeeg_config_path(*, config_root: Path, explicit_config_path: str | None) -> Path:
    if explicit_config_path:
        return Path(explicit_config_path).resolve()
    return discover_config_root(config_root) / "induced" / "alpha_beta_lmeeeg.yaml"


def resolve_sensor_figure_manifest_path(
    *,
    out_dir: Path,
    config_path: Path,
    figure_kind: str,
) -> Path:
    if figure_kind not in {"figures", "qc"}:
        raise ValueError(f"Unsupported induced figure kind: {figure_kind}")

    config_name = config_path.name
    if figure_kind == "qc":
        base_dir = out_dir / "figures" / "qc" / "induced" / "sensor_lmeeeg"
        if config_name == "alpha_beta_conf_disc.yaml":
            return base_dir / "conf_disc" / "figure_manifest.json"
        if config_name == "alpha_beta_cycle_position.yaml":
            return base_dir / "cycle_position" / "figure_manifest.json"
        return base_dir / "figure_manifest.json"

    if config_name == "alpha_beta_conf_disc.yaml":
        return out_dir / "figures" / "supp" / "induced" / "sensor_lmeeeg" / "conf_disc" / "figure_manifest.json"
    if config_name == "alpha_beta_cycle_position.yaml":
        return out_dir / "figures" / "supp" / "induced" / "sensor_lmeeeg" / "cycle_position" / "figure_manifest.json"
    return out_dir / "figures" / "main" / "induced" / "sensor_lmeeeg" / "figure_manifest.json"
