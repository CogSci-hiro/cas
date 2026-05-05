"""Configuration loading for the behavioral hazard pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def _resolve_path(base: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (base / candidate).resolve()


def _expand_template(value: str, mapping: dict[str, str]) -> str:
    expanded = str(value)
    for key, replacement in mapping.items():
        expanded = expanded.replace("{" + key + "}", replacement)
    return expanded


@dataclass(frozen=True, slots=True)
class BehaviorHazardPaths:
    output_dir: Path
    behavior_root: Path
    hazard_root: Path
    figures_main_behavior: Path
    figures_supp_behavior: Path
    figures_qc_behavior: Path


@dataclass(frozen=True, slots=True)
class BehaviorHazardConfig:
    path: Path
    raw: dict[str, Any]
    paths_config_path: Path
    paths_config: dict[str, Any]
    paths: BehaviorHazardPaths

    @property
    def inputs(self) -> dict[str, Any]:
        return dict(self.raw.get("inputs") or {})

    @property
    def behavior(self) -> dict[str, Any]:
        return dict(self.raw.get("behavior") or {})

    @property
    def behavior_root(self) -> dict[str, Any]:
        return self.behavior

    @property
    def hazard(self) -> dict[str, Any]:
        return dict(self.behavior_root.get("hazard") or {})

    @property
    def bin_size_ms(self) -> int:
        return int(self.hazard.get("bin_size_ms", 50))

    @property
    def candidate_lags_ms(self) -> list[int]:
        values = self.hazard.get("candidate_lags_ms") or self.behavior_root.get("candidate_lags_ms") or [0, 50, 100, 150, 200, 250, 300, 400, 500]
        return [
            int(value)
            for value in list(values)
        ]

    @property
    def controls(self) -> list[str]:
        return [str(value) for value in list(self.hazard.get("controls") or [])]

    @property
    def standardization(self) -> dict[str, Any]:
        return dict(self.hazard.get("standardization") or {})

    @property
    def diagnostics(self) -> dict[str, Any]:
        return dict(self.hazard.get("diagnostics") or {})

    @property
    def figures(self) -> dict[str, Any]:
        return dict(self.hazard.get("figures") or {})

    @property
    def models(self) -> dict[str, Any]:
        return dict(self.hazard.get("models") or {})

    @property
    def lag_selection(self) -> dict[str, Any]:
        return dict(self.hazard.get("lag_selection") or {})

    @property
    def modeling(self) -> dict[str, Any]:
        return dict(self.hazard.get("modeling") or {})

    @property
    def model_backend(self) -> str:
        raw_backend = str(
            self.behavior_root.get("model_backend")
            or self.modeling.get("model_backend")
            or self.modeling.get("backend")
            or "glm"
        ).strip()
        normalized = raw_backend.lower()
        if normalized in {"glmm", "r_glmmtmb", "r_glmmtmb_binomial_mixed"}:
            return "glmm"
        if normalized in {"glm", "fixed", "fixed_effect", "fixed_effects"}:
            return "glm"
        raise ValueError(
            f"Unsupported behavior.model_backend / behavior.hazard.modeling.model_backend/backend value: {raw_backend!r}"
        )

    @property
    def lag_selection_criterion(self) -> str:
        raw_value = str(
            self.behavior_root.get("lag_selection_criterion")
            or self.lag_selection.get("criterion")
            or "bic"
        ).strip()
        normalized = raw_value.lower()
        if normalized in {"bic", "aic_bic_bic", "min_bic"}:
            return "bic"
        if normalized in {"log_likelihood", "loglik", "ll", "max_loglik"}:
            return "log_likelihood"
        raise ValueError(f"Unsupported behavior.lag_selection_criterion value: {raw_value!r}")

    @property
    def glm_covariance(self) -> str:
        value = str(self.modeling.get("glm_covariance") or "model_based").strip().lower()
        if value not in {"model_based", "cluster_robust"}:
            raise ValueError(f"Unsupported behavior.hazard.modeling.glm_covariance value: {value!r}")
        return value

    @property
    def glm_cluster(self) -> str | None:
        raw_value = self.modeling.get("glm_cluster")
        if raw_value is None:
            return "subject" if self.glm_covariance == "cluster_robust" else None
        value = str(raw_value).strip()
        if not value:
            return None
        if value not in {"subject", "dyad_id"}:
            raise ValueError(f"Unsupported behavior.hazard.modeling.glm_cluster value: {value!r}")
        return value

    def model_group_enabled(self, name: str) -> bool:
        return bool((self.models.get(name) or {}).get("enabled", False))

    def diagnostics_enabled(self, name: str) -> bool:
        return bool((self.diagnostics.get(name)))

    def figure_enabled(self, section: str, name: str) -> bool:
        section_config = dict((self.figures.get(section) or {}))
        return bool(section_config.get(name, False))


def load_behavior_hazard_config(path: str | Path) -> BehaviorHazardConfig:
    config_path = Path(path).resolve()
    raw = _load_yaml(config_path)
    paths_config_value = str(raw.get("paths_config") or "config/paths.yaml")
    paths_config_path = _resolve_path(config_path.parent, paths_config_value)
    paths_config = _load_yaml(paths_config_path)
    output_dir_value = str(paths_config.get("output_dir") or paths_config.get("derivatives_root") or "").strip()
    if not output_dir_value:
        raise ValueError(f"`output_dir` is missing from {paths_config_path}")
    output_dir = Path(output_dir_value).resolve()

    path_mapping = {
        "output_dir": str(output_dir),
        "behavior": str(output_dir / "behavior"),
        "figures_main": str(output_dir / "figures" / "main"),
        "figures_supp": str(output_dir / "figures" / "supp"),
        "figures_qc": str(output_dir / "figures" / "qc"),
    }
    structured_paths = dict(paths_config.get("paths") or {})
    behavior_root = Path(_expand_template(str(structured_paths.get("behavior") or "{output_dir}/behavior"), path_mapping)).resolve()
    figures_main = Path(_expand_template(str(structured_paths.get("figures_main") or "{output_dir}/figures/main"), path_mapping)).resolve()
    figures_supp = Path(_expand_template(str(structured_paths.get("figures_supp") or "{output_dir}/figures/supp"), path_mapping)).resolve()
    figures_qc = Path(_expand_template(str(structured_paths.get("figures_qc") or "{output_dir}/figures/qc"), path_mapping)).resolve()
    return BehaviorHazardConfig(
        path=config_path,
        raw=raw,
        paths_config_path=paths_config_path,
        paths_config=paths_config,
        paths=BehaviorHazardPaths(
            output_dir=output_dir,
            behavior_root=behavior_root,
            hazard_root=behavior_root / "hazard",
            figures_main_behavior=figures_main / "behavior",
            figures_supp_behavior=figures_supp / "behavior",
            figures_qc_behavior=figures_qc / "behavior",
        ),
    )
