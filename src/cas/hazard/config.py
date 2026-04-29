"""Configuration models for the partner-onset hazard analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

DEFAULT_DYADS_CSV_PATH = "dyads.csv"
DEFAULT_PAIRING_ISSUES_FILENAME = "pairing_issues.csv"


@dataclass(frozen=True, slots=True)
class InputConfig:
    """Input paths for the hazard analysis.

    Parameters
    ----------
    tde_hmm_results_dir
        Directory containing existing TDE-HMM outputs.
    events_table_path
        Path to the canonical paired events CSV.
    pairing_issues_table_path
        Optional path to the canonical pairing-issues CSV. When omitted, the
        analysis looks for ``pairing_issues.csv`` next to ``events_table_path``.
    dyads_csv_path
        Optional dyad mapping CSV used to map dyad speaker labels to subject
        IDs. Missing rows fall back to the repository's odd/even dyad
        convention.
    """

    tde_hmm_results_dir: Path
    events_table_path: Path
    pairing_issues_table_path: Path | None = None
    dyads_csv_path: Path | None = None


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output paths for the hazard analysis."""

    output_dir: Path


@dataclass(frozen=True, slots=True)
class EventDefinitionConfig:
    """Event-table column settings and label filters."""

    partner_onset_column: str
    target_onset_column: str
    target_label_column: str
    partner_label_column: str
    fpp_label_prefixes: tuple[str, ...]
    event_id_column: str
    recording_id_column: str
    run_column: str
    partner_speaker_column: str
    target_speaker_column: str
    issue_partner_tier_column: str = "fpp_tier"
    issue_code_column: str = "issue_code"
    issue_partner_label_column: str = "fpp_label"
    issue_partner_onset_column: str = "fpp_onset"
    issue_partner_offset_column: str = "fpp_offset"
    censoring_issue_codes: tuple[str, ...] = ("no_opposite_spp",)


@dataclass(frozen=True, slots=True)
class TimeAxisConfig:
    """Forward-running hazard clock settings."""

    observation_window_seconds: float
    bin_size_seconds: float
    entropy_lag_seconds: float = 0.15
    exclude_initial_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class EntropyConfig:
    """Entropy feature settings."""

    normalize_by_log_k: bool = True
    epsilon: float = 1.0e-12
    zscore_within_subject: bool = True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Hazard-model fitting settings."""

    fitting_backend: str = "auto"
    include_quadratic_time: bool = True
    prefer_random_intercept_subject: bool = True


@dataclass(frozen=True, slots=True)
class QcConfig:
    """QC artifact toggles."""

    make_entropy_trajectory_plot: bool = True
    make_entropy_histogram: bool = True
    make_predicted_hazard_plot: bool = True
    make_event_rate_by_entropy_quantile_plot: bool = True
    make_observed_hazard_by_time_bin_plot: bool = True
    make_observed_hazard_by_time_bin_smoothed_plot: bool = True
    make_observed_event_rate_by_entropy_quantile_plot: bool = True
    make_observed_hazard_by_time_and_entropy_quantile_plot: bool = True
    make_model_vs_observed_hazard_plot: bool = True
    make_entropy_distribution_terminal_vs_nonterminal_plot: bool = True


@dataclass(frozen=True, slots=True)
class PlottingConfig:
    """Plotting settings for empirical hazard summaries."""

    entropy_quantile_count: int = 5
    entropy_group_count_for_time_plot: int = 3
    min_at_risk_per_bin: int = 10
    smoothing_window_bins: int = 3


@dataclass(frozen=True, slots=True)
class MiscConfig:
    """Miscellaneous analysis settings."""

    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class NeuralInputConfig:
    """Inputs for low-level neural hazard analysis."""

    surprisal_paths: tuple[Path, ...]
    lowlevel_neural_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class NeuralEpisodeConfig:
    """Partner-IPU episode construction options for neural hazard analyses."""

    ipu_gap_threshold_s: float = 0.300
    max_followup_s: float = 6.0
    include_censored: bool = True


@dataclass(frozen=True, slots=True)
class NeuralWindowConfig:
    """Causal neural-window settings."""

    start_lag_s: float = 0.500
    end_lag_s: float = 0.100
    epsilon: float = 1.0e-12


@dataclass(frozen=True, slots=True)
class NeuralPcaConfig:
    """Band-specific PCA configuration."""

    mode: Literal["count", "variance"] = "variance"
    n_components: int = 3
    variance_threshold: float = 0.90


@dataclass(frozen=True, slots=True)
class NeuralModelConfig:
    """Low-level neural model settings."""

    fitting_backend: Literal["python_glm", "glmmTMB"] = "glmmTMB"
    baseline_spline_df: int = 6
    baseline_spline_degree: int = 3
    information_rate_lag_ms: int = 150
    prop_expected_lag_ms: int = 700


@dataclass(frozen=True, slots=True)
class NeuralHazardConfig:
    """Neural low-level partner-IPU hazard analysis configuration."""

    enabled: bool = False
    event_types: tuple[Literal["fpp", "spp"], ...] = ("fpp", "spp")
    bin_size_s: float = 0.050
    events_path: Path | None = None
    out_dir: Path | None = None
    input: NeuralInputConfig | None = None
    episode: NeuralEpisodeConfig = field(default_factory=NeuralEpisodeConfig)
    window: NeuralWindowConfig = field(default_factory=NeuralWindowConfig)
    pca: NeuralPcaConfig = field(default_factory=NeuralPcaConfig)
    model: NeuralModelConfig = field(default_factory=NeuralModelConfig)
    select_neural_lags: bool = False
    neural_lag_grid_ms: tuple[tuple[int, int], ...] = (
        (50, 250),
        (100, 300),
        (100, 500),
        (300, 500),
        (300, 700),
        (500, 900),
    )
    neural_lag_selection_criterion: Literal["bic"] = "bic"
    neural_null_permutations: int = 100
    skip_spp_on_failure: bool = True

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.events_path is None:
            raise ValueError("Neural hazard mode requires `neural.events_path`.")
        if self.out_dir is None:
            raise ValueError("Neural hazard mode requires `neural.out_dir`.")
        if self.input is None:
            raise ValueError("Neural hazard mode requires `neural.input` paths.")
        if not self.input.surprisal_paths:
            raise ValueError("Neural hazard mode requires at least one surprisal file.")
        if not self.input.lowlevel_neural_paths:
            raise ValueError("Neural hazard mode requires at least one low-level neural table.")
        if self.bin_size_s <= 0.0:
            raise ValueError("`neural.bin_size_s` must be positive.")
        if self.episode.max_followup_s <= 0.0:
            raise ValueError("`neural.episode.max_followup_s` must be positive.")
        if self.window.start_lag_s <= self.window.end_lag_s:
            raise ValueError("Neural window requires start_lag_s > end_lag_s.")
        if self.window.end_lag_s < 0.0:
            raise ValueError("`neural.window.end_lag_s` must be non-negative.")
        if self.pca.mode not in {"count", "variance"}:
            raise ValueError("`neural.pca.mode` must be one of count or variance.")
        if self.pca.n_components < 1:
            raise ValueError("`neural.pca.n_components` must be at least 1.")
        if not 0.0 < self.pca.variance_threshold <= 1.0:
            raise ValueError("`neural.pca.variance_threshold` must be in (0, 1].")
        if self.model.baseline_spline_df < 3:
            raise ValueError("`neural.model.baseline_spline_df` must be at least 3.")
        if self.model.baseline_spline_degree < 1:
            raise ValueError("`neural.model.baseline_spline_degree` must be at least 1.")
        if self.model.information_rate_lag_ms < 0 or self.model.prop_expected_lag_ms < 0:
            raise ValueError("Neural behavioural-control lags must be non-negative.")
        if self.neural_lag_selection_criterion != "bic":
            raise ValueError("`neural.neural_lag_selection_criterion` must be `bic`.")
        if self.neural_null_permutations < 0:
            raise ValueError("`neural.neural_null_permutations` must be non-negative.")
        if not self.neural_lag_grid_ms:
            raise ValueError("`neural.neural_lag_grid_ms` must contain at least one lag window.")
        for lag_start_ms, lag_end_ms in self.neural_lag_grid_ms:
            if int(lag_start_ms) <= 0:
                raise ValueError("Neural lag windows must satisfy lag_start_ms > 0.")
            if int(lag_end_ms) <= int(lag_start_ms):
                raise ValueError("Neural lag windows must satisfy lag_end_ms > lag_start_ms.")
        unsupported_events = sorted(set(self.event_types) - {"fpp", "spp"})
        if unsupported_events:
            raise ValueError(f"Unsupported neural event type(s): {unsupported_events}")


@dataclass(frozen=True, slots=True)
class HazardAnalysisConfig:
    """Full partner-onset hazard-analysis configuration."""

    input: InputConfig
    output: OutputConfig
    event_definition: EventDefinitionConfig
    time_axis: TimeAxisConfig
    entropy: EntropyConfig
    model: ModelConfig
    qc: QcConfig
    plotting: PlottingConfig
    misc: MiscConfig
    mode: Literal["legacy_entropy", "neural_lowlevel"] = "legacy_entropy"
    neural: NeuralHazardConfig = field(default_factory=NeuralHazardConfig)

    def validate(self) -> None:
        """Validate the configuration for conservative first-pass use."""

        if self.time_axis.observation_window_seconds <= 0.0:
            raise ValueError("`observation_window_seconds` must be positive.")
        if self.time_axis.bin_size_seconds <= 0.0:
            raise ValueError("`bin_size_seconds` must be positive.")
        if self.time_axis.entropy_lag_seconds < 0.0:
            raise ValueError("`entropy_lag_seconds` must be non-negative.")
        if self.time_axis.exclude_initial_seconds < 0.0:
            raise ValueError("`exclude_initial_seconds` must be non-negative.")
        if self.time_axis.exclude_initial_seconds >= self.time_axis.observation_window_seconds:
            raise ValueError("`exclude_initial_seconds` must be smaller than the observation window.")
        if not self.event_definition.fpp_label_prefixes:
            raise ValueError("At least one FPP label prefix is required.")
        if not self.event_definition.censoring_issue_codes:
            raise ValueError("At least one censoring issue code is required.")
        if self.entropy.epsilon <= 0.0:
            raise ValueError("`entropy.epsilon` must be positive.")
        if self.model.fitting_backend not in {"auto", "gee", "glm_cluster"}:
            raise ValueError("`model.fitting_backend` must be one of auto, gee, glm_cluster.")
        if self.plotting.entropy_quantile_count < 2:
            raise ValueError("`entropy_quantile_count` must be at least 2.")
        if self.plotting.entropy_group_count_for_time_plot < 2:
            raise ValueError("`entropy_group_count_for_time_plot` must be at least 2.")
        if self.plotting.min_at_risk_per_bin < 1:
            raise ValueError("`min_at_risk_per_bin` must be at least 1.")
        if self.plotting.smoothing_window_bins < 1:
            raise ValueError("`smoothing_window_bins` must be at least 1.")
        if self.mode not in {"legacy_entropy", "neural_lowlevel"}:
            raise ValueError("`mode` must be one of legacy_entropy or neural_lowlevel.")
        self.neural.validate()


def load_hazard_analysis_config(config_path: str | Path) -> HazardAnalysisConfig:
    """Load and validate a hazard-analysis config from YAML."""

    resolved_config_path = Path(config_path).resolve()
    with resolved_config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {resolved_config_path}.")

    config_root = resolved_config_path.parent
    input_payload = dict(payload.get("input") or {})
    output_payload = dict(payload.get("output") or {})
    event_payload = dict(payload.get("event_definition") or {})
    time_axis_payload = dict(payload.get("time_axis") or {})
    entropy_payload = dict(payload.get("entropy") or {})
    model_payload = dict(payload.get("model") or {})
    qc_payload = dict(payload.get("qc") or {})
    plotting_payload = dict(payload.get("plotting") or {})
    misc_payload = dict(payload.get("misc") or {})
    neural_payload = dict(payload.get("neural") or {})
    neural_input_payload = dict(neural_payload.get("input") or {})
    neural_episode_payload = dict(neural_payload.get("episode") or {})
    neural_window_payload = dict(neural_payload.get("window") or {})
    neural_pca_payload = dict(neural_payload.get("pca") or {})
    neural_model_payload = dict(neural_payload.get("model") or {})

    events_table_path = _resolve_path(input_payload["events_table_path"], config_root=config_root)
    config = HazardAnalysisConfig(
        input=InputConfig(
            tde_hmm_results_dir=_resolve_path(input_payload["tde_hmm_results_dir"], config_root=config_root),
            events_table_path=events_table_path,
            pairing_issues_table_path=(
                _resolve_path(input_payload["pairing_issues_table_path"], config_root=config_root)
                if input_payload.get("pairing_issues_table_path")
                else _default_pairing_issues_path(events_table_path)
            ),
            dyads_csv_path=(
                _resolve_path(input_payload["dyads_csv_path"], config_root=config_root)
                if input_payload.get("dyads_csv_path")
                else _default_dyads_path(config_root)
            ),
        ),
        output=OutputConfig(
            output_dir=_resolve_path(output_payload["output_dir"], config_root=config_root),
        ),
        event_definition=EventDefinitionConfig(
            partner_onset_column=str(event_payload["partner_onset_column"]),
            target_onset_column=str(event_payload["target_onset_column"]),
            target_label_column=str(event_payload["target_label_column"]),
            partner_label_column=str(event_payload.get("partner_label_column", "fpp_label")),
            fpp_label_prefixes=tuple(str(value) for value in event_payload["fpp_label_prefixes"]),
            event_id_column=str(event_payload.get("event_id_column", "pair_id")),
            recording_id_column=str(event_payload.get("recording_id_column", "recording_id")),
            run_column=str(event_payload.get("run_column", "run")),
            partner_speaker_column=str(event_payload.get("partner_speaker_column", "speaker_fpp")),
            target_speaker_column=str(event_payload.get("target_speaker_column", "speaker_spp")),
            issue_partner_tier_column=str(event_payload.get("issue_partner_tier_column", "fpp_tier")),
            issue_code_column=str(event_payload.get("issue_code_column", "issue_code")),
            issue_partner_label_column=str(event_payload.get("issue_partner_label_column", "fpp_label")),
            issue_partner_onset_column=str(event_payload.get("issue_partner_onset_column", "fpp_onset")),
            issue_partner_offset_column=str(event_payload.get("issue_partner_offset_column", "fpp_offset")),
            censoring_issue_codes=tuple(
                str(value) for value in event_payload.get("censoring_issue_codes", ("no_opposite_spp",))
            ),
        ),
        time_axis=TimeAxisConfig(
            observation_window_seconds=float(time_axis_payload["observation_window_seconds"]),
            bin_size_seconds=float(time_axis_payload["bin_size_seconds"]),
            entropy_lag_seconds=float(time_axis_payload.get("entropy_lag_seconds", 0.15)),
            exclude_initial_seconds=float(time_axis_payload.get("exclude_initial_seconds", 0.0)),
        ),
        entropy=EntropyConfig(
            normalize_by_log_k=bool(entropy_payload.get("normalize_by_log_k", True)),
            epsilon=float(entropy_payload.get("epsilon", 1.0e-12)),
            zscore_within_subject=bool(entropy_payload.get("zscore_within_subject", True)),
        ),
        model=ModelConfig(
            fitting_backend=str(model_payload.get("fitting_backend", "auto")),
            include_quadratic_time=bool(model_payload.get("include_quadratic_time", True)),
            prefer_random_intercept_subject=bool(
                model_payload.get("prefer_random_intercept_subject", True)
            ),
        ),
        qc=QcConfig(
            make_entropy_trajectory_plot=bool(qc_payload.get("make_entropy_trajectory_plot", True)),
            make_entropy_histogram=bool(qc_payload.get("make_entropy_histogram", True)),
            make_predicted_hazard_plot=bool(qc_payload.get("make_predicted_hazard_plot", True)),
            make_event_rate_by_entropy_quantile_plot=bool(
                qc_payload.get("make_event_rate_by_entropy_quantile_plot", True)
            ),
            make_observed_hazard_by_time_bin_plot=bool(
                qc_payload.get("make_observed_hazard_by_time_bin_plot", True)
            ),
            make_observed_hazard_by_time_bin_smoothed_plot=bool(
                qc_payload.get("make_observed_hazard_by_time_bin_smoothed_plot", True)
            ),
            make_observed_event_rate_by_entropy_quantile_plot=bool(
                qc_payload.get("make_observed_event_rate_by_entropy_quantile_plot", True)
            ),
            make_observed_hazard_by_time_and_entropy_quantile_plot=bool(
                qc_payload.get("make_observed_hazard_by_time_and_entropy_quantile_plot", True)
            ),
            make_model_vs_observed_hazard_plot=bool(
                qc_payload.get("make_model_vs_observed_hazard_plot", True)
            ),
            make_entropy_distribution_terminal_vs_nonterminal_plot=bool(
                qc_payload.get("make_entropy_distribution_terminal_vs_nonterminal_plot", True)
            ),
        ),
        plotting=PlottingConfig(
            entropy_quantile_count=int(plotting_payload.get("entropy_quantile_count", 5)),
            entropy_group_count_for_time_plot=int(
                plotting_payload.get("entropy_group_count_for_time_plot", 3)
            ),
            min_at_risk_per_bin=int(plotting_payload.get("min_at_risk_per_bin", 10)),
            smoothing_window_bins=int(plotting_payload.get("smoothing_window_bins", 3)),
        ),
        misc=MiscConfig(
            overwrite=bool(misc_payload.get("overwrite", False)),
        ),
        mode=str(payload.get("mode", "legacy_entropy")),
        neural=NeuralHazardConfig(
            enabled=bool(neural_payload.get("enabled", False)),
            event_types=tuple(
                str(value).lower()
                for value in neural_payload.get("event_types", ("fpp", "spp"))
            ),
            bin_size_s=float(neural_payload.get("bin_size_s", 0.050)),
            events_path=(
                _resolve_path(neural_payload["events_path"], config_root=config_root)
                if neural_payload.get("events_path")
                else None
            ),
            out_dir=(
                _resolve_path(neural_payload["out_dir"], config_root=config_root)
                if neural_payload.get("out_dir")
                else None
            ),
            input=(
                NeuralInputConfig(
                    surprisal_paths=tuple(
                        _resolve_path(path_value, config_root=config_root)
                        for path_value in neural_input_payload.get("surprisal_paths", [])
                    ),
                    lowlevel_neural_paths=tuple(
                        _resolve_path(path_value, config_root=config_root)
                        for path_value in neural_input_payload.get("lowlevel_neural_paths", [])
                    ),
                )
                if neural_input_payload
                else None
            ),
            episode=NeuralEpisodeConfig(
                ipu_gap_threshold_s=float(neural_episode_payload.get("ipu_gap_threshold_s", 0.300)),
                max_followup_s=float(neural_episode_payload.get("max_followup_s", 6.0)),
                include_censored=bool(neural_episode_payload.get("include_censored", True)),
            ),
            window=NeuralWindowConfig(
                start_lag_s=float(neural_window_payload.get("start_lag_s", 0.500)),
                end_lag_s=float(neural_window_payload.get("end_lag_s", 0.100)),
                epsilon=float(neural_window_payload.get("epsilon", 1.0e-12)),
            ),
            pca=NeuralPcaConfig(
                mode=str(neural_pca_payload.get("mode", "variance")),
                n_components=int(neural_pca_payload.get("n_components", 3)),
                variance_threshold=float(neural_pca_payload.get("variance_threshold", 0.90)),
            ),
            model=NeuralModelConfig(
                fitting_backend=str(neural_model_payload.get("fitting_backend", "glmmTMB")),
                baseline_spline_df=int(neural_model_payload.get("baseline_spline_df", 6)),
                baseline_spline_degree=int(neural_model_payload.get("baseline_spline_degree", 3)),
                information_rate_lag_ms=int(neural_model_payload.get("information_rate_lag_ms", 150)),
                prop_expected_lag_ms=int(neural_model_payload.get("prop_expected_lag_ms", 700)),
            ),
            select_neural_lags=bool(neural_payload.get("select_neural_lags", False)),
            neural_lag_grid_ms=tuple(
                (int(lag_window["lag_start_ms"]), int(lag_window["lag_end_ms"]))
                for lag_window in neural_payload.get(
                    "neural_lag_grid_ms",
                    (
                        {"lag_start_ms": 50, "lag_end_ms": 250},
                        {"lag_start_ms": 100, "lag_end_ms": 300},
                        {"lag_start_ms": 100, "lag_end_ms": 500},
                        {"lag_start_ms": 300, "lag_end_ms": 500},
                        {"lag_start_ms": 300, "lag_end_ms": 700},
                        {"lag_start_ms": 500, "lag_end_ms": 900},
                    ),
                )
            ),
            neural_lag_selection_criterion=str(
                neural_payload.get("neural_lag_selection_criterion", "bic")
            ),
            neural_null_permutations=int(neural_payload.get("neural_null_permutations", 100)),
            skip_spp_on_failure=bool(neural_payload.get("skip_spp_on_failure", True)),
        ),
    )
    config.validate()
    return config


def _resolve_path(value: str | Path, *, config_root: Path) -> Path:
    """Resolve a config path relative to the config file directory."""

    path = Path(value)
    if path.is_absolute():
        return path
    return (config_root / path).resolve()


def _default_dyads_path(config_root: Path) -> Path | None:
    """Resolve the default dyads CSV if it exists."""

    candidate = (config_root / DEFAULT_DYADS_CSV_PATH).resolve()
    return candidate if candidate.exists() else None


def _default_pairing_issues_path(events_table_path: Path) -> Path | None:
    """Resolve the default pairing-issues CSV next to the events table."""

    candidate = events_table_path.with_name(DEFAULT_PAIRING_ISSUES_FILENAME)
    return candidate if candidate.exists() else None


def config_to_metadata_dict(config: HazardAnalysisConfig) -> dict[str, Any]:
    """Convert the config into a JSON-friendly metadata mapping."""

    return {
        "input": {
            "tde_hmm_results_dir": str(config.input.tde_hmm_results_dir),
            "events_table_path": str(config.input.events_table_path),
            "pairing_issues_table_path": (
                None
                if config.input.pairing_issues_table_path is None
                else str(config.input.pairing_issues_table_path)
            ),
            "dyads_csv_path": None if config.input.dyads_csv_path is None else str(config.input.dyads_csv_path),
        },
        "output": {"output_dir": str(config.output.output_dir)},
        "event_definition": {
            "partner_onset_column": config.event_definition.partner_onset_column,
            "target_onset_column": config.event_definition.target_onset_column,
            "target_label_column": config.event_definition.target_label_column,
            "partner_label_column": config.event_definition.partner_label_column,
            "fpp_label_prefixes": list(config.event_definition.fpp_label_prefixes),
            "event_id_column": config.event_definition.event_id_column,
            "recording_id_column": config.event_definition.recording_id_column,
            "run_column": config.event_definition.run_column,
            "partner_speaker_column": config.event_definition.partner_speaker_column,
            "target_speaker_column": config.event_definition.target_speaker_column,
            "issue_partner_tier_column": config.event_definition.issue_partner_tier_column,
            "issue_code_column": config.event_definition.issue_code_column,
            "issue_partner_label_column": config.event_definition.issue_partner_label_column,
            "issue_partner_onset_column": config.event_definition.issue_partner_onset_column,
            "issue_partner_offset_column": config.event_definition.issue_partner_offset_column,
            "censoring_issue_codes": list(config.event_definition.censoring_issue_codes),
        },
        "time_axis": {
            "observation_window_seconds": config.time_axis.observation_window_seconds,
            "bin_size_seconds": config.time_axis.bin_size_seconds,
            "entropy_lag_seconds": config.time_axis.entropy_lag_seconds,
            "exclude_initial_seconds": config.time_axis.exclude_initial_seconds,
        },
        "entropy": {
            "normalize_by_log_k": config.entropy.normalize_by_log_k,
            "epsilon": config.entropy.epsilon,
            "zscore_within_subject": config.entropy.zscore_within_subject,
        },
        "model": {
            "fitting_backend": config.model.fitting_backend,
            "include_quadratic_time": config.model.include_quadratic_time,
            "prefer_random_intercept_subject": config.model.prefer_random_intercept_subject,
        },
        "qc": {
            "make_entropy_trajectory_plot": config.qc.make_entropy_trajectory_plot,
            "make_entropy_histogram": config.qc.make_entropy_histogram,
            "make_predicted_hazard_plot": config.qc.make_predicted_hazard_plot,
            "make_event_rate_by_entropy_quantile_plot": config.qc.make_event_rate_by_entropy_quantile_plot,
            "make_observed_hazard_by_time_bin_plot": config.qc.make_observed_hazard_by_time_bin_plot,
            "make_observed_hazard_by_time_bin_smoothed_plot": config.qc.make_observed_hazard_by_time_bin_smoothed_plot,
            "make_observed_event_rate_by_entropy_quantile_plot": (
                config.qc.make_observed_event_rate_by_entropy_quantile_plot
            ),
            "make_observed_hazard_by_time_and_entropy_quantile_plot": (
                config.qc.make_observed_hazard_by_time_and_entropy_quantile_plot
            ),
            "make_model_vs_observed_hazard_plot": config.qc.make_model_vs_observed_hazard_plot,
            "make_entropy_distribution_terminal_vs_nonterminal_plot": (
                config.qc.make_entropy_distribution_terminal_vs_nonterminal_plot
            ),
        },
        "plotting": {
            "entropy_quantile_count": config.plotting.entropy_quantile_count,
            "entropy_group_count_for_time_plot": config.plotting.entropy_group_count_for_time_plot,
            "min_at_risk_per_bin": config.plotting.min_at_risk_per_bin,
            "smoothing_window_bins": config.plotting.smoothing_window_bins,
        },
        "misc": {"overwrite": config.misc.overwrite},
        "mode": config.mode,
        "neural": {
            "enabled": config.neural.enabled,
            "event_types": list(config.neural.event_types),
            "bin_size_s": config.neural.bin_size_s,
            "events_path": None if config.neural.events_path is None else str(config.neural.events_path),
            "out_dir": None if config.neural.out_dir is None else str(config.neural.out_dir),
            "input": (
                None
                if config.neural.input is None
                else {
                    "surprisal_paths": [str(path) for path in config.neural.input.surprisal_paths],
                    "lowlevel_neural_paths": [str(path) for path in config.neural.input.lowlevel_neural_paths],
                }
            ),
            "episode": {
                "ipu_gap_threshold_s": config.neural.episode.ipu_gap_threshold_s,
                "max_followup_s": config.neural.episode.max_followup_s,
                "include_censored": config.neural.episode.include_censored,
            },
            "window": {
                "start_lag_s": config.neural.window.start_lag_s,
                "end_lag_s": config.neural.window.end_lag_s,
                "epsilon": config.neural.window.epsilon,
            },
            "pca": {
                "mode": config.neural.pca.mode,
                "n_components": config.neural.pca.n_components,
                "variance_threshold": config.neural.pca.variance_threshold,
            },
            "model": {
                "fitting_backend": config.neural.model.fitting_backend,
                "baseline_spline_df": config.neural.model.baseline_spline_df,
                "baseline_spline_degree": config.neural.model.baseline_spline_degree,
                "information_rate_lag_ms": config.neural.model.information_rate_lag_ms,
                "prop_expected_lag_ms": config.neural.model.prop_expected_lag_ms,
            },
            "select_neural_lags": config.neural.select_neural_lags,
            "neural_lag_grid_ms": [
                {"lag_start_ms": int(lag_start_ms), "lag_end_ms": int(lag_end_ms)}
                for lag_start_ms, lag_end_ms in config.neural.neural_lag_grid_ms
            ],
            "neural_lag_selection_criterion": config.neural.neural_lag_selection_criterion,
            "neural_null_permutations": config.neural.neural_null_permutations,
            "skip_spp_on_failure": config.neural.skip_spp_on_failure,
        },
    }
