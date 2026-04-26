"""Configuration models for the behavioural FPP hazard pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

UnmatchedSurprisalStrategy = Literal["drop", "zero", "keep_nan"]
TokenAvailability = Literal["onset", "offset"]
ExpectedInfoGroup = Literal["partner_ipu_class", "partner_role", "global"]
EpisodeAnchor = Literal["partner_ipu", "legacy_fpp_previous_partner"]
NeuralPcaMode = Literal["by_family", "combined"]


@dataclass(frozen=True, slots=True)
class BehaviourHazardConfig:
    """Configuration for the behavioural FPP-onset hazard pipeline.

    Parameters
    ----------
    events_path
        Path to the paired or minimally annotated events CSV.
    surprisal_paths
        One or more LM surprisal TSV paths.
    out_dir
        Output directory root.
    bin_size_s
        Discrete-time hazard bin size in seconds.
    information_rate_window_s
        Width of the causal information-rate window in seconds.
    minimum_episode_duration_s
        Minimum valid episode duration in seconds.
    unmatched_surprisal_strategy
        How to treat unmatched surprisal rows.
    baseline_spline_df
        Degrees of freedom for the baseline spline.
    baseline_spline_degree
        Spline degree for the baseline spline.
    ipu_gap_threshold_s
        Maximum gap between adjacent partner tokens when inferring IPUs.
    max_followup_s
        Follow-up duration for censored negative episodes.
    episode_anchor
        Whether episodes are anchored to partner IPUs or legacy FPP-linked anchors.
    include_censored
        Whether to construct censored negative episodes when possible.
    token_availability
        Whether surprisal becomes available at token onset or offset.
    expected_info_group
        Grouping used for expected total information.
    cluster_column
        Optional preferred clustering column.
    overwrite
        Whether existing output files may be overwritten.
    save_riskset
        Whether to persist the full risk-set table.
    clip_proportions
        Whether to clip proportion variables after construction.
    clip_range
        Lower and upper bounds used when clipping proportions.
    default_output_prefix
        Stable prefix for output artifacts.
    default_expected_info_group_column
        Default column name used for expected information grouping.
    default_model_family
        Name of the modelling family used for metadata output.

    Usage example
    -------------
        from pathlib import Path
        from cas.hazard_behavior.config import BehaviourHazardConfig

        config = BehaviourHazardConfig(
            events_path=Path("data/events/events.csv"),
            surprisal_paths=[Path("data/features/run-1_desc-lmSurprisal_features.tsv")],
            out_dir=Path("results/hazard_behavior_fpp"),
        )
    """

    events_path: Path
    surprisal_paths: tuple[Path, ...]
    out_dir: Path
    bin_size_s: float = 0.050
    information_rate_window_s: float = 0.500
    minimum_episode_duration_s: float = 0.100
    unmatched_surprisal_strategy: UnmatchedSurprisalStrategy = "drop"
    baseline_spline_df: int = 6
    baseline_spline_degree: int = 3
    ipu_gap_threshold_s: float = 0.300
    max_followup_s: float = 6.0
    episode_anchor: EpisodeAnchor = "partner_ipu"
    include_censored: bool = True
    token_availability: TokenAvailability = "onset"
    expected_info_group: ExpectedInfoGroup = "partner_ipu_class"
    target_fpp_label_prefix: str = "FPP_"
    require_partner_offset_before_fpp: bool = True
    partner_offset_fpp_tolerance_s: float = 0.020
    overlapping_episode_strategy: Literal["exclude", "truncate", "keep"] = "exclude"
    cluster_column: str | None = None
    overwrite: bool = False
    save_riskset: bool = True
    clip_proportions: bool = False
    clip_range: tuple[float, float] = (0.0, 1.5)
    lag_grid_ms: tuple[int, ...] = (0, 100, 200, 300, 500, 700, 1000)
    primary_lagged_predictor: str = "information_rate"
    lagged_feature_fill_value: float = 0.0
    fit_primary_behaviour_models: bool = True
    fit_timing_control_models: bool = False
    select_lags_with_timing_controls: bool = False
    fit_primary_stat_tests: bool = True
    make_primary_publication_figures: bool = True
    run_primary_leave_one_cluster: bool = False
    run_behaviour_model_suite: bool = True
    primary_information_rate_lag_ms: int = 0
    primary_prop_expected_lag_ms: int = 300
    fit_neural_lowlevel_models: bool = False
    neural_features: tuple[Path, ...] = ()
    neural_time_column: str = "time"
    neural_speaker_column: str = "speaker"
    neural_window_s: float = 0.500
    neural_guard_s: float = 0.100
    neural_feature_prefixes: tuple[str, ...] = ("amp_", "alpha_", "beta_")
    neural_include_amplitude: bool = True
    neural_include_alpha: bool = True
    neural_include_beta: bool = True
    neural_pca_variance_threshold: float = 0.90
    neural_pca_max_components: int = 10
    neural_pca_min_components: int = 1
    neural_pca_mode: NeuralPcaMode = "by_family"
    neural_standardize_features: bool = True
    neural_cluster_column: str | None = None
    primary_model_baseline_spline_df: int = 6
    primary_model_baseline_spline_degree: int = 3
    fit_lagged_models: bool = True
    save_lagged_feature_table: bool = True
    default_output_prefix: str = "hazard_behavior_fpp"
    default_expected_info_group_column: str = "partner_ipu_class"
    default_model_family: str = "binomial_glm"
    drop_unmatched_surprisal: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "surprisal_paths", tuple(Path(path) for path in self.surprisal_paths))
        object.__setattr__(self, "neural_features", tuple(Path(path) for path in self.neural_features))
        object.__setattr__(self, "drop_unmatched_surprisal", self.unmatched_surprisal_strategy == "drop")
        self.validate()

    def validate(self) -> None:
        """Validate configuration values conservatively."""

        if self.bin_size_s <= 0.0:
            raise ValueError("`bin_size_s` must be positive.")
        if self.information_rate_window_s <= 0.0:
            raise ValueError("`information_rate_window_s` must be positive.")
        if self.minimum_episode_duration_s <= 0.0:
            raise ValueError("`minimum_episode_duration_s` must be positive.")
        if self.baseline_spline_df < 3:
            raise ValueError("`baseline_spline_df` must be at least 3.")
        if self.baseline_spline_degree < 1:
            raise ValueError("`baseline_spline_degree` must be at least 1.")
        if self.ipu_gap_threshold_s < 0.0:
            raise ValueError("`ipu_gap_threshold_s` must be non-negative.")
        if self.max_followup_s <= 0.0:
            raise ValueError("`max_followup_s` must be positive.")
        if self.episode_anchor not in {"partner_ipu", "legacy_fpp_previous_partner"}:
            raise ValueError("`episode_anchor` must be one of partner_ipu, legacy_fpp_previous_partner.")
        if self.unmatched_surprisal_strategy not in {"drop", "zero", "keep_nan"}:
            raise ValueError("`unmatched_surprisal_strategy` must be one of drop, zero, keep_nan.")
        if self.token_availability not in {"onset", "offset"}:
            raise ValueError("`token_availability` must be one of onset, offset.")
        if self.expected_info_group not in {"partner_ipu_class", "partner_role", "global"}:
            raise ValueError("`expected_info_group` must be one of partner_ipu_class, partner_role, global.")
        if self.partner_offset_fpp_tolerance_s < 0.0:
            raise ValueError("`partner_offset_fpp_tolerance_s` must be non-negative.")
        if self.overlapping_episode_strategy not in {"exclude", "truncate", "keep"}:
            raise ValueError("`overlapping_episode_strategy` must be one of exclude, truncate, keep.")
        lower, upper = self.clip_range
        if lower >= upper:
            raise ValueError("`clip_range` must be an increasing interval.")
        if not self.target_fpp_label_prefix:
            raise ValueError("`target_fpp_label_prefix` must be non-empty.")
        if not self.lag_grid_ms:
            raise ValueError("`lag_grid_ms` must contain at least one lag.")
        if any(int(lag_ms) < 0 for lag_ms in self.lag_grid_ms):
            raise ValueError("`lag_grid_ms` must contain only non-negative integers.")
        if self.primary_information_rate_lag_ms < 0:
            raise ValueError("`primary_information_rate_lag_ms` must be non-negative.")
        if self.primary_prop_expected_lag_ms < 0:
            raise ValueError("`primary_prop_expected_lag_ms` must be non-negative.")
        if self.neural_window_s <= 0.0:
            raise ValueError("`neural_window_s` must be positive.")
        if self.neural_guard_s < 0.0:
            raise ValueError("`neural_guard_s` must be non-negative.")
        if self.neural_guard_s >= self.neural_window_s:
            raise ValueError("`neural_guard_s` must be smaller than `neural_window_s`.")
        if not self.neural_time_column:
            raise ValueError("`neural_time_column` must be non-empty.")
        if not self.neural_speaker_column:
            raise ValueError("`neural_speaker_column` must be non-empty.")
        if not self.neural_feature_prefixes:
            raise ValueError("`neural_feature_prefixes` must contain at least one prefix.")
        if self.neural_pca_variance_threshold <= 0.0 or self.neural_pca_variance_threshold > 1.0:
            raise ValueError("`neural_pca_variance_threshold` must be in the interval (0, 1].")
        if self.neural_pca_max_components < 1:
            raise ValueError("`neural_pca_max_components` must be at least 1.")
        if self.neural_pca_min_components < 1:
            raise ValueError("`neural_pca_min_components` must be at least 1.")
        if self.neural_pca_min_components > self.neural_pca_max_components:
            raise ValueError("`neural_pca_min_components` must be <= `neural_pca_max_components`.")
        if self.neural_pca_mode not in {"by_family", "combined"}:
            raise ValueError("`neural_pca_mode` must be one of by_family, combined.")
        if self.primary_model_baseline_spline_df < 3:
            raise ValueError("`primary_model_baseline_spline_df` must be at least 3.")
        if self.primary_model_baseline_spline_degree < 1:
            raise ValueError("`primary_model_baseline_spline_degree` must be at least 1.")
        if self.primary_lagged_predictor not in {
            "information_rate",
            "cumulative_info",
            "prop_actual_cumulative_info",
            "prop_expected_cumulative_info",
        }:
            raise ValueError("`primary_lagged_predictor` must name a supported information feature.")

    @property
    def include_censored_episodes(self) -> bool:
        return self.include_censored

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""

        payload = asdict(self)
        payload["events_path"] = str(self.events_path)
        payload["surprisal_paths"] = [str(path) for path in self.surprisal_paths]
        payload["neural_features"] = [str(path) for path in self.neural_features]
        payload["out_dir"] = str(self.out_dir)
        return payload

    def to_json(self) -> str:
        """Serialize the config to stable JSON."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
