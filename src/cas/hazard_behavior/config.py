"""Configuration models for the behavioural FPP hazard pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

UnmatchedSurprisalStrategy = Literal["drop", "zero", "keep_nan"]
TokenAvailability = Literal["onset", "offset"]
ExpectedInfoGroup = Literal["partner_ipu_class", "partner_role", "global"]
EpisodeAnchor = Literal["partner_ipu"]
RGlmmBackend = Literal["glmmTMB", "glmer"]
RGlmmPropExpectedMode = Literal["after_best_rate", "matched_lag"]


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
        Episodes are anchored to partner IPUs.
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
    require_partner_offset_before_fpp: bool = False
    partner_offset_fpp_tolerance_s: float = 0.020
    overlapping_episode_strategy: Literal["exclude", "truncate", "keep"] = "keep"
    cluster_column: str | None = None
    overwrite: bool = False
    save_riskset: bool = True
    clip_proportions: bool = False
    clip_range: tuple[float, float] = (0.0, 1.5)
    lag_grid_ms: tuple[int, ...] = (0, 50, 100, 150, 200, 300, 500, 700, 1000)
    lagged_feature_fill_value: float = 0.0
    fit_timing_control_models: bool = True
    select_lags_with_timing_controls: bool = True
    run_r_glmm_lag_sweep: bool = False
    r_glmm_lag_grid_ms: tuple[int, ...] = (0, 50, 100, 150, 200, 300, 500, 700, 1000)
    r_glmm_onset_spline_df: int = 5
    r_glmm_offset_spline_df: int = 4
    r_glmm_backend: RGlmmBackend = "glmmTMB"
    r_glmm_include_run_random_effect: bool = False
    r_glmm_prop_expected_mode: RGlmmPropExpectedMode = "after_best_rate"
    r_glmm_include_prop_expected_in_final: bool = False
    primary_information_rate_lag_ms: int = 0
    primary_prop_expected_lag_ms: int = 300
    primary_model_baseline_spline_df: int = 6
    primary_model_baseline_spline_degree: int = 3
    save_lagged_feature_table: bool = False
    default_output_prefix: str = "hazard_behavior_fpp"
    default_expected_info_group_column: str = "partner_ipu_class"
    default_model_family: str = "binomial_glm"
    drop_unmatched_surprisal: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "surprisal_paths", tuple(Path(path) for path in self.surprisal_paths))
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
        if self.episode_anchor != "partner_ipu":
            raise ValueError("`episode_anchor` must be `partner_ipu` for the active behavioural hazard pipeline.")
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
        if not self.r_glmm_lag_grid_ms:
            raise ValueError("`r_glmm_lag_grid_ms` must contain at least one lag.")
        if any(int(lag_ms) < 0 for lag_ms in self.r_glmm_lag_grid_ms):
            raise ValueError("`r_glmm_lag_grid_ms` must contain only non-negative integers.")
        if self.r_glmm_onset_spline_df < 1:
            raise ValueError("`r_glmm_onset_spline_df` must be at least 1.")
        if self.r_glmm_offset_spline_df < 1:
            raise ValueError("`r_glmm_offset_spline_df` must be at least 1.")
        if self.r_glmm_backend not in {"glmmTMB", "glmer"}:
            raise ValueError("`r_glmm_backend` must be one of glmmTMB, glmer.")
        if self.r_glmm_prop_expected_mode not in {"after_best_rate", "matched_lag"}:
            raise ValueError(
                "`r_glmm_prop_expected_mode` must be one of after_best_rate, matched_lag."
            )
        if self.primary_information_rate_lag_ms < 0:
            raise ValueError("`primary_information_rate_lag_ms` must be non-negative.")
        if self.primary_prop_expected_lag_ms < 0:
            raise ValueError("`primary_prop_expected_lag_ms` must be non-negative.")
        if self.primary_model_baseline_spline_df < 3:
            raise ValueError("`primary_model_baseline_spline_df` must be at least 3.")
        if self.primary_model_baseline_spline_degree < 1:
            raise ValueError("`primary_model_baseline_spline_degree` must be at least 1.")

    @property
    def include_censored_episodes(self) -> bool:
        return self.include_censored

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""

        payload = asdict(self)
        payload["events_path"] = str(self.events_path)
        payload["surprisal_paths"] = [str(path) for path in self.surprisal_paths]
        payload["out_dir"] = str(self.out_dir)
        return payload

    def to_json(self) -> str:
        """Serialize the config to stable JSON."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
