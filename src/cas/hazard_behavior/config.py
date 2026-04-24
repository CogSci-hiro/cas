"""Configuration models for the behavioural FPP hazard pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

UnmatchedSurprisalStrategy = Literal["drop", "zero", "keep_nan"]
TokenAvailability = Literal["onset", "offset"]
ExpectedInfoGroup = Literal["partner_ipu_class", "partner_role", "global"]


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
    max_followup_s: float = 5.0
    include_censored: bool = True
    token_availability: TokenAvailability = "onset"
    expected_info_group: ExpectedInfoGroup = "partner_ipu_class"
    require_partner_offset_before_fpp: bool = True
    partner_offset_fpp_tolerance_s: float = 0.020
    overlapping_episode_strategy: Literal["exclude", "truncate", "keep"] = "exclude"
    cluster_column: str | None = None
    overwrite: bool = False
    save_riskset: bool = True
    clip_proportions: bool = False
    clip_range: tuple[float, float] = (0.0, 1.5)
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
