"""Typed configuration loading for the source-level DICS pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LONG_TABLE_CHUNK_ROWS = 250_000
DEFAULT_TFR_FREQ_SAMPLES = 5
DEFAULT_TFR_N_CYCLES = 7.0
DEFAULT_TFR_TIME_BANDWIDTH = 4.0


def _resolve_path(path_value: str | None, *, config_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def _require_mapping(payload: Any, *, label: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping.")
    return dict(payload)


@dataclass(frozen=True, slots=True)
class PathConfig:
    events_csv: Path
    epochs_dir: Path
    source_dir: Path
    derivatives_dir: Path
    lmeeeg_dir: Path
    bids_root: Path | None = None
    preprocessed_eeg_root: Path | None = None

    @property
    def qc_dir(self) -> Path:
        return self.derivatives_dir / "qc"

    @property
    def filters_dir(self) -> Path:
        return self.source_dir / "filters"

    @property
    def trial_power_dir(self) -> Path:
        return self.source_dir / "trial_power"

    @property
    def metadata_dir(self) -> Path:
        return self.source_dir / "metadata"

    @property
    def long_table_dir(self) -> Path:
        return self.derivatives_dir / "source_power_long"


@dataclass(frozen=True, slots=True)
class EventsConfig:
    anchor_types: tuple[str, ...]
    subject_column: str
    dyad_column: str
    run_column: str
    anchor_type_column: str
    onset_column: str
    duration_column: str
    latency_column: str
    label_column: str


@dataclass(frozen=True, slots=True)
class EpochingConfig:
    tmin: float
    tmax: float
    baseline: tuple[float | None, float | None] | None
    reject_by_annotation: bool


@dataclass(frozen=True, slots=True)
class BandConfig:
    fmin: float
    fmax: float


@dataclass(frozen=True, slots=True)
class DicsConfig:
    method: str
    common_filter: bool
    filter_tmin: float
    filter_tmax: float
    analysis_tmin: float
    analysis_tmax: float
    bands: dict[str, BandConfig]
    csd_method: str
    mt_bandwidth: float | None
    regularization: float
    pick_ori: str | None
    weight_norm: str | None
    reduce_rank: bool
    real_filter: bool
    n_jobs: int
    tfr_freq_samples: int = DEFAULT_TFR_FREQ_SAMPLES
    tfr_n_cycles: float = DEFAULT_TFR_N_CYCLES
    tfr_time_bandwidth: float = DEFAULT_TFR_TIME_BANDWIDTH


@dataclass(frozen=True, slots=True)
class SourceSpaceConfig:
    kind: str
    mode: str
    spacing: str
    subjects_dir: Path
    subject: str
    trans: str
    bem: Path | None
    forward_template: Path | None
    parcellation: str | None = None
    aggregate_to_labels: bool = False
    aggregation: str = "mean"


@dataclass(frozen=True, slots=True)
class OutputConfig:
    save_filters: bool
    save_trial_power: bool
    save_long_table: bool
    save_qc: bool
    overwrite: bool
    long_table_chunk_rows: int = DEFAULT_LONG_TABLE_CHUNK_ROWS


@dataclass(frozen=True, slots=True)
class LmEEGConfig:
    enabled: bool
    formula: str
    dependent_variable: str
    predictors: tuple[str, ...]
    test_predictors: tuple[str, ...] = ()
    contrast_of_interest: str | None = None


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    verbose: bool
    progress: bool


@dataclass(frozen=True, slots=True)
class SourceDicsConfig:
    config_path: Path
    paths: PathConfig
    events: EventsConfig
    epoching: EpochingConfig
    dics: DicsConfig
    source_space: SourceSpaceConfig
    output: OutputConfig
    lmeeeg: LmEEGConfig
    logging: LoggingConfig

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _parse_baseline(payload: Any) -> tuple[float | None, float | None] | None:
    if payload is None:
        return None
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        raise ValueError("epoching.baseline must be null or a two-item sequence.")
    start, end = payload
    return (
        None if start is None else float(start),
        None if end is None else float(end),
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _parse_bands(payload: Any) -> dict[str, BandConfig]:
    mapping = _require_mapping(payload, label="dics.bands")
    if not mapping:
        raise ValueError("dics.bands must define at least one frequency band.")
    bands: dict[str, BandConfig] = {}
    for band_name, band_payload in mapping.items():
        band_mapping = _require_mapping(band_payload, label=f"dics.bands.{band_name}")
        fmin = float(band_mapping["fmin"])
        fmax = float(band_mapping["fmax"])
        if fmin >= fmax:
            raise ValueError(f"dics.bands.{band_name} must satisfy fmin < fmax.")
        bands[str(band_name)] = BandConfig(fmin=fmin, fmax=fmax)
    return bands


def _validate_time_windows(epoching: EpochingConfig, dics: DicsConfig) -> None:
    if not epoching.tmin < dics.filter_tmin < dics.filter_tmax <= epoching.tmax:
        raise ValueError(
            "Expected epoching.tmin < dics.filter_tmin < dics.filter_tmax <= epoching.tmax."
        )
    if not epoching.tmin < dics.analysis_tmin < dics.analysis_tmax <= epoching.tmax:
        raise ValueError(
            "Expected epoching.tmin < dics.analysis_tmin < dics.analysis_tmax <= epoching.tmax."
        )
    if dics.filter_tmin > dics.analysis_tmin:
        raise ValueError("dics.filter_tmin must be <= dics.analysis_tmin.")
    if dics.filter_tmax > dics.analysis_tmax:
        raise ValueError("dics.filter_tmax must be <= dics.analysis_tmax.")
    if dics.filter_tmax > 0.0:
        raise ValueError("dics.filter_tmax must be <= 0.0 to keep filter estimation pre-event only.")
    if dics.analysis_tmax > 0.0:
        raise ValueError("dics.analysis_tmax must be <= 0.0 to keep final statistics pre-event only.")


def _validate_source_space(source_space: SourceSpaceConfig) -> None:
    if source_space.kind not in {"surface", "volume"}:
        raise ValueError("source_space.kind must be either 'surface' or 'volume'.")
    if source_space.aggregate_to_labels and not source_space.parcellation:
        raise ValueError(
            "source_space.parcellation is required when source_space.aggregate_to_labels is true."
        )


def _create_output_dirs(paths: PathConfig) -> None:
    for path in (
        paths.source_dir,
        paths.derivatives_dir,
        paths.lmeeeg_dir,
        paths.qc_dir,
        paths.filters_dir,
        paths.trial_power_dir,
        paths.metadata_dir,
        paths.long_table_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_source_dics_config(config_path: str | Path) -> SourceDicsConfig:
    """Load and validate the source-level DICS YAML config.

    Usage example
    -------------
    >>> from pathlib import Path
    >>> cfg = load_source_dics_config(Path("config/source_dics_fpp_spp_alpha_beta.yaml"))
    >>> cfg.dics.analysis_tmax
    -0.1
    """

    resolved_config_path = Path(config_path).resolve()
    with resolved_config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {resolved_config_path}.")
    config_dir = resolved_config_path.parent.parent

    paths_payload = _require_mapping(payload.get("paths"), label="paths")
    events_payload = _require_mapping(payload.get("events"), label="events")
    epoching_payload = _require_mapping(payload.get("epoching"), label="epoching")
    dics_payload = _require_mapping(payload.get("dics"), label="dics")
    source_space_payload = _require_mapping(payload.get("source_space"), label="source_space")
    output_payload = _require_mapping(payload.get("output"), label="output")
    lmeeeg_payload = _require_mapping(payload.get("lmeeeg"), label="lmeeeg")
    logging_payload = _require_mapping(payload.get("logging"), label="logging")

    paths = PathConfig(
        events_csv=_resolve_path(str(paths_payload["events_csv"]), config_dir=config_dir),
        epochs_dir=_resolve_path(str(paths_payload["epochs_dir"]), config_dir=config_dir),
        source_dir=_resolve_path(str(paths_payload["source_dir"]), config_dir=config_dir),
        derivatives_dir=_resolve_path(str(paths_payload["derivatives_dir"]), config_dir=config_dir),
        lmeeeg_dir=_resolve_path(str(paths_payload["lmeeeg_dir"]), config_dir=config_dir),
        bids_root=_resolve_path(paths_payload.get("bids_root"), config_dir=config_dir),
        preprocessed_eeg_root=_resolve_path(
            paths_payload.get("preprocessed_eeg_root"),
            config_dir=config_dir,
        ),
    )
    events = EventsConfig(
        anchor_types=tuple(str(anchor).upper() for anchor in events_payload.get("anchor_types", ())),
        subject_column=str(events_payload.get("subject_column", "subject")),
        dyad_column=str(events_payload.get("dyad_column", "dyad")),
        run_column=str(events_payload.get("run_column", "run")),
        anchor_type_column=str(events_payload.get("anchor_type_column", "anchor_type")),
        onset_column=str(events_payload.get("onset_column", "onset")),
        duration_column=str(events_payload.get("duration_column", "duration")),
        latency_column=str(events_payload.get("latency_column", "latency")),
        label_column=str(events_payload.get("label_column", "label")),
    )
    if not events.anchor_types:
        raise ValueError("events.anchor_types must not be empty.")
    epoching = EpochingConfig(
        tmin=float(epoching_payload["tmin"]),
        tmax=float(epoching_payload["tmax"]),
        baseline=_parse_baseline(epoching_payload.get("baseline")),
        reject_by_annotation=bool(epoching_payload.get("reject_by_annotation", True)),
    )
    dics = DicsConfig(
        method=str(dics_payload.get("method", "dics")),
        common_filter=bool(dics_payload.get("common_filter", True)),
        filter_tmin=float(dics_payload["filter_tmin"]),
        filter_tmax=float(dics_payload["filter_tmax"]),
        analysis_tmin=float(dics_payload["analysis_tmin"]),
        analysis_tmax=float(dics_payload["analysis_tmax"]),
        bands=_parse_bands(dics_payload.get("bands")),
        csd_method=str(dics_payload.get("csd_method", "multitaper")),
        mt_bandwidth=None
        if dics_payload.get("mt_bandwidth") is None
        else float(dics_payload["mt_bandwidth"]),
        regularization=float(dics_payload.get("regularization", 0.05)),
        pick_ori=None if dics_payload.get("pick_ori") is None else str(dics_payload["pick_ori"]),
        weight_norm=None
        if dics_payload.get("weight_norm") is None
        else str(dics_payload["weight_norm"]),
        reduce_rank=bool(dics_payload.get("reduce_rank", True)),
        real_filter=bool(dics_payload.get("real_filter", True)),
        n_jobs=int(dics_payload.get("n_jobs", 1)),
        tfr_freq_samples=int(dics_payload.get("tfr_freq_samples", DEFAULT_TFR_FREQ_SAMPLES)),
        tfr_n_cycles=float(dics_payload.get("tfr_n_cycles", DEFAULT_TFR_N_CYCLES)),
        tfr_time_bandwidth=float(
            dics_payload.get("tfr_time_bandwidth", DEFAULT_TFR_TIME_BANDWIDTH)
        ),
    )
    source_space = SourceSpaceConfig(
        kind=str(
            source_space_payload.get(
                "kind",
                "volume"
                if str(source_space_payload.get("mode", "")).strip().lower().startswith("fsaverage_vol")
                else "surface",
            )
        ).strip().lower(),
        mode=str(source_space_payload.get("mode", "fsaverage_vol_or_surface")),
        spacing=str(source_space_payload.get("spacing", "oct6")),
        subjects_dir=_resolve_path(str(source_space_payload["subjects_dir"]), config_dir=config_dir),
        subject=str(source_space_payload.get("subject", "fsaverage")),
        trans=str(source_space_payload.get("trans", "fsaverage")),
        bem=_resolve_path(source_space_payload.get("bem"), config_dir=config_dir),
        forward_template=_resolve_path(
            source_space_payload.get("forward_template"),
            config_dir=config_dir,
        ),
        parcellation=source_space_payload.get("parcellation"),
        aggregate_to_labels=bool(source_space_payload.get("aggregate_to_labels", False)),
        aggregation=str(source_space_payload.get("aggregation", "mean")),
    )
    output = OutputConfig(
        save_filters=bool(output_payload.get("save_filters", True)),
        save_trial_power=bool(output_payload.get("save_trial_power", True)),
        save_long_table=bool(output_payload.get("save_long_table", True)),
        save_qc=bool(output_payload.get("save_qc", True)),
        overwrite=bool(output_payload.get("overwrite", False)),
        long_table_chunk_rows=int(
            output_payload.get("long_table_chunk_rows", DEFAULT_LONG_TABLE_CHUNK_ROWS)
        ),
    )
    lmeeeg = LmEEGConfig(
        enabled=bool(lmeeeg_payload.get("enabled", True)),
        formula=str(lmeeeg_payload.get("formula", "")),
        dependent_variable=str(lmeeeg_payload.get("dependent_variable", "power")),
        predictors=tuple(str(value) for value in lmeeeg_payload.get("predictors", ())),
        test_predictors=tuple(str(value) for value in lmeeeg_payload.get("test_predictors", ())),
        contrast_of_interest=(
            None
            if lmeeeg_payload.get("contrast_of_interest") in {None, ""}
            else str(lmeeeg_payload.get("contrast_of_interest"))
        ),
    )
    logging = LoggingConfig(
        verbose=bool(logging_payload.get("verbose", True)),
        progress=bool(logging_payload.get("progress", True)),
    )

    config = SourceDicsConfig(
        config_path=resolved_config_path,
        paths=paths,
        events=events,
        epoching=epoching,
        dics=dics,
        source_space=source_space,
        output=output,
        lmeeeg=lmeeeg,
        logging=logging,
    )
    _validate_time_windows(config.epoching, config.dics)
    _validate_source_space(config.source_space)
    _create_output_dirs(config.paths)
    return config
