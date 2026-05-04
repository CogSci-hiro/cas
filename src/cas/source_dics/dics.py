"""DICS-specific computational helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import pickle

import numpy as np

from cas.source_dics.config import BandConfig, SourceDicsConfig
from cas.source_dics.io import configure_mne_runtime

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourcePowerResult:
    anchor_type: str
    band_name: str
    source_ids: list[str]
    times: np.ndarray
    power: np.ndarray
    stcs: list[object]


def _average_band_csd(csd, *, band_name: str, band: BandConfig):
    averaged = csd.mean(fmin=band.fmin, fmax=band.fmax)
    LOGGER.info(
        "Averaged multitaper CSD across %.1f..%.1f Hz to build one common %s filter.",
        band.fmin,
        band.fmax,
        band_name,
    )
    return averaged


def _resolve_fsaverage_subjects_root(subjects_dir: Path, *, subject: str) -> tuple[Path, Path]:
    resolved = subjects_dir.resolve()
    if (resolved / subject).exists():
        return resolved, resolved / subject
    if resolved.name == subject and resolved.exists():
        return resolved.parent, resolved
    raise FileNotFoundError(
        f"Could not resolve fsaverage subject directory from subjects_dir={subjects_dir} and subject={subject}."
    )


def _resolve_template_source_space_paths(config: SourceDicsConfig) -> tuple[Path, Path, Path]:
    subjects_root, subject_dir = _resolve_fsaverage_subjects_root(
        config.source_space.subjects_dir,
        subject=config.source_space.subject,
    )
    bem_dir = subject_dir / "bem"
    if config.source_space.kind == "surface":
        src_path = bem_dir / "fsaverage-ico-5-src.fif"
    elif config.source_space.kind == "volume":
        src_path = bem_dir / "fsaverage-vol-5-src.fif"
    else:
        raise ValueError(
            f"Unsupported source_space.kind {config.source_space.kind!r}; expected 'surface' or 'volume'."
        )
    bem_candidates = [
        config.source_space.bem,
        bem_dir / "fsaverage-5120-5120-5120-bem-sol.fif",
        bem_dir / "fsaverage-inner_skull-bem.fif",
    ]
    bem_path = next((path for path in bem_candidates if path is not None and path.exists()), None)
    if bem_path is None:
        raise FileNotFoundError(f"No fsaverage BEM file found in {bem_dir}.")
    trans_candidates = [
        bem_dir / "fsaverage-trans.fif",
        subject_dir / "fsaverage-trans.fif",
    ]
    trans_path = next((path for path in trans_candidates if path.exists()), None)
    if trans_path is None:
        raise FileNotFoundError(f"No fsaverage trans file found in {subject_dir} or {bem_dir}.")
    if not src_path.exists():
        raise FileNotFoundError(f"No fsaverage source-space file found at {src_path}.")
    return subjects_root, src_path, bem_path


def resolve_forward_model(config: SourceDicsConfig, *, info=None):
    """Load the configured forward model or raise a clear actionable error.

    Usage example
    -------------
    >>> cfg = load_source_dics_config("config/induced/source_localisation.yaml")  # doctest: +SKIP
    >>> resolve_forward_model(cfg)  # doctest: +SKIP
    """

    configure_mne_runtime()
    import mne

    forward_template = config.source_space.forward_template
    if forward_template is not None and forward_template.exists():
        LOGGER.warning(
            "Using template forward model %s for source localisation. "
            "This is a conservative fallback and should be documented in downstream QC.",
            forward_template,
        )
        return mne.read_forward_solution(forward_template, verbose="ERROR")

    if info is None:
        raise FileNotFoundError(
            "No forward model is configured and no MNE info object was provided to build an fsaverage template forward."
        )

    subjects_root, src_path, bem_path = _resolve_template_source_space_paths(config)
    trans_path = (
        subjects_root / config.source_space.subject / "bem" / "fsaverage-trans.fif"
        if (subjects_root / config.source_space.subject / "bem" / "fsaverage-trans.fif").exists()
        else subjects_root / config.source_space.subject / "fsaverage-trans.fif"
    )
    LOGGER.warning(
        "Building an fsaverage template forward model on the fly from src=%s, bem=%s, trans=%s.",
        src_path,
        bem_path,
        trans_path,
    )
    src = mne.read_source_spaces(src_path, verbose="ERROR")
    return mne.make_forward_solution(
        info,
        trans=str(trans_path),
        src=src,
        bem=str(bem_path),
        meg=False,
        eeg=True,
        mindist=0.0,
        n_jobs=config.dics.n_jobs,
        verbose="ERROR",
    )


def compute_common_dics_filters(epochs, *, band_name: str, band: BandConfig, forward, config: SourceDicsConfig):
    configure_mne_runtime()
    import mne

    if config.dics.csd_method != "multitaper":
        raise ValueError(
            f"Unsupported dics.csd_method {config.dics.csd_method!r}. Only 'multitaper' is supported."
        )
    LOGGER.info(
        "Computing pooled common DICS filter for band=%s over %.3f..%.3f s using pooled FPP+SPP epochs.",
        band_name,
        config.dics.filter_tmin,
        config.dics.filter_tmax,
    )
    filter_epochs = epochs.copy().crop(
        tmin=config.dics.filter_tmin,
        tmax=config.dics.filter_tmax,
        include_tmax=True,
    )
    csd = mne.time_frequency.csd_multitaper(
        filter_epochs,
        fmin=band.fmin,
        fmax=band.fmax,
        tmin=config.dics.filter_tmin,
        tmax=config.dics.filter_tmax,
        bandwidth=config.dics.mt_bandwidth,
        n_jobs=config.dics.n_jobs,
        verbose="ERROR",
    )
    csd = _average_band_csd(csd, band_name=band_name, band=band)
    return mne.beamformer.make_dics(
        epochs.info,
        forward,
        csd,
        reg=config.dics.regularization,
        pick_ori=config.dics.pick_ori,
        weight_norm=config.dics.weight_norm,
        reduce_rank=config.dics.reduce_rank,
        real_filter=config.dics.real_filter,
        verbose="ERROR",
    )


def save_filters(filters, output_path: Path, *, overwrite: bool) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        LOGGER.info("Reusing existing HDF5 filter export at %s.", output_path)
        return output_path
    try:
        filters.save(output_path, overwrite=overwrite)
        return output_path
    except (RuntimeError, OSError) as exc:
        fallback_path = output_path.with_suffix(".pkl")
        LOGGER.warning(
            "Falling back to pickle filter export because HDF5 beamformer saving is unavailable: %s",
            exc,
        )
        if fallback_path.exists() and not overwrite:
            LOGGER.info("Reusing existing fallback filter export at %s.", fallback_path)
            return fallback_path
        with fallback_path.open("wb") as handle:
            pickle.dump(filters, handle)
        return fallback_path


def _build_band_frequencies(band: BandConfig, *, n_samples: int) -> np.ndarray:
    if n_samples <= 1:
        return np.asarray([0.5 * (band.fmin + band.fmax)], dtype=float)
    return np.linspace(band.fmin, band.fmax, num=n_samples, dtype=float)


def _format_source_ids(stc) -> list[str]:
    vertices = stc.vertices
    if isinstance(vertices, list):
        labels: list[str] = []
        hemis = ("lh", "rh")
        for hemisphere, vertex_ids in zip(hemis, vertices):
            labels.extend(f"{hemisphere}:{int(vertex_id)}" for vertex_id in np.asarray(vertex_ids))
        return labels
    return [str(int(vertex_id)) for vertex_id in np.asarray(vertices)]


def _average_frequency_stcs(stcs_for_epoch: list[object]):
    data = np.stack([np.abs(np.asarray(stc.data)) ** 2 for stc in stcs_for_epoch], axis=0)
    averaged = stcs_for_epoch[0].copy()
    averaged._data = data.mean(axis=0).astype(np.float32, copy=False)
    return averaged


def _aggregate_to_labels(stcs: list[object], forward, config: SourceDicsConfig) -> tuple[np.ndarray, list[str]]:
    configure_mne_runtime()
    import mne

    if not config.source_space.aggregate_to_labels:
        first_stc = stcs[0]
        data = np.stack([np.asarray(stc.data, dtype=np.float32) for stc in stcs], axis=0)
        return data, _format_source_ids(first_stc)

    labels = mne.read_labels_from_annot(
        config.source_space.subject,
        parc=str(config.source_space.parcellation),
        subjects_dir=str(config.source_space.subjects_dir),
        verbose="ERROR",
    )
    extracted = mne.extract_label_time_course(
        stcs,
        labels,
        forward["src"],
        mode=config.source_space.aggregation,
        return_generator=False,
        verbose="ERROR",
    )
    return np.asarray(extracted, dtype=np.float32), [label.name for label in labels]


def apply_common_filters_to_epochs(
    epochs,
    *,
    filters,
    forward,
    anchor_type: str,
    band_name: str,
    band: BandConfig,
    config: SourceDicsConfig,
) -> SourcePowerResult:
    configure_mne_runtime()
    import mne
    import mne.beamformer._dics as mne_dics

    LOGGER.info(
        "Applying pooled DICS filter to anchor=%s band=%s and cropping final output to %.3f..%.3f s.",
        anchor_type,
        band_name,
        config.dics.analysis_tmin,
        config.dics.analysis_tmax,
    )
    frequencies = _build_band_frequencies(band, n_samples=config.dics.tfr_freq_samples)
    tfr = epochs.compute_tfr(
        method="multitaper",
        freqs=frequencies,
        tmin=config.epoching.tmin,
        tmax=config.epoching.tmax,
        output="complex",
        average=False,
        return_itc=False,
        n_jobs=config.dics.n_jobs,
        time_bandwidth=config.dics.tfr_time_bandwidth,
        n_cycles=config.dics.tfr_n_cycles,
        verbose="ERROR",
    )
    if tfr.data.ndim == 5:
        # MNE multitaper complex TFRs can carry an explicit taper axis with shape
        # (epochs, channels, tapers, freqs, times), but apply_dics_tfr_epochs
        # expects the tapers to already be combined.
        tfr._data = tfr.data.mean(axis=2)
    if len(filters["weights"]) != 1:
        raise ValueError(
            "Expected one band-averaged DICS filter per band. "
            f"Observed {len(filters['weights'])} filters for band={band_name}."
        )

    freq_stcs_per_epoch: list[list[object]] = [[] for _ in range(len(epochs))]
    for frequency_index, frequency_hz in enumerate(frequencies):
        LOGGER.info(
            "Applying common %s filter to anchor=%s at sampled frequency %.2f Hz (%d/%d).",
            band_name,
            anchor_type,
            float(frequency_hz),
            frequency_index + 1,
            len(frequencies),
        )
        single_frequency_data = tfr.data[:, :, frequency_index : frequency_index + 1, :]
        per_epoch_generator = mne_dics._apply_dics(
            single_frequency_data,
            filters,
            tfr.info,
            tfr.tmin,
            tfr=True,
        )
        for epoch_index, stc in enumerate(per_epoch_generator):
            freq_stcs_per_epoch[epoch_index].append(stc)

    averaged_stcs = [_average_frequency_stcs(stcs_for_epoch) for stcs_for_epoch in freq_stcs_per_epoch]
    power, source_ids = _aggregate_to_labels(averaged_stcs, forward=forward, config=config)
    times = np.asarray(averaged_stcs[0].times, dtype=float)
    time_mask = (times >= config.dics.analysis_tmin) & (times <= config.dics.analysis_tmax)
    return SourcePowerResult(
        anchor_type=anchor_type,
        band_name=band_name,
        source_ids=source_ids,
        times=times[time_mask],
        power=power[:, :, time_mask],
        stcs=averaged_stcs,
    )
