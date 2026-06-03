from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml

from cas.epochs.io import (
    write_epoch_events_array,
    write_epoch_metadata,
    write_epoch_summary,
    write_epochs,
)
from cas.induced_epochs.transform import (
    build_induced_epochs,
    resolve_induced_band_limits_hz,
    resolve_induced_band_names,
)


def build_subject_induced_epochs(
    subject: str,
    source_epoch_paths: list[str | Path],
    config_path: str | Path,
    output_root: str | Path,
    *,
    root_subdir: str = "induced_epochs",
) -> None:
    source_paths = [Path(path) for path in source_epoch_paths]
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    output_root_path = Path(output_root)

    source_epochs = [
        mne.read_epochs(path, preload=True, verbose="ERROR")
        for path in source_paths
    ]
    concatenated = mne.concatenate_epochs(
        source_epochs,
        add_offset=True,
        on_mismatch="raise",
        verbose="ERROR",
    )

    if concatenated.metadata is None:
        metadata_df = pd.DataFrame(index=np.arange(len(concatenated)))
    else:
        metadata_df = concatenated.metadata.copy().reset_index(drop=True)

    written_bands: list[dict[str, str]] = []
    for band_name in resolve_induced_band_names(config):
        low_hz, high_hz = resolve_induced_band_limits_hz(band_name, config)
        induced_epochs = build_induced_epochs(
            concatenated,
            band_name=band_name,
            config=config,
        )

        band_dir = output_root_path / root_subdir / band_name / f"sub-{subject}"
        epochs_output = band_dir / "epochs-time_s.fif"
        metadata_output = band_dir / "metadata-time_s.csv"
        events_array_output = band_dir / "events-time_s.npy"
        band_summary_output = band_dir / "epoching_summary-time_s.json"

        band_summary = {
            "status": "ok",
            "band_name": band_name,
            "band_limits_hz": [low_hz, high_hz],
            "subject_id": f"sub-{subject}",
            "source_epochs_paths": [str(path) for path in source_paths],
            "n_source_files": len(source_paths),
            "n_epochs": int(len(induced_epochs)),
            "n_channels": int(len(induced_epochs.ch_names)),
            "n_times": int(len(induced_epochs.times)),
            "tmin_s": float(induced_epochs.times[0]) if len(induced_epochs.times) else 0.0,
            "tmax_s": float(induced_epochs.times[-1]) if len(induced_epochs.times) else 0.0,
            "sampling_frequency_hz": float(induced_epochs.info["sfreq"]),
            "method": "morlet_induced_power",
        }

        write_epochs(induced_epochs, epochs_output)
        write_epoch_metadata(metadata_df, metadata_output)
        write_epoch_events_array(induced_epochs.events.copy(), events_array_output)
        write_epoch_summary(band_summary, band_summary_output)
        written_bands.append(
            {
                "band_name": band_name,
                "metadata_output": str(metadata_output),
                "epochs_output": str(epochs_output),
                "events_array_output": str(events_array_output),
                "summary_output": str(band_summary_output),
            }
        )

    subject_summary = {
        "status": "ok",
        "subject_id": f"sub-{subject}",
        "bands": written_bands,
    }
    output_path = output_root_path / root_subdir / f"sub-{subject}" / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(subject_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
