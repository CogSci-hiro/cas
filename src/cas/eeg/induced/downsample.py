from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from cas.epochs.io import (
    write_epoch_events_array,
    write_epoch_metadata,
    write_epoch_summary,
    write_epochs,
)
from cas.induced_epochs.downsample import (
    downsample_power_time,
    resolve_downsampling_config,
)
from cas.induced_epochs.io import build_induced_epoch_band_paths


def _load_lmeeeg_workflow_config(config_path: str | Path) -> dict[str, object]:
    import yaml

    with Path(config_path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {config_path}.")
    section = payload.get("lmeeeg", payload)
    if not isinstance(section, dict):
        raise ValueError(f"`lmeeeg` in {config_path} must be a mapping.")
    loaded = dict(section)
    if "induced_epochs" in payload:
        loaded["induced_epochs"] = payload["induced_epochs"]
    return loaded


def downsample_subject_induced_epochs(
    subject: str,
    summary_path: str | Path,
    config_path: str | Path,
    output_root: str | Path,
) -> None:
    source_summary_path = Path(summary_path)
    if not source_summary_path.exists():
        raise FileNotFoundError(f"Source induced-epoch summary not found: {source_summary_path}")

    lmeeeg_config = _load_lmeeeg_workflow_config(config_path)
    downsample_cfg = resolve_downsampling_config(lmeeeg_config)
    if not bool(downsample_cfg.get("enabled", False)):
        raise ValueError("post_power_downsampling.enabled must be true for this target.")

    output_root_path = Path(output_root)
    target_sfreq = float(downsample_cfg.get("target_sfreq", 20.0))
    method = str(downsample_cfg.get("method", "mean_bin"))
    root_subdir = str(
        (dict(lmeeeg_config.get("input") or {})).get(
            "induced_epochs_subdir",
            "induced_epochs_fpp_spp_conf_disc_alpha_beta_lmeeeg",
        )
    )
    output_summary_path = output_root_path / root_subdir / f"sub-{subject}" / "summary.json"

    workflow_config = {"paths": {"out_dir": str(output_root_path)}}
    row = {"subject_id": f"sub-{subject}"}
    written_bands: list[dict[str, str]] = []
    for band_name in [
        str(value)
        for value in lmeeeg_config.get("induced_epochs", {}).get("bands", ["alpha", "beta"])
    ]:
        source_dir = output_root_path / "induced_epochs" / band_name / f"sub-{subject}"
        source_epochs = mne.read_epochs(
            source_dir / "epochs-time_s.fif",
            preload=True,
            verbose="ERROR",
        )
        source_metadata = pd.read_csv(source_dir / "metadata-time_s.csv")
        source_events = np.load(source_dir / "events-time_s.npy")

        source_data = np.asarray(source_epochs.get_data(copy=True), dtype=np.float32)
        downsampled_data, downsampled_times = downsample_power_time(
            source_data,
            np.asarray(source_epochs.times, dtype=float),
            target_sfreq=target_sfreq,
        )

        sfreq = 1.0 / float(np.median(np.diff(downsampled_times)))
        downsampled_info = mne.create_info(
            ch_names=source_epochs.ch_names,
            sfreq=sfreq,
            ch_types=source_epochs.get_channel_types(),
        )
        downsampled_info["bads"] = list(source_epochs.info.get("bads", []))
        downsampled_epochs = mne.EpochsArray(
            downsampled_data.astype(np.float32, copy=False),
            downsampled_info,
            events=source_epochs.events.copy(),
            tmin=float(downsampled_times[0]),
            event_id=dict(source_epochs.event_id),
            metadata=source_metadata.copy().reset_index(drop=True),
            verbose="ERROR",
        )
        source_montage = source_epochs.get_montage()
        if source_montage is not None:
            downsampled_epochs.set_montage(source_montage, on_missing="ignore")

        output_paths = build_induced_epoch_band_paths(
            workflow_config,
            row,
            band_name=band_name,
            root_subdir=root_subdir,
        )

        band_summary = {
            "status": "ok",
            "band_name": band_name,
            "subject_id": f"sub-{subject}",
            "source_epochs_path": str(source_dir / "epochs-time_s.fif"),
            "n_epochs": int(len(downsampled_epochs)),
            "n_channels": int(len(downsampled_epochs.ch_names)),
            "n_times_original": int(source_data.shape[-1]),
            "n_times_downsampled": int(downsampled_data.shape[-1]),
            "tmin_s": float(downsampled_times[0]),
            "tmax_s": float(downsampled_times[-1]),
            "sampling_frequency_hz": float(sfreq),
            "target_sfreq_hz": target_sfreq,
            "method": method,
        }

        write_epochs(downsampled_epochs, output_paths.epochs_output_path)
        write_epoch_metadata(source_metadata, output_paths.metadata_output_path)
        write_epoch_events_array(source_events, output_paths.events_array_output_path)
        write_epoch_summary(band_summary, output_paths.summary_output_path)
        written_bands.append(
            {
                "band_name": band_name,
                "epochs_output": str(output_paths.epochs_output_path),
                "metadata_output": str(output_paths.metadata_output_path),
                "events_array_output": str(output_paths.events_array_output_path),
                "summary_output": str(output_paths.summary_output_path),
            }
        )

    output_path = Path(summary_path)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "subject_id": f"sub-{subject}",
                "source_summary_path": str(source_summary_path),
                "target_sfreq_hz": target_sfreq,
                "method": method,
                "bands": written_bands,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
