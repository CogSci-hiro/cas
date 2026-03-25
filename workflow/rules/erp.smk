EPOCHS_SETTINGS = EPOCHS_CONFIG["epochs"]
EPOCHS_OUTPUT_ROOT = os.path.join(OUT_DIR, EPOCHS_SETTINGS["output"]["root"])
EPOCHS_OUTPUT_PATTERN = (
    f"{EPOCHS_OUTPUT_ROOT}/sub-{{subject}}/eeg/"
    f"sub-{{subject}}_task-{{task}}_run-{{run}}_desc-tasklocked_epo.fif"
)


def _normalize_run_label(run: str) -> str:
    return str(int(run))


def _subject_speaker_label(subject: str) -> str:
    return "A" if int(subject) % 2 == 1 else "B"


def _epochs_recording_id(subject: str) -> str:
    subject_number = int(subject)
    dyad_number = (subject_number + 1) // 2
    return f"dyad-{dyad_number:03d}"


def _discover_epoch_records() -> list[dict[str, str]]:
    return list(PREPROCESSED_EEG_RECORDS)


EPOCH_RECORDS = _discover_epoch_records()
EPOCH_OUTPUTS = expand(
    EPOCHS_OUTPUT_PATTERN,
    zip,
    subject=[record["subject"] for record in EPOCH_RECORDS],
    task=[record["task"] for record in EPOCH_RECORDS],
    run=[record["run"] for record in EPOCH_RECORDS],
)


rule make_epochs:
    input:
        eeg=PREPROCESSED_EEG_OUTPUT_PATTERN,
        events_csv=EVENTS_CSV_OUTPUT,
    output:
        epochs=EPOCHS_OUTPUT_PATTERN,
    run:
        import numpy as np
        import pandas as pd
        import mne

        raw = mne.io.read_raw_fif(input.eeg, preload=True, verbose="ERROR")
        events_df = pd.read_csv(input.events_csv)

        recording_id = _epochs_recording_id(wildcards.subject)
        run_label = _normalize_run_label(wildcards.run)
        speaker_label = _subject_speaker_label(wildcards.subject)
        event_types = tuple(EPOCHS_SETTINGS.get("event_types", ("fpp", "spp")))

        recording_events = events_df.loc[
            (events_df["recording_id"] == recording_id)
            & (events_df["run"].astype(str) == run_label)
        ].copy()

        metadata_frames = []
        for event_type in event_types:
            if event_type == "fpp":
                speaker_column = "speaker_fpp"
                onset_column = "fpp_onset"
            elif event_type == "spp":
                speaker_column = "speaker_spp"
                onset_column = "spp_onset"
            else:
                raise ValueError(f"Unsupported epoch event type: {event_type}")

            selected = recording_events.loc[
                recording_events[speaker_column].astype(str) == speaker_label
            ].copy()
            if selected.empty:
                continue

            selected["event_type"] = event_type
            selected["event_source_column"] = onset_column
            selected["event_onset_s"] = pd.to_numeric(selected[onset_column], errors="coerce")
            selected["subject_id"] = f"sub-{wildcards.subject}"
            selected["task"] = wildcards.task
            selected["run"] = run_label
            selected["recording_id"] = recording_id
            selected["speaker_current"] = speaker_label
            selected["speaker_role"] = speaker_label
            selected["pair_member"] = np.where(
                selected["speaker_fpp"].astype(str) == speaker_label,
                "fpp",
                "spp",
            )
            metadata_frames.append(selected)

        if metadata_frames:
            metadata = pd.concat(metadata_frames, ignore_index=True)
            metadata = metadata.loc[metadata["event_onset_s"].notna()].copy()
            metadata = metadata.sort_values(["event_onset_s", "pair_id", "event_type"]).reset_index(drop=True)
        else:
            metadata = pd.DataFrame(
                columns=[
                    "recording_id",
                    "run",
                    "file_path",
                    "part",
                    "response",
                    "speaker_fpp",
                    "speaker_spp",
                    "fpp_label",
                    "spp_label",
                    "fpp_onset",
                    "fpp_offset",
                    "spp_onset",
                    "spp_offset",
                    "fpp_duration",
                    "spp_duration",
                    "latency",
                    "pair_id",
                    "event_type",
                    "event_source_column",
                    "event_onset_s",
                    "subject_id",
                    "task",
                    "speaker_current",
                    "speaker_role",
                    "pair_member",
                ]
            )

        event_id = {event_type: index + 1 for index, event_type in enumerate(event_types)}
        if len(metadata):
            recording_duration_s = raw.n_times / raw.info["sfreq"]
            metadata = metadata.loc[
                (metadata["event_onset_s"] >= 0.0)
                & (metadata["event_onset_s"] < recording_duration_s)
            ].reset_index(drop=True)
            sample_events = np.empty((len(metadata), 3), dtype=int)
            onset_samples = np.rint(metadata["event_onset_s"].to_numpy(dtype=float) * raw.info["sfreq"]).astype(int)
            sample_events[:, 0] = onset_samples
            sample_events[:, 1] = 0
            sample_events[:, 2] = metadata["event_type"].map(event_id).to_numpy(dtype=int)
        else:
            sample_events = np.empty((0, 3), dtype=int)

        baseline_s = EPOCHS_SETTINGS.get("timing", {}).get("baseline_s")
        baseline = None if baseline_s is None else tuple(float(value) for value in baseline_s)
        tmin_s = float(EPOCHS_SETTINGS["timing"]["tmin_s"])
        tmax_s = float(EPOCHS_SETTINGS["timing"]["tmax_s"])

        if len(metadata):
            observed_event_types = list(dict.fromkeys(metadata["event_type"].tolist()))
            observed_event_id = {event_type: event_id[event_type] for event_type in observed_event_types}
            epochs = mne.Epochs(
                raw,
                sample_events,
                event_id=observed_event_id,
                tmin=tmin_s,
                tmax=tmax_s,
                baseline=baseline,
                metadata=metadata,
                preload=True,
                reject_by_annotation=bool(EPOCHS_SETTINGS.get("reject_by_annotation", True)),
                event_repeated="drop",
                verbose="ERROR",
            )
        else:
            dummy_sample = max(0, int(np.ceil(abs(tmin_s) * raw.info["sfreq"])))
            dummy_events = np.array([[dummy_sample, 0, 1]], dtype=int)
            dummy_metadata = pd.DataFrame([{"event_type": "__empty__"}])
            epochs = mne.Epochs(
                raw,
                dummy_events,
                event_id={"__empty__": 1},
                tmin=tmin_s,
                tmax=tmax_s,
                baseline=baseline,
                metadata=dummy_metadata,
                preload=True,
                reject_by_annotation=bool(EPOCHS_SETTINGS.get("reject_by_annotation", True)),
                event_repeated="drop",
                verbose="ERROR",
            )
            epochs.drop([0], reason="no_matching_events")

        os.makedirs(os.path.dirname(output.epochs), exist_ok=True)
        epochs.save(output.epochs, overwrite=True)
