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


def _other_speaker_label(subject: str) -> str:
    return "B" if _subject_speaker_label(subject) == "A" else "A"


def _subject_dyad_id(subject: str) -> str:
    subject_id = f"sub-{subject}"
    dyad_id = SUBJECT_TO_DYAD_MAP.get(subject_id)
    if dyad_id:
        return dyad_id

    subject_number = int(subject)
    dyad_number = (subject_number + 1) // 2
    return f"dyad-{dyad_number:03d}"


def _epochs_recording_id(subject: str) -> str:
    return _subject_dyad_id(subject)


def _epoch_annotation_path(subject: str, run: str) -> str | None:
    annotation_path = os.path.join(
        ANNOTATIONS_DIR,
        f"{_subject_dyad_id(subject)}_run-{run}_combined.TextGrid",
    )
    if os.path.exists(annotation_path):
        return annotation_path
    return None


def epoch_annotation_input(wildcards):
    annotation_path = _epoch_annotation_path(wildcards.subject, wildcards.run)
    if annotation_path is None:
        subject_id = f"sub-{wildcards.subject}"
        raise FileNotFoundError(
            f"No combined annotation file found for {subject_id}, task {wildcards.task}, run {wildcards.run}."
        )
    return annotation_path


def _discover_epoch_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for record in PREPROCESSED_EEG_RECORDS:
        if _epoch_annotation_path(record["subject"], record["run"]) is None:
            continue
        records.append(record)
    return records


EPOCH_RECORDS = _discover_epoch_records()
EPOCH_OUTPUTS = expand(
    EPOCHS_OUTPUT_PATTERN,
    zip,
    subject=[record["subject"] for record in EPOCH_RECORDS],
    task=[record["task"] for record in EPOCH_RECORDS],
    run=[record["run"] for record in EPOCH_RECORDS],
)
LMEEEG_CONFIG_PATH = f"{CONFIG_DIR}/lmeeeg.yaml"
LMEEEG_OUTPUT_DIR_PATTERN = f"{OUT_DIR}/lmeeeg/sub-{{subject}}/task-{{task}}/run-{{run}}"
LMEEEG_SUMMARY_OUTPUT_PATTERN = f"{LMEEEG_OUTPUT_DIR_PATTERN}/lmeeeg_analysis_summary.json"
LMEEEG_OUTPUTS = expand(
    LMEEEG_SUMMARY_OUTPUT_PATTERN,
    zip,
    subject=[record["subject"] for record in EPOCH_RECORDS],
    task=[record["task"] for record in EPOCH_RECORDS],
    run=[record["run"] for record in EPOCH_RECORDS],
)


rule make_epochs:
    input:
        eeg=PREPROCESSED_EEG_OUTPUT_PATTERN,
        raw=preprocess_raw_input,
        annotation=epoch_annotation_input,
        events_csv=EVENTS_CSV_OUTPUT,
    output:
        epochs=EPOCHS_OUTPUT_PATTERN,
    run:
        import numpy as np
        import pandas as pd
        import mne

        raw = mne.io.read_raw_fif(input.eeg, preload=True, verbose="ERROR")
        raw_source = mne.io.read_raw(input.raw, preload=False, verbose="ERROR")
        events_df = pd.read_csv(input.events_csv)

        status_channel = "Status" if "Status" in raw_source.ch_names else None
        anchor_kwargs = {"shortest_event": 1, "verbose": "ERROR"}
        if status_channel is not None:
            anchor_kwargs["stim_channel"] = status_channel
        anchor_events = mne.find_events(raw_source, **anchor_kwargs)
        if anchor_events.size == 0:
            raise ValueError(
                f"No conversation-start trigger found for sub-{wildcards.subject}, "
                f"task {wildcards.task}, run {wildcards.run}."
            )

        anchor_sample = int(anchor_events[0, 0])
        anchor_time_s = float((anchor_sample - raw_source.first_samp) / raw_source.info["sfreq"])

        recording_id = _epochs_recording_id(wildcards.subject)
        run_label = _normalize_run_label(wildcards.run)
        self_speaker_label = _subject_speaker_label(wildcards.subject)
        other_speaker_label = _other_speaker_label(wildcards.subject)
        event_types = tuple(EPOCHS_SETTINGS.get("event_types", ("fpp", "spp")))

        recording_events = events_df.loc[
            (events_df["recording_id"] == recording_id)
            & (events_df["run"].astype(str) == run_label)
        ].copy()

        metadata_frames = []
        for event_type in event_types:
            if event_type == "self_fpp_onset":
                speaker_column = "speaker_fpp"
                onset_column = "fpp_onset"
                offset_column = "fpp_offset"
                speaker_value = self_speaker_label
                event_role = "self"
                event_family = "fpp"
                event_lock = "onset"
            elif event_type == "self_fpp_offset":
                speaker_column = "speaker_fpp"
                onset_column = "fpp_onset"
                offset_column = "fpp_offset"
                speaker_value = self_speaker_label
                event_role = "self"
                event_family = "fpp"
                event_lock = "offset"
            elif event_type == "other_fpp_onset":
                speaker_column = "speaker_fpp"
                onset_column = "fpp_onset"
                offset_column = "fpp_offset"
                speaker_value = other_speaker_label
                event_role = "other"
                event_family = "fpp"
                event_lock = "onset"
            elif event_type == "other_fpp_offset":
                speaker_column = "speaker_fpp"
                onset_column = "fpp_onset"
                offset_column = "fpp_offset"
                speaker_value = other_speaker_label
                event_role = "other"
                event_family = "fpp"
                event_lock = "offset"
            elif event_type == "self_spp_onset":
                speaker_column = "speaker_spp"
                onset_column = "spp_onset"
                offset_column = "spp_offset"
                speaker_value = self_speaker_label
                event_role = "self"
                event_family = "spp"
                event_lock = "onset"
            elif event_type == "self_spp_offset":
                speaker_column = "speaker_spp"
                onset_column = "spp_onset"
                offset_column = "spp_offset"
                speaker_value = self_speaker_label
                event_role = "self"
                event_family = "spp"
                event_lock = "offset"
            elif event_type == "other_spp_onset":
                speaker_column = "speaker_spp"
                onset_column = "spp_onset"
                offset_column = "spp_offset"
                speaker_value = other_speaker_label
                event_role = "other"
                event_family = "spp"
                event_lock = "onset"
            elif event_type == "other_spp_offset":
                speaker_column = "speaker_spp"
                onset_column = "spp_onset"
                offset_column = "spp_offset"
                speaker_value = other_speaker_label
                event_role = "other"
                event_family = "spp"
                event_lock = "offset"
            else:
                raise ValueError(f"Unsupported epoch event type: {event_type}")

            selected = recording_events.loc[
                recording_events[speaker_column].astype(str) == speaker_value
            ].copy()
            if selected.empty:
                continue

            selected["event_type"] = event_type
            selected["event_role"] = event_role
            selected["event_family"] = event_family
            selected["event_lock"] = event_lock
            selected["event_speaker_label"] = speaker_value
            selected["event_source_column"] = onset_column if event_lock == "onset" else offset_column
            selected["event_source_offset_column"] = offset_column
            selected["event_onset_conversation_s"] = pd.to_numeric(selected[onset_column], errors="coerce")
            selected["event_offset_conversation_s"] = pd.to_numeric(selected[offset_column], errors="coerce")
            selected["conversation_anchor_sample"] = anchor_sample
            selected["conversation_anchor_time_s"] = anchor_time_s
            selected["event_latency_conversation_s"] = np.where(
                selected["event_lock"] == "onset",
                selected["event_onset_conversation_s"],
                selected["event_offset_conversation_s"],
            )
            selected["event_onset_s"] = selected["event_latency_conversation_s"] + anchor_time_s
            selected["event_offset_s"] = selected["event_offset_conversation_s"] + anchor_time_s
            selected["subject_id"] = f"sub-{wildcards.subject}"
            selected["task"] = wildcards.task
            selected["run"] = run_label
            selected["recording_id"] = recording_id
            selected["speaker_current"] = self_speaker_label
            selected["speaker_role"] = event_role
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
                    "event_role",
                    "event_family",
                    "event_lock",
                    "event_speaker_label",
                    "event_source_column",
                    "event_source_offset_column",
                    "event_onset_conversation_s",
                    "event_offset_conversation_s",
                    "event_latency_conversation_s",
                    "conversation_anchor_sample",
                    "conversation_anchor_time_s",
                    "event_onset_s",
                    "event_offset_s",
                    "subject_id",
                    "task",
                    "speaker_current",
                    "speaker_role",
                ]
            )

        event_id = {event_type: index + 1 for index, event_type in enumerate(event_types)}
        if len(metadata):
            onset_samples = raw.first_samp + np.rint(metadata["event_onset_s"].to_numpy(dtype=float) * raw.info["sfreq"]).astype(int)
            metadata = metadata.loc[
                (metadata["event_onset_s"] >= 0.0)
                & (onset_samples >= raw.first_samp)
                & (onset_samples <= raw.last_samp)
            ].reset_index(drop=True)
            sample_events = np.empty((len(metadata), 3), dtype=int)
            onset_samples = raw.first_samp + np.rint(metadata["event_onset_s"].to_numpy(dtype=float) * raw.info["sfreq"]).astype(int)
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


rule run_lmeeeg:
    input:
        epochs=EPOCHS_OUTPUT_PATTERN,
        config=LMEEEG_CONFIG_PATH,
    output:
        summary=LMEEEG_SUMMARY_OUTPUT_PATTERN,
    run:
        output_dir = os.path.dirname(output.summary)
        shell(
            'PYTHONPATH="{src_dir}" "{python_bin}" -m cas.cli.main lmeeeg '
            '--epochs "{epochs}" '
            '--config "{config}" '
            '--output-dir "{output_dir}"'.format(
                src_dir=SRC_DIR,
                python_bin=PYTHON_BIN,
                epochs=input.epochs,
                config=input.config,
                output_dir=output_dir,
            )
        )
