from __future__ import annotations

from pathlib import Path

from cas.preprocessing.config import (
    build_preprocessing_run_paths,
    resolve_preprocessing_output_layout,
)


def test_preprocessing_layout_resolves_structured_paths_against_output_dir(tmp_path: Path) -> None:
    paths_config = {
        "output_dir": str(tmp_path / "derived"),
        "paths": {
            "preprocessing": "{output_dir}/preprocessing",
            "features": "{output_dir}/features",
        },
    }
    preprocessing_config = {
        "preprocessing": {
            "final_dir": "{paths.preprocessing}/eeg",
            "intermediate_dir": "{paths.preprocessing}/eeg_verbose",
            "tables_dir": "{paths.preprocessing}/tables",
            "qc_dir": "{paths.preprocessing}/qc",
            "save_intermediates": True,
        }
    }

    layout = resolve_preprocessing_output_layout(
        paths_config,
        preprocessing_config,
        base_dir=tmp_path,
    )

    assert layout.preprocessing_root == (tmp_path / "derived" / "preprocessing")
    assert layout.final_dir == (tmp_path / "derived" / "preprocessing" / "eeg")
    assert layout.intermediate_dir == (tmp_path / "derived" / "preprocessing" / "eeg_verbose")
    assert layout.tables_dir == (tmp_path / "derived" / "preprocessing" / "tables")
    assert layout.qc_dir == (tmp_path / "derived" / "preprocessing" / "qc")
    assert layout.save_intermediates is True


def test_preprocessing_run_paths_follow_new_output_layout(tmp_path: Path) -> None:
    layout = resolve_preprocessing_output_layout(
        {
            "output_dir": str(tmp_path / "derived"),
            "paths": {"preprocessing": "{output_dir}/preprocessing"},
        },
        {"preprocessing": {"save_intermediates": True}},
        base_dir=tmp_path,
    )

    run_paths = build_preprocessing_run_paths(
        layout=layout,
        subject="001",
        task="conversation",
        run="2",
        dyad_id="dyad-001",
    )

    assert run_paths.eeg_path == (
        tmp_path
        / "derived"
        / "preprocessing"
        / "eeg"
        / "sub-001"
        / "task-conversation"
        / "run-2"
        / "preprocessed_eeg.fif"
    )
    assert run_paths.events_path.name == "sub-001_task-conversation_run-2_events.tsv"
    assert run_paths.summary_path.name == "sub-001_task-conversation_run-2_summary.json"
    assert run_paths.intermediates_dir == (
        tmp_path
        / "derived"
        / "preprocessing"
        / "eeg_verbose"
        / "dyad-001"
        / "sub-001"
        / "run-2"
    )
