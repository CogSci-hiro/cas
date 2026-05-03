from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

from cas.stats.lmeeeg_pipeline import (
    _apply_duration_controls,
    _configure_mne_runtime,
    _build_adjacency,
    _patch_mne_cluster_level,
    _prepare_model_inputs,
    _augment_lmeeeg_metadata,
    derive_fpp_spp_conf_disc_class,
    _fit_one_model,
    _model_output_dir,
    _normalize_effect_name,
    _run_permutation_inference,
    _resolve_model_formula,
    _resolve_term_tests,
    _resolve_test_effects,
    _resolve_term_test_effects,
    _run_model_inference,
    _row_from_epochs_path,
    _resolve_pooled_source_paths,
    _resolve_project_out_dir,
    load_lmeeeg_config,
    load_epochs_with_metadata,
    run_pooled_lmeeeg_analysis,
    select_epochs_from_config,
)


class FakeEpochs:
    def __init__(self, metadata: pd.DataFrame, selection: np.ndarray | None = None):
        self.metadata = metadata.reset_index(drop=True)
        self.selection = np.arange(len(self.metadata), dtype=int) if selection is None else np.asarray(selection, dtype=int)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, item):
        if isinstance(item, str):
            mask = self.metadata["event_type"].astype(str) == item
        else:
            values = np.asarray(item)
            if values.dtype == bool:
                mask = values
            else:
                mask = np.zeros(len(self.metadata), dtype=bool)
                mask[values.astype(int)] = True
        return FakeEpochs(self.metadata.loc[mask].reset_index(drop=True), self.selection[mask])


def test_load_lmeeeg_config_supports_section_wrapper(tmp_path):
    config_path = tmp_path / "lmeeeg.yaml"
    config_path.write_text(
        """
lmeeeg:
  selection:
    event_type: "self_fpp_onset"
  models:
    demo:
      formula: "~ proximity"
      test_predictors: ["proximity"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    loaded = load_lmeeeg_config(config_path)
    assert loaded["selection"]["event_type"] == "self_fpp_onset"
    assert "demo" in loaded["models"]


def test_select_epochs_from_config_applies_event_type_and_metadata_query():
    metadata = pd.DataFrame(
        {
            "event_type": ["self_fpp_onset", "self_fpp_onset", "other_fpp_onset"],
            "latency": [0.1, 0.4, 0.9],
            "subject_id": ["sub-001", "sub-001", "sub-002"],
        }
    )
    epochs = FakeEpochs(metadata)
    config = {
        "selection": {
            "event_type": "self_fpp_onset",
            "metadata_query": "latency >= 0.3",
        },
        "models": {"demo": {"formula": "~ latency", "test_predictors": ["latency"]}},
    }

    selected_epochs, selected_metadata = select_epochs_from_config(epochs, metadata, config)

    assert len(selected_epochs) == 1
    assert selected_epochs.selection.tolist() == [1]
    assert selected_metadata["latency"].tolist() == [0.4]


def test_select_epochs_from_config_raises_on_empty_selection():
    metadata = pd.DataFrame({"event_type": ["self_fpp_onset"], "latency": [0.1]})
    epochs = FakeEpochs(metadata)
    config = {
        "selection": {"event_type": "other_fpp_onset"},
        "models": {"demo": {"formula": "~ latency", "test_predictors": ["latency"]}},
    }

    with pytest.raises(ValueError, match="zero rows"):
        select_epochs_from_config(epochs, metadata, config)


def test_augment_lmeeeg_metadata_derives_duration_from_event_family():
    metadata = pd.DataFrame(
        {
            "event_family": ["spp", "fpp"],
            "spp_duration": [0.25, 0.40],
            "fpp_duration": [1.2, 1.5],
        }
    )

    augmented = _augment_lmeeeg_metadata(metadata)

    assert augmented["duration_s"].tolist() == [0.25, 1.5]


def test_augment_lmeeeg_metadata_splits_label_classes():
    metadata = pd.DataFrame(
        {
            "fpp_label": ["FPP_RFC_DECL", "FPP_RFC_INT"],
            "spp_label": ["SPP_CONF_SIMP", "SPP_DISC_CORR"],
        }
    )

    augmented = _augment_lmeeeg_metadata(metadata)

    assert augmented["fpp_class_1"].tolist() == ["RFC", "RFC"]
    assert augmented["fpp_class_2"].tolist() == ["DECL", "INT"]
    assert augmented["spp_class_1"].tolist() == ["CONF", "DISC"]
    assert augmented["spp_class_2"].tolist() == ["SIMP", "CORR"]


def test_derive_fpp_spp_conf_disc_class_maps_requested_labels_and_durations():
    metadata = pd.DataFrame(
        {
            "event_family": ["fpp", "spp", "spp", "spp"],
            "fpp_duration": [0.8, 0.8, 0.8, 0.8],
            "spp_duration": [0.3, 0.4, 0.5, 0.6],
            "spp_label": ["", "SPP_CONF_SIMP", "SPP_DISC_CORR", "SPP_OVERLAP"],
        }
    )

    derived = derive_fpp_spp_conf_disc_class(metadata)

    assert derived["class_3"].tolist() == ["FPP", "SPP_CONF", "SPP_DISC", "OTHER"]
    assert np.isclose(derived.loc[0, "duration"], 0.8)
    assert np.isclose(derived.loc[1, "duration"], 0.4)
    assert np.isclose(derived.loc[2, "log_duration"], np.log(0.5))


def test_derive_fpp_spp_conf_disc_class_centers_log_duration_within_class():
    metadata = pd.DataFrame(
        {
            "event_family": ["fpp", "fpp", "spp", "spp"],
            "fpp_duration": [1.0, 2.0, 1.0, 2.0],
            "spp_duration": [0.5, 0.5, 1.5, 3.0],
            "spp_label": ["", "", "SPP_CONF_SIMP", "SPP_CONF_EXP"],
        }
    )

    derived = derive_fpp_spp_conf_disc_class(metadata)

    means = derived.groupby("class_3")["log_duration_within_class"].mean()
    assert np.isclose(means.loc["FPP"], 0.0)
    assert np.isclose(means.loc["SPP_CONF"], 0.0)


def test_derive_fpp_spp_conf_disc_class_prefers_event_family_over_part():
    metadata = pd.DataFrame(
        {
            "event_family": ["fpp", "spp"],
            "part": ["SPP", "SPP"],
            "fpp_duration": [1.2, 1.2],
            "spp_duration": [0.5, 0.7],
            "spp_label": ["SPP_CONF_SIMP", "SPP_DISC_CORR"],
        }
    )

    derived = derive_fpp_spp_conf_disc_class(metadata)

    assert derived["class_3"].tolist() == ["FPP", "SPP_DISC"]


def test_prepare_model_inputs_applies_eligibility_standardization_and_formula_alignment():
    runtime_config = {
        "lmeeeg": {
            "random_effects": {"group": "subject_id"},
            "models": {
                "demo": {
                    "formula": "~ fpp_class_2 + latency + duration_s + run",
                    "eligibility": {
                        "not_null": ["fpp_class_2", "latency", "duration_s"],
                        "min_values": {"duration_s": 0.10},
                    },
                    "standardize": ["latency", "duration_s"],
                }
            },
        }
    }
    eeg = np.arange(4, dtype=float).reshape(4, 1, 1)
    metadata = pd.DataFrame(
        {
            "fpp_class_2": ["DECL", None, "INT", "DECL"],
            "latency": [1.0, 2.0, None, 4.0],
            "duration_s": [0.20, 0.30, 0.40, 0.05],
            "run": [1, 1, 2, 2],
            "subject_id": ["sub-001", "sub-001", "sub-002", "sub-002"],
        }
    )

    cleaned_eeg, cleaned_metadata = _prepare_model_inputs(
        runtime_config,
        model_name="demo",
        eeg_data=eeg,
        metadata=metadata,
    )

    assert cleaned_eeg.shape == (1, 1, 1)
    assert cleaned_eeg[:, 0, 0].tolist() == [0.0]
    assert cleaned_metadata["fpp_class_2"].tolist() == ["DECL"]
    assert cleaned_metadata["latency"].tolist() == [0.0]
    assert cleaned_metadata["duration_s"].tolist() == [0.0]


def test_prepare_model_inputs_drops_formula_missing_rows_even_without_explicit_eligibility():
    runtime_config = {
        "lmeeeg": {
            "random_effects": {"group": "subject_id"},
            "models": {
                "demo": {
                    "formula": "~ latency + run",
                }
            },
        }
    }
    eeg = np.arange(3, dtype=float).reshape(3, 1, 1)
    metadata = pd.DataFrame(
        {
            "latency": [0.1, None, 0.3],
            "run": [1, 1, 2],
            "subject_id": ["sub-001", "sub-001", "sub-002"],
        }
    )

    cleaned_eeg, cleaned_metadata = _prepare_model_inputs(
        runtime_config,
        model_name="demo",
        eeg_data=eeg,
        metadata=metadata,
    )

    assert cleaned_eeg[:, 0, 0].tolist() == [0.0, 2.0]
    assert cleaned_metadata["latency"].tolist() == [0.1, 0.3]


def test_prepare_model_inputs_applies_fpp_spp_cycle_position_design_recipe():
    runtime_config = {
        "lmeeeg": {
            "analysis_name": "fpp_spp_cycle_position",
            "models": {
                "cycle_position": {
                    "formula": "power ~ pair_position + z_event_duration + z_latency + run + z_time_within_run + (1 | subject)",
                }
            },
            "design": {
                "column_mapping": {
                    "subject": "subject_id",
                    "pair_position": "event_family",
                    "event_duration": "duration_s",
                    "time_within_run": "event_onset_conversation_s",
                },
                "value_mapping": {
                    "pair_position": {"fpp": "FPP", "spp": "SPP"},
                },
                "predictors": {
                    "categorical": ["pair_position"],
                    "continuous": ["run", "event_duration", "latency", "time_within_run"],
                },
                "zscore": {
                    "event_duration": "z_event_duration",
                    "latency": "z_latency",
                    "time_within_run": "z_time_within_run",
                },
                "required_columns": [
                    "subject",
                    "run",
                    "pair_position",
                    "event_duration",
                    "latency",
                    "time_within_run",
                ],
                "reference_levels": {"pair_position": "SPP"},
            },
        }
    }
    eeg = np.arange(4, dtype=float).reshape(4, 1, 1)
    metadata = pd.DataFrame(
        {
            "subject_id": ["sub-001", "sub-001", "sub-002", "sub-002"],
            "run": ["1", "1", "2", "2"],
            "event_family": ["fpp", "spp", "fpp", "spp"],
            "duration_s": [0.4, 0.5, 0.6, 0.7],
            "latency": [0.2, 0.2, 0.4, 0.4],
            "event_onset_conversation_s": [1.0, 2.0, 3.0, 4.0],
        }
    )

    cleaned_eeg, cleaned_metadata = _prepare_model_inputs(
        runtime_config,
        model_name="cycle_position",
        eeg_data=eeg,
        metadata=metadata,
    )

    assert cleaned_eeg.shape == (4, 1, 1)
    assert {"z_event_duration", "z_latency", "z_time_within_run"}.issubset(cleaned_metadata.columns)
    assert list(cleaned_metadata["pair_position"].cat.categories) == ["SPP", "FPP"]
    assert cleaned_metadata["subject"].tolist() == ["sub-001", "sub-001", "sub-002", "sub-002"]
    assert cleaned_metadata["run"].dtype.kind in {"i", "u", "f"}
    assert np.isclose(cleaned_metadata["z_event_duration"].mean(), 0.0)


def test_prepare_model_inputs_rejects_invalid_pair_position_value():
    runtime_config = {
        "lmeeeg": {
            "models": {"cycle_position": {"formula": "~ pair_position + latency"}},
            "design": {
                "column_mapping": {
                    "subject": "subject_id",
                    "pair_position": "event_family",
                    "event_duration": "duration_s",
                    "time_within_run": "event_onset_conversation_s",
                },
                "value_mapping": {"pair_position": {"fpp": "FPP", "spp": "SPP"}},
                "predictors": {
                    "categorical": ["pair_position"],
                    "continuous": ["run", "event_duration", "latency", "time_within_run"],
                },
                "required_columns": ["subject", "run", "pair_position", "event_duration", "latency", "time_within_run"],
                "reference_levels": {"pair_position": "SPP"},
            },
        }
    }

    metadata = pd.DataFrame(
        {
            "subject_id": ["sub-001", "sub-001"],
            "run": ["1", "1"],
            "event_family": ["fpp", "third"],
            "duration_s": [0.4, 0.5],
            "latency": [0.2, 0.3],
            "event_onset_conversation_s": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="invalid levels"):
        _prepare_model_inputs(
            runtime_config,
            model_name="cycle_position",
            eeg_data=np.arange(2, dtype=float).reshape(2, 1, 1),
            metadata=metadata,
        )


def test_prepare_model_inputs_rejects_missing_required_column():
    runtime_config = {
        "lmeeeg": {
            "models": {"cycle_position": {"formula": "~ pair_position + latency"}},
            "design": {
                "required_columns": ["subject", "run", "pair_position", "event_duration", "latency", "time_within_run"],
            },
        }
    }

    metadata = pd.DataFrame({"subject": ["sub-001"], "run": ["1"], "pair_position": ["SPP"]})

    with pytest.raises(ValueError, match="missing required design columns"):
        _prepare_model_inputs(
            runtime_config,
            model_name="cycle_position",
            eeg_data=np.arange(1, dtype=float).reshape(1, 1, 1),
            metadata=metadata,
        )


def test_prepare_model_inputs_rejects_non_positive_event_duration():
    runtime_config = {
        "lmeeeg": {
            "models": {"cycle_position": {"formula": "~ pair_position + latency"}},
            "design": {
                "predictors": {
                    "categorical": ["pair_position"],
                    "continuous": ["run", "event_duration", "latency", "time_within_run"],
                },
                "required_columns": ["subject", "run", "pair_position", "event_duration", "latency", "time_within_run"],
                "reference_levels": {"pair_position": "SPP"},
            },
        }
    }
    metadata = pd.DataFrame(
        {
            "subject": ["sub-001", "sub-001"],
            "run": ["1", "1"],
            "pair_position": ["SPP", "FPP"],
            "event_duration": [0.0, 0.5],
            "latency": [0.2, 0.3],
            "time_within_run": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="event_duration"):
        _prepare_model_inputs(
            runtime_config,
            model_name="cycle_position",
            eeg_data=np.arange(2, dtype=float).reshape(2, 1, 1),
            metadata=metadata,
        )


def test_apply_duration_controls_creates_log_spline_and_bin_features():
    runtime_config = {
        "lmeeeg": {
            "models": {
                "demo": {
                    "duration_controls": {
                        "duration_column": "duration_s",
                        "log_duration": {
                            "enabled": True,
                            "output_column": "z_log_spp_duration",
                            "offset_s": 0.001,
                            "standardize": True,
                        },
                        "spline_duration": {
                            "enabled": True,
                            "source": "z_log_spp_duration",
                            "output_prefix": "spline_log_spp_duration",
                            "df": 4,
                            "degree": 3,
                            "include_intercept": False,
                            "standardize": True,
                        },
                        "duration_bins": {
                            "enabled": True,
                            "output_column": "spp_duration_bin",
                            "n_bins": 4,
                            "labels_prefix": "dur_q",
                            "min_unique_bins": 3,
                        },
                    },
                    "term_tests": {
                        "class_effect": {"enabled": True, "terms": ["spp_class_1"]},
                    },
                }
            }
        }
    }
    metadata = pd.DataFrame(
        {
            "duration_s": [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75],
            "spp_class_1": ["CONF", "CONF", "CONF", "DISC", "DISC", "DISC", "DISC", "DISC"],
        }
    )

    prepared, artifacts = _apply_duration_controls(metadata, runtime_config, model_name="demo")

    assert "z_log_spp_duration" in prepared.columns
    assert np.isclose(prepared["z_log_spp_duration"].mean(), 0.0)
    spline_columns = [column for column in prepared.columns if column.startswith("spline_log_spp_duration_")]
    assert len(spline_columns) == 3
    assert "spp_duration_bin" in prepared.columns
    assert artifacts["term_aliases"]["spline_log_spp_duration"] == spline_columns
    design = np.column_stack(
        [
            np.ones(len(prepared), dtype=float),
            prepared["z_log_spp_duration"].to_numpy(dtype=float),
            prepared[spline_columns].to_numpy(dtype=float),
        ]
    )
    assert np.linalg.matrix_rank(design) == design.shape[1]
    for column in spline_columns:
        reduced_columns = [
            np.ones(len(prepared), dtype=float),
            prepared["z_log_spp_duration"].to_numpy(dtype=float),
        ]
        reduced_columns.extend(
            prepared[other].to_numpy(dtype=float)
            for other in spline_columns
            if other != column
        )
        reduced = np.column_stack(reduced_columns)
        effect = prepared[column].to_numpy(dtype=float)
        residualized = effect - reduced @ (np.linalg.pinv(reduced) @ effect)
        assert float(residualized @ residualized) > 1e-8


def test_apply_duration_controls_keeps_zero_duration_when_log_offset_is_configured():
    runtime_config = {
        "lmeeeg": {
            "models": {
                "demo": {
                    "duration_controls": {
                        "duration_column": "duration_s",
                        "log_duration": {
                            "enabled": True,
                            "output_column": "z_log_spp_duration",
                            "offset_s": 0.01,
                            "standardize": False,
                        },
                    }
                }
            }
        }
    }
    metadata = pd.DataFrame({"duration_s": [0.0, 0.2, 0.4]})

    prepared, _ = _apply_duration_controls(metadata, runtime_config, model_name="demo")

    assert len(prepared) == 3
    assert np.isfinite(prepared["z_log_spp_duration"]).all()


def test_apply_duration_controls_handles_duplicate_quantile_edges():
    runtime_config = {
        "lmeeeg": {
            "models": {
                "demo": {
                    "duration_controls": {
                        "duration_column": "duration_s",
                        "duration_bins": {
                            "enabled": True,
                            "output_column": "spp_duration_bin",
                            "n_bins": 5,
                            "labels_prefix": "dur_q",
                            "min_unique_bins": 3,
                        },
                    }
                }
            }
        }
    }
    metadata = pd.DataFrame({"duration_s": [0.1, 0.1, 0.2, 0.2, 0.3, 0.4]})

    prepared, artifacts = _apply_duration_controls(metadata, runtime_config, model_name="demo")

    assert "spp_duration_bin" in prepared.columns
    assert prepared["spp_duration_bin"].nunique() >= 3
    assert artifacts["transforms"][-1]["n_bins_observed"] >= 3


def test_apply_duration_controls_common_support_filters_nonoverlap():
    runtime_config = {
        "lmeeeg": {
            "models": {
                "demo": {
                    "duration_controls": {
                        "duration_column": "duration_s",
                        "common_support": {
                            "enabled": True,
                            "class_column": "spp_class_1",
                            "lower_quantile": 0.0,
                            "upper_quantile": 1.0,
                        },
                    }
                }
            }
        }
    }
    metadata = pd.DataFrame(
        {
            "duration_s": [0.2, 0.4, 0.6, 0.4, 0.8, 1.0],
            "spp_class_1": ["CONF", "CONF", "CONF", "DISC", "DISC", "DISC"],
        }
    )

    prepared, artifacts = _apply_duration_controls(metadata, runtime_config, model_name="demo")

    assert prepared["duration_s"].tolist() == [0.4, 0.6, 0.4]
    assert artifacts["common_support"]["counts_before"] == {"CONF": 3, "DISC": 3}
    assert artifacts["common_support"]["counts_after"] == {"CONF": 2, "DISC": 1}


def test_resolve_model_formula_keeps_old_style_when_duration_controls_are_absent():
    runtime_config = {
        "lmeeeg": {
            "models": {"demo": {"formula": "~ spp_class_1 + latency + duration_s + run"}}
        }
    }

    formula, formula_rhs, _ = _resolve_model_formula(runtime_config, model_name="demo")

    assert formula == "y ~ spp_class_1 + latency + duration_s + run + (1|subject_id)"
    assert formula_rhs == "spp_class_1 + latency + duration_s + run"


def test_resolve_model_formula_adds_log_only_duration_term():
    runtime_config = {"lmeeeg": {"models": {"demo": {"formula": "~ spp_class_1 + latency + run"}}}}
    duration_artifacts = {"formula_terms": ["z_log_spp_duration"]}

    formula, formula_rhs, _ = _resolve_model_formula(
        runtime_config,
        model_name="demo",
        duration_artifacts=duration_artifacts,
    )

    assert formula_rhs == "spp_class_1 + latency + run + z_log_spp_duration"
    assert "z_log_spp_duration" in formula


def test_resolve_model_formula_adds_spline_and_bin_terms_to_one_full_model():
    runtime_config = {"lmeeeg": {"models": {"demo": {"formula": "~ spp_class_1 + latency + run"}}}}
    duration_artifacts = {
        "formula_terms": [
            "z_log_spp_duration",
            "spline_log_spp_duration_1",
            "spline_log_spp_duration_2",
            "spp_duration_bin",
        ]
    }

    _, formula_rhs, _ = _resolve_model_formula(
        runtime_config,
        model_name="demo",
        duration_artifacts=duration_artifacts,
    )

    assert formula_rhs == (
        "spp_class_1 + latency + run + z_log_spp_duration + "
        "spline_log_spp_duration_1 + spline_log_spp_duration_2 + spp_duration_bin"
    )


def test_resolve_term_tests_groups_duration_terms():
    runtime_config = {
        "lmeeeg": {
            "models": {
                "demo": {
                    "term_tests": {
                        "class_effect": {"enabled": True, "terms": ["spp_class_1"]},
                        "duration_log": {"enabled": True, "terms": ["z_log_spp_duration"]},
                        "duration_spline": {"enabled": True, "terms": ["spline_log_spp_duration"]},
                        "duration_bin": {"enabled": True, "terms": ["spp_duration_bin"]},
                    }
                }
            }
        }
    }
    metadata = pd.DataFrame(
        {
            "spp_class_1": pd.Categorical(["CONF", "DISC"]),
            "z_log_spp_duration": [0.1, 0.2],
            "spp_duration_bin": pd.Categorical(["dur_q1", "dur_q2"]),
        }
    )
    duration_artifacts = {
        "term_aliases": {"spline_log_spp_duration": ["spline_log_spp_duration_1", "spline_log_spp_duration_2"]}
    }

    groups = _resolve_term_tests(
        runtime_config,
        model_name="demo",
        duration_artifacts=duration_artifacts,
        metadata=metadata,
    )

    assert [group["group_name"] for group in groups] == [
        "class_effect",
        "duration_log",
        "duration_spline",
        "duration_bin",
    ]


def test_fit_one_model_uses_fixed_column_names(tmp_path, monkeypatch):
    class DummyDesignSpec:
        fixed_column_names = ["Intercept", "latency"]

    class DummyFitResult:
        design_spec = DummyDesignSpec()
        ols_betas = {
            "Intercept": np.ones((1, 2), dtype=float),
            "latency": np.full((1, 2), 2.0, dtype=float),
        }
        ols_t_values = {
            "Intercept": np.full((1, 2), 3.0, dtype=float),
            "latency": np.full((1, 2), 4.0, dtype=float),
        }

    monkeypatch.setattr(
        "lmeeeg.fit_lmm_mass_univariate",
        lambda **kwargs: DummyFitResult(),
    )

    runtime_config = {
        "paths": {"out_dir": str(tmp_path)},
        "lmeeeg": {"models": {"demo": {"formula": "~ latency", "test_predictors": ["latency"]}}},
    }
    trial_data = SimpleNamespace(
        eeg_data=np.ones((2, 1, 2), dtype=float),
        trial_metadata=pd.DataFrame(
            {"latency": [0.1, 0.2], "subject_id": ["sub-001", "sub-002"]}
        ),
        channel_names=["Cz"],
        times=np.array([0.1, 0.2], dtype=float),
    )

    summary = _fit_one_model(runtime_config, trial_data, model_name="demo")

    assert summary["betas_shape"] == [2, 1, 2]
    assert summary["test_predictors"] == ["latency"]
    assert summary["contrast_of_interest"] == "latency"
    assert summary["resolved_test_effects"] == {"latency": ["latency"]}
    assert (tmp_path / "lmeeeg" / "demo" / "column_names.json").exists()
    saved_summary = json.loads((tmp_path / "lmeeeg" / "demo" / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["contrast_of_interest"] == "latency"


def test_resolve_test_effects_expands_categorical_predictor():
    fixed_column_names = [
        "Intercept",
        "fpp_class_2[T.INF]",
        "fpp_class_2[T.INT]",
        "latency",
    ]

    assert _resolve_test_effects(fixed_column_names, "latency") == ["latency"]
    assert _resolve_test_effects(fixed_column_names, "fpp_class_2") == [
        "fpp_class_2[T.INF]",
        "fpp_class_2[T.INT]",
    ]


def test_resolve_test_effects_accepts_normalized_effect_name():
    fixed_column_names = ["Intercept", "pair_position[T.FPP]", "z_latency"]

    assert _resolve_test_effects(fixed_column_names, "pair_positionFPP") == ["pair_position[T.FPP]"]
    assert _normalize_effect_name("pair_position[T.FPP]") == "pair_positionFPP"


def test_resolve_term_test_effects_maps_grouped_duration_terms():
    fixed_column_names = [
        "Intercept",
        "spp_class_1[T.DISC]",
        "z_log_spp_duration",
        "spline_log_spp_duration_1",
        "spline_log_spp_duration_2",
        "spp_duration_bin[T.dur_q2]",
    ]
    term_tests = [
        {"group_name": "class_effect", "contrast_type": "categorical", "terms": ["spp_class_1"]},
        {"group_name": "duration_log", "contrast_type": "continuous", "terms": ["z_log_spp_duration"]},
        {"group_name": "duration_spline", "contrast_type": "grouped_continuous", "terms": ["spline_log_spp_duration"]},
        {"group_name": "duration_bin", "contrast_type": "categorical", "terms": ["spp_duration_bin"]},
    ]

    resolved = _resolve_term_test_effects(fixed_column_names, term_tests)

    assert resolved[0]["group_name"] == "class_effect"
    assert resolved[0]["terms"][0]["raw_effects"] == ["spp_class_1[T.DISC]"]
    assert resolved[1]["group_name"] == "duration_log"
    assert resolved[1]["terms"][0]["raw_effects"] == ["z_log_spp_duration"]
    assert resolved[2]["group_name"] == "duration_spline"
    assert resolved[2]["terms"][0]["raw_effects"] == [
        "spline_log_spp_duration_1",
        "spline_log_spp_duration_2",
    ]
    assert resolved[3]["group_name"] == "duration_bin"
    assert resolved[3]["terms"][0]["raw_effects"] == ["spp_duration_bin[T.dur_q2]"]


def test_run_model_inference_runs_each_expanded_effect(tmp_path, monkeypatch):
    runtime_config = {
        "paths": {"out_dir": str(tmp_path)},
        "lmeeeg": {
            "models": {"demo": {"formula": "~ fpp_class_2 + latency", "test_predictors": ["fpp_class_2"]}},
            "test": {"method": "maxstat", "n_permutations": 8, "seed": 0, "tail": 0},
        },
    }
    trial_data = SimpleNamespace(
        eeg_data=np.ones((3, 1, 2), dtype=float),
        trial_metadata=pd.DataFrame({"subject_id": ["sub-001", "sub-002", "sub-003"]}),
        channel_names=["Cz"],
        times=np.array([0.1, 0.2], dtype=float),
    )

    class DummyDesignSpec:
        fixed_column_names = ["Intercept", "fpp_class_2[T.INF]", "fpp_class_2[T.INT]", "latency"]

    fit_result = SimpleNamespace(design_spec=DummyDesignSpec())
    seen_effects: list[str] = []

    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline._refit_for_inference",
        lambda runtime_config, trial_data, *, model_name, formula_override=None: fit_result,
    )

    def fake_permute_fixed_effect(fit_result, *, effect, **kwargs):
        seen_effects.append(effect)
        return SimpleNamespace(
            observed_statistic=np.ones((1, 2), dtype=float),
            corrected_p_values=np.full((1, 2), 0.04, dtype=float),
        )

    monkeypatch.setattr("lmeeeg.permute_fixed_effect", fake_permute_fixed_effect)

    results = _run_model_inference(runtime_config, trial_data, model_name="demo")

    assert seen_effects == ["fpp_class_2[T.INF]", "fpp_class_2[T.INT]"]
    assert [result["effect"] for result in results] == seen_effects
    assert all(result["requested_predictor"] == "fpp_class_2" for result in results)
    assert all(result["contrast_of_interest"] == "fpp_class_2" for result in results)
    csv_path = Path(results[0]["corrected_p_values_csv"])
    assert csv_path.exists()
    csv_rows = pd.read_csv(csv_path)
    assert csv_rows.columns.tolist() == ["p_values", "channel", "time"]
    assert csv_rows["channel"].tolist() == ["Cz", "Cz"]
    assert csv_rows["time"].tolist() == [0.1, 0.2]
    assert csv_rows["p_values"].tolist() == [0.04, 0.04]


def test_run_model_inference_writes_grouped_duration_summary(tmp_path, monkeypatch):
    runtime_config = {
        "paths": {"out_dir": str(tmp_path)},
        "lmeeeg": {
            "models": {
                "demo": {
                    "formula": "~ spp_class_1 + z_log_spp_duration",
                    "term_tests": {
                        "class_effect": {"enabled": True, "terms": ["spp_class_1"]},
                        "duration_log": {"enabled": True, "terms": ["z_log_spp_duration"]},
                        "duration_spline": {"enabled": True, "terms": ["spline_log_spp_duration"]},
                    },
                }
            },
            "test": {"method": "maxstat", "n_permutations": 8, "seed": 0, "tail": 0},
        },
    }
    trial_data = SimpleNamespace(
        eeg_data=np.ones((4, 1, 2), dtype=float),
        trial_metadata=pd.DataFrame(
            {
                "subject_id": ["sub-001", "sub-002", "sub-003", "sub-004"],
                "spp_class_1": pd.Categorical(["CONF", "DISC", "CONF", "DISC"]),
                "z_log_spp_duration": [0.1, 0.2, 0.3, 0.4],
            }
        ),
        channel_names=["Cz"],
        times=np.array([0.1, 0.2], dtype=float),
    )

    class DummyDesignSpec:
        fixed_column_names = [
            "Intercept",
            "spp_class_1[T.DISC]",
            "z_log_spp_duration",
            "spline_log_spp_duration_1",
            "spline_log_spp_duration_2",
        ]

    fit_result = SimpleNamespace(design_spec=DummyDesignSpec())

    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline._refit_for_inference",
        lambda runtime_config, trial_data, *, model_name, formula_override=None: fit_result,
    )
    monkeypatch.setattr(
        "lmeeeg.permute_fixed_effect",
        lambda fit_result, *, effect, **kwargs: SimpleNamespace(
            observed_statistic=np.asarray([[1.0, -2.0]], dtype=float),
            corrected_p_values=np.asarray([[0.01, 0.20]], dtype=float),
        ),
    )

    results = _run_model_inference(
        runtime_config,
        trial_data,
        model_name="demo",
        fit_result=fit_result,
        term_tests_override=[
            {"group_name": "class_effect", "contrast_type": "categorical", "terms": ["spp_class_1"]},
            {"group_name": "duration_log", "contrast_type": "continuous", "terms": ["z_log_spp_duration"]},
            {"group_name": "duration_spline", "contrast_type": "grouped_continuous", "terms": ["spline_log_spp_duration"]},
        ],
    )

    assert {result["term_group"] for result in results} == {
        "class_effect",
        "duration_log",
        "duration_spline",
    }
    summary_path = tmp_path / "lmeeeg" / "demo" / "inference_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert {group["term_group"] for group in payload["groups"]} == {
        "class_effect",
        "duration_log",
        "duration_spline",
    }


def test_run_model_inference_skips_adjacency_for_maxstat(tmp_path, monkeypatch):
    runtime_config = {
        "paths": {"out_dir": str(tmp_path)},
        "lmeeeg": {
            "models": {"demo": {"formula": "~ pair_position", "test_predictors": ["pair_position"]}},
            "test": {"method": "maxstat", "n_permutations": 8, "seed": 0, "tail": 0},
        },
    }
    trial_data = SimpleNamespace(
        eeg_data=np.ones((3, 1, 2), dtype=float),
        trial_metadata=pd.DataFrame({"subject_id": ["sub-001", "sub-002", "sub-003"]}),
        channel_names=["Cz"],
        times=np.array([0.1, 0.2], dtype=float),
    )

    class DummyDesignSpec:
        fixed_column_names = ["Intercept", "pair_position[T.FPP]"]

    fit_result = SimpleNamespace(design_spec=DummyDesignSpec())

    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline._refit_for_inference",
        lambda runtime_config, trial_data, *, model_name, formula_override=None: fit_result,
    )
    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline._build_adjacency",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("adjacency should not be built")),
    )
    monkeypatch.setattr(
        "lmeeeg.permute_fixed_effect",
        lambda fit_result, *, effect, **kwargs: SimpleNamespace(
            observed_statistic=np.ones((1, 2), dtype=float),
            corrected_p_values=np.full((1, 2), 0.04, dtype=float),
        ),
    )

    results = _run_model_inference(runtime_config, trial_data, model_name="demo")

    assert len(results) == 1
    assert results[0]["effect"] == "pair_position[T.FPP]"


def test_run_model_inference_reuses_existing_fit_result(tmp_path, monkeypatch):
    runtime_config = {
        "paths": {"out_dir": str(tmp_path)},
        "lmeeeg": {
            "models": {"demo": {"formula": "~ pair_position", "test_predictors": ["pair_position"]}},
            "test": {"method": "maxstat", "n_permutations": 8, "seed": 0, "tail": 0},
        },
    }
    trial_data = SimpleNamespace(
        eeg_data=np.ones((3, 1, 2), dtype=float),
        trial_metadata=pd.DataFrame({"subject_id": ["sub-001", "sub-002", "sub-003"]}),
        channel_names=["Cz"],
        times=np.array([0.1, 0.2], dtype=float),
    )

    class DummyDesignSpec:
        fixed_column_names = ["Intercept", "pair_position[T.FPP]"]

    fit_result = SimpleNamespace(design_spec=DummyDesignSpec())

    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline._refit_for_inference",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not refit")),
    )
    monkeypatch.setattr(
        "lmeeeg.permute_fixed_effect",
        lambda fit_result, *, effect, **kwargs: SimpleNamespace(
            observed_statistic=np.ones((1, 2), dtype=float),
            corrected_p_values=np.full((1, 2), 0.04, dtype=float),
        ),
    )

    results = _run_model_inference(
        runtime_config,
        trial_data,
        model_name="demo",
        fit_result=fit_result,
    )

    assert len(results) == 1
    assert results[0]["effect"] == "pair_position[T.FPP]"


def test_run_permutation_inference_skips_on_backend_failure(monkeypatch):
    calls: list[str] = []

    def fake_permute_fixed_effect(fit_result, *, effect, correction, **kwargs):
        calls.append(correction)
        raise AttributeError("'function' object has no attribute 'get_call_template'")

    monkeypatch.setattr("lmeeeg.permute_fixed_effect", fake_permute_fixed_effect)

    result = _run_permutation_inference(
        SimpleNamespace(),
        effect_name="latency",
        correction="cluster",
        n_permutations=8,
        seed=0,
        tail=0,
        threshold=2.0,
        adjacency=np.eye(2),
    )

    assert calls == ["cluster"]
    assert result["status"] == "skipped"
    assert result["correction"] == "cluster"
    assert "get_call_template" in result["error"]
    assert "Traceback" in result["traceback"]
    assert "fake_permute_fixed_effect" in result["traceback"]


def test_build_adjacency_uses_biosemi64_spatial_graph(monkeypatch):
    calls: dict[str, object] = {}
    spatial_adjacency = object()

    def fake_create_info(channel_names, sfreq, ch_types):
        calls["create_info"] = {
            "channel_names": list(channel_names),
            "sfreq": sfreq,
            "ch_types": ch_types,
        }
        return SimpleNamespace(set_montage=lambda montage, on_missing=None: calls.update({
            "set_montage": {"montage": montage, "on_missing": on_missing}
        }))

    def fake_make_standard_montage(name):
        calls["montage_name"] = name
        return f"montage:{name}"

    def fake_find_ch_adjacency(info, ch_type):
        calls["find_ch_adjacency"] = {"info": info, "ch_type": ch_type}
        return spatial_adjacency, ["Cz", "Pz"]

    fake_mne = SimpleNamespace(
        create_info=fake_create_info,
        channels=SimpleNamespace(
            make_standard_montage=fake_make_standard_montage,
            find_ch_adjacency=fake_find_ch_adjacency,
        ),
    )
    monkeypatch.setitem(sys.modules, "mne", fake_mne)

    adjacency = _build_adjacency(["Cz", "Pz"], 17, {"adjacency": "mne_default", "montage": "biosemi64"})

    assert adjacency is spatial_adjacency
    assert calls["montage_name"] == "biosemi64"


def test_build_adjacency_configures_mne_runtime_before_import(monkeypatch):
    calls: dict[str, int] = {"configure": 0}

    def fake_configure_mne_runtime():
        calls["configure"] += 1

    monkeypatch.setattr(
        "lmeeeg.backends.correction._regression.configure_mne_runtime",
        fake_configure_mne_runtime,
    )

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mne":
            assert calls["configure"] == 1
            return SimpleNamespace(
                create_info=lambda channel_names, sfreq, ch_types: SimpleNamespace(
                    set_montage=lambda montage, on_missing=None: None
                ),
                channels=SimpleNamespace(
                    make_standard_montage=lambda name: f"montage:{name}",
                    find_ch_adjacency=lambda info, ch_type: ("spatial", ["Cz"]),
                ),
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "mne", raising=False)
    monkeypatch.setattr("builtins.__import__", fake_import)

    adjacency = _build_adjacency(["Cz"], 5, {"adjacency": "mne_default"})

    assert adjacency == "spatial"
    assert calls["configure"] == 1


def test_configure_mne_runtime_disables_mne_numba(monkeypatch):
    calls: dict[str, int] = {"configure": 0}

    def fake_configure_mne_runtime():
        calls["configure"] += 1

    monkeypatch.setattr(
        "lmeeeg.backends.correction._regression.configure_mne_runtime",
        fake_configure_mne_runtime,
    )
    monkeypatch.delenv("MNE_USE_NUMBA", raising=False)

    _configure_mne_runtime()

    assert calls["configure"] == 1
    assert os.environ["MNE_USE_NUMBA"] == "false"


def test_patch_mne_cluster_level_replaces_numba_helpers(monkeypatch):
    calls: dict[str, int] = {"configure": 0}

    def fake_configure_mne_runtime():
        calls["configure"] += 1

    fake_cluster_level = SimpleNamespace(
        _masked_sum="old_masked_sum",
        _masked_sum_power="old_masked_sum_power",
    )

    monkeypatch.setattr(
        "lmeeeg.backends.correction._regression.configure_mne_runtime",
        fake_configure_mne_runtime,
    )
    monkeypatch.setitem(sys.modules, "mne", SimpleNamespace(stats=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "mne.stats", SimpleNamespace(cluster_level=fake_cluster_level))
    monkeypatch.setitem(sys.modules, "mne.stats.cluster_level", fake_cluster_level)

    _patch_mne_cluster_level()

    assert calls["configure"] == 1
    assert callable(fake_cluster_level._masked_sum)
    assert callable(fake_cluster_level._masked_sum_power)
    assert fake_cluster_level._masked_sum(np.array([1.0, 2.0, 3.0]), np.array([0, 2])) == 4.0


def test_load_epochs_with_metadata_configures_runtime_before_mne_import(monkeypatch, tmp_path):
    calls: dict[str, int] = {"configure": 0}

    def fake_configure_mne_runtime():
        calls["configure"] += 1

    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("value\n1\n", encoding="utf-8")
    fake_epochs = SimpleNamespace(metadata=None)

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mne":
            assert calls["configure"] == 1
            return SimpleNamespace(read_epochs=lambda *args, **kwargs: fake_epochs)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(
        "lmeeeg.backends.correction._regression.configure_mne_runtime",
        fake_configure_mne_runtime,
    )
    monkeypatch.delitem(sys.modules, "mne", raising=False)
    monkeypatch.setattr("builtins.__import__", fake_import)

    epochs, metadata = load_epochs_with_metadata(tmp_path / "epochs.fif", metadata_csv=metadata_path)

    assert epochs is fake_epochs
    assert metadata["value"].tolist() == [1]
    assert calls["configure"] == 1


def test_select_epochs_from_config_uses_metadata_event_type_fallback():
    metadata = pd.DataFrame(
        {
            "event_type": ["other_spp_onset", "self_spp_onset"],
            "latency": [0.2, 0.4],
        }
    )
    epochs = FakeEpochs(metadata)
    config = {
        "selection": {"event_type": "other_spp_onset"},
        "models": {"demo": {"formula": "~ latency", "test_predictors": ["latency"]}},
    }

    selected_epochs, selected_metadata = select_epochs_from_config(epochs, metadata, config)

    assert len(selected_epochs) == 1
    assert selected_metadata["event_type"].tolist() == ["other_spp_onset"]


def test_row_from_epochs_path_parses_subject_task_and_run():
    row = _row_from_epochs_path("/tmp/sub-001_task-conversation_run-2_desc-tasklocked_epo.fif")

    assert row == {
        "subject_id": "sub-001",
        "task": "conversation",
        "run": "2",
    }


def test_run_pooled_lmeeeg_analysis_uses_model_specific_input_modality(tmp_path, monkeypatch):
    config_path = tmp_path / "lmeeeg.yaml"
    config_path.write_text(
        """
lmeeeg:
  models:
    demo:
      modality: "induced"
      formula: "~ run"
      test_predictors: ["run"]
induced_epochs:
  bands: ["theta"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    class DummyEpochs:
        def __init__(self) -> None:
            self.ch_names = ["Cz"]
            self.times = np.asarray([0.1], dtype=float)

        def __len__(self) -> int:
            return 1

        def get_data(self, copy: bool = True) -> np.ndarray:
            return np.ones((1, 1, 1), dtype=float)

    loaded_paths: list[tuple[Path, Path | None]] = []

    def fake_load_epochs_with_metadata(epochs_path, *, metadata_csv=None):
        loaded_paths.append((Path(epochs_path), None if metadata_csv is None else Path(metadata_csv)))
        metadata = pd.DataFrame({"run": [1], "event_type": ["self_fpp_onset"]})
        return DummyEpochs(), metadata

    monkeypatch.setattr("cas.stats.lmeeeg_pipeline.load_epochs_with_metadata", fake_load_epochs_with_metadata)
    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline.build_lmeeeg_trial_data_from_arrays",
        lambda **kwargs: SimpleNamespace(
            eeg_data=kwargs["eeg_data"],
            channel_names=kwargs["channel_names"],
            times=kwargs["times"],
        ),
    )
    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline._fit_one_model",
        lambda runtime_config, trial_data, *, model_name, band_name=None, **kwargs: {"summary_output": f"{model_name}.json"},
    )
    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline._run_model_inference",
        lambda runtime_config, trial_data, *, model_name, band_name=None, fit_result=None, **kwargs: [],
    )

    summary = run_pooled_lmeeeg_analysis(
        epochs_paths=[tmp_path / "sub-001_task-conversation_run-1_desc-tasklocked_epo.fif"],
        config_path=config_path,
        output_dir=tmp_path / "out" / "lmeeeg",
    )

    assert loaded_paths
    assert loaded_paths[0][0].as_posix().endswith("/out/induced_epochs/theta/sub-001/epochs-time_s.fif")
    assert loaded_paths[0][1].as_posix().endswith("/out/induced_epochs/theta/sub-001/metadata-time_s.csv")
    assert summary["n_files_input"] == 1


def test_run_pooled_lmeeeg_analysis_expands_induced_model_over_multiple_bands(tmp_path, monkeypatch):
    config_path = tmp_path / "lmeeeg.yaml"
    config_path.write_text(
        """
lmeeeg:
  models:
    demo:
      modality: "induced"
      formula: "~ run"
      test_predictors: ["run"]
induced_epochs:
  bands: ["theta", "alpha", "beta"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    class DummyEpochs:
        def __init__(self) -> None:
            self.ch_names = ["Cz"]
            self.times = np.asarray([0.1], dtype=float)

        def __len__(self) -> int:
            return 1

        def get_data(self, copy: bool = True) -> np.ndarray:
            return np.ones((1, 1, 1), dtype=float)

    loaded_paths: list[tuple[Path, Path | None]] = []
    fit_band_names: list[str | None] = []

    def fake_load_epochs_with_metadata(epochs_path, *, metadata_csv=None):
        loaded_paths.append((Path(epochs_path), None if metadata_csv is None else Path(metadata_csv)))
        metadata = pd.DataFrame({"run": [1], "event_type": ["self_fpp_onset"]})
        return DummyEpochs(), metadata

    monkeypatch.setattr("cas.stats.lmeeeg_pipeline.load_epochs_with_metadata", fake_load_epochs_with_metadata)
    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline.build_lmeeeg_trial_data_from_arrays",
        lambda **kwargs: SimpleNamespace(
            eeg_data=kwargs["eeg_data"],
            channel_names=kwargs["channel_names"],
            times=kwargs["times"],
        ),
    )

    def fake_fit_one_model(runtime_config, trial_data, *, model_name, band_name=None, **kwargs):
        fit_band_names.append(band_name)
        return {"summary_output": f"{model_name}.json"}

    monkeypatch.setattr("cas.stats.lmeeeg_pipeline._fit_one_model", fake_fit_one_model)
    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline._run_model_inference",
        lambda runtime_config, trial_data, *, model_name, band_name=None, fit_result=None, **kwargs: [],
    )

    summary = run_pooled_lmeeeg_analysis(
        epochs_paths=[tmp_path / "sub-001_task-conversation_run-1_desc-tasklocked_epo.fif"],
        config_path=config_path,
        output_dir=tmp_path / "out" / "lmeeeg",
    )

    assert [path[0].parent.parent.name for path in loaded_paths] == ["theta", "alpha", "beta"]
    assert fit_band_names == ["theta", "alpha", "beta"]
    assert len(summary["models"]) == 3


def test_run_pooled_lmeeeg_analysis_skips_files_with_zero_selected_rows(tmp_path, monkeypatch):
    config_path = tmp_path / "lmeeeg.yaml"
    config_path.write_text(
        """
lmeeeg:
  models:
    demo:
      formula: "~ run"
      selection:
        event_type: "other_spp_onset"
      test_predictors: ["run"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    class DummyEpochs:
        def __init__(self, metadata: pd.DataFrame) -> None:
            self._metadata = metadata
            self.ch_names = ["Cz"]
            self.times = np.asarray([0.1], dtype=float)

        @property
        def metadata(self) -> pd.DataFrame:
            return self._metadata

        def __len__(self) -> int:
            return len(self._metadata)

        def get_data(self, copy: bool = True) -> np.ndarray:
            return np.ones((len(self._metadata), 1, 1), dtype=float)

    def fake_load_epochs_with_metadata(epochs_path, *, metadata_csv=None):
        path = Path(epochs_path)
        if "run-1" in path.name:
            metadata = pd.DataFrame({"event_type": ["other_spp_onset"], "run": [1], "subject_id": ["sub-001"]})
        else:
            metadata = pd.DataFrame({"event_type": ["self_fpp_onset"], "run": [2], "subject_id": ["sub-001"]})
        return DummyEpochs(metadata), metadata

    monkeypatch.setattr("cas.stats.lmeeeg_pipeline.load_epochs_with_metadata", fake_load_epochs_with_metadata)
    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline._fit_one_model",
        lambda runtime_config, trial_data, *, model_name, band_name=None, **kwargs: {
            "summary_output": f"{model_name}.json",
            "output_dir": str(tmp_path / "out" / "lmeeeg" / model_name),
        },
    )
    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline._run_model_inference",
        lambda runtime_config, trial_data, *, model_name, band_name=None, fit_result=None, **kwargs: [],
    )

    summary = run_pooled_lmeeeg_analysis(
        epochs_paths=[
            tmp_path / "sub-001_task-conversation_run-1_desc-tasklocked_epo.fif",
            tmp_path / "sub-001_task-conversation_run-2_desc-tasklocked_epo.fif",
        ],
        config_path=config_path,
        output_dir=tmp_path / "out" / "lmeeeg",
    )

    assert summary["models"][0]["fit"]["summary_output"] == "demo.json"


def test_resolve_pooled_source_paths_keeps_evoked_input_path():
    runtime_config = {
        "paths": {"out_dir": "/tmp/out"},
        "lmeeeg": {
            "models": {
                "demo": {
                    "formula": "~ run",
                }
            }
        },
    }

    epochs_path, metadata_path = _resolve_pooled_source_paths(
        runtime_config=runtime_config,
        model_name="demo",
        epochs_path="/tmp/out/epochs/sub-001/eeg/sub-001_task-conversation_run-1_desc-tasklocked_epo.fif",
    )

    assert epochs_path.as_posix().endswith("/tmp/out/epochs/sub-001/eeg/sub-001_task-conversation_run-1_desc-tasklocked_epo.fif")
    assert metadata_path is None


def test_model_output_dir_namespaces_analysis_name():
    runtime_config = {
        "paths": {"out_dir": "/tmp/out"},
        "lmeeeg": {
            "analysis_name": "fpp_spp_cycle_position",
            "models": {
                "demo": {
                    "formula": "~ run",
                    "modality": "induced",
                }
            },
        },
    }

    output_dir = _model_output_dir(runtime_config, model_name="demo", band_name="theta")

    assert output_dir.as_posix().endswith("/tmp/out/lmeeeg/fpp_spp_cycle_position/induced/theta/demo")


def test_resolve_project_out_dir_supports_namespaced_analysis_output():
    assert _resolve_project_out_dir("/tmp/out/lmeeeg") == Path("/tmp/out")
    assert _resolve_project_out_dir("/tmp/out/lmeeeg/fpp_spp_cycle_position") == Path("/tmp/out")
