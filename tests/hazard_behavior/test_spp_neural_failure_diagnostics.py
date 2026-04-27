from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from cas.cli.main import main
from cas.hazard.config import NeuralHazardConfig
from cas.hazard_behavior.diagnose_spp_neural_failure import (
    compute_design_diagnostics,
    compute_event_count_diagnostics,
    compute_separation_summary,
    diagnose_spp_neural_hazard_failure,
    same_rows_match,
    summarize_missingness_by_feature,
)
from cas.hazard_behavior.neural_lowlevel import FittedFormulaModel


class _DummyResult:
    def __init__(self) -> None:
        self.aic = 10.0
        self.bic = 11.0
        self.llf = -3.0
        self.converged = True
        self.params = pd.Series([0.0, 0.5], index=["Intercept", "x1"])
        self.bse = pd.Series([0.1, 0.2], index=["Intercept", "x1"])
        self.tvalues = pd.Series([0.0, 2.5], index=["Intercept", "x1"])
        self.pvalues = pd.Series([1.0, 0.01], index=["Intercept", "x1"])

    def conf_int(self) -> pd.DataFrame:
        return pd.DataFrame({0: [-0.1, 0.1], 1: [0.1, 0.9]})


def _fake_spp_riskset() -> pd.DataFrame:
    n = 12
    table = pd.DataFrame(
        {
            "episode_id": [f"ep-{i // 2}" for i in range(n)],
            "dyad_id": ["d1"] * 6 + ["d2"] * 6,
            "run": ["1"] * 4 + ["2"] * 4 + ["3"] * 4,
            "participant_speaker": ["A"] * 3 + ["B"] * 3 + ["A"] * 3 + ["B"] * 3,
            "bin_start": np.arange(n) * 0.05,
            "bin_end": np.arange(1, n + 1) * 0.05,
            "time_from_partner_onset": np.linspace(0.05, 0.60, n),
            "time_from_partner_offset": np.linspace(-0.20, 0.35, n),
            "event_spp": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            "z_information_rate_lag_150ms": np.linspace(-1.0, 1.0, n),
            "z_prop_expected_cumulative_info_lag_700ms": np.linspace(1.0, -1.0, n),
            "z_alpha_pc1": np.linspace(-0.5, 0.5, n),
            "z_alpha_pc2": np.linspace(0.2, 1.2, n),
            "z_beta_pc1": np.linspace(0.5, -0.5, n),
            "z_beta_pc2": np.linspace(1.0, -1.0, n),
        }
    )
    table.loc[[1, 4], "z_alpha_pc1"] = np.nan
    table.loc[[2, 8], "z_beta_pc1"] = np.nan
    table.loc[[1, 2, 4, 8], "z_beta_pc2"] = np.nan
    return table


def test_event_count_diagnostics_handles_neural_filtering() -> None:
    riskset = _fake_spp_riskset()
    qc = compute_event_count_diagnostics(
        riskset,
        event_column="event_spp",
        alpha_features=["z_alpha_pc1", "z_alpha_pc2"],
        beta_features=["z_beta_pc1", "z_beta_pc2"],
        alpha_beta_features=["z_alpha_pc1", "z_alpha_pc2", "z_beta_pc1", "z_beta_pc2"],
        neural_config=NeuralHazardConfig(enabled=True),
        row_ids=(riskset["episode_id"].astype(str) + "|" + riskset["bin_start"].astype(str)),
    )
    assert qc["n_rows_total"] == 12
    assert qc["n_events_total"] == 3
    assert qc["n_participant_speaker_ids_total"] == 4
    assert qc["n_events_neural_complete_alpha"] == 1
    assert qc["n_events_neural_complete_alpha_beta"] == 0


def test_spp_diagnostics_grouping_uses_participant_speaker_id() -> None:
    riskset = pd.DataFrame(
        {
            "episode_id": [f"ep-{index // 2}" for index in range(12)],
            "dyad_id": ["d1"] * 4 + ["d2"] * 4 + ["d3"] * 4,
            "run": ["1"] * 12,
            "participant_speaker": ["A", "A", "B", "B"] * 3,
            "bin_start": np.arange(12) * 0.05,
            "bin_end": np.arange(1, 13) * 0.05,
            "time_from_partner_onset": np.linspace(0.05, 0.60, 12),
            "time_from_partner_offset": np.linspace(-0.20, 0.35, 12),
            "event_spp": [1, 0, 0, 0] * 3,
            "z_information_rate_lag_150ms": np.linspace(-1.0, 1.0, 12),
            "z_prop_expected_cumulative_info_lag_700ms": np.linspace(1.0, -1.0, 12),
            "z_alpha_pc1": np.linspace(-0.5, 0.5, 12),
            "z_beta_pc1": np.linspace(0.5, -0.5, 12),
        }
    )
    qc = compute_event_count_diagnostics(
        riskset,
        event_column="event_spp",
        alpha_features=["z_alpha_pc1"],
        beta_features=["z_beta_pc1"],
        alpha_beta_features=["z_alpha_pc1", "z_beta_pc1"],
        neural_config=NeuralHazardConfig(enabled=True),
        row_ids=(riskset["episode_id"].astype(str) + "|" + riskset["bin_start"].astype(str)),
    )
    assert qc["n_participant_speaker_ids_total"] == 6
    assert qc["identity_validation"]["participant_speaker_id_valid"] is True


def test_missingness_by_feature_summary_is_correct() -> None:
    riskset = _fake_spp_riskset()
    summary = summarize_missingness_by_feature(
        riskset,
        event_column="event_spp",
        alpha_features=["z_alpha_pc1"],
        beta_features=["z_beta_pc1"],
    )
    alpha_row = summary.loc[summary["feature"] == "z_alpha_pc1"].iloc[0]
    assert int(alpha_row["n_missing"]) == 2
    assert int(alpha_row["n_events_missing"]) == 2


def test_same_row_assertion_requires_exact_row_ids() -> None:
    parent = pd.Series(["a", "b", "c"])
    child_same = pd.Series(["a", "b", "c"])
    child_diff = pd.Series(["a", "c", "b"])
    assert same_rows_match(parent, child_same)
    assert not same_rows_match(parent, child_diff)


def test_design_diagnostics_flag_duplicate_columns() -> None:
    subset = pd.DataFrame(
        {
            "event_spp": [0, 1, 0, 1, 0, 1],
            "time_from_partner_onset": np.linspace(0.0, 0.5, 6),
            "time_from_partner_offset": np.linspace(-0.2, 0.3, 6),
            "z_information_rate_lag_150ms": [0, 1, 0, 1, 0, 1],
            "z_prop_expected_cumulative_info_lag_700ms": [0, 1, 0, 1, 0, 1],
            "z_alpha_pc1": [0, 1, 0, 1, 0, 1],
            "z_alpha_pc2": [0, 1, 0, 1, 0, 1],
        }
    )
    row = compute_design_diagnostics(
        subset,
        formula="event_spp ~ z_alpha_pc1 + z_alpha_pc2 + z_information_rate_lag_150ms + z_prop_expected_cumulative_info_lag_700ms",
        event_column="event_spp",
        model_family="alpha",
        row_ids=pd.Series([str(i) for i in range(len(subset))]),
        neural_features=["z_alpha_pc1", "z_alpha_pc2"],
        max_design_rows=100,
    )
    assert row["rank_deficiency"] > 0
    assert float(row["maximum_absolute_correlation"]) >= 0.95


def test_separation_diagnostics_flag_perfect_separator() -> None:
    table = pd.DataFrame(
        {
            "event_spp": [0, 0, 0, 1, 1, 1],
            "z_alpha_pc1": [-3, -2, -1, 1, 2, 3],
        }
    )
    summary = compute_separation_summary(table, event_column="event_spp", predictors=["z_alpha_pc1"])
    assert summary["n_possible_separation_predictors"] == 1


def test_incremental_fit_failure_handling_continues(tmp_path: Path) -> None:
    riskset = _fake_spp_riskset()
    riskset_path = tmp_path / "riskset.csv"
    riskset.to_csv(riskset_path, index=False)

    def fit_stub(**kwargs) -> FittedFormulaModel:  # type: ignore[no-untyped-def]
        if kwargs["model_name"] == "SPP_M_alpha_all":
            return FittedFormulaModel(
                model_name=kwargs["model_name"],
                formula=kwargs["formula"],
                result=None,
                n_rows=int(len(kwargs["riskset_table"])),
                n_events=int(kwargs["riskset_table"][kwargs["event_column"]].sum()),
                n_predictors=0,
                converged=False,
                fit_warnings=["diagnostic warning"],
                error_message="simulated failure",
            )
        return FittedFormulaModel(
            model_name=kwargs["model_name"],
            formula=kwargs["formula"],
            result=_DummyResult(),
            n_rows=int(len(kwargs["riskset_table"])),
            n_events=int(kwargs["riskset_table"][kwargs["event_column"]].sum()),
            n_predictors=2,
            converged=True,
            fit_warnings=[],
            error_message=None,
        )

    result = diagnose_spp_neural_hazard_failure(
        riskset_path=riskset_path,
        models_dir=None,
        output_dir=tmp_path / "diagnostics",
        run_ridge_diagnostic=False,
        fit_model_fn=fit_stub,
    )
    incremental = pd.read_csv(result.incremental_diagnostics_path)
    failed = incremental.loc[incremental["model_name"] == "SPP_M_alpha_all"].iloc[0]
    assert failed["status"] == "failed"
    assert result.report_path.exists()


def test_report_and_qc_are_written(tmp_path: Path) -> None:
    riskset = _fake_spp_riskset()
    riskset_path = tmp_path / "riskset.csv"
    riskset.to_csv(riskset_path, index=False)
    result = diagnose_spp_neural_hazard_failure(
        riskset_path=riskset_path,
        models_dir=None,
        output_dir=tmp_path / "diagnostics",
        run_ridge_diagnostic=False,
    )
    assert result.report_path.exists()
    assert result.qc_path.exists()


def test_hazard_pipeline_isolation_imports_without_spp_diagnostics_dependency() -> None:
    from cas.hazard_behavior.pipeline import run_behaviour_hazard_pipeline  # noqa: F401


def test_cli_diagnose_spp_neural_hazard_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    riskset = _fake_spp_riskset()
    riskset_path = tmp_path / "riskset.csv"
    riskset.to_csv(riskset_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cas",
            "diagnose-spp-neural-hazard-failure",
            "--riskset-path",
            str(riskset_path),
            "--output-dir",
            str(tmp_path / "diagnostics"),
        ],
    )
    assert main() == 0
