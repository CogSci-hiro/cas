from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cas.hazard_behavior.neural_lowlevel import FittedFormulaModel
from cas.hazard_behavior.neural_permutation_null import (
    circular_shift_events_within_episode,
    compute_empirical_p_value,
    prepare_family_complete_case_data,
    run_fpp_neural_permutation_null,
)
from cas.hazard_behavior.plot_neural_permutation_null import plot_permutation_null_outputs


class _DummyResult:
    def __init__(self, *, aic: float, bic: float, llf: float, n_params: int, converged: bool = True) -> None:
        self.aic = float(aic)
        self.bic = float(bic)
        self.llf = float(llf)
        self.converged = bool(converged)
        self.params = pd.Series(np.zeros(n_params, dtype=float))
        self.bse = pd.Series(np.ones(n_params, dtype=float))
        self.tvalues = pd.Series(np.zeros(n_params, dtype=float))
        self.pvalues = pd.Series(np.ones(n_params, dtype=float))


def _base_riskset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_fpp": [0, 1, 0, 0, 0, 0, 1, 0],
            "episode_id": ["ep1", "ep1", "ep1", "ep2", "ep2", "ep3", "ep3", "ep3"],
            "dyad_id": ["d1"] * 8,
            "run": [1] * 8,
            "speaker": ["A"] * 8,
            "participant_speaker_id": ["d1_A"] * 8,
            "time_from_partner_onset": np.linspace(0.1, 0.8, 8),
            "time_from_partner_offset": np.linspace(0.2, 0.9, 8),
            "z_information_rate_lag_best": np.linspace(-1.0, 1.0, 8),
            "z_prop_expected_cumulative_info_lag_best": np.linspace(1.0, -1.0, 8),
            "z_alpha_pc1": np.linspace(-0.2, 0.2, 8),
            "z_beta_pc1": np.linspace(0.3, -0.3, 8),
        }
    )


def _successful_fit(
    *,
    riskset_table: pd.DataFrame,
    model_name: str,
    formula: str,
    event_column: str,
) -> FittedFormulaModel:
    del formula
    is_parent = "_sample" in model_name
    result = _DummyResult(
        aic=100.0 if is_parent else 95.0,
        bic=110.0 if is_parent else 104.0,
        llf=-50.0 if is_parent else -46.0,
        n_params=4 if is_parent else 5,
    )
    return FittedFormulaModel(
        model_name=model_name,
        formula="",
        result=result,
        n_rows=int(len(riskset_table)),
        n_events=int(pd.to_numeric(riskset_table[event_column], errors="coerce").fillna(0).sum()),
        n_predictors=int(len(result.params)),
        converged=True,
        fit_warnings=[],
        error_message=None,
    )


def test_circular_shift_preserves_event_count() -> None:
    table = pd.DataFrame({"episode_id": ["ep"] * 4, "event_fpp": [0, 1, 0, 0]})
    shifted = circular_shift_events_within_episode(table, "event_fpp", "episode_id", np.random.default_rng(7))
    assert int(shifted.sum()) == 1


def test_circular_shift_changes_event_location() -> None:
    table = pd.DataFrame({"episode_id": ["ep"] * 4, "event_fpp": [0, 1, 0, 0]})
    shifted = circular_shift_events_within_episode(table, "event_fpp", "episode_id", np.random.default_rng(7))
    assert not shifted.equals(table["event_fpp"])


def test_censored_episodes_unchanged() -> None:
    table = pd.DataFrame({"episode_id": ["ep"] * 4, "event_fpp": [0, 0, 0, 0]})
    shifted = circular_shift_events_within_episode(table, "event_fpp", "episode_id", np.random.default_rng(2))
    pd.testing.assert_series_equal(shifted.reset_index(drop=True), table["event_fpp"], check_names=False)


def test_length_one_episode_handled() -> None:
    table = pd.DataFrame({"episode_id": ["ep"], "event_fpp": [1]})
    shifted = circular_shift_events_within_episode(table, "event_fpp", "episode_id", np.random.default_rng(2))
    assert shifted.tolist() == [1]


def test_multiple_event_episode_preserves_count() -> None:
    table = pd.DataFrame({"episode_id": ["ep"] * 5, "event_fpp": [1, 0, 1, 0, 0]})
    shifted = circular_shift_events_within_episode(table, "event_fpp", "episode_id", np.random.default_rng(9))
    assert int(shifted.sum()) == 2


def test_same_row_fitting_uses_identical_complete_case_rows() -> None:
    table = _base_riskset()
    table.loc[2, "z_alpha_pc1"] = np.nan
    prepared = prepare_family_complete_case_data(
        riskset_table=table,
        event_column="event_fpp",
        episode_column="episode_id",
        participant_column="participant_speaker_id",
        run_column="run",
        neural_family="alpha",
        information_rate_column="z_information_rate_lag_best",
        prop_expected_column="z_prop_expected_cumulative_info_lag_best",
    )
    assert len(prepared.data) == 7
    assert prepared.row_ids.index.equals(prepared.data.index)


def test_delta_convention(tmp_path: Path) -> None:
    result = run_fpp_neural_permutation_null(
        riskset_path=_write_riskset_csv(_base_riskset(), tmp_path),
        output_dir=tmp_path / "out1",
        neural_family="beta",
        n_permutations=2,
        seed=1,
        fit_model_fn=_successful_fit,
    )
    comparison = pd.read_csv(result.family_output_dirs["beta"] / "fpp_neural_permutation_real_comparison.csv")
    assert (comparison["delta_bic"] == (comparison["child_bic"] - comparison["parent_bic"])).all()
    assert (comparison["delta_aic"] == (comparison["child_aic"] - comparison["parent_aic"])).all()


def test_empirical_p_value_direction() -> None:
    assert compute_empirical_p_value(-10.0, np.array([-5.0, -8.0, -12.0, 0.0])) == pytest.approx(0.4)


def test_failed_permutation_handling(tmp_path: Path) -> None:
    call_counter = {"count": 0}

    def sometimes_failing_fit(**kwargs) -> FittedFormulaModel:  # type: ignore[no-untyped-def]
        call_counter["count"] += 1
        if call_counter["count"] > 2 and call_counter["count"] <= 4:
            return FittedFormulaModel(
                model_name=kwargs["model_name"],
                formula=kwargs["formula"],
                result=None,
                n_rows=int(len(kwargs["riskset_table"])),
                n_events=int(pd.to_numeric(kwargs["riskset_table"][kwargs["event_column"]], errors="coerce").fillna(0).sum()),
                n_predictors=0,
                converged=False,
                fit_warnings=["test warning"],
                error_message="synthetic failure",
            )
        return _successful_fit(**kwargs)

    riskset_path = tmp_path / "riskset.csv"
    _base_riskset().to_csv(riskset_path, index=False)
    result = run_fpp_neural_permutation_null(
        riskset_path=riskset_path,
        output_dir=tmp_path / "out",
        neural_family="beta",
        n_permutations=2,
        seed=1,
        fit_model_fn=sometimes_failing_fit,
    )
    distribution = pd.read_csv(result.family_output_dirs["beta"] / "fpp_neural_permutation_null_distribution.csv")
    summary = pd.read_json(result.family_output_dirs["beta"] / "fpp_neural_permutation_summary.json", typ="series")
    assert "failed" in distribution["status"].tolist()
    assert distribution["error_message"].fillna("").str.contains("synthetic failure").any()
    assert int(summary["n_permutations_failed"]) >= 1
    assert int(summary["n_permutations_successful"]) >= 0


def test_plotting_smoke(tmp_path: Path) -> None:
    real = pd.DataFrame([{"delta_bic": -10.0, "delta_aic": -8.0}])
    null = pd.DataFrame(
        {
            "status": ["ok", "ok", "failed"],
            "delta_bic": [-5.0, -11.0, np.nan],
            "delta_aic": [-4.0, -9.0, np.nan],
            "proportion_event_rows_changed": [0.25, 0.50, np.nan],
            "proportion_event_episodes_changed": [1.0, 1.0, np.nan],
            "n_events_after_shift": [2, 2, np.nan],
        }
    )
    summary = {"real_delta_bic": -10.0, "real_delta_aic": -8.0, "empirical_p_delta_bic": 0.5}
    plot_permutation_null_outputs(real_comparison=real, null_distribution=null, summary=summary, figures_dir=tmp_path)
    assert (tmp_path / "fpp_neural_permutation_delta_bic_null.png").exists()
    assert (tmp_path / "fpp_neural_permutation_delta_bic_ecdf.png").exists()
    assert (tmp_path / "fpp_neural_permutation_real_vs_null_summary.png").exists()
    assert (tmp_path / "fpp_neural_permutation_shift_qc.png").exists()


def _write_riskset_csv(table: pd.DataFrame, directory: Path) -> Path:
    path = directory / "riskset.csv"
    table.to_csv(path, index=False)
    return path
