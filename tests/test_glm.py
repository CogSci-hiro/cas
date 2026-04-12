from __future__ import annotations

import numpy as np
import pandas as pd

from cas.glm import (
    aggregate_subject_betas,
    build_predictor_column_map,
    extract_predictor_betas,
    fit_subject_glm,
    prepare_glm_design,
)


class DummyEpochs:
    def __init__(self, data: np.ndarray, *, ch_names: list[str], times: np.ndarray):
        self._data = np.asarray(data, dtype=float)
        self.ch_names = list(ch_names)
        self.times = np.asarray(times, dtype=float)

    def __len__(self) -> int:
        return int(self._data.shape[0])

    def get_data(self, copy: bool = True) -> np.ndarray:
        return self._data.copy() if copy else self._data


def test_fit_subject_glm_extracts_named_predictors() -> None:
    metadata_df = pd.DataFrame(
        {
            "trial": [0, 1, 2, 3],
            "condition": [0.0, 1.0, 0.0, 1.0],
        }
    )
    config = {
        "glm": {
            "formula": "1 + condition",
            "test_predictors": ["condition"],
        }
    }
    design_result = prepare_glm_design(metadata_df, config)

    data = np.array(
        [
            [[1.0, 2.0], [0.5, 1.5]],
            [[3.0, 4.0], [1.5, 2.5]],
            [[1.0, 2.0], [0.5, 1.5]],
            [[3.0, 4.0], [1.5, 2.5]],
        ]
    )
    epochs = DummyEpochs(data, ch_names=["Cz", "Pz"], times=np.array([0.0, 0.1]))

    result = fit_subject_glm(epochs, design_result)
    predictor_map = build_predictor_column_map(result)
    condition_betas = extract_predictor_betas(result, "condition")

    assert result.betas.shape == (2, 2, 2)
    assert result.standard_errors is not None
    assert result.t_values is not None
    assert predictor_map == {"condition": ["condition"]}
    assert np.allclose(condition_betas["condition"], np.array([[2.0, 2.0], [1.0, 1.0]]))


def test_aggregate_subject_betas_stacks_consistently() -> None:
    beta_a = np.ones((2, 3, 4))
    beta_b = np.full((2, 3, 4), 2.0)

    stacked = aggregate_subject_betas([beta_a, beta_b])

    assert stacked.shape == (2, 2, 3, 4)
    assert np.allclose(stacked[0], beta_a)
    assert np.allclose(stacked[1], beta_b)
