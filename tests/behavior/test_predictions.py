from __future__ import annotations

import numpy as np
import pandas as pd

from cas.behavior.predictions import _conditional_median_curve


def test_conditional_median_curve_tracks_empirical_timing_manifold() -> None:
    table = pd.DataFrame(
        {
            "time_from_partner_onset_s": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "time_from_partner_offset_s": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        }
    )

    conditioned = _conditional_median_curve(
        table,
        x_column="time_from_partner_onset_s",
        y_column="time_from_partner_offset_s",
        x_values=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
    )

    assert np.allclose(conditioned, [-0.75, -0.25, 0.25, 0.75, 1.25])
