from __future__ import annotations

import numpy as np

from cas.hazard.entropy import compute_posterior_entropy, normalize_entropy


def test_compute_posterior_entropy_matches_known_values() -> None:
    posterior = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=float)
    entropy = compute_posterior_entropy(posterior)
    assert np.isclose(entropy[0], 0.0, atol=1.0e-8)
    assert np.isclose(entropy[1], np.log(2.0), atol=1.0e-8)


def test_normalize_entropy_divides_by_log_k() -> None:
    entropy = np.array([0.0, np.log(4.0)], dtype=float)
    normalized = normalize_entropy(entropy, 4)
    assert np.allclose(normalized, np.array([0.0, 1.0], dtype=float))
