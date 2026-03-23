"""Nested leave-one-run-out CV for TRF models."""

from __future__ import annotations

import inspect

import numpy as np


def _validate_runs(X_runs: list[np.ndarray], Y_runs: list[np.ndarray]) -> None:
    if len(X_runs) != len(Y_runs):
        raise ValueError("`X_runs` and `Y_runs` must have the same number of runs.")
    if len(X_runs) < 3:
        raise ValueError("Need at least 3 runs for nested CV.")
    for i, (x, y) in enumerate(zip(X_runs, Y_runs)):
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)
        if xx.ndim != 2 or yy.ndim != 2:
            raise ValueError(f"Run {i} must be 2D: X=(samples, features), Y=(samples, channels).")
        if xx.shape[0] != yy.shape[0]:
            raise ValueError(f"Run {i} sample mismatch: X={xx.shape[0]} vs Y={yy.shape[0]}.")
        if not np.isfinite(xx).all() or not np.isfinite(yy).all():
            raise ValueError(f"Run {i} contains NaN or infinite values.")


def _import_trf_class():
    try:
        import spyeeg  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("`spyeeg` is required for `loro_nested_cv`.") from e

    candidates = [
        ("models", "TRF"),
        ("models.trf", "TRF"),
        (None, "TRF"),
    ]
    for module_name, class_name in candidates:
        try:
            if module_name is None:
                return getattr(spyeeg, class_name)
            mod = __import__(f"spyeeg.{module_name}", fromlist=[class_name])
            return getattr(mod, class_name)
        except Exception:
            continue
    raise ImportError("Could not find a TRF class in `spyeeg`.")


def _fit_model(trf_cls, X: np.ndarray, Y: np.ndarray, alpha: float):
    init_sig = inspect.signature(trf_cls)
    init_kwargs = {}
    for k in ("alpha", "reg_alpha", "l2", "lambda_"):
        if k in init_sig.parameters:
            init_kwargs[k] = alpha
            break
    model = trf_cls(**init_kwargs)

    fit_sig = inspect.signature(model.fit)
    fit_kwargs = {}
    for k in ("alpha", "reg_alpha", "l2", "lambda_"):
        if k in fit_sig.parameters:
            fit_kwargs[k] = alpha
            break
    model.fit(X, Y, **fit_kwargs)
    return model


def _score_model(model, X: np.ndarray, Y: np.ndarray) -> float:
    if hasattr(model, "score"):
        s = model.score(X, Y)
        return float(np.nanmean(np.asarray(s, dtype=float)))
    if not hasattr(model, "predict"):
        raise RuntimeError("TRF model has neither `score` nor `predict`.")
    pred = np.asarray(model.predict(X), dtype=float)
    if pred.shape != Y.shape:
        raise RuntimeError(f"Prediction shape mismatch: pred={pred.shape}, y={Y.shape}.")
    vals = []
    for ch in range(Y.shape[1]):
        y = Y[:, ch]
        p = pred[:, ch]
        if np.std(y) == 0 or np.std(p) == 0:
            vals.append(0.0)
        else:
            vals.append(float(np.corrcoef(y, p)[0, 1]))
    return float(np.mean(vals))


def _coef_from_model(model) -> np.ndarray:
    for attr in ("coef_", "coef", "weights_", "w_", "betas_"):
        if hasattr(model, attr):
            return np.asarray(getattr(model, attr), dtype=float)
    raise RuntimeError("Could not extract coefficients from TRF model.")


def _stack_runs(runs: list[np.ndarray], indices: list[int]) -> np.ndarray:
    return np.concatenate([np.asarray(runs[i], dtype=float) for i in indices], axis=0)


def loro_nested_cv(
    X_runs: list[np.ndarray],
    Y_runs: list[np.ndarray],
    alphas: list[float],
):
    """Nested LORO CV with inner LORO alpha selection on training runs."""
    _validate_runs(X_runs, Y_runs)
    if len(alphas) == 0:
        raise ValueError("`alphas` must not be empty.")
    if any(a <= 0 for a in alphas):
        raise ValueError("All alphas must be > 0.")

    trf_cls = _import_trf_class()
    n_runs = len(X_runs)
    fold_scores: list[dict[str, float | int]] = []
    fold_coefficients: list[np.ndarray] = []

    for test_idx in range(n_runs):
        outer_train = [i for i in range(n_runs) if i != test_idx]
        alpha_to_scores: dict[float, list[float]] = {float(a): [] for a in alphas}

        for val_idx in outer_train:
            inner_train = [i for i in outer_train if i != val_idx]
            Xtr = _stack_runs(X_runs, inner_train)
            Ytr = _stack_runs(Y_runs, inner_train)
            Xval = np.asarray(X_runs[val_idx], dtype=float)
            Yval = np.asarray(Y_runs[val_idx], dtype=float)

            for alpha in alphas:
                model = _fit_model(trf_cls, Xtr, Ytr, float(alpha))
                alpha_to_scores[float(alpha)].append(_score_model(model, Xval, Yval))

        best_alpha = max(alpha_to_scores, key=lambda a: float(np.mean(alpha_to_scores[a])))

        Xtr_outer = _stack_runs(X_runs, outer_train)
        Ytr_outer = _stack_runs(Y_runs, outer_train)
        Xte = np.asarray(X_runs[test_idx], dtype=float)
        Yte = np.asarray(Y_runs[test_idx], dtype=float)

        final_model = _fit_model(trf_cls, Xtr_outer, Ytr_outer, best_alpha)
        test_score = _score_model(final_model, Xte, Yte)
        fold_scores.append(
            {"test_run": test_idx, "best_alpha": float(best_alpha), "score": float(test_score)}
        )
        fold_coefficients.append(_coef_from_model(final_model))

    return fold_scores, fold_coefficients

