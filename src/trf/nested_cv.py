"""Leave-one-run-out nested CV utilities for TRF fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _progress_iter(iterable, *, enabled: bool, desc: str, total: int | None = None):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, total=total, leave=False)
    except ModuleNotFoundError:
        return iterable


class _LocalTRFEstimator:
    """Minimal ridge-based forward TRF estimator used when `spyeeg` is unavailable."""

    def __init__(self, *, srate: float, tmin_s: float, tmax_s: float, alpha: float, fit_intercept: bool):
        self.srate = float(srate)
        self.tmin_s = float(tmin_s)
        self.tmax_s = float(tmax_s)
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.lags = np.arange(
            int(np.rint(self.tmin_s * self.srate)),
            int(np.rint(self.tmax_s * self.srate)) + 1,
            dtype=int,
        )
        self.intercept_: np.ndarray | None = None
        self.beta_: np.ndarray | None = None
        self.n_feats_: int | None = None
        self.n_chans_: int | None = None

    def _lag_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_samples, n_feats = X.shape
        design = np.zeros((n_samples, len(self.lags) * n_feats), dtype=float)
        for lag_index, lag in enumerate(self.lags):
            start = lag_index * n_feats
            stop = start + n_feats
            if lag < 0:
                shifted = np.zeros_like(X)
                shifted[:lag, :] = X[-lag:, :]
            elif lag > 0:
                shifted = np.zeros_like(X)
                shifted[lag:, :] = X[:-lag, :]
            else:
                shifted = X
            design[:, start:stop] = shifted
        return design

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_feats_ = int(X.shape[1])
        self.n_chans_ = int(y.shape[1])
        design = self._lag_matrix(X)
        if self.fit_intercept:
            design = np.concatenate([design, np.ones((design.shape[0], 1), dtype=float)], axis=1)
        gram = design.T @ design
        ridge = self.alpha * np.eye(gram.shape[0], dtype=float)
        if self.fit_intercept:
            ridge[-1, -1] = 0.0
        weights = np.linalg.solve(gram + ridge, design.T @ y)
        if self.fit_intercept:
            self.beta_ = weights[:-1, :]
            self.intercept_ = weights[-1, :]
        else:
            self.beta_ = weights
            self.intercept_ = np.zeros(self.n_chans_, dtype=float)
        return self.beta_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta_ is None or self.intercept_ is None:
            raise RuntimeError("Estimator must be fit before calling predict().")
        design = self._lag_matrix(np.asarray(X, dtype=float))
        yhat = design @ self.beta_ + self.intercept_
        return yhat[..., np.newaxis]

    def score(self, Xtest: np.ndarray, ytrue: np.ndarray, scoring: str = "corr") -> np.ndarray:
        if scoring != "corr":
            raise NotImplementedError("Local TRF fallback currently supports corr scoring only.")
        yhat = np.asarray(self.predict(Xtest)[..., 0], dtype=float)
        ytrue = np.asarray(ytrue, dtype=float)
        scores = np.full((ytrue.shape[1], 1), np.nan, dtype=float)
        for channel_index in range(ytrue.shape[1]):
            observed = ytrue[:, channel_index]
            predicted = yhat[:, channel_index]
            if np.std(observed) == 0.0 or np.std(predicted) == 0.0:
                scores[channel_index, 0] = 0.0
            else:
                scores[channel_index, 0] = float(np.corrcoef(predicted, observed)[0, 1])
        return scores

    def get_coef(self) -> np.ndarray:
        if self.beta_ is None or self.n_feats_ is None or self.n_chans_ is None:
            raise RuntimeError("Estimator must be fit before calling get_coef().")
        reshaped = np.asarray(self.beta_, dtype=float).reshape(len(self.lags), self.n_feats_, self.n_chans_)
        return reshaped[:, :, :, np.newaxis]


def _make_trf_estimator(*, srate: float, tmin_s: float, tmax_s: float, alpha: float, fit_intercept: bool):
    import os

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    try:
        from spyeeg.models.TRF import TRFEstimator

        return TRFEstimator(
            tmin=float(tmin_s),
            tmax=float(tmax_s),
            srate=float(srate),
            alpha=[float(alpha)],
            fit_intercept=bool(fit_intercept),
            mtype="forward",
        )
    except ModuleNotFoundError:
        return _LocalTRFEstimator(
            srate=float(srate),
            tmin_s=float(tmin_s),
            tmax_s=float(tmax_s),
            alpha=float(alpha),
            fit_intercept=bool(fit_intercept),
        )


def _concat_runs(runs: list[np.ndarray], indices: list[int]) -> np.ndarray:
    return np.concatenate([np.asarray(runs[index], dtype=float) for index in indices], axis=0)


def _safe_standardize(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0, keepdims=True)
    std = np.nanstd(train, axis=0, keepdims=True)
    std = np.where(std > 0.0, std, 1.0)
    return (train - mean) / std, (test - mean) / std


@dataclass(frozen=True, slots=True)
class FoldResult:
    test_run_index: int
    selected_alpha: float
    mean_validation_score: float
    channel_scores: list[float]


def loro_nested_cv(
    *,
    X_runs: list[np.ndarray],
    Y_runs: list[np.ndarray],
    alphas: list[float],
    srate: float,
    tmin_s: float,
    tmax_s: float,
    fit_intercept: bool = False,
    scoring: str = "corr",
    standardize_X: bool = True,
    standardize_Y: bool = False,
    verbose: bool = False,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    """Fit a leave-one-run-out nested CV TRF and return fold summaries."""

    if len(X_runs) != len(Y_runs):
        raise ValueError("X_runs and Y_runs must have the same length.")
    if len(X_runs) < 2:
        raise ValueError("At least two runs are required for leave-one-run-out CV.")
    if not alphas:
        raise ValueError("At least one ridge alpha must be provided.")

    fold_scores: list[dict[str, object]] = []
    fold_coefficients: list[np.ndarray] = []
    run_indices = list(range(len(X_runs)))

    if verbose:
        print(
            f"[trf] starting nested CV: n_runs={len(X_runs)} "
            f"n_alphas={len(alphas)} srate={float(srate):.3f}Hz "
            f"lag_window=[{float(tmin_s):.3f}, {float(tmax_s):.3f}]s"
        )

    outer_iterator = _progress_iter(
        run_indices,
        enabled=verbose,
        desc="TRF outer folds",
        total=len(run_indices),
    )
    for outer_test_index in outer_iterator:
        outer_train_indices = [index for index in run_indices if index != outer_test_index]
        if verbose:
            print(
                f"[trf] outer fold test_run={outer_test_index + 1} "
                f"train_runs={[index + 1 for index in outer_train_indices]}"
            )

        alpha_validation_means: list[float] = []
        alpha_iterator = _progress_iter(
            alphas,
            enabled=verbose,
            desc=f"fold {outer_test_index + 1} alphas",
            total=len(alphas),
        )
        for alpha in alpha_iterator:
            inner_fold_means: list[float] = []
            for inner_valid_index in outer_train_indices:
                inner_train_indices = [
                    index for index in outer_train_indices if index != inner_valid_index
                ]
                if verbose:
                    print(
                        f"[trf]  inner fold valid_run={inner_valid_index + 1} "
                        f"alpha={float(alpha):.6g} train_runs={[index + 1 for index in inner_train_indices]}"
                    )
                X_train = _concat_runs(X_runs, inner_train_indices)
                Y_train = _concat_runs(Y_runs, inner_train_indices)
                X_valid = np.asarray(X_runs[inner_valid_index], dtype=float)
                Y_valid = np.asarray(Y_runs[inner_valid_index], dtype=float)

                if standardize_X:
                    X_train, X_valid = _safe_standardize(X_train, X_valid)
                if standardize_Y:
                    Y_train, Y_valid = _safe_standardize(Y_train, Y_valid)

                model = _make_trf_estimator(
                    srate=srate,
                    tmin_s=tmin_s,
                    tmax_s=tmax_s,
                    alpha=float(alpha),
                    fit_intercept=fit_intercept,
                )
                model.fit(X_train, Y_train)
                valid_scores = np.asarray(model.score(X_valid, Y_valid, scoring=scoring), dtype=float)
                inner_fold_means.append(float(np.nanmean(valid_scores)))

            if verbose:
                print(
                    f"[trf]  alpha={float(alpha):.6g} "
                    f"inner_mean={float(np.nanmean(inner_fold_means)):.6f}"
                )
            alpha_validation_means.append(float(np.nanmean(inner_fold_means)))

        best_alpha_index = int(np.nanargmax(np.asarray(alpha_validation_means, dtype=float)))
        selected_alpha = float(alphas[best_alpha_index])
        if verbose:
            print(
                f"[trf] selected alpha for outer test_run={outer_test_index + 1}: "
                f"{selected_alpha:.6g}"
            )

        X_outer_train = _concat_runs(X_runs, outer_train_indices)
        Y_outer_train = _concat_runs(Y_runs, outer_train_indices)
        X_outer_test = np.asarray(X_runs[outer_test_index], dtype=float)
        Y_outer_test = np.asarray(Y_runs[outer_test_index], dtype=float)
        if standardize_X:
            X_outer_train, X_outer_test = _safe_standardize(X_outer_train, X_outer_test)
        if standardize_Y:
            Y_outer_train, Y_outer_test = _safe_standardize(Y_outer_train, Y_outer_test)

        final_model = _make_trf_estimator(
            srate=srate,
            tmin_s=tmin_s,
            tmax_s=tmax_s,
            alpha=selected_alpha,
            fit_intercept=fit_intercept,
        )
        final_model.fit(X_outer_train, Y_outer_train)
        test_scores = np.asarray(final_model.score(X_outer_test, Y_outer_test, scoring=scoring), dtype=float)
        channel_scores = np.asarray(test_scores[..., 0], dtype=float).reshape(-1)
        coefficients = np.asarray(final_model.get_coef()[..., 0], dtype=float)

        if verbose:
            print(
                f"[trf] completed outer fold test_run={outer_test_index + 1} "
                f"selected_alpha={selected_alpha:.6g} mean_score={float(np.nanmean(channel_scores)):.6f}"
            )

        fold_scores.append(
            {
                "test_run": int(outer_test_index + 1),
                "selected_alpha": selected_alpha,
                "inner_mean_scores_by_alpha": [float(value) for value in alpha_validation_means],
                "channel_scores": [float(value) for value in channel_scores],
                "mean_score": float(np.nanmean(channel_scores)),
            }
        )
        fold_coefficients.append(coefficients)

    if verbose:
        mean_scores = [float(fold["mean_score"]) for fold in fold_scores]
        print(
            f"[trf] completed nested CV: mean_fold_score={float(np.nanmean(mean_scores)):.6f} "
            f"sd_fold_score={float(np.nanstd(mean_scores)):.6f}"
        )
    return fold_scores, fold_coefficients
