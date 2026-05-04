"""Publication-oriented plotting for source-level DICS lmeEEG outputs.

Usage example
-------------
>>> from pathlib import Path
>>> from cas.source_dics.config import load_source_dics_config
>>> cfg = load_source_dics_config(Path("config/induced/source_localisation.yaml"))  # doctest: +SKIP
>>> run_source_dics_plotting(cfg, roi_only=True, surface_only=False)  # doctest: +SKIP
PosixPath('.../figures/index/figure_index.csv')
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Final

import matplotlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from cas.source_dics.broad_rois import add_broad_roi_columns, normalize_aparc_label
from cas.source_dics.config import SourceDicsConfig
from cas.source_dics.plotting_io import load_source_statistics_table

LOGGER = logging.getLogger(__name__)

MISSING_SURFACE_BACKEND_FILE: Final[str] = "missing_surface_backend.txt"


@dataclass(frozen=True, slots=True)
class TimeSummarySpec:
    """Resolved time-summary specification for one plotting pass."""

    name: str
    tmin: float
    tmax: float
    mode: str
    cluster_id: str | None = None


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return token.strip("._-") or "item"


def _plot_title(config: SourceDicsConfig, predictor: str) -> str:
    return config.plots.predictor_labels.get(predictor, predictor)


def _resolve_requested_items(requested: list[str] | None, configured: tuple[str, ...]) -> list[str]:
    if requested is None:
        return list(configured)
    return [item for item in requested if item in set(configured)]


def _apply_configured_pvalue_aliases(table: pd.DataFrame, config: SourceDicsConfig) -> pd.DataFrame:
    out = table.copy()
    fallback_p = config.plots.significance.fallback_p_column
    fallback_pcorr = config.plots.significance.fallback_corrected_p_column

    if fallback_p in out.columns and "p_value" in out.columns:
        out["p_value"] = out["p_value"].where(
            out["p_value"].notna(),
            pd.to_numeric(out[fallback_p], errors="coerce"),
        )
    if fallback_pcorr in out.columns and "p_value_corrected" in out.columns:
        out["p_value_corrected"] = out["p_value_corrected"].where(
            out["p_value_corrected"].notna(),
            pd.to_numeric(out[fallback_pcorr], errors="coerce"),
        )
    return out


def apply_significance_rule(
    table: pd.DataFrame,
    *,
    alpha: float,
    use_cluster_mask_if_available: bool,
) -> tuple[pd.DataFrame, str]:
    """Apply significance masking with cluster/corrected/raw priority.

    Parameters
    ----------
    table
        Standardized source-level statistics table.
    alpha
        Significance threshold.
    use_cluster_mask_if_available
        Whether cluster masks should override p-value masking.

    Returns
    -------
    tuple[pd.DataFrame, str]
        Updated table and a short string naming the applied rule.

    Usage example
    -------------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"p_value": [0.01, 0.2], "p_value_corrected": [0.07, 0.9]})
    >>> out, rule = apply_significance_rule(frame, alpha=0.05, use_cluster_mask_if_available=True)
    >>> rule
    'corrected_p'
    """

    out = table.copy()
    cluster_significant = out.get("cluster_significant")
    cluster_p_value = out.get("cluster_p_value")

    if use_cluster_mask_if_available and (
        (cluster_significant is not None)
        or (cluster_p_value is not None and pd.to_numeric(cluster_p_value, errors="coerce").notna().any())
    ):
        cluster_sig_series = (
            cluster_significant.astype(bool)
            if cluster_significant is not None
            else pd.Series(False, index=out.index, dtype=bool)
        )
        cluster_p_series = (
            pd.to_numeric(cluster_p_value, errors="coerce")
            if cluster_p_value is not None
            else pd.Series(np.nan, index=out.index, dtype=float)
        )
        out["significant"] = cluster_sig_series | (cluster_p_series < float(alpha))
        return out, "cluster"

    corrected_series = pd.to_numeric(out.get("p_value_corrected"), errors="coerce")
    if corrected_series.notna().any():
        out["significant"] = corrected_series < float(alpha)
        return out, "corrected_p"

    p_series = pd.to_numeric(out.get("p_value"), errors="coerce")
    if p_series.notna().any():
        out["significant"] = p_series < float(alpha)
        LOGGER.warning(
            "Falling back to uncorrected p-values for significance masking because corrected/cluster masks were unavailable."
        )
        return out, "uncorrected_p"

    if "significant" in out.columns:
        out["significant"] = out["significant"].astype(bool)
        LOGGER.warning("Using precomputed `significant` column because no p-value columns were found.")
        return out, "provided_significant"

    out["significant"] = False
    LOGGER.warning("No significance columns were found; all vertices are treated as non-significant.")
    return out, "none"


def _window_slice(table: pd.DataFrame, *, tmin: float, tmax: float) -> pd.DataFrame:
    return table.loc[(table["time"] >= float(tmin)) & (table["time"] <= float(tmax))].copy()


def summarize_time_window(
    table: pd.DataFrame,
    *,
    spec: TimeSummarySpec,
) -> pd.DataFrame:
    """Summarize source-level t-values within a time window.

    Supported modes are ``mean`` and ``peak_abs``.

    Usage example
    -------------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"source_id": ["1", "1"], "hemi": ["lh", "lh"], "time": [-1.0, -0.5], "t_value": [1.0, -2.0], "significant": [False, True]})
    >>> out = summarize_time_window(frame, spec=TimeSummarySpec(name="demo", tmin=-1.5, tmax=-0.1, mode="peak_abs"))
    >>> float(out.loc[0, "t_value"])
    -2.0
    """

    window = _window_slice(table, tmin=spec.tmin, tmax=spec.tmax)
    if window.empty:
        return pd.DataFrame()

    group_cols = ["band", "predictor", "hemi", "source_id"]
    passthrough_cols = ["label", "roi"]

    if spec.mode == "mean":
        grouped = (
            window.groupby(group_cols, observed=True)
            .agg(
                t_value=("t_value", "mean"),
                significant=("significant", "any"),
                time=("time", "mean"),
            )
            .reset_index()
        )
        for column in passthrough_cols:
            if column in window.columns:
                first_values = (
                    window.groupby(group_cols, observed=True)[column]
                    .first()
                    .reset_index(drop=True)
                )
                grouped[column] = first_values
        grouped["time_summary"] = spec.name
        return grouped

    if spec.mode == "peak_abs":
        peak_idx = (
            window.assign(abs_t=window["t_value"].abs())
            .groupby(group_cols, observed=True)["abs_t"]
            .idxmax()
            .dropna()
            .astype(int)
        )
        peak = window.loc[peak_idx].copy().reset_index(drop=True)
        peak["time_summary"] = spec.name
        return peak

    raise ValueError(f"Unsupported time summary mode: {spec.mode}")


def _cluster_summary_specs(
    table: pd.DataFrame,
    *,
    window_name: str,
    tmin: float,
    tmax: float,
    alpha: float,
) -> list[TimeSummarySpec]:
    if "cluster_id" not in table.columns:
        return []
    if table["cluster_id"].isna().all():
        return []

    window = _window_slice(table, tmin=tmin, tmax=tmax)
    if window.empty:
        return []

    cluster_mask = pd.Series(False, index=window.index, dtype=bool)
    if "cluster_significant" in window.columns:
        cluster_mask |= window["cluster_significant"].astype(bool)
    if "cluster_p_value" in window.columns:
        cluster_mask |= pd.to_numeric(window["cluster_p_value"], errors="coerce") < float(alpha)

    significant_clusters = sorted(window.loc[cluster_mask, "cluster_id"].dropna().astype(str).unique().tolist())
    if not significant_clusters:
        return []

    specs = [
        TimeSummarySpec(
            name=f"{window_name}_cluster_peak_all_significant_clusters",
            tmin=tmin,
            tmax=tmax,
            mode="peak_abs",
            cluster_id="__all__",
        )
    ]
    specs.extend(
        TimeSummarySpec(
            name=f"{window_name}_cluster_peak_cluster_{_safe_token(cluster_id)}",
            tmin=tmin,
            tmax=tmax,
            mode="peak_abs",
            cluster_id=cluster_id,
        )
        for cluster_id in significant_clusters
    )
    return specs


def _resolve_time_summary_specs(table: pd.DataFrame, config: SourceDicsConfig) -> list[TimeSummarySpec]:
    specs: list[TimeSummarySpec] = []
    mode = config.plots.time_summary.mode.strip().lower()
    include_mean = mode in {"mean", "mean_only", "cluster_peak_and_mean", "mean_and_peak"}
    include_peak = mode in {"peak", "peak_abs", "cluster_peak_and_mean", "mean_and_peak"}

    for window_name, window in config.plots.time_summary.windows.items():
        if include_mean:
            specs.append(
                TimeSummarySpec(
                    name=f"{window_name}_mean",
                    tmin=window.tmin,
                    tmax=window.tmax,
                    mode="mean",
                )
            )
        if include_peak:
            specs.append(
                TimeSummarySpec(
                    name=f"{window_name}_peak_abs",
                    tmin=window.tmin,
                    tmax=window.tmax,
                    mode="peak_abs",
                )
            )
        if mode == "cluster_peak_and_mean":
            specs.extend(
                _cluster_summary_specs(
                    table,
                    window_name=window_name,
                    tmin=window.tmin,
                    tmax=window.tmax,
                    alpha=config.plots.significance.alpha,
                )
            )

    deduped: dict[str, TimeSummarySpec] = {}
    for spec in specs:
        deduped[spec.name] = spec
    return list(deduped.values())


def _source_to_vertex(source_id: str) -> int | None:
    token = str(source_id).strip()
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    return None


def _write_placeholder_image(path: Path, *, title: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4), facecolor="white")
    axis.axis("off")
    axis.text(0.5, 0.5, title, ha="center", va="center", fontsize=12)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _combine_surface_pair(left_path: Path, right_path: Path, output_path: Path, *, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    left = plt.imread(left_path)
    right = plt.imread(right_path)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
    axes[0].imshow(left)
    axes[0].axis("off")
    axes[1].imshow(right)
    axes[1].axis("off")
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _expand_roi_rows_to_vertices(
    table: pd.DataFrame,
    *,
    subject: str,
    subjects_dir: Path,
    parc: str,
) -> tuple[pd.DataFrame, bool]:
    """Expand ROI-level rows into vertex-level rows using FreeSurfer labels."""

    import mne

    labels = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=str(subjects_dir), verbose="ERROR")
    lookup: dict[tuple[str, str], np.ndarray] = {}
    for label in labels:
        hemi = "lh" if label.hemi == "lh" else "rh"
        normalized = normalize_aparc_label(label.name)
        lookup[(hemi, normalized)] = np.asarray(label.vertices, dtype=int)

    rows: list[dict[str, Any]] = []
    matched_any = False
    for row in table.to_dict(orient="records"):
        hemi = str(row.get("hemi", "unknown"))
        label_text = str(row.get("label") or row.get("roi") or "")
        normalized = normalize_aparc_label(label_text)

        candidate_hemis = [hemi] if hemi in {"lh", "rh"} else ["lh", "rh"]
        for candidate_hemi in candidate_hemis:
            vertices = lookup.get((candidate_hemi, normalized))
            if vertices is None or len(vertices) == 0:
                continue
            matched_any = True
            for vertex in vertices:
                mapped = dict(row)
                mapped["hemi"] = candidate_hemi
                mapped["source_id"] = str(int(vertex))
                rows.append(mapped)

    return pd.DataFrame(rows), matched_any


def _render_surface_images(
    summary_table: pd.DataFrame,
    *,
    config: SourceDicsConfig,
    lh_path: Path,
    rh_path: Path,
    pair_path: Path,
) -> tuple[str, str | None]:
    """Render lh/rh lateral surfaces and a combined panel.

    Returns
    -------
    tuple[str, str | None]
        Surface-level mode (``vertex-level`` or ``roi-level``) and optional
        backend-error message.
    """

    plot_table = summary_table.copy()
    plot_table["source_vertex"] = plot_table["source_id"].map(_source_to_vertex)

    mode = "vertex-level"
    if plot_table["source_vertex"].isna().all():
        mode = "roi-level"
        if not {"label", "roi"}.intersection(set(plot_table.columns)):
            raise ValueError("ROI-level surface plotting requires `label` or `roi` columns.")
        expanded, matched = _expand_roi_rows_to_vertices(
            plot_table,
            subject=config.plots.source_space,
            subjects_dir=config.source_space.subjects_dir,
            parc=config.plots.roi_plot.atlas,
        )
        if not matched or expanded.empty:
            raise ValueError("Could not map ROI-level labels onto fsaverage vertices.")
        plot_table = expanded
        plot_table["source_vertex"] = plot_table["source_id"].map(_source_to_vertex)

    if config.plots.significance.mask_non_significant:
        plot_table = plot_table.loc[plot_table["significant"].astype(bool)].copy()

    if plot_table.empty:
        _write_placeholder_image(lh_path, title="No significant left-hemisphere sources", dpi=config.plots.surface_plot.dpi)
        _write_placeholder_image(rh_path, title="No significant right-hemisphere sources", dpi=config.plots.surface_plot.dpi)
        _combine_surface_pair(lh_path, rh_path, pair_path, dpi=config.plots.surface_plot.dpi)
        return mode, None

    import mne

    lh_rows = plot_table.loc[(plot_table["hemi"] == "lh") & plot_table["source_vertex"].notna()].copy()
    rh_rows = plot_table.loc[(plot_table["hemi"] == "rh") & plot_table["source_vertex"].notna()].copy()

    if lh_rows.empty and rh_rows.empty:
        _write_placeholder_image(lh_path, title="No plottable source vertices", dpi=config.plots.surface_plot.dpi)
        _write_placeholder_image(rh_path, title="No plottable source vertices", dpi=config.plots.surface_plot.dpi)
        _combine_surface_pair(lh_path, rh_path, pair_path, dpi=config.plots.surface_plot.dpi)
        return mode, None

    lh_vertices = lh_rows["source_vertex"].astype(int).to_numpy()
    rh_vertices = rh_rows["source_vertex"].astype(int).to_numpy()
    lh_values = lh_rows["t_value"].to_numpy(dtype=float)
    rh_values = rh_rows["t_value"].to_numpy(dtype=float)

    all_values = np.concatenate([lh_values, rh_values]) if len(lh_values) or len(rh_values) else np.array([0.0])
    vmax = float(np.nanmax(np.abs(all_values)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    stc = mne.SourceEstimate(
        data=np.concatenate([lh_values, rh_values])[:, np.newaxis].astype(float),
        vertices=[lh_vertices.astype(int), rh_vertices.astype(int)],
        tmin=0.0,
        tstep=1.0,
        subject=config.plots.source_space,
    )

    view = config.plots.views[0] if config.plots.views else "lateral"
    clim = {"kind": "value", "pos_lims": [vmax * 0.33, vmax * 0.66, vmax]}

    for hemi, output_path in (("lh", lh_path), ("rh", rh_path)):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        has_hemi_vertices = (hemi == "lh" and len(lh_vertices) > 0) or (hemi == "rh" and len(rh_vertices) > 0)
        if not has_hemi_vertices:
            _write_placeholder_image(output_path, title=f"No significant {hemi} vertices", dpi=config.plots.surface_plot.dpi)
            continue
        try:
            brain = stc.plot(
                subject=config.plots.source_space,
                hemi=hemi,
                surface=config.plots.surface,
                views=[view],
                subjects_dir=str(config.source_space.subjects_dir),
                background=config.plots.surface_plot.background,
                transparent=config.plots.surface_plot.transparent_background,
                colormap=config.plots.surface_plot.cmap,
                clim=clim,
                colorbar=config.plots.surface_plot.show_colorbar,
                time_viewer=False,
                verbose="ERROR",
            )
            brain.save_image(str(output_path))
            brain.close()
        except Exception as exc:  # noqa: BLE001
            return mode, str(exc)

    _combine_surface_pair(lh_path, rh_path, pair_path, dpi=config.plots.surface_plot.dpi)
    return mode, None


def _compute_roi_summary(summary_table: pd.DataFrame) -> pd.DataFrame:
    roi_label_column = "label" if "label" in summary_table.columns else "roi" if "roi" in summary_table.columns else None
    if roi_label_column is None:
        fallback = summary_table.copy()
        fallback["label"] = "unknown"
        roi_label_column = "label"
    else:
        fallback = summary_table.copy()

    with_rois = add_broad_roi_columns(
        fallback,
        label_column=roi_label_column,
        hemisphere_column="hemi" if "hemi" in fallback.columns else None,
    )

    rows: list[dict[str, Any]] = []
    for (hemi, broad_roi), group in with_rois.groupby(["hemisphere", "broad_roi"], observed=True):
        significant_mask = group["significant"].astype(bool)
        significant_group = group.loc[significant_mask]
        n_sources = int(group["source_id"].nunique())
        n_significant = int(significant_group["source_id"].nunique())

        rows.append(
            {
                "hemisphere": str(hemi),
                "broad_roi": str(broad_roi),
                "mean_t_value": float(group["t_value"].mean()),
                "mean_abs_t_value": float(group["t_value"].abs().mean()),
                "max_abs_t_value": float(group["t_value"].abs().max()),
                "n_sources": n_sources,
                "n_significant_sources": n_significant,
                "proportion_significant_sources": float(n_significant / n_sources) if n_sources else 0.0,
                "mean_t_value_significant_only": (
                    float(significant_group["t_value"].mean()) if not significant_group.empty else np.nan
                ),
                "mean_abs_t_value_significant_only": (
                    float(significant_group["t_value"].abs().mean()) if not significant_group.empty else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def _plot_roi_bars(
    roi_summary: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    hemis = [hemi for hemi in ("lh", "rh", "unknown") if hemi in set(roi_summary["hemisphere"])]
    if not hemis:
        hemis = ["unknown"]

    figure, axes = plt.subplots(1, len(hemis), figsize=(6 * len(hemis), 8), sharex=True, facecolor="white")
    if len(hemis) == 1:
        axes = [axes]

    for axis, hemi in zip(axes, hemis):
        subset = roi_summary.loc[roi_summary["hemisphere"] == hemi].sort_values("broad_roi", kind="mergesort")
        y = np.arange(len(subset))
        x = subset["mean_t_value"].to_numpy(dtype=float)
        colors = ["#2c7fb8" if value >= 0 else "#d7191c" for value in x]

        axis.barh(y, x, color=colors, alpha=0.85)
        axis.set_yticks(y)
        axis.set_yticklabels(subset["broad_roi"].tolist())
        axis.axvline(0.0, color="black", linewidth=1.0)
        axis.set_title(f"{hemi.upper()} hemisphere")
        axis.set_xlabel("Mean t-value")

        for row_index, (_, row) in enumerate(subset.iterrows()):
            axis.text(
                float(row["mean_t_value"]),
                row_index,
                f"  n_sig={int(row['n_significant_sources'])}",
                va="center",
                fontsize=8,
            )

    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _write_index_files(
    rows: list[dict[str, Any]],
    *,
    index_dir: Path,
    write_csv: bool,
    write_markdown: bool,
) -> tuple[Path, Path | None]:
    index_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)

    csv_path = index_dir / "figure_index.csv"
    if write_csv:
        table.to_csv(csv_path, index=False)
    else:
        csv_path.write_text("artifact_type,path\n", encoding="utf-8")

    md_path: Path | None = None
    if write_markdown:
        md_path = index_dir / "figure_index.md"
        if table.empty:
            md_path.write_text("# Figure Index\n\nNo figures were generated.\n", encoding="utf-8")
        else:
            markdown = ["# Figure Index", "", "| artifact_type | band | predictor | time_summary | path |", "|---|---|---|---|---|"]
            for row in table.to_dict(orient="records"):
                markdown.append(
                    f"| {row.get('artifact_type', '')} | {row.get('band', '')} | {row.get('predictor', '')} | {row.get('time_summary', '')} | {row.get('path', '')} |"
                )
            md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    return csv_path, md_path


def run_source_dics_plotting(
    config: SourceDicsConfig,
    *,
    predictors: list[str] | None = None,
    bands: list[str] | None = None,
    time_summaries: list[str] | None = None,
    surface_only: bool = False,
    roi_only: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
) -> Path:
    """Run batch plotting for source-level DICS lmeEEG outputs.

    Parameters
    ----------
    config
        Parsed source-DICS configuration.
    predictors
        Optional predictor subset.
    bands
        Optional frequency-band subset.
    time_summaries
        Optional time-summary name subset.
    surface_only
        If true, skip ROI summaries.
    roi_only
        If true, skip surface figures.
    overwrite
        If false, existing output files are retained and indexed.
    verbose
        If true, use info-level logging.

    Returns
    -------
    Path
        Path to ``figure_index.csv``.

    Usage example
    -------------
    >>> from pathlib import Path
    >>> from cas.source_dics.config import load_source_dics_config
    >>> cfg = load_source_dics_config(Path("config/induced/source_localisation.yaml"))  # doctest: +SKIP
    >>> run_source_dics_plotting(cfg, roi_only=True)  # doctest: +SKIP
    PosixPath('.../figure_index.csv')
    """

    level = logging.INFO if (verbose or config.logging.verbose) else logging.WARNING
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s", force=True)

    output_root = config.plots.output_dir
    surface_root = output_root / "surface"
    roi_root = output_root / "roi"
    index_root = output_root / "index"
    output_root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)

    requested_bands = _resolve_requested_items(bands, config.plots.bands)
    requested_predictors = _resolve_requested_items(predictors, config.plots.predictors)

    load_result = load_source_statistics_table(
        config.plots.statistics_dir,
        predictors=set(requested_predictors) if requested_predictors else None,
        bands=set(requested_bands) if requested_bands else None,
    )
    adapted_table = _apply_configured_pvalue_aliases(load_result.table, config)
    table, significance_rule = apply_significance_rule(
        adapted_table,
        alpha=config.plots.significance.alpha,
        use_cluster_mask_if_available=config.plots.significance.use_cluster_mask_if_available,
    )

    found_predictors = sorted(table["predictor"].astype(str).unique().tolist())
    missing_predictors = sorted(set(requested_predictors) - set(found_predictors))
    found_bands = sorted(table["band"].astype(str).unique().tolist())

    LOGGER.info("Loaded %d rows from %d compatible statistics files.", load_result.rows_loaded, load_result.files_loaded)
    LOGGER.info("Significance rule used: %s", significance_rule)
    LOGGER.info("Predictors found: %d", len(found_predictors))
    if missing_predictors:
        LOGGER.warning("Requested predictors missing from statistics: %s", ", ".join(missing_predictors))
    LOGGER.info("Bands found: %d", len(found_bands))

    summary_specs = _resolve_time_summary_specs(table, config)
    if time_summaries is not None:
        requested_time_set = set(time_summaries)
        summary_specs = [spec for spec in summary_specs if spec.name in requested_time_set]
    if not summary_specs:
        raise ValueError("No time summaries were resolved for plotting.")

    index_rows: list[dict[str, Any]] = []
    surface_enabled = not roi_only
    surface_backend_error: str | None = None

    band_iter = tqdm(found_bands, desc="Bands", disable=not config.logging.progress)
    for band_name in band_iter:
        band_table = table.loc[table["band"] == band_name].copy()
        predictor_iter = tqdm(
            [p for p in found_predictors if p in set(band_table["predictor"].unique())],
            desc=f"{band_name} predictors",
            leave=False,
            disable=not config.logging.progress,
        )
        for predictor_name in predictor_iter:
            predictor_table = band_table.loc[band_table["predictor"] == predictor_name].copy()
            summary_iter = tqdm(
                summary_specs,
                desc=f"{band_name}/{predictor_name} summaries",
                leave=False,
                disable=not config.logging.progress,
            )
            for spec in summary_iter:
                working = predictor_table.copy()
                if spec.cluster_id is not None and spec.cluster_id != "__all__":
                    working = working.loc[working["cluster_id"].astype(str) == spec.cluster_id]
                if spec.cluster_id == "__all__":
                    cluster_mask = pd.Series(False, index=working.index, dtype=bool)
                    if "cluster_significant" in working.columns:
                        cluster_mask |= working["cluster_significant"].astype(bool)
                    if "cluster_p_value" in working.columns:
                        cluster_mask |= pd.to_numeric(working["cluster_p_value"], errors="coerce") < float(
                            config.plots.significance.alpha
                        )
                    working = working.loc[cluster_mask]

                summary = summarize_time_window(working, spec=spec)
                if summary.empty:
                    continue

                n_significant = int(summary.loc[summary["significant"].astype(bool), "source_id"].nunique())
                LOGGER.info(
                    "%s | %s | %s -> %d significant sources",
                    band_name,
                    predictor_name,
                    spec.name,
                    n_significant,
                )

                predictor_token = _safe_token(predictor_name)
                summary_token = _safe_token(spec.name)

                if surface_enabled and surface_backend_error is None:
                    lh_path = surface_root / band_name / predictor_token / f"{predictor_token}_{band_name}_{summary_token}_lh_lateral.png"
                    rh_path = surface_root / band_name / predictor_token / f"{predictor_token}_{band_name}_{summary_token}_rh_lateral.png"
                    pair_path = surface_root / band_name / predictor_token / f"{predictor_token}_{band_name}_{summary_token}_lateral_pair.png"

                    if overwrite or not (lh_path.exists() and rh_path.exists() and pair_path.exists()):
                        try:
                            surface_mode, backend_error = _render_surface_images(
                                summary,
                                config=config,
                                lh_path=lh_path,
                                rh_path=rh_path,
                                pair_path=pair_path,
                            )
                        except Exception as exc:  # noqa: BLE001
                            surface_mode = "unavailable"
                            backend_error = str(exc)

                        if backend_error is not None:
                            surface_backend_error = backend_error
                            message = (
                                "Surface plotting backend is unavailable. "
                                f"Reason: {backend_error}\n"
                                "ROI plotting completed; surface figures were skipped.\n"
                            )
                            (index_root / MISSING_SURFACE_BACKEND_FILE).write_text(message, encoding="utf-8")
                            LOGGER.warning(message.strip())
                            surface_enabled = False
                        else:
                            for surface_path, kind in (
                                (lh_path, "surface_lh"),
                                (rh_path, "surface_rh"),
                                (pair_path, "surface_pair"),
                            ):
                                index_rows.append(
                                    {
                                        "artifact_type": kind,
                                        "band": band_name,
                                        "predictor": predictor_name,
                                        "time_summary": spec.name,
                                        "path": str(surface_path),
                                        "surface_level": surface_mode,
                                        "n_significant_sources": n_significant,
                                    }
                                )
                    else:
                        for surface_path, kind in (
                            (lh_path, "surface_lh"),
                            (rh_path, "surface_rh"),
                            (pair_path, "surface_pair"),
                        ):
                            index_rows.append(
                                {
                                    "artifact_type": kind,
                                    "band": band_name,
                                    "predictor": predictor_name,
                                    "time_summary": spec.name,
                                    "path": str(surface_path),
                                    "surface_level": "cached",
                                    "n_significant_sources": n_significant,
                                }
                            )

                if surface_only:
                    continue
                if not config.plots.roi_plot.enabled:
                    continue

                roi_summary = _compute_roi_summary(summary)
                if roi_summary.empty:
                    continue

                roi_summary.insert(0, "time_summary", spec.name)
                roi_summary.insert(0, "predictor", predictor_name)
                roi_summary.insert(0, "band", band_name)

                roi_dir = roi_root / band_name
                csv_path = roi_dir / f"{predictor_token}_{band_name}_{summary_token}_broad_roi_tvalues.csv"
                png_path = roi_dir / f"{predictor_token}_{band_name}_{summary_token}_broad_roi_tvalues.png"

                if overwrite or not csv_path.exists():
                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                    roi_summary.to_csv(csv_path, index=False)
                if overwrite or not png_path.exists():
                    _plot_roi_bars(
                        roi_summary,
                        title=f"{_plot_title(config, predictor_name)} | {band_name} | {spec.name}",
                        output_path=png_path,
                        dpi=config.plots.roi_plot.dpi,
                    )

                index_rows.append(
                    {
                        "artifact_type": "roi_csv",
                        "band": band_name,
                        "predictor": predictor_name,
                        "time_summary": spec.name,
                        "path": str(csv_path),
                        "surface_level": "",
                        "n_significant_sources": n_significant,
                    }
                )
                index_rows.append(
                    {
                        "artifact_type": "roi_plot",
                        "band": band_name,
                        "predictor": predictor_name,
                        "time_summary": spec.name,
                        "path": str(png_path),
                        "surface_level": "",
                        "n_significant_sources": n_significant,
                    }
                )

    if surface_backend_error is not None:
        index_rows.append(
            {
                "artifact_type": "note",
                "band": "",
                "predictor": "",
                "time_summary": "",
                "path": str(index_root / MISSING_SURFACE_BACKEND_FILE),
                "surface_level": "",
                "n_significant_sources": 0,
            }
        )

    figure_index_path, _ = _write_index_files(
        index_rows,
        index_dir=index_root,
        write_csv=config.plots.report.write_csv_index,
        write_markdown=config.plots.report.write_markdown_index,
    )
    LOGGER.info("Wrote figure index to %s", figure_index_path)
    return figure_index_path
