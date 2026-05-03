"""Broad ROI mapping utilities for source-DICS plotting.

Usage example
-------------
>>> import pandas as pd
>>> table = pd.DataFrame({"label": ["ctx-lh-superiorfrontal", "rh.precentral"]})
>>> out = add_broad_roi_columns(table, label_column="label", hemisphere_column=None)
>>> out["broad_roi"].tolist()
['Frontal', 'Motor']
"""

from __future__ import annotations

import re
from typing import Final

import pandas as pd

_LABEL_SANITIZE_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z]+")

_BROAD_ROI_LABELS: Final[dict[str, tuple[str, ...]]] = {
    "Frontal": (
        "superiorfrontal",
        "rostralmiddlefrontal",
        "caudalmiddlefrontal",
        "parsopercularis",
        "parsorbitalis",
        "parstriangularis",
        "lateralorbitofrontal",
        "medialorbitofrontal",
        "frontalpole",
    ),
    "Motor": (
        "precentral",
        "paracentral",
    ),
    "Somatosensory": (
        "postcentral",
    ),
    "Parietal": (
        "superiorparietal",
        "inferiorparietal",
        "supramarginal",
        "precuneus",
    ),
    "Temporal": (
        "superiortemporal",
        "middletemporal",
        "inferiortemporal",
        "bankssts",
        "transversetemporal",
        "temporalpole",
        "entorhinal",
        "parahippocampal",
        "fusiform",
    ),
    "Occipital": (
        "lateraloccipital",
        "cuneus",
        "pericalcarine",
        "lingual",
    ),
    "Cingulate/medial": (
        "rostralanteriorcingulate",
        "caudalanteriorcingulate",
        "posteriorcingulate",
        "isthmuscingulate",
    ),
    "Insula": (
        "insula",
    ),
}

_OTHER_ROI_NAME: Final[str] = "Other/unknown"
_BROAD_LOOKUP: Final[dict[str, str]] = {
    _LABEL_SANITIZE_RE.sub("", label.lower()): broad_roi
    for broad_roi, labels in _BROAD_ROI_LABELS.items()
    for label in labels
}


def _extract_hemisphere(label: str) -> tuple[str | None, str]:
    """Extract hemisphere and base label from diverse FreeSurfer naming styles."""

    token = str(label).strip()
    if not token:
        return None, ""

    lowered = token.lower()
    if lowered.startswith("ctx-lh-"):
        return "lh", token[7:]
    if lowered.startswith("ctx-rh-"):
        return "rh", token[7:]
    if lowered.startswith("lh."):
        return "lh", token[3:]
    if lowered.startswith("rh."):
        return "rh", token[3:]
    if lowered.endswith("-lh"):
        return "lh", token[:-3]
    if lowered.endswith("-rh"):
        return "rh", token[:-3]

    return None, token


def normalize_aparc_label(label: str) -> str:
    """Normalize a label name to a robust lower-case aparc token.

    Parameters
    ----------
    label
        Original label string. Supports variants such as
        ``ctx-lh-superiorfrontal`` and ``superiorfrontal-lh``.

    Returns
    -------
    str
        Normalized token, e.g. ``superiorfrontal``.

    Usage example
    -------------
    >>> normalize_aparc_label("ctx-lh-superiorfrontal")
    'superiorfrontal'
    """

    _, base = _extract_hemisphere(label)
    return _LABEL_SANITIZE_RE.sub("", base.lower())


def map_aparc_to_broad_roi(label: str) -> str:
    """Map an aparc-style label to a broad interpretable ROI group.

    Parameters
    ----------
    label
        Label text in any common FreeSurfer style.

    Returns
    -------
    str
        Broad ROI name. Unmapped labels return ``Other/unknown``.

    Usage example
    -------------
    >>> map_aparc_to_broad_roi("lh.superiorfrontal")
    'Frontal'
    """

    normalized = normalize_aparc_label(label)
    return _BROAD_LOOKUP.get(normalized, _OTHER_ROI_NAME)


def add_broad_roi_columns(
    frame: pd.DataFrame,
    *,
    label_column: str,
    hemisphere_column: str | None,
) -> pd.DataFrame:
    """Attach ``broad_roi`` and ``hemisphere`` columns to a source table.

    Parameters
    ----------
    frame
        Input data frame containing a label column.
    label_column
        Column name containing aparc-style ROI labels.
    hemisphere_column
        Optional hemisphere column. If missing or null, hemisphere is inferred
        from label text when possible.

    Returns
    -------
    pd.DataFrame
        Copy of the input table with added ``broad_roi`` and ``hemisphere``
        columns.

    Usage example
    -------------
    >>> import pandas as pd
    >>> table = pd.DataFrame({"label": ["lh.precentral", "ctx-rh-insula"]})
    >>> out = add_broad_roi_columns(table, label_column="label", hemisphere_column=None)
    >>> out[["hemisphere", "broad_roi"]].to_dict(orient="records")
    [{'hemisphere': 'lh', 'broad_roi': 'Motor'}, {'hemisphere': 'rh', 'broad_roi': 'Insula'}]
    """

    out = frame.copy()
    labels = out[label_column].fillna("").astype(str)

    inferred_hemi: list[str | None] = []
    broad_rois: list[str] = []
    for label in labels:
        hemi, _ = _extract_hemisphere(label)
        inferred_hemi.append(hemi)
        broad_rois.append(map_aparc_to_broad_roi(label))

    if hemisphere_column is not None and hemisphere_column in out.columns:
        hemi_values = out[hemisphere_column].where(out[hemisphere_column].notna(), pd.Series(inferred_hemi))
        out["hemisphere"] = hemi_values.fillna("unknown").astype(str)
    else:
        out["hemisphere"] = pd.Series(inferred_hemi).fillna("unknown").astype(str)

    out["broad_roi"] = broad_rois
    return out
