"""Nested comparison specifications for the behavioral hazard pipeline."""

from __future__ import annotations


MODEL_COMPARISONS = {
    "fpp": [("M_0", "M_1"), ("M_0", "M_2"), ("M_1", "M_3"), ("M_2", "M_3"), ("M_3", "M_4")],
    "spp": [("M_0", "M_1"), ("M_0", "M_2"), ("M_1", "M_3"), ("M_2", "M_3"), ("M_3", "M_4")],
    "pooled": [("M_pooled_main", "M_pooled_anchor_interaction")],
}
