"""Comparison specifications for the behavioral hazard pipeline."""

from __future__ import annotations


MODEL_COMPARISONS = {
    "primary_fpp": [("A0_timing", "A1_information_rate"), ("A0_timing", "A2_expected_cum_info"), ("A0_timing", "A3_joint_information")],
    "fpp_spp_control": [("B1_shared_information", "B2_anchor_x_information")],
    "timing_moderation": [("A3_joint_information", "C1_onset_x_rate"), ("A3_joint_information", "C2_offset_x_rate")],
    "exploratory": [("D1_two_way_reference", "D2_three_way")],
}
