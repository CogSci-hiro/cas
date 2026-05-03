# Final Behavioral Hazard Analysis

## Purpose
This branch runs a final, auditable behavioral discrete-time hazard analysis to test whether partner information dynamics during FPPs predict response hazard above partner-onset and partner-offset timing, with SPP as a matched negative-control anchor.

## Model Sequence
For each anchor type (`fpp`, `spp`) at a shared selected lag `L*`:
- `timing_only`: `event_bin ~ z_time_from_partner_onset_s + z_time_from_partner_offset_s + z_time_from_partner_offset_s_squared`
- `information_rate`: timing terms + `information_rate_lag_<L*>ms_z`
- `full_information`: timing terms + `information_rate_lag_<L*>ms_z` + `prop_expected_cumulative_info_lag_<L*>ms_z`
- `timing_information_rate_interaction`: full-information terms + `z_time_from_partner_onset_s:information_rate_lag_<L*>ms_z` + `z_time_from_partner_offset_s:information_rate_lag_<L*>ms_z`

The primary behavioural-final analysis uses the stable shared linear/quadratic timing baseline for both FPP and SPP.

## Lag-Selection Policy
- Selection uses **FPP only**.
- One shared lag is chosen by minimum BIC across candidate full-information models.
- The chosen lag is frozen and reused for FPP, SPP, and pooled FPP-vs-SPP comparison.
- SPP does **not** reselect lag.

## Pooled Interaction Model
The pooled model tests anchor interactions:
- `event_bin ~ anchor_type * (z_time_from_partner_onset_s + z_time_from_partner_offset_s + z_time_from_partner_offset_s_squared + information_rate_lag_<L*>ms_z + prop_expected_cumulative_info_lag_<L*>ms_z)`

## Timing x Information-Rate Interaction Analysis
As a secondary final analysis, we tested whether the information-rate effect
varied as a function of timing relative to partner onset and offset. Using the
same FPP-selected lag as the primary analysis, we fit an interaction model adding
timing-by-information-rate interaction terms for z-scored time from partner onset
and z-scored time from partner offset. This model was compared against the full-information additive
model. The same interaction model was also fit to SPP-anchored episodes as a
negative-control analysis.

This analysis:
- uses the same selected lag as the primary model
- does not perform separate lag selection
- compares against the additive full-information model
- should be interpreted as a timing-dependent information-rate effect

## Output Structure
Outputs are written under `results/hazard_behavior/final/`, including:
- `lag_selection/` artifacts (`fpp_lag_selection_table.csv`, `selected_lag.json`, `lag_selection_plot.png`)
- `fpp/` and `spp_control/` risksets, episode tables, standardization stats, model summaries, coefficients, and comparisons
- `fpp_vs_spp/` pooled interaction outputs, contrast table, markdown/json reports, and QC plots

## How To Run
Snakemake target:

```bash
snakemake behavior_final_all
```

Equivalent CLI sequence:

```bash
cas behavior-final-select-lag --config config/behavior.yaml --out-dir results/hazard_behavior/final/lag_selection
cas behavior-final-fit --config config/behavior.yaml --anchor-type fpp --selected-lag results/hazard_behavior/final/lag_selection/selected_lag.json --out-dir results/hazard_behavior/final/fpp
cas behavior-final-fit --config config/behavior.yaml --anchor-type spp --selected-lag results/hazard_behavior/final/lag_selection/selected_lag.json --out-dir results/hazard_behavior/final/spp_control
cas behavior-final-compare --config config/behavior.yaml --selected-lag results/hazard_behavior/final/lag_selection/selected_lag.json --fpp-riskset results/hazard_behavior/final/fpp/riskset.parquet --spp-riskset results/hazard_behavior/final/spp_control/riskset.parquet --out-dir results/hazard_behavior/final/fpp_vs_spp
```
