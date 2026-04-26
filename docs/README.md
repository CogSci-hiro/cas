## Behavioural Hazard Final Inference

The behavioural hazard final-inference workflow now runs in three stages:

1. Python builds the partner-IPU-anchored behavioural risk set and exports a compact GLMM-ready CSV with `export-behaviour-glmm-data`.
2. R fits the final logistic mixed-effects models with `glmmTMB` using `scripts/r/fit_behaviour_hazard_glmm.R`.
3. Python reads the R outputs and renders publication-style figures with `plot-behaviour-glmm-results`.

This final inference path is behavioural only. It does not add neural features and it does not implement HMM, GLHMM, entropy, or K-selection.

### Model family

The final behavioural model is a logistic mixed-effects model for the binary hazard outcome `event in {0, 1}`, not an LMM. The default random intercept is `participant_speaker`.

The final R model sequence is:

- `M0_timing`: onset and offset timing splines plus `(1 | participant_speaker)`
- `M1_rate`: `M0_timing` plus `z_information_rate_lag_best`
- `M2_expected`: `M1_rate` plus `z_prop_expected_cumulative_info_lag_best`

### Lag usage

Lag selection remains in Python for fast pooled screening and QC. Final inference uses the lags selected from the timing-controlled pooled lag-selection stage, usually from `behaviour_timing_control_selected_lags.json`. Explicit CLI lag overrides are supported when those selected lags are unavailable.

### Model comparison convention

For behavioural GLMM model comparisons:

- `delta_aic = child_aic - parent_aic`
- negative `delta_aic` favours the child model

### Example flow

```bash
python -m cas.cli.main export-behaviour-glmm-data \
  --input-riskset results/hazard_behavior/riskset/hazard_behavior_riskset_with_timing_controls.tsv \
  --selected-lags-json results/hazard_behavior/models/behaviour_timing_control_selected_lags.json \
  --output-csv results/hazard_behavior/r_exports/behaviour_glmm_data.csv \
  --output-qc-json results/hazard_behavior/r_exports/behaviour_glmm_export_qc.json

Rscript scripts/r/fit_behaviour_hazard_glmm.R \
  --input-csv results/hazard_behavior/r_exports/behaviour_glmm_data.csv \
  --output-dir results/hazard_behavior/r_models

python -m cas.cli.main plot-behaviour-glmm-results \
  --r-results-dir results/hazard_behavior/r_models \
  --output-dir results/hazard_behavior/figures
```
