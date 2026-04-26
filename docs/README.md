## Behavioural Hazard Final Inference

The behavioural hazard final-inference workflow now runs in three stages:

1. Python builds the partner-IPU-anchored behavioural risk set, computes timing controls, runs timing-controlled lag selection, and exports a compact GLMM-ready CSV with `export-behaviour-glmm-data`.
2. R fits the final logistic mixed-effects models with `glmmTMB` using `scripts/r/fit_behaviour_hazard_glmm.R`.
3. Python reads the R outputs and renders the active publication figure suite with `plot-behaviour-hazard-results`.

This final inference path is behavioural only. It does not add neural features and it does not implement HMM, GLHMM, entropy, or K-selection.

The active unit of analysis is the partner-IPU anchored discrete-time hazard episode, with 50 ms risk bins, first-FPP-by-the-other-speaker as the event, and censored episodes retained.

### Model family

The final behavioural model is a logistic mixed-effects model for the binary hazard outcome `event in {0, 1}`, not an LMM. The default random intercept is `participant_speaker`.

The final R model sequence is:

- `M0_timing`: onset and offset timing splines plus `(1 | participant_speaker)`
- `M1_rate`: `M0_timing` plus `z_information_rate_lag_best`
- `M2_expected`: `M1_rate` plus `z_prop_expected_cumulative_info_lag_best`

Active behavioural predictors are limited to onset timing, offset timing, information rate, and expected cumulative information.

### Lag usage

Lag handling is now split into two stages:

- Python pooled lag screening for QC only
- R GLMM lag sweep for final mixed-effects lag inference

The pooled screen remains useful for quick diagnostics, but final behavioural lag conclusions should be based on the R GLMM lag-sweep outputs and the selected lag JSON written by that stage.

### Model comparison convention

For behavioural GLMM model comparisons:

- `delta_aic = child_aic - parent_aic`
- negative `delta_aic` favours the child model

### Active figures

Top-level final behavioural figures in `figures/` are:

- `behaviour_r_glmm_delta_bic_by_lag.png`
- `behaviour_r_glmm_coefficient_by_lag.png`
- `behaviour_r_glmm_odds_ratio_by_lag.png`
- `behaviour_r_glmm_final_model_comparison.png`
- `behaviour_r_glmm_final_predicted_hazard_information_rate.png`

Behavioural QC figures are stored separately under `qc_plots/`, with lag-screening output at:

- `qc_plots/lag_selection/behaviour_pooled_delta_bic_by_lag.png`

Optional latency diagnostics may be generated separately. Older behavioural analyses and figures have been moved under `legacy/behavior/` for provenance only. Exploratory `prop_actual` and raw `cumulative_info` outputs are not part of the active primary model.

A separate exploratory event-only Stan analysis is also available for offset-relative FPP latency regimes. It is opt-in, does not replace the primary hazard model, and lives under `results/hazard_behavior/latency_regime/`. See [docs/behaviour_latency_regime_stan.md](/Users/hiro/Projects/active/cas/docs/behaviour_latency_regime_stan.md).

### Example flow

```bash
python -m cas.cli.main export-behaviour-glmm-data \
  --input-riskset results/hazard_behavior/riskset/hazard_behavior_riskset_with_timing_controls.tsv \
  --selected-lags-json results/hazard_behavior/models/behaviour_timing_control_selected_lags.json \
  --lag-grid-ms 0,50,100,150,200,300,500,700,1000 \
  --output-csv results/hazard_behavior/models/exports/r_behaviour_glmm_lag_sweep_input.csv \
  --output-qc-json results/hazard_behavior/models/exports/r_behaviour_glmm_lag_sweep_export_qc.json

Rscript scripts/r/fit_behaviour_glmm_lag_sweep.R \
  --input-csv results/hazard_behavior/models/exports/r_behaviour_glmm_lag_sweep_input.csv \
  --output-dir results/hazard_behavior/models

python -m cas.cli.main plot-behaviour-hazard-results \
  --r-results-dir results/hazard_behavior/models \
  --timing-control-models-dir results/hazard_behavior/models/lag_selection \
  --qc-output-dir results/hazard_behavior/qc_plots/lag_selection \
  --output-dir results/hazard_behavior/figures

python -m cas.cli.main export-behaviour-latency-regime-data \
  --input-riskset results/hazard_behavior/riskset/hazard_behavior_riskset_with_timing_controls.tsv

Rscript scripts/r/fit_behaviour_latency_regime_stan.R \
  --input-csv results/hazard_behavior/latency_regime/behaviour_latency_regime_data.csv \
  --output-dir results/hazard_behavior/latency_regime/stan_models

python -m cas.cli.main plot-behaviour-latency-regime-results \
  --stan-results-dir results/hazard_behavior/latency_regime/stan_models \
  --event-data-csv results/hazard_behavior/latency_regime/behaviour_latency_regime_data.csv \
  --output-dir results/hazard_behavior/latency_regime/figures
```

## Neural Hazard (Low-Level Alpha/Beta)

The active neural hazard path now mirrors the behavioural partner-IPU structure and runs two separate analyses:

1. FPP hazard: event is first participant-speaker FPP within each partner-IPU episode.
2. SPP hazard: event is first participant-speaker SPP within each partner-IPU episode.

Both analyses use the same episode construction, risk grid, timing controls, behavioural controls, neural windowing, PCA flow, and model-comparison convention.

### Episode and risk-set definition

- Episodes are anchored to each partner IPU onset.
- Episode end is the earliest of next partner IPU onset or `max_followup_s` (default `6.0`).
- Events during overlap/interruption are valid.
- `event_latency_from_partner_offset_s` is retained, with phase labels:
  - `during_partner_ipu` if latency `< 0`
  - `post_partner_ipu` otherwise
- Risk bins are discrete-time (default `bin_size_s = 0.050`).

### Behavioural baseline controls in neural models

Each neural baseline includes:

- `bs(time_from_partner_onset, df=6, degree=3)`
- `bs(time_from_partner_offset, df=6, degree=3)`
- `z_information_rate_lag_150ms`
- `z_prop_expected_cumulative_info_lag_700ms`
- participant and dyad grouping IDs retained for mixed-effects compatibility

`z_information_rate_lag_150ms` remains the primary information-flow behavioural control.  
`z_prop_expected_cumulative_info_lag_700ms` is retained as a control variable and should not be read as direct confirmation of a positive accumulation-threshold claim.

### Primary neural features

Primary model families use participant-speaker low-level EEG only:

- alpha power
- beta power
- alpha+beta power

For each risk bin, neural features use a guarded causal window:

- start: `bin_end - 0.500 s`
- end: `bin_end - 0.100 s`

This guard avoids using data at or after the hazard bin onset and reduces motor/speech-onset leakage.

Alpha and beta are reduced separately:

- z-score channel/window features
- run PCA separately for alpha and beta
- retain configurable PCs (count or variance threshold)
- z-score PC scores for modelling

Raw amplitude and HMM entropy are excluded from this primary low-level neural model family.

### Model families and comparison

For each event type (FPP, SPP):

- behavioural baseline
- baseline + alpha PCs
- baseline + beta PCs
- baseline + alpha+beta PCs

Model comparison uses:

- `delta_bic = BIC(child) - BIC(parent)`
- `delta_aic = AIC(child) - AIC(parent)`

Negative deltas favour the neural child model.

When available, the preferred backend is `glmmTMB` (matching behavioural GLMM usage). If unavailable in the runtime, the pipeline falls back to a documented Python binomial-GLM path for continuity.

### Neural outputs

The neural run writes:

- `riskset/neural_fpp_hazard_table.parquet` (or `.csv`)
- `riskset/neural_spp_hazard_table.parquet` (or `.csv`)
- `models/neural_model_comparison.csv`
- `models/neural_coefficients.csv`
- `models/neural_fit_metrics.json`
- figures:
  - `neural_delta_bic_fpp_vs_spp.png`
  - `neural_delta_aic_fpp_vs_spp.png`
  - `neural_coefficients_fpp_vs_spp.png`
  - `neural_power_by_partner_time.png`

### Interpretation policy

- Similar alpha/beta improvement for FPP and SPP suggests generic speech preparation.
- FPP-only improvement is a candidate FPP-relevant signal.
- SPP-only improvement is not evidence for FPP specificity.
- Larger FPP than SPP improvement is a stronger FPP-specificity candidate, to be tested later with direct FPP-vs-SPP models.
