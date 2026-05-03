# Behavioural Latency-Regime Stan Analysis

This analysis is a secondary, event-only exploratory follow-up to the primary behavioural hazard model.

- It does not replace the primary hazard analysis.
- It is not part of the default hazard pipeline.
- It should be interpreted cautiously.

The outcome is offset-relative FPP latency:

- `latency_from_partner_offset = fpp_onset - partner_ipu_offset`

Negative latencies are valid and indicate overlap or an early response relative to partner IPU offset.

The analysis uses only observed FPP events (`event == 1`) and only active timing-controlled predictors:

- `z_information_rate_lag_best`
- `z_prop_expected_cumulative_info_lag_best`

These lag-best predictors come from `behaviour_timing_control_selected_lags.json` unless explicit lag overrides are supplied at export time.

## Active exploratory model set

1. Model A: one Student-t latency distribution.
2. Model S: one skewed unimodal latency distribution (skew-normal competitor).
3. Model B: two Student-t mixture with a constant late-component weight.
4. Model C: mixture-of-experts where information variables predict late-component probability.
5. Model R1: single-regime Student-t location regression.
6. Model R2: single-regime Student-t location plus scale regression.
7. Model R3: single-regime shifted-lognormal location regression with a fixed negative-latency-preserving shift.
8. Model R4: single-regime shifted-lognormal location plus scale regression with the same fixed shift.

The constrained Model D has been retired from the active exploratory pipeline and preserved under `legacy/behavior/` for provenance.

Shifted lognormal is the only additional latency family used in this refactor. The exploratory comparison does not include Gamma, ex-Gaussian, Wald, or inverse-Gaussian alternatives.

## Current interpretation target

The active question is whether offset-relative latency is better described by:

- a single symmetric heavy-tailed process (Model A),
- a single skewed process (Model S),
- a constant-weight mixture (Model B),
- a covariate-dependent mixture separating main latency component versus longer-latency/right-tail component (Model C),
- or a single-regime regression alternative where information variables continuously shift latency (Models R1-R4).

Model comparison should rely on LOO and posterior predictive checks rather than likelihood-ratio tests.

- Model C vs Model B asks whether information variables predict latent component membership better than a constant mixture.
- Model C vs Model R asks whether mixture-of-experts structure is needed beyond a single-regime latency regression alternative.
- If R models are competitive with Model C, prefer a continuous latency-shift interpretation.
- If Model C beats R models and the later component is broad/right-tailed, describe it as information-dependent prediction of long-latency/right-tail membership rather than proof of two psychological regimes.
