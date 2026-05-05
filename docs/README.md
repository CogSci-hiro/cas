## Active Surfaces

The active behavioral workflow is the `behavior_hazard_*` pipeline rooted at `config/behavior/hazard.yaml` and `workflow/rules/behavior.smk`.

For behavioral hazard modeling, `behavior.model_backend` selects either:

- `glmm`: the current R `glmmTMB` mixed-effects backend with random intercepts for `dyad_id` and `subject`
- `glm`: a fixed-effect binomial GLM comparison backend intended to approximate the old pre-refactor `behavior_final.py` path by dropping random effects while keeping the same preprocessing, lag grid, the per-anchor `M_0`-`M_4` sequence, and the pooled omnibus anchor-interaction model

The active behavioral target now uses one shared FPP-selected lag chosen from `M_3`, with the lag-selection rule controlled by `behavior.lag_selection_criterion` (`bic` or `log_likelihood`).

The `glm` backend is a compatibility/comparison mode, not a claim of exact historical restoration. Cluster-robust covariance is not yet matched to the old workflow in the active pipeline, so `glm` currently uses model-based standard errors even if cluster-robust output is requested in config.

Supported active workflow areas in this cleanup phase:

- preprocessing
- events
- acoustic features
- epochs
- behavior hazard
- pending induced/source/TRF surfaces that still participate in the current Snakemake workflow

Deprecated HMM, GLHMM, TDE-HMM, Renyi, neural-hazard, behavior-final, hazard-behavior, latency-regime, and Stan/GLMM export workflows have been removed from the active repository surface.
