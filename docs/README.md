## Active Surfaces

The active behavioral workflow is the `behavior_hazard_*` pipeline rooted at `config/behavior/hazard.yaml` and `workflow/rules/behavior.smk`.

Supported active workflow areas in this cleanup phase:

- preprocessing
- events
- acoustic features
- epochs
- behavior hazard
- pending induced/source/TRF surfaces that still participate in the current Snakemake workflow

Deprecated HMM, GLHMM, TDE-HMM, Renyi, neural-hazard, behavior-final, hazard-behavior, latency-regime, and Stan/GLMM export workflows have been removed from the active repository surface.
