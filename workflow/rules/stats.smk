rule write_manifest:
    input:
        # all confirmatory outputs
        f"{OUT_DIR}/models/lagged_logistic/results.parquet",
        f"{OUT_DIR}/models/hazard/results.parquet",
        f"{OUT_DIR}/models/mediation/results.parquet",
        f"{OUT_DIR}/stats/n400/results.parquet"
    output:
        f"{OUT_DIR}/_meta/run_complete.txt"
    shell:
        """
        python -m cas.meta.write_manifest \
            --out-dir {OUT_DIR}
        touch {output}
        """