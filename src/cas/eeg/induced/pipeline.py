from __future__ import annotations

import subprocess


def run_induced_workflow_target(target: str) -> int:
    """Execute one induced Snakemake target through the canonical grouped CLI."""
    result = subprocess.run(["snakemake", target], check=False)
    return int(result.returncode)
