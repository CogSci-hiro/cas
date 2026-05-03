from __future__ import annotations

import subprocess
import sys


def test_info_rate_induced_lmeeg_cli_help_works() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cas.cli.main", "info-rate-induced-lmeeg", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "info-rate-induced-lmeeg" in result.stdout
    assert "--config" in result.stdout
