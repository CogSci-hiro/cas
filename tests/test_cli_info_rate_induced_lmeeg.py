from __future__ import annotations

import subprocess
import sys


def test_induced_cli_help_works() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cas.cli.main", "induced", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "sensor-lmeeeg" in result.stdout
    assert "source-lmeeeg" in result.stdout
    assert "figures" in result.stdout
