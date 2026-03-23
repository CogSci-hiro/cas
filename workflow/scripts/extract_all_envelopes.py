from __future__ import annotations

import subprocess
from pathlib import Path


out_dir = Path("/Volumes/work-4T/speech-rate-testing")

for audio_path in snakemake.input:
    audio_path = Path(str(audio_path))
    subject_dir = audio_path.parents[1].name
    stem = audio_path.stem
    envelope_path = out_dir / "features" / "envelope" / subject_dir / f"{stem}_envelope.npy"
    envelope_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "python",
            "-m",
            "cas.cli.main",
            "envelope",
            "--input",
            str(audio_path),
            "--output",
            str(envelope_path),
        ],
        check=True,
        env={
            **__import__("os").environ,
            "PYTHONPATH": "/Users/hiro/Projects/active/cas/src",
        },
    )

Path(str(snakemake.output[0])).touch()
