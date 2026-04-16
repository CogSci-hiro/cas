"""VoxAtlas-backed acoustic feature extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile


def _ensure_local_voxatlas_on_path() -> None:
    """Expose a sibling local VoxAtlas checkout when the package is not installed."""
    project_root = Path(__file__).resolve().parents[3]
    candidate = project_root.parent / "voxatlas" / "src"
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


try:
    from voxatlas.audio.audio import Audio
    from voxatlas.features.acoustic.envelope.hilbert import HilbertEnvelope
    from voxatlas.features.acoustic.pitch.f0 import F0Extractor
    from voxatlas.features.feature_input import FeatureInput
except ModuleNotFoundError as exc:
    if exc.name != "voxatlas":
        raise
    _ensure_local_voxatlas_on_path()
    from voxatlas.audio.audio import Audio
    from voxatlas.features.acoustic.envelope.hilbert import HilbertEnvelope
    from voxatlas.features.acoustic.pitch.f0 import F0Extractor
    from voxatlas.features.feature_input import FeatureInput


@dataclass(frozen=True)
class AcousticVectorResult:
    """Container for one frame-aligned acoustic feature."""

    feature_name: str
    time_s: np.ndarray
    values: np.ndarray
    params: dict[str, float | int | str]


@dataclass(frozen=True)
class AcousticFeatureBundle:
    """Container for the full set of extracted acoustic features."""

    sampling_rate_hz: float
    envelope: AcousticVectorResult
    f0: AcousticVectorResult


def load_mono_audio(audio_path: str | Path) -> tuple[np.ndarray, float]:
    """Load a WAV file and collapse multi-channel audio to mono."""
    path = Path(audio_path)
    sampling_rate_hz, signal = wavfile.read(path)
    signal_array = np.asarray(signal, dtype=np.float32)
    if signal_array.ndim == 2:
        signal_array = signal_array.mean(axis=1)
    if signal_array.ndim != 1:
        raise ValueError(f"Expected mono or stereo waveform, got shape {signal_array.shape!r}.")
    return signal_array, float(sampling_rate_hz)


def _frame_step_from_time_axis(time_s: np.ndarray) -> float | None:
    if time_s.size < 2:
        return None
    return float(np.median(np.diff(time_s)))


def _build_feature_input(signal: np.ndarray, sampling_rate_hz: float, source_path: str | None) -> FeatureInput:
    audio = Audio(
        waveform=np.asarray(signal, dtype=np.float32),
        sample_rate=int(round(float(sampling_rate_hz))),
        path=source_path,
    )
    return FeatureInput(audio=audio, units=None, context={})


def extract_acoustic_features(
    signal: np.ndarray,
    sampling_rate_hz: float,
    config: dict,
    *,
    source_path: str | None = None,
) -> AcousticFeatureBundle:
    """Extract frame-aligned envelope and F0 using VoxAtlas."""
    acoustic_config = config.get("acoustic", config)
    if not isinstance(acoustic_config, dict):
        raise ValueError("Acoustic config must be a mapping.")

    envelope_config = acoustic_config.get("envelope", {})
    f0_config = acoustic_config.get("f0", {})
    if not isinstance(envelope_config, dict) or not isinstance(f0_config, dict):
        raise ValueError("Acoustic config must define mapping entries for 'envelope' and 'f0'.")

    feature_input = _build_feature_input(signal, sampling_rate_hz, source_path)

    envelope_params = {
        "frame_length": float(envelope_config.get("frame_length_s", 0.025)),
        "frame_step": float(envelope_config.get("frame_step_s", 0.010)),
        "smoothing": int(envelope_config.get("smoothing", 1)),
        "peak_threshold": float(envelope_config.get("peak_threshold", 0.1)),
    }
    envelope_output = HilbertEnvelope().compute(feature_input, envelope_params)

    f0_params = {
        "fmin": float(f0_config.get("fmin_hz", 75.0)),
        "fmax": float(f0_config.get("fmax_hz", 500.0)),
        "frame_length": float(f0_config.get("frame_length_s", 0.040)),
        "frame_step": float(f0_config.get("frame_step_s", 0.010)),
    }
    f0_output = F0Extractor().compute(feature_input, f0_params)

    return AcousticFeatureBundle(
        sampling_rate_hz=float(sampling_rate_hz),
        envelope=AcousticVectorResult(
            feature_name=str(envelope_config.get("extractor", envelope_output.feature)),
            time_s=np.asarray(envelope_output.time, dtype=np.float32),
            values=np.asarray(envelope_output.values, dtype=np.float32),
            params=envelope_params,
        ),
        f0=AcousticVectorResult(
            feature_name=str(f0_config.get("extractor", f0_output.feature)),
            time_s=np.asarray(f0_output.time, dtype=np.float32),
            values=np.asarray(f0_output.values, dtype=np.float32),
            params=f0_params,
        ),
    )


def build_feature_summary(
    *,
    input_path: str | Path,
    output_path: str | Path,
    sampling_rate_hz: float,
    result: AcousticVectorResult,
) -> dict[str, object]:
    """Build a compact JSON-serializable summary for one extracted feature."""
    time_s = np.asarray(result.time_s, dtype=np.float32)
    values = np.asarray(result.values, dtype=np.float32)
    finite_mask = np.isfinite(values)
    return {
        "input": str(Path(input_path)),
        "output": str(Path(output_path)),
        "feature": result.feature_name,
        "sampling_rate_hz": float(sampling_rate_hz),
        "n_frames": int(values.shape[0]),
        "frame_step_s": _frame_step_from_time_axis(time_s),
        "time_start_s": float(time_s[0]) if time_s.size else None,
        "time_end_s": float(time_s[-1]) if time_s.size else None,
        "n_finite_frames": int(np.count_nonzero(finite_mask)),
        "params": result.params,
    }
