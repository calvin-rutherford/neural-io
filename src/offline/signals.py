"""Explicit recording schema; NPZ storage never enables pickle."""

from dataclasses import asdict, dataclass
import json
from numbers import Real
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SignalMetadata:
    recording_id: str
    split_group: str
    sample_rate_hz: float
    channels: tuple[str, ...]
    units: tuple[str, ...]
    source: str  # synthetic or recorded; neither implies EEG/EMG validation
    label: str | None = None
    schema_version: str = "neural-io.signal.v1"

    def __post_init__(self):
        if self.schema_version != "neural-io.signal.v1":
            raise ValueError("Unsupported signal schema")
        if not all(isinstance(v, str) and v.strip() for v in (self.recording_id, self.split_group)):
            raise ValueError("recording_id and split_group must be nonempty")
        if not isinstance(self.sample_rate_hz, Real) or isinstance(self.sample_rate_hz, bool) or not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        if not self.channels or any(not isinstance(c, str) or not c.strip() for c in self.channels):
            raise ValueError("Named channels are required")
        if len(set(self.channels)) != len(self.channels) or len(self.units) != len(self.channels):
            raise ValueError("Channels must be unique and have one unit each")
        if any(u not in {"V", "mV", "uV"} for u in self.units):
            raise ValueError("Supported voltage units: V, mV, uV")
        if self.source not in {"synthetic", "recorded"}:
            raise ValueError("source must be synthetic or recorded")
        if self.label is not None and (not isinstance(self.label, str) or not self.label.strip()):
            raise ValueError("label must be nonempty text or null")


@dataclass(frozen=True)
class Recording:
    metadata: SignalMetadata
    samples: np.ndarray  # channels x time, uniformly sampled from relative t=0

    def __post_init__(self):
        arr = np.asarray(self.samples)
        if arr.ndim != 2 or arr.shape[0] != len(self.metadata.channels) or arr.shape[1] == 0:
            raise ValueError("samples must be nonempty channels x time")
        if arr.dtype.kind not in "fiu" or not np.isfinite(arr).all():
            raise ValueError("samples must be finite real numbers")
        object.__setattr__(self, "samples", arr)

    def microvolts(self):
        factors = np.array([{"V": 1e6, "mV": 1e3, "uV": 1}[u] for u in self.metadata.units])
        with np.errstate(over="ignore"):
            result = np.asarray(self.samples, dtype=np.float64) * factors[:, None]
        if not np.isfinite(result).all():
            raise ValueError("Voltage conversion overflow")
        return result


def save_recording(path: str | Path, recording: Recording):
    np.savez_compressed(path, samples=recording.samples,
                        metadata=np.array(json.dumps(asdict(recording.metadata))))


def load_recording(path: str | Path) -> Recording:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"samples", "metadata"}:
            raise ValueError("Expected exactly samples and metadata")
        metadata = json.loads(str(archive["metadata"].item()))
        metadata["channels"] = tuple(metadata["channels"])
        metadata["units"] = tuple(metadata["units"])
        return Recording(SignalMetadata(**metadata), archive["samples"].copy())


def synthetic_recordings(seed=42, groups=12):
    """Two designed frequency classes, NOT simulated human EEG or intent."""
    rng = np.random.default_rng(seed)
    for group in range(groups):
        for label, frequency in (("tone_low", 10), ("tone_high", 22)):
            time = np.arange(768) / 128
            phase = rng.uniform(0, 2 * np.pi, (2, 1))
            samples = 15 * np.sin(2 * np.pi * frequency * time + phase)
            samples += rng.normal(0, 2, samples.shape)
            yield Recording(SignalMetadata(
                recording_id=f"synthetic-{group:02d}-{label}", split_group=f"session-{group:02d}",
                sample_rate_hz=128, channels=("CH1", "CH2"), units=("uV", "uV"),
                source="synthetic", label=label), samples)
