"""Causal, chunk-independent replay and leakage-aware partitions."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, welch

from .signals import Recording, SignalMetadata


@dataclass(frozen=True)
class Preprocessing:
    low_hz: float = 1
    high_hz: float = 40
    window_samples: int = 256
    step_samples: int = 128
    warmup_samples: int = 128

    def validate(self, sample_rate_hz):
        if not 0 < self.low_hz < self.high_hz < sample_rate_hz / 2:
            raise ValueError("Filter requires 0 < low < high < Nyquist")
        values = (self.window_samples, self.step_samples, self.warmup_samples)
        if any(type(v) is not int for v in values):
            raise ValueError("Window, step and warmup must be integers")
        if self.window_samples < 16 or not 0 < self.step_samples <= self.window_samples or self.warmup_samples < 0:
            raise ValueError("Invalid window, step or warmup")


@dataclass(frozen=True)
class Window:
    metadata: SignalMetadata
    start_sample: int
    end_sample: int  # exclusive; prediction available only after this sample arrives
    samples_uv: np.ndarray
    quality_ok: bool
    preprocessing: Preprocessing = Preprocessing()


class ReplayProcessor:
    """One processor per continuous recording. No lookahead, no dropped tail window."""

    def __init__(self, metadata: SignalMetadata, config=Preprocessing()):
        config.validate(metadata.sample_rate_hz)
        self.metadata, self.config = metadata, config
        self.sos = butter(4, [config.low_hz, config.high_hz], btype="bandpass",
                          fs=metadata.sample_rate_hz, output="sos")
        self.state = np.zeros((len(self.sos), len(metadata.channels), 2))
        self.buffer = np.empty((len(metadata.channels), 0))
        self.raw_buffer = self.buffer.copy()
        self.offset = 0
        self.next_start = config.warmup_samples

    def push(self, samples) -> list[Window]:
        raw = Recording(self.metadata, samples).microvolts()
        filtered, self.state = sosfilt(self.sos, raw, axis=-1, zi=self.state)
        if not np.isfinite(filtered).all():
            raise ValueError("Filter output is not finite")
        self.buffer = np.concatenate((self.buffer, filtered), axis=-1)
        self.raw_buffer = np.concatenate((self.raw_buffer, raw), axis=-1)
        windows = []
        while self.next_start + self.config.window_samples <= self.offset + self.buffer.shape[1]:
            local = self.next_start - self.offset
            end = local + self.config.window_samples
            values = self.buffer[:, local:end].copy()
            # Detect flat channels using raw samples: filter startup can turn DC into a transient.
            quality = bool(np.all(np.std(self.raw_buffer[:, local:end], axis=-1) > 1e-9))
            windows.append(Window(self.metadata, self.next_start,
                                  self.next_start + self.config.window_samples, values, quality, self.config))
            self.next_start += self.config.step_samples
        discard = min(self.next_start - self.offset, self.buffer.shape[1])
        self.buffer = self.buffer[:, discard:]
        self.raw_buffer = self.raw_buffer[:, discard:]
        self.offset += discard
        return windows


def replay(recording: Recording, config=Preprocessing(), chunk_samples=37):
    if type(chunk_samples) is not int or chunk_samples <= 0:
        raise ValueError("chunk_samples must be a positive integer")
    processor = ReplayProcessor(recording.metadata, config)
    for start in range(0, recording.samples.shape[1], chunk_samples):
        yield from processor.push(recording.samples[:, start:start + chunk_samples])


BANDS = ((4, 8), (8, 16), (16, 30), (30, 40))


def bandpower(window: Window):
    """Log mean spectral density in fixed half-open bands, per channel."""
    fs = window.metadata.sample_rate_hz
    if fs / 2 <= BANDS[-1][1]:
        raise ValueError("Feature bands require Nyquist > 40 Hz")
    frequencies, power = welch(window.samples_uv, fs=fs, nperseg=window.samples_uv.shape[1], axis=-1)
    features = []
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        if not mask.any():
            raise ValueError("Window too short to resolve feature band")
        features.append(np.log10(np.maximum(power[:, mask].mean(axis=-1), 1e-12)))
    return np.stack(features, axis=-1).ravel()


def split_recordings(recordings, seed=42):
    """Partition independent groups BEFORE preprocessing/windowing. No fitted transforms."""
    if len({r.metadata.recording_id for r in recordings}) != len(recordings):
        raise ValueError("Recording ids must be unique")
    groups = sorted({r.metadata.split_group for r in recordings})
    if len(groups) < 6:
        raise ValueError("At least six independent split groups required")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    size = max(1, len(groups) // 4)
    assignments = dict(train=groups[2 * size:], validation=groups[:size], test=groups[size:2 * size])
    splits = {name: [r for r in recordings if r.metadata.split_group in selected]
              for name, selected in assignments.items()}
    labels = {r.metadata.label for r in recordings}
    if None in labels or len(labels) < 2:
        raise ValueError("At least two explicit labels required for evaluation")
    if any({r.metadata.label for r in subset} != labels for subset in splits.values()):
        raise ValueError("Every partition must contain all labels; revise grouping/data")
    return splits
