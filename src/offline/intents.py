"""A local mock boundary. This module has no transport, execution or credentials."""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .pipeline import Window


@dataclass(frozen=True)
class IntentEvent:
    event_id: str
    recording_id: str
    start_sample: int
    end_sample: int
    source: str
    predicted_label: str | None
    confidence: float | None
    status: str
    candidate_intent: str | None
    requires_human_confirmation: bool = True
    dispatched: bool = False
    schema_version: str = "neural-io.intent.v1"


class DryRunBoundary:
    """Empty mapping is the independent default. Even mapped events cannot execute."""

    def __init__(self, mapping=None, threshold=0.9):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a probability")
        self.mapping = dict(mapping or {})
        if any(value not in {"preview_status", "preview_pause"} for value in self.mapping.values()):
            raise ValueError("Only mock preview intents are supported")
        self.threshold = threshold

    def predict(self, window: Window, classes, probabilities) -> IntentEvent:
        try:
            values = np.asarray(probabilities, dtype=float)
        except (ValueError, TypeError):
            values = np.array([])
        valid = (values.shape == (len(classes),) and len(classes) >= 2
                 and all(isinstance(label, str) and label for label in classes)
                 and len(set(classes)) == len(classes)
                 and np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
                 and np.isclose(values.sum(), 1))
        label, confidence = None, None
        status, candidate = "abstained_invalid_prediction", None
        if valid:
            winner = int(np.argmax(values))
            label, confidence = str(classes[winner]), float(values[winner])
            if not window.quality_ok:
                status = "abstained_signal_quality"
            elif confidence < self.threshold or np.count_nonzero(values == values[winner]) > 1:
                status = "abstained_low_confidence"
            elif label not in self.mapping:
                status = "no_op_unmapped"
            else:
                status, candidate = "pending_human_confirmation", self.mapping[label]
        identity = json.dumps([window.metadata.recording_id, window.start_sample,
                               window.end_sample, label, confidence, status, candidate])
        return IntentEvent(hashlib.sha256(identity.encode()).hexdigest()[:24],
                           window.metadata.recording_id, window.start_sample, window.end_sample,
                           window.metadata.source, label, confidence, status, candidate)

    @staticmethod
    def serialize(event):
        return json.dumps(asdict(event), allow_nan=False, sort_keys=True)
