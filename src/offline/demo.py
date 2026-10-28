"""Run a reproducible synthetic replay-to-prediction-to-dry-run demonstration."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from .intents import DryRunBoundary
from .models import BandpowerClassifier, CNNClassifier
from .pipeline import Preprocessing, replay, split_recordings
from .signals import load_recording, save_recording, synthetic_recordings


def evaluate(model, windows, threshold):
    probabilities = model.probabilities(windows)
    actual = [w.metadata.label for w in windows]
    predicted = [model.classes[i] for i in probabilities.argmax(axis=1)]
    accepted = probabilities.max(axis=1) >= threshold
    return {
        "windows": len(windows), "classes": model.classes,
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=model.classes).tolist(),
        "threshold_coverage": float(accepted.mean()),
        "accepted_accuracy": accuracy_score(np.array(actual)[accepted], np.array(predicted)[accepted]) if accepted.any() else None,
    }


def run(output: Path, seed=42, mapped=False):
    # Refuse reuse so a prior run's artifacts cannot be mistaken for this run's evidence.
    output.mkdir(parents=True, exist_ok=False)
    recordings_dir = output / "recordings"
    recordings_dir.mkdir()
    paths = []
    for recording in synthetic_recordings(seed):
        path = recordings_dir / f"{recording.metadata.recording_id}.npz"
        save_recording(path, recording)
        paths.append(path)
    recordings = [load_recording(path) for path in paths]
    config = Preprocessing()
    splits = split_recordings(recordings, seed)
    windows = {name: [window for recording in subset for window in replay(recording, config)
                      if window.quality_ok] for name, subset in splits.items()}
    train = windows["train"]
    labels = [window.metadata.label for window in train]
    threshold = 0.9  # Fixed demo policy; not calibrated or selected using the test set.
    models = {"bandpower_logistic": BandpowerClassifier().fit(train, labels, seed),
              "temporal_cnn": CNNClassifier().fit(train, labels, seed)}
    report = {
        "schema_version": "neural-io.evaluation.v1", "source": "synthetic",
        "claim": "Software demonstration only; no human signal or intent validation",
        "seed": seed, "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "torch")},
        "preprocessing": asdict(config), "threshold": threshold,
        "confidence_calibrated": False,
        "split_unit": "split_group (synthetic session)",
        "splits": {name: {"groups": sorted({r.metadata.split_group for r in subset}),
                          "recordings": [r.metadata.recording_id for r in subset]}
                   for name, subset in splits.items()},
        "recording_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "models": {},
    }
    for name, model in models.items():
        path = output / f"{name}.npz"
        model.save(path)
        report["models"][name] = {
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "validation": evaluate(model, windows["validation"], threshold),
            "test": evaluate(model, windows["test"], threshold),
        }
    # Fixed reference model, not a winner selected on held-out test scores.
    model = models["bandpower_logistic"]
    boundary = DryRunBoundary({"tone_low": "preview_status", "tone_high": "preview_pause"} if mapped else {}, threshold)
    events_path = output / "intent-events.jsonl"
    with events_path.open("w") as stream:
        # Replay fresh chunks, producing one prediction as soon as each window is available.
        for recording in splits["test"]:
            for window in replay(recording, config):
                probabilities = model.probabilities([window])[0]
                stream.write(boundary.serialize(boundary.predict(window, model.classes, probabilities)) + "\n")
    report["intent_policy"] = "mock_preview_mapping" if mapped else "default_no_op"
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/offline-demo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock-intents", action="store_true", help="Map synthetic tones to local preview events; never dispatch")
    args = parser.parse_args()
    report = run(args.output, args.seed, args.mock_intents)
    print(json.dumps({"output": str(args.output), "source": report["source"],
                      "intent_policy": report["intent_policy"],
                      "test": {name: item["test"] for name, item in report["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
