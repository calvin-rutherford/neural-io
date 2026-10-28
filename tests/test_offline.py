from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.offline.demo import run
from src.offline.intents import DryRunBoundary
from src.offline.models import BandpowerClassifier, CNNClassifier
from src.offline.pipeline import Preprocessing, bandpower, replay, split_recordings
from src.offline.signals import Recording, SignalMetadata, load_recording, save_recording, synthetic_recordings


class SignalTests(unittest.TestCase):
    def setUp(self):
        self.recording = next(synthetic_recordings())

    def test_schema_rejects_invalid_metadata_and_samples(self):
        for changes in ({"sample_rate_hz": 0}, {"sample_rate_hz": float("nan")},
                        {"sample_rate_hz": "128"},
                        {"channels": ("X", "X")}, {"units": ("counts", "uV")},
                        {"units": ("uV",)}, {"split_group": ""}, {"source": "EEG"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.recording.metadata, **changes)
        for values in (np.zeros((3, 100)), np.full((2, 100), np.nan), np.empty((2, 0)), np.ones((2, 8), dtype=complex)):
            with self.assertRaises(ValueError):
                Recording(self.recording.metadata, values)

    def test_units_and_non_pickle_roundtrip(self):
        metadata = replace(self.recording.metadata, units=("V", "mV"))
        converted = Recording(metadata, self.recording.samples / np.array([[1e6], [1e3]]))
        np.testing.assert_allclose(converted.microvolts(), self.recording.samples)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.npz"
            save_recording(path, converted)
            restored = load_recording(path)
            self.assertEqual(restored.metadata, metadata)
            np.testing.assert_array_equal(restored.samples, converted.samples)
            np.savez(path, samples=np.array([{}], dtype=object), metadata=np.array("{}"))
            with self.assertRaises((ValueError, KeyError)):
                load_recording(path)

    def test_causal_chunks_match_whole_recording_and_include_final_window(self):
        expected = list(replay(self.recording, chunk_samples=768))
        self.assertEqual([w.start_sample for w in expected], [128, 256, 384, 512])
        for chunk in (1, 37, 128, 256, 1000):
            actual = list(replay(self.recording, chunk_samples=chunk))
            self.assertEqual(len(actual), 4)
            for left, right in zip(actual, expected):
                np.testing.assert_allclose(left.samples_uv, right.samples_uv, rtol=1e-12, atol=1e-12)
        prefix = Recording(self.recording.metadata, self.recording.samples[:, :384])
        np.testing.assert_array_equal(list(replay(prefix))[0].samples_uv, expected[0].samples_uv)

    def test_short_flat_and_invalid_filter_inputs(self):
        self.assertEqual(list(replay(Recording(self.recording.metadata, np.ones((2, 100))))), [])
        flat = list(replay(Recording(self.recording.metadata, np.ones((2, 768)))))
        self.assertTrue(all(not window.quality_ok for window in flat))
        for config in (Preprocessing(high_hz=64), Preprocessing(step_samples=0),
                       Preprocessing(window_samples=2), Preprocessing(warmup_samples=-1)):
            with self.assertRaises(ValueError):
                list(replay(self.recording, config))
        with self.assertRaises(ValueError):
            list(replay(self.recording, chunk_samples=0))

    def test_features_retain_designed_frequency_difference(self):
        low, high = list(synthetic_recordings())[:2]
        low_features, high_features = [bandpower(next(replay(item))).reshape(2, 4) for item in (low, high)]
        self.assertTrue((low_features[:, 1] > low_features[:, 2]).all())
        self.assertTrue((high_features[:, 2] > high_features[:, 1]).all())

    def test_split_groups_never_overlap_and_partition_all_recordings(self):
        recordings = list(synthetic_recordings())
        split = split_recordings(recordings)
        groups = [{item.metadata.split_group for item in subset} for subset in split.values()]
        self.assertEqual(sum(map(len, split.values())), len(recordings))
        for index, group in enumerate(groups):
            for other in groups[index + 1:]:
                self.assertFalse(group & other)
        self.assertEqual([r.metadata.recording_id for r in split["test"]],
                         [r.metadata.recording_id for r in split_recordings(recordings)["test"]])
        with self.assertRaises(ValueError):
            split_recordings(recordings + [recordings[0]])
        with self.assertRaises(ValueError):
            split_recordings(recordings[:4])


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.window = next(replay(next(synthetic_recordings())))
        self.classes = ["tone_low", "tone_high"]

    def test_default_no_op_and_explicit_mock_confirmation(self):
        event = DryRunBoundary().predict(self.window, self.classes, [0.99, 0.01])
        self.assertEqual(event.status, "no_op_unmapped")
        boundary = DryRunBoundary({"tone_low": "preview_status"})
        event = boundary.predict(self.window, self.classes, [0.99, 0.01])
        self.assertEqual(event.status, "pending_human_confirmation")
        self.assertTrue(event.requires_human_confirmation)
        self.assertFalse(event.dispatched)
        self.assertEqual(json.loads(boundary.serialize(event))["candidate_intent"], "preview_status")
        with self.assertRaises(ValueError):
            DryRunBoundary({"tone_low": "deploy"})

    def test_abstention_invalid_low_confidence_tie_and_flat_signal(self):
        boundary = DryRunBoundary({"tone_low": "preview_status"})
        for values in ([np.nan, 0], [1.1, -0.1], [0.9], [0.4, 0.4], ["bad", "prediction"]):
            event = boundary.predict(self.window, self.classes, values)
            self.assertEqual(event.status, "abstained_invalid_prediction")
            self.assertIsNone(event.candidate_intent)
            json.loads(boundary.serialize(event))
        self.assertEqual(boundary.predict(self.window, self.classes, [0.6, 0.4]).status, "abstained_low_confidence")
        self.assertEqual(DryRunBoundary(threshold=0.5).predict(self.window, self.classes, [0.5, 0.5]).status, "abstained_low_confidence")
        event = boundary.predict(replace(self.window, quality_ok=False), self.classes, [0.99, 0.01])
        self.assertEqual(event.status, "abstained_signal_quality")


class EndToEndTests(unittest.TestCase):
    def test_reload_preserves_probabilities_and_training_scaler(self):
        recordings = list(synthetic_recordings())[:4]
        windows = [window for recording in recordings for window in replay(recording)]
        labels = [window.metadata.label for window in windows]
        baseline = BandpowerClassifier().fit(windows, labels)
        np.testing.assert_allclose(baseline.scaler.mean_, np.stack([bandpower(w) for w in windows]).mean(axis=0))
        cnn = CNNClassifier().fit(windows, labels, epochs=3)
        np.testing.assert_allclose(cnn.mean, np.stack([w.samples_uv for w in windows]).mean(axis=(0, 2), keepdims=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            baseline.save(path)
            classes, probabilities = BandpowerClassifier.saved_probabilities(path, windows)
            self.assertEqual(classes, baseline.classes)
            np.testing.assert_allclose(probabilities, baseline.probabilities(windows), rtol=1e-12, atol=1e-12)
            cnn.save(path)
            np.testing.assert_array_equal(CNNClassifier.load(path).probabilities(windows), cnn.probabilities(windows))
            incompatible = [
                replace(windows[0], preprocessing=Preprocessing(low_hz=2)),
                replace(windows[0], metadata=replace(windows[0].metadata, sample_rate_hz=256)),
                replace(windows[0], samples_uv=windows[0].samples_uv[:, :-1]),
            ]
            for window in incompatible:
                with self.assertRaises(ValueError):
                    cnn.probabilities([window])
            baseline.save(path)
            for window in incompatible:
                with self.assertRaises(ValueError):
                    BandpowerClassifier.saved_probabilities(path, [window])

    def test_demo_models_reload_reproducibility_and_no_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "default"
            report = run(output)
            mapped = run(Path(directory) / "mapped", mapped=True)
            self.assertEqual(report["models"], mapped["models"])
            self.assertEqual(report["recording_sha256"], mapped["recording_sha256"])
            self.assertEqual(report["source"], "synthetic")
            self.assertEqual(report["intent_policy"], "default_no_op")
            paths = sorted((output / "recordings").glob("*.npz"))
            window = next(replay(load_recording(paths[0])))
            windows = [window]
            classes, probabilities = BandpowerClassifier.saved_probabilities(output / "bandpower_logistic.npz", windows)
            self.assertEqual(len(classes), 2)
            self.assertAlmostEqual(float(probabilities.sum()), 1)
            cnn = CNNClassifier.load(output / "temporal_cnn.npz")
            self.assertEqual(cnn.probabilities(windows).shape, (1, 2))
            changed = replace(window, metadata=replace(window.metadata, channels=("CH2", "CH1")))
            with self.assertRaises(ValueError):
                cnn.probabilities([changed])
            with self.assertRaises(ValueError):
                BandpowerClassifier.saved_probabilities(output / "bandpower_logistic.npz", [changed])
            for folder, expected in ((output, "no_op_unmapped"), (Path(directory) / "mapped", "pending_human_confirmation")):
                events = [json.loads(line) for line in (folder / "intent-events.jsonl").read_text().splitlines()]
                self.assertEqual(len(events), 24)
                self.assertTrue(all(not event["dispatched"] for event in events))
                self.assertTrue(all(event["requires_human_confirmation"] for event in events))
                self.assertTrue(all(event["status"] == expected for event in events))
            with self.assertRaises(FileExistsError):
                run(output)


if __name__ == "__main__":
    unittest.main()
