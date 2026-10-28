"""Two reproducible baselines with the same probability interface."""

import json
from dataclasses import asdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from .pipeline import bandpower


def signature(window):
    return {"sample_rate_hz": window.metadata.sample_rate_hz,
            "channels": list(window.metadata.channels), "shape": list(window.samples_uv.shape),
            "preprocessing": asdict(window.preprocessing)}


def validate_windows(windows, expected):
    if not windows or any(signature(window) != expected for window in windows):
        raise ValueError("Model requires the trained sample rate, channels, shape and preprocessing")


class BandpowerClassifier:
    def fit(self, windows, labels, seed=42):
        self.signature = signature(windows[0])
        validate_windows(windows, self.signature)
        self.scaler = StandardScaler().fit(np.stack([bandpower(w) for w in windows]))
        self.model = LogisticRegression(random_state=seed, max_iter=1000).fit(
            self.scaler.transform(np.stack([bandpower(w) for w in windows])), labels)
        self.classes = self.model.classes_.tolist()
        return self

    def probabilities(self, windows):
        validate_windows(windows, self.signature)
        return self.model.predict_proba(self.scaler.transform(np.stack([bandpower(w) for w in windows])))

    def save(self, path):
        # Numeric parameters only; no executable pickle artifacts.
        np.savez_compressed(path, classes=np.array(self.classes), mean=self.scaler.mean_,
                            scale=self.scaler.scale_, coefficients=self.model.coef_,
                            intercept=self.model.intercept_, signature=np.array(json.dumps(self.signature)))

    @staticmethod
    def saved_probabilities(path, windows):
        with np.load(path, allow_pickle=False) as model:
            validate_windows(windows, json.loads(str(model["signature"].item())))
            features = (np.stack([bandpower(w) for w in windows]) - model["mean"]) / model["scale"]
            logits = features @ model["coefficients"].T + model["intercept"]
            if logits.shape[1] == 1:
                logits = np.concatenate((np.zeros_like(logits), logits), axis=1)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            return model["classes"].tolist(), probabilities / probabilities.sum(axis=1, keepdims=True)


class TemporalCNN(nn.Module):
    """Small 1-D temporal CNN, not EEGNet and not a neuroscience-validated model."""

    def __init__(self, channels, classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, 8, kernel_size=17, padding=8), nn.ELU(),
            nn.AvgPool1d(4), nn.Conv1d(8, 16, kernel_size=7, padding=3), nn.ELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, classes))

    def forward(self, values):
        return self.layers(values)


class CNNClassifier:
    def fit(self, windows, labels, seed=42, epochs=40):
        self.signature = signature(windows[0])
        validate_windows(windows, self.signature)
        # Fixed CPU seed, thread count and deterministic ops for this tiny reference run.
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.classes = sorted(set(labels))
        values = np.stack([w.samples_uv for w in windows])
        self.mean = values.mean(axis=(0, 2), keepdims=True)
        self.scale = np.maximum(values.std(axis=(0, 2), keepdims=True), 1e-9)
        inputs = torch.tensor((values - self.mean) / self.scale, dtype=torch.float32)
        targets = torch.tensor([self.classes.index(label) for label in labels])
        self.model = TemporalCNN(values.shape[1], len(self.classes))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(self.model(inputs), targets)
            loss.backward()
            optimizer.step()
        self.model.eval()
        return self

    def probabilities(self, windows):
        validate_windows(windows, self.signature)
        values = np.stack([w.samples_uv for w in windows])
        inputs = torch.tensor((values - self.mean) / self.scale, dtype=torch.float32)
        with torch.no_grad():
            return self.model(inputs).softmax(dim=1).numpy()

    def save(self, path):
        arrays = {key: value.numpy() for key, value in self.model.state_dict().items()}
        np.savez_compressed(path, classes=np.array(self.classes), mean=self.mean, scale=self.scale,
                            signature=np.array(json.dumps(self.signature)), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            result = cls()
            result.signature = json.loads(str(arrays["signature"].item()))
            result.classes = arrays["classes"].tolist()
            result.mean, result.scale = arrays["mean"], arrays["scale"]
            result.model = TemporalCNN(result.mean.shape[1], len(result.classes))
            result.model.load_state_dict({key: torch.from_numpy(arrays[key].copy())
                                         for key in result.model.state_dict()})
            result.model.eval()
            return result
