import numpy as np
from scipy.signal import butter, filtfilt
import os

def zscore_normalize(data_dict):
    out = {}
    for label, arr in data_dict.items():
        mean = arr.mean(axis=-1, keepdims=True)
        std = arr.std(axis=-1, keepdims=True)
        out[label] = (arr - mean) / std
    return out

def bandpass_filter(arr, low=1, high=40, fs=250):
    nyq = fs / 2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    filtered = np.zeros_like(arr)
    for trials in range(arr.shape[0]):
        for ch in range(arr.shape[1]):
            filtered[trials, ch, :] = filtfilt(b, a, arr[trials, ch, :])
    return filtered

def epoch_data(arr, window_size=500, step=250):
    epochs = []
    _, channels, length = arr.shape
    for t in range(0, length - window_size, step):
        epoch = arr[:, :, t:t+window_size]
        epochs.append(epoch)
    return np.concatenate(epochs, axis=0)

if __name__ == "__main__":
    data = np.load("data/dummy.npy", allow_pickle=True).item()
    norm = zscore_normalize(data)
    filtered = {label:bandpass_filter(arr) for label, arr in norm.items()}
    epoched = {label: epoch_data(arr) for label, arr in filtered.items()}

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/preprocessed.npy", epoched)
    print("saved preprocessed data to data/processed/preprocessed.npy")