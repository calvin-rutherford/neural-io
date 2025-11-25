import os
import numpy as np

CLASSES = ["up", "down", "left", "right"]

def generate_dummy_eeg(n_samples=50, n_channels=4, length=1000):
    data = {}
    for i, label in enumerate(CLASSES):
        waves = np.random.randn(n_samples, n_channels, length)
        waves = waves +  i * 0.5
        data[label] = waves
    return data

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    d = generate_dummy_eeg()
    np.save("data/dummy.npy", d)

    print("Saved dummy data to data/dummy.npy")