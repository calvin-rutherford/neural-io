import numpy as np

def load_dummy(path="data/dummy.npy"):
    data = np.load(path, allow_pickle=True).item()
    return data

if __name__ == "__main__":
    data = load_dummy()
    print("Classes:", list(data.keys()))
    print("Shape for 'up' :", data["up"].shape)