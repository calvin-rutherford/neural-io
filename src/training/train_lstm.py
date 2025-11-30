import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

from src.models.lstm_baseline import LSTMBaseline

LABELS = ["up", "down", "left", "right"]

def load_preprocessed(path="data/processed/preprocessed.npy"):
    """
    Load epoched data and build X, y arrays.

    Expected format:
        dict[label] -> array of shape (n_epochs_label, channels, window)
    Returns:
        X: (N, seq_len, input_dim)  # (N, time, channels)
        y: (N,) int labels
    """
    d = np.load(path, allow_pickle=True).item()

    X_list = []
    y_list = []

    for idx, label in enumerate(LABELS):
        arr = d[label]  # (n_epochs, channels, window)
        # Move time dimension to the middle for LSTM: (n_epochs, window, channels)
        arr = np.transpose(arr, (0, 2, 1))
        X_list.append(arr)
        y_list.append(np.full((arr.shape[0],), idx, dtype=np.int64))

    X = np.concatenate(X_list, axis=0)  # (N, seq_len, channels)
    y = np.concatenate(y_list, axis=0)  # (N,)

    return X, y

def make_loaders(X, y, batch_size=32, test_size=0.2, seed=42):
    """
    Split into train/val and wrap in DataLoaders.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)
    y_val   = torch.tensor(y_val,   dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train.shape[2]

def train_model(train_loader, val_loader, input_dim, num_classes=4, epochs=10, lr=1e-3, device="cpu"):
    model = LSTMBaseline(input_dim=input_dim, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    return model

if __name__ == "__main__":
    X, y = load_preprocessed()
    print("Data shape:", X.shape, y.shape)

    train_loader, val_loader, input_dim = make_loaders(X, y)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = train_model(train_loader, val_loader, input_dim=input_dim, device=device)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_baseline.pt")
    print("Saved model to models/lstm_baseline.pt")
