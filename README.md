# Neural IO — synthetic sequence-classification baseline

An exploratory PyTorch LSTM pipeline using generated arrays with EEG-like shapes.
The current repository does **not** demonstrate imagined-speech or thought-to-text
decoding, live EEG acquisition, CNN/EEGNet models, wavelet processing, or a mobile
accessibility application.

## What the code does

1. `src/utils/generate_dummy.py` generates four labels (`up`, `down`, `left`,
   `right`), each with 50 random trials, four channels and 1,000 samples. Noise
   receives a label-dependent constant offset (`label_index * 0.5`). This is
   synthetic data, not human EEG or evidence of a neural signal.
2. `src/preprocessing/prepare.py` normalizes each trial/channel, applies a fourth-
   order 1–40 Hz Butterworth filter assuming 250 Hz sampling, and creates windows
   of 500 samples with step 250. Its current loop excludes the final valid window:
   the default input produces two overlapping windows per trial (400 total).
3. `src/training/train_lstm.py` transposes to time × channels, makes an 80/20
   stratified split, trains a one-layer, 64-unit LSTM plus linear classifier for
   ten epochs, prints train/validation statistics, and saves a state dictionary.

## Reproduce the environment and run

The pinned environment targets **Python 3.12 on Linux x86_64, CPU only**.
`requirements.in` records direct dependencies; `requirements.txt` pins the resolved
transitive versions. Setup was resolved with uv 0.12.15. Use uv's CPU backend so
it retrieves the pinned `+cpu` Torch wheel from the official PyTorch index:

```sh
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --torch-backend cpu requirements.txt
source .venv/bin/activate
python -m src.utils.generate_dummy
python -m src.preprocessing.load_data
python -m src.preprocessing.prepare
python -m src.training.train_lstm
```

Run commands from the repository root. They write `data/dummy.npy`,
`data/processed/preprocessed.npy` and `models/lstm_baseline.pt`. All three already
exist in repository history; regeneration overwrites the local copies. Use a
scratch checkout or copy them elsewhere if you need the existing artifacts.
Generated output is ignored for future additions; existing tracked artifacts
remain tracked in this documentation change. Other operating systems and GPU
builds have not been qualified by this environment.

## Interpretation and limitations

- Per-channel z-score normalization largely removes the generator's constant
  label offsets. The resulting random classes do not establish real neural
  discrimination. Any printed accuracy describes this artificial setup only.
- Overlapping windows are split *after* epoching, so related windows from a trial
  can occur in both training and validation. There is no held-out participant,
  recording session or independent test dataset.
- Only the split uses a fixed seed (42). NumPy generation and Torch initialization
  are unseeded, so a pinned environment does not imply identical weights or scores.
- Normalization has no zero-variance guard, and filtering/epoching assumes the
  expected shapes and sampling rate. Arbitrary recordings need validation.
- The `.npy` loader uses `allow_pickle=True`. Load only repository-generated or
  otherwise trusted files; this is not an untrusted-upload interface. The saved
  model has no committed evaluation or provenance record proving its quality.

## Possible next work

Before making real EEG claims: document a consented dataset and license, split by
trial/session/participant before windowing, control seeds, validate preprocessing,
and compare baselines on an untouched test set. CNN/EEGNet, wavelets, live devices
and speech decoding are ideas for future work, not implemented capabilities.
