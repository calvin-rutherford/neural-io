# Neural IO — offline signal replay and inference

A working, hardware-independent pipeline: explicit recording metadata → causal
preprocessing → windowed features or a temporal CNN → predictions → local dry-run
intent events. The demo learns two designed synthetic tone classes. Its accuracy
measures this software path only, **not brain decoding, human EEG/EMG performance,
or recognition of user intent**.

## Run the offline demo

The pinned environment targets **Python 3.12 on Linux x86_64, CPU only**.
`requirements.in` lists direct dependencies; `requirements.txt` pins the resolved
versions. No additional dependencies are needed for the offline path.

```sh
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --torch-backend cpu requirements.txt
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m src.offline.demo --output runs/offline-demo --seed 42
python -m src.offline.demo --output runs/mock-preview --seed 42 --mock-intents
```

Run from the repository root. Each output directory must be new; the command
refuses to overwrite an earlier run. The demo downloads nothing and uses no
device, service, credential or network connection. Outputs are ignored by Git:

- `recordings/*.npz`: 24 synthetic recordings with JSON metadata and numeric samples.
- `bandpower_logistic.npz`, `temporal_cnn.npz`: numeric parameters, training-only
  scaling and input compatibility signatures; no model pickle.
- `evaluation.json`: seeds, versions, preprocessing, group assignments, hashes,
  class order, validation/test metrics and confidence limitations.
- `intent-events.jsonl`: 24 held-out window predictions passed to a local mock.

The default emits `no_op_unmapped` for confident predictions. `--mock-intents`
maps tone labels to `preview_status` / `preview_pause` candidates; they remain
`pending_human_confirmation`, with `dispatched: false`. These mappings are a
demonstration, not evidence that a person requested either action.

## Implemented capability

The signal schema specifies voltage units per named channel, sample rate,
recording identity, provenance and an independent split group. Numeric NPZ files
load with `allow_pickle=False`. Replay validates samples, converts to microvolts,
carries causal filter state across arbitrary chunks, detects flat channels and
includes the final complete window. Incomplete tails are dropped. Replay runs as
fast as the CPU permits; there is no acquisition or wall-clock scheduler.

The feature baseline uses log mean spectral density in four frequency bands with
logistic regression. The temporal CNN has two actual `Conv1d` layers, ELU
activations, pooling and a linear head. It is a compact reference model, not
EEGNet. Both expose probabilities and numeric model persistence. Inference
rejects incompatible sample rates, channel order, window shapes or preprocessing.

The demo partitions 12 synthetic session groups into six training, three
validation and three test groups **before** preprocessing/windowing. Scaling
uses training windows only. Seeds and fixed hyperparameters are recorded; test
scores select neither the model nor the threshold. Group separation prevents
cross-partition windows from one recording, but synthetic groups do not establish
generalization across people or real sessions.

See [the detailed contracts](docs/offline-foundation.md) for recorded-file replay,
timing, models, evaluation and the inspected Magistrate seam.

## Limits and next decisions

The fixed 0.9 confidence threshold is uncalibrated. High confidence does not prove
intent or detect unfamiliar data. Flat-channel detection is not an EEG/EMG
artifact-rejection system. Filter/feature bands are engineering defaults, not a
validated protocol. There is no live dispatch or confirmation executor, EEGNet,
wavelet processing, device adapter, imagined-speech decoding or human-recording
training in this increment.

Next choose the available device or recording format and a concrete control task.
Human-signal evaluation needs consented provenance, suitable participant/session
splits, labels, artifact handling, calibration and independent acceptance criteria.
Future agent control must preserve explicit confirmation and destination authority.

## Legacy LSTM experiment

The earlier commands remain available separately:

```sh
python -m src.utils.generate_dummy
python -m src.preprocessing.load_data
python -m src.preprocessing.prepare
python -m src.training.train_lstm
```

They overwrite historical tracked arrays/model in `data/` and `models/`; use a
scratch checkout. This legacy path uses unseeded offset noise, normalization that
removes much of the class difference, a loop omitting the final valid window and
a split after overlapping windows. It lacks a flat-channel guard and its trusted
`.npy` loaders enable pickle. Saved LSTM weights have no validated provenance.
The offline demo neither loads nor modifies these artifacts; its safeguards do
not apply to the legacy commands.
