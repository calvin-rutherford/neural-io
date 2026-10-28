# Offline foundation contracts

## Recording input: `neural-io.signal.v1`

One NPZ contains exactly `samples` (finite real numeric array, channels × time)
and `metadata` (scalar JSON string). Loading never enables pickle. This is a
local-file API, not a resource-limited untrusted-upload service. Example metadata:

```json
{
  "schema_version": "neural-io.signal.v1",
  "recording_id": "synthetic-00-tone_low",
  "split_group": "session-00",
  "sample_rate_hz": 128,
  "channels": ["CH1", "CH2"],
  "units": ["uV", "uV"],
  "source": "synthetic",
  "label": "tone_low"
}
```

Channel names must be unique and nonempty. Each has an explicit voltage unit:
`V`, `mV` or `uV`. Device counts need conversion before this interface; gain and
electrode locations are not inferred. `source` is `synthetic` or `recorded`, not
a declaration of EEG/EMG validity. `label` can be null for inference. Supervised
splitting requires at least two labels present in every partition and unique
recording ids within the collection.

Samples are uniformly sampled and contiguous, relative to time zero; sample `i`
occurs at `i / sample_rate_hz` seconds. Irregular timestamps, gaps, dropped samples
and rate changes are unsupported. Start a new processor at a discontinuity. No
personal identity or medical interpretation is carried by this schema.

`split_group` is the caller's independence boundary. Put trials/fragments from
one continuous recording together. For a participant-held-out study, group all
sessions from each participant together. The code cannot infer these relations
or detect duplicated signals under different ids. Session grouping alone does
not support a participant-held-out claim. `split_recordings` requires at least
six groups and rejects partitions missing classes; it never falls back to
splitting windows.

Recorded-file replay uses the same API:

```python
from src.offline.signals import load_recording
from src.offline.pipeline import Preprocessing, replay

recording = load_recording("path/to/local-recording.npz")
for window in replay(recording, Preprocessing(), chunk_samples=37):
    print(window.start_sample, window.end_sample, window.quality_ok)
```

This example only preprocesses local data. The demo trains exclusively on
generated tones. There is no real-recording training CLI, dataset downloader,
device adapter or automatic adaptation to human signals. Matching array shapes
do not make a synthetic model scientifically applicable.

## Preprocessing and timing

Defaults: fourth-order Butterworth bandpass design, 1–40 Hz, second-order sections
and causal stateful filtering. The bandpass transformation produces four SOS
sections. It requires `0 < low < high < Nyquist`; no resampling occurs.
[SciPy documents the filter state semantics](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html).

Zero initial state resets once per recording. Replay skips 128 initial samples,
then emits 256-sample windows every 128 samples. This fixed warmup does not prove
all transients have settled. A prediction uses only received samples and
`end_sample` is exclusive. At 128 Hz, the first window is `[128,384)`, available
after three seconds of acquisition; later windows arrive every second. Filter
phase delay is not corrected, so event bounds are input positions, not validated
physiological onsets. No hard real-time hardware latency is established.

`ReplayProcessor.push` preserves state/overlap across chunks for one recording.
`replay` yields windows without sleeping. The final complete window is included;
incomplete tails are dropped. Retained state is bounded by window/overlap, though
each push temporarily holds its supplied chunk.

Finite samples are required. `quality_ok` requires every raw channel's window
standard deviation to exceed `1e-9 uV`, detecting flat channels even when filter
startup produces a transient. This numerical check does not detect eye/muscle
artifacts, clipping, mains interference, movement or unfamiliar input. No clinical
amplitude cutoff is inferred.

## Models, fitting and evaluation

`bandpower(window)` uses Welch spectral density on the full window and log10 of
mean density in `[4,8)`, `[8,16)`, `[16,30)` and `[30,40)` Hz, floored at `1e-12`.
Features are channel-major, four bands per channel. Despite the short function
name, these are mean spectral density, not integrated band energy. Using one
segment per window means no averaging over several shorter segments.
[SciPy documents the estimator and density units](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).

`BandpowerClassifier.fit(windows, labels, seed)` fits training-only standardization
and logistic regression (`max_iter=1000`). `CNNClassifier.fit(windows, labels,
seed, epochs=40)` fits training-only per-channel scaling, then:

```text
channels × 256 → Conv1d(channels,8,k=17,p=8) → ELU → AvgPool1d(4)
               → Conv1d(8,16,k=7,p=3) → ELU → AdaptiveAvgPool1d(1)
               → flatten → Linear(16,number_of_classes)
```

CNN fitting uses full-batch Adam (`lr=0.01`) and cross entropy. It seeds Torch,
sets one CPU thread and enables deterministic algorithms, changing process-global
Torch settings. This compact comparison is not EEGNet; no EEGNet architecture or
benchmark is claimed. Wavelets await a task/evidence motivating that transform.

Both expose `probabilities(windows)` with columns in `classes` order. NPZ artifacts
contain numeric parameters and a JSON signature: sample rate, ordered channels,
window shape and preprocessing. Conversion to microvolts precedes inference, so
alternate declared voltage units are compatible after conversion. Mismatched
signatures raise `ValueError`. Keep the evaluation report, provenance and hashes
with the model; this is not a general registry or untrusted-model interface.

```python
from src.offline.models import BandpowerClassifier, CNNClassifier

# windows must use preprocessing from the corresponding evaluation.json.
classes, probabilities = BandpowerClassifier.saved_probabilities(
    "runs/offline-demo/bandpower_logistic.npz", windows)
cnn = CNNClassifier.load("runs/offline-demo/temporal_cnn.npz")
cnn_probabilities = cnn.probabilities(windows)
```

Generation creates 12 synthetic sessions, each with two six-second, two-channel
recordings at 128 Hz: designed 10 Hz and 22 Hz tones with random phase and noise.
They are not human EEG, EMG or simulated intentions. The seed controls generation
and group shuffle. Six groups train; three validate; three test. Four windows per
recording give 48 training, 24 validation and 24 test windows. Overlapping windows
remain correlated: the test partition contains three independent synthetic groups,
not 24 independent participants or trials.

Both models fit once using fixed hyperparameters. Validation/test are evaluated
separately. The baseline is the fixed event source and 0.9 is the fixed threshold;
test accuracy selects neither. Reports include class order, confusion matrices,
accuracy, balanced accuracy, threshold coverage and accepted accuracy (null when
none pass). The synthetic demo has no quality failures; metrics do not establish
artifact rejection or human generalization. No calibration, statistical test,
confidence interval or unfamiliar-input detector is implemented. Tests check
repeat metrics/hashes in the pinned environment, not bitwise portability across
Torch versions or platforms.

## Dry-run intent: `neural-io.intent.v1`

Events record recording identity, sample bounds, source, predicted label,
confidence, status, optional candidate intent and a content-derived id. This id
supports demo inspection, not cross-model execution idempotency. Invalid vectors,
nonfinite predictions, ties, low confidence and flat channels abstain. Confident
unmapped labels emit `no_op_unmapped`. Optional local `preview_status` and
`preview_pause` mappings emit `pending_human_confirmation`.

Every event has `requires_human_confirmation: true` and `dispatched: false`.
There is no confirmation method, command body, transport, shell execution or
live agent connection. A prediction grants no authority. Scores are uncalibrated
model probabilities, not the probability that a person intended an action.

## Inspected Magistrate seam; no production integration

Read-only inspection used local `/home/spectre/Magistrate` at
`9492b7c87a710f4359e5cd5d7bc641eae644f76b` on `melkezic-dev-01`. This is an inspected
local ref, not a statement about public main or deployment. No Magistrate files,
PR128 work or settings were modified; no runtime calls or credentials were used.
A replacement project name remains undecided.

At that ref:

- `gateway/app/contracts.py` defines `UniversalInputContract`: source, modality,
  type, text, target and optional message/execution metadata.
- `gateway/app/main.py` exposes `POST /api/v1/captain/prompt` under command scope.
  It executes prompts and is not a harmless prediction sink.
- Attention prepare/execute routes require server-issued action keys and
  confirmation tokens, with target/revision and owner authority checks documented
  in `docs/architecture.md`.
- `docs/magi-roadmap-implementation-report.md` makes neural/subvocal input
  hardware-contingent future work with more conservative confirmation.

Neural IO events use their own schema; they are not Magistrate requests. No
compatibility adapter or production integration is claimed. A future adapter
needs a reviewed mapping from a validated task to a bounded request, fresh human
confirmation and destination authorization, target, revision and idempotency
guarantees. It must never convert raw scores to attention approvals/rejections or
bypass those controls.

## Verification and next decisions

`python -m unittest discover -s tests -v` checks schema/units, pickle-free files,
causal chunk invariance, final windows, short/flat/invalid inputs, spectral
features, separated groups, probability/quality abstention, default no-op and
mock mappings, reload equivalence, compatibility and two-run reproducibility.

Next required input: available device or recording format, the first specific
control task and an appropriate independent evaluation unit. Acquisition,
human-signal validation and consequential actions remain outside this increment.
