"""
vad -- voice activity detection, so the hot path stops paying for silence.

WHY THIS EXISTS, WITH NUMBERS.

`watcher.py` classifies speech by energy alone. Measured against signals whose
truth we know, the energy-only segmenter reports the SAME fraction of "speech"
for every one of these -- 89% -- because it marks the whole clip:

    steady 60 Hz hum         89% "speech"
    hum + harmonics          89%
    white noise              89%
    digital silence          89%
    speech-like modulation   89%

That sameness is the whole finding: in a noisy room the segmentation carries
no information at all, a room with a fan emits no pause events, and every
table built from pauses is wrong in exactly the same direction.

THE NUMBERS BELOW ARE FROM REAL SPEECH, NOT FROM A TONE GENERATOR, and the
two disagree. Measured over 90 s of LibriVox
(`examples/realaudio/alice01_excerpt.wav`, resampled to 16 kHz by
`wavio.read_wav`) with ONE gate held across the whole stream:

    signal                       mean      frac of frames >= 0.5
    real speech                  0.7300    0.728
    120 Hz hum, amplitude 0.05   0.1579    0.000
    hum + harmonics              0.0205    0.000
    white noise                  0.0118    0.000
    digital silence              0.0037    0.000

The earlier version of this docstring quoted 0.7727 for "speech-like
modulation" and 0.0047-0.0138 for the non-speech cases. Those were measured
on synthesised signals and were optimistic: a synthetic amplitude-modulated
tone is not speech, and the real quiet end is lower than that. Both numbers
are corrected above rather than quietly replaced, because the direction of the
error is itself the lesson -- the synthetic corpus flattered the gate.

Cost, one thread, this machine, per the whole watcher (not this module alone):

    watcher.py, no gate       31.8% of a core    (39.13 ms per 100 ms)
    watcher.py, with gate      2.4% of a core    (1.2 ms per 100 ms)

The reason the gate has to sit in front is that the watcher's dominant cost is
`pitch.PitchTracker` running across every speech segment -- which is exactly
when it is wrong about a hum.

WHAT THE GATE IS NOT, AND WHY THAT DECIDES THE INTEGRATION.

Measured against the non-speech it must catch, it is excellent. Measured
against something else, it is not: pure tones at 150, 175 and 200 Hz score
0.66-0.82, and a harmonic stack at 200 Hz scores 0.97. The model's training
contains voiced periodic sound, and a periodic sound with that structure is
what it is looking for. So it CANNOT decide what gets reported.

It is therefore wired into the watcher for exactly one purpose -- deciding
whether running YIN over a segment is worth the CPU -- and every utterance
record carries the probability so the decision stays auditable. This is the
same rule `policy.backend_may_gate` already enforces for local models, and it
applies with more force to a 2 MB graph on a peripheral.

THE DIRECTION THAT ACTUALLY MATTERS.

A gate built for a threat and only tested against that threat always looks
good. The adversarial battery of this project was retracted for precisely
that: it scored an attack at 0.867 and ordinary narration at 0.980, blocking
normal conversation harder than the attack. So the question asked here is not
"does it catch a hum" but "does it refuse to throw away a person talking":

    real speech segments vetoed, over 20 segments of the fixture
        0 dB   0/20        -30 dB   0/20
        -20 dB 0/20        -40 dB   1/20
                               -50 dB  20/20

So there is a floor around -40 dB below which this gate cannot be trusted, and
any claim that the system "always hears the room" would be false under it.
That is asserted, not just written here: `TestDeclaredLimits`.

WHY SILERO AND NOT SOMETHING ELSE.

    size          2,327,524 bytes, 16 kHz variant
    latency       0.072-0.135 ms per 32 ms chunk, single thread, CPU only
    licence       MIT (model and code)
    separation    0.0037 (silence) .. 0.1579 (loudest hum) | 0.7300 (speech)

The 0.5 threshold sits inside a gap roughly 4x wide on the hum side and 3x on
the quiet-speech side. Not enormous, and the `max` statistic is what makes it
usable: over 20 real segments `max` vetoed 0/20 where `mean` vetoed 1/20.

THE DEPENDENCY TRAP, WHICH DECIDES THE IMPLEMENTATION.

`pip install silero-vad` installs **torch** regardless of extras --
`pyproject.toml:32` puts `torch>=1.12.0` in *base* dependencies and extras can
only ever add. torch is not present here and is not wanted in a hot path that
must survive on a peripheral. So this module talks to the ONNX graph directly
with numpy + onnxruntime, ~90 lines, and nothing else.

`Watcher` treats the gate as OPTIONAL. Without onnxruntime or without the
weights it degrades to energy-only behaviour, unchanged, and says so once.

MIT licensed. Weights are fetched, not vendored -- `fetch_model()` does it and
tests use a synthetic stand-in where possible so the suite does not need a
2.2 MB blob.
"""

from __future__ import annotations

import math
import os

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512        # 32 ms at 16 kHz -- the model's native frame
CONTEXT_SAMPLES = 64       # the model's required left context
STATE_SHAPE = (2, 1, 128)
DEFAULT_THRESHOLD = 0.5

_MODEL_URLS = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx",
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "files/silero_vad.onnx",
)

_REPO_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "silero_vad.onnx")


def model_path() -> str | None:
    """Where the weights are, or None if they have not been fetched."""
    for candidate in (_REPO_MODEL,
                      os.environ.get("SILERO_VAD_MODEL", "")):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def fetch_model(dest: str | None = None) -> str:
    """
    Download the weights. Kept out of `import` so that merely reading this
    module never reaches the network -- tests and offline machines must not be
    surprised by a download.
    """
    import urllib.request
    dest = dest or _REPO_MODEL
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last = None
    for url in _MODEL_URLS:
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as exc:                            # noqa: BLE001
            last = exc
    raise RuntimeError(f"could not fetch silero weights: {last}")


class GateUnavailable(RuntimeError):
    """onnxruntime or the weights are missing. Not a crash -- a downgrade."""


class SileroGate:
    """
    Stateful frame-wise speech probability.

    Stateful on purpose: the model carries an LSTM state across frames, and a
    stateless implementation scores every 32 ms in isolation, which measurably
    worsens separation at utterance edges. That is the same class of mistake as
    the watcher's file-global thresholds.

        g = SileroGate()
        for chunk in stream:                 # any chunk size
            for prob in g.push(chunk):
                ...                            # one prob per 32 ms frame

    `push` accepts any chunk length and returns one probability per complete
    32 ms frame, carrying the remainder. Audio pushed mid-frame is not dropped
    or double-counted.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD,
                 model_path: str | None = None,
                 sample_rate: int = SAMPLE_RATE):
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"the bundled graph is 16 kHz only; got {sample_rate}. "
                "Resample before calling, or the scores are meaningless.")
        self.threshold = threshold
        self.sample_rate = sample_rate
        path = model_path or _locate_model()
        if path is None:
            raise GateUnavailable(
                "silero weights not found; run fetch_model() or set "
                "SILERO_VAD_MODEL")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise GateUnavailable(f"onnxruntime missing: {exc}") from None
        # One thread on purpose. A peripheral runs alongside other things, and
        # ORT's default thread pool would make this the most demanding thing on
        # the machine despite being the least important.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        # ANY failure to open the graph becomes GateUnavailable, including a
        # bad explicit `model_path`. onnxruntime raises NoSuchFile (an
        # onnxruntime exception, not an OSError) for a missing file, so a
        # caller catching OSError -- the obvious thing to write -- misses it
        # entirely and the documented "degrade to energy-only" path never
        # runs. Found by a test that passed a nonexistent path and got a
        # NoSuchFile instead of a downgrade.
        try:
            self._sess = ort.InferenceSession(path, opts,
                                              providers=["CPUExecutionProvider"])
        except Exception as exc:                            # noqa: BLE001
            raise GateUnavailable(f"could not open silero graph at {path}: "
                                  f"{exc}") from None
        self._state = None
        self._sr = __import__("numpy").array(sample_rate, dtype="int64")
        self._carry = []

    def reset(self) -> None:
        self._state = None
        self._carry = []

    def push(self, samples) -> list:
        """One probability per complete 32 ms frame."""
        import numpy as np
        if len(samples) == 0:
            return []
        buf = np.concatenate([np.asarray(self._carry, dtype=np.float32),
                              np.asarray(samples, dtype=np.float32)]) \
            if self._carry else np.asarray(samples, dtype=np.float32)
        n_frames = len(buf) // FRAME_SAMPLES
        if n_frames == 0:
            self._carry = buf.tolist()
            return []
        out = []
        for i in range(n_frames):
            frame = buf[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
            inp = np.concatenate(
                [np.zeros(CONTEXT_SAMPLES, np.float32), frame])[None, :]
            if self._state is None:
                self._state = np.zeros(STATE_SHAPE, np.float32)
            res = self._sess.run(None, {"input": inp,
                                        "state": self._state,
                                        "sr": self._sr})
            self._state = res[1]
            out.append(float(np.ravel(res[0])[0]))
        self._carry = buf[n_frames * FRAME_SAMPLES:].tolist()
        return out

    def is_speech(self, samples) -> bool:
        """Any frame over threshold across the chunk."""
        return any(p >= self.threshold for p in self.push(samples))

    def mean_probability(self, samples) -> float:
        probs = self.push(samples)
        return sum(probs) / len(probs) if probs else 0.0


def _locate_model() -> str | None:
    env = os.environ.get("SILERO_VAD_MODEL")
    if env and os.path.isfile(env):
        return env
    return model_path() if os.path.isfile(_REPO_MODEL) else None


# -- a stand-in with no weights, for tests and offline machines ---------------

class ScriptedGate:
    """
    A gate whose probabilities are supplied rather than computed.

    Used by tests so the SUITE does not require a 2.2 MB blob or onnxruntime,
    and used by anyone who wants the watcher's gate behaviour without the
    dependency. `script` is a list of probabilities consumed one per 32 ms
    frame; when exhausted it repeats the last value.
    """

    def __init__(self, script, threshold: float = DEFAULT_THRESHOLD):
        self._script = list(script)
        self._i = 0
        self.threshold = threshold
        self.calls = 0

    def reset(self):
        self._i = 0

    def push(self, samples) -> list:
        import numpy as np
        n = len(samples) // FRAME_SAMPLES
        out = []
        for _ in range(n):
            v = (self._script[self._i] if self._i < len(self._script)
                 else (self._script[-1] if self._script else 0.0))
            out.append(float(v))
            self._i += 1
        self.calls += 1
        return out

    def is_speech(self, samples) -> bool:
        return any(p >= self.threshold for p in self.push(samples))

    def mean_probability(self, samples) -> float:
        probs = self.push(samples)
        return sum(probs) / len(probs) if probs else 0.0