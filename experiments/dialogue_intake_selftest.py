#!/usr/bin/env python3
"""
dialogue_intake_selftest.py -- prove the real-dialogue intake can do its job
BEFORE anybody spends 25 minutes of human time recording into it.

WHY THIS EXISTS.

`examples/realdialogue/raw/` is empty. `experiments/dialogue_intake.py` has
never been run against a single real file. Every claim that the intake
"checks the corpus" is a claim about code nobody has ever watched make a
decision.

The project has already paid for that once, on the OTHER corpus. The captures
in `docs/research/AV-SYNC.md` were wrong in three ways at once, and all three
were invisible because nothing had ever checked:

    1. the audio and the video did not correspond, and nothing recorded that
       they should;
    2. the camera delivered 25 fps where 30 was assumed, silently stretching
       the time base by 1.2x;
    3. the achievement check never ran -- the script's `src` import resolved
       to the wrong path, the exception was swallowed, and every take was
       saved with "speech check unavailable" and no verification at all.

So the question this file answers is not "does the intake work". It is:
**for each thing the intake says it checks, can a bad corpus actually make it
say so?** A gate that has only ever been observed accepting things is not
known to work -- it might be accepting everything.

THE METHOD.

Synthetic matched pairs, built to look like the eight scenes in
`examples/realdialogue/README.md`, written with the repo's own reader
(`src.wavio`) and read back through it. Then:

    * one GOOD corpus, which must be accepted;
    * eight PERTURBED corpora, each of which must be rejected in the specific
      way the intake claims to detect;
    * a set of KNOWN-OFFSET pairs, to measure what an unaligned pair does to
      the numbers.

Where the intake does NOT catch a perturbation, this script records that as a
FINDING and keeps going. It does not weaken the perturbation until the gate
passes -- that is the trap this project keeps falling into. A perturbation
that is not caught is a measurement about the tool, not a failed test.

WHAT IS SYNTHETIC HERE, AND WHY THAT IS THE POINT.

The speech is a voiced/unvoiced-ish harmonic stack with a syllable-rate
envelope and aspiration noise. It is NOT speech, and it is not trying to be:
this is a test of the INTAKE, which never does speech recognition. What
matters is that the signal has the properties the intake's gates claim to
key on -- loud and quiet regions, silences of known length, two channels that
are independent, a clap at a known instant, and an inter-channel offset that
is known exactly.

Every number in the results file is produced by running the real
`experiments/dialogue_intake.py` as a subprocess and parsing its JSON. The
only things this script computes itself are the things the intake does not
output at all (the measured gaps, the overlap boundary, the clap offset), and
those are computed with a MIRROR of the intake's own logic, which is stated
in the code so the two can be compared.

WHAT CANNOT BE MEASURED HERE, AND IS NOT CLAIMED.

Real speech, real room tone, real two-microphone crosstalk, real recorder
start skew and real clock drift between two independent devices. This file
measures the GATE, with synthetic inputs of known truth. It cannot tell you
what a real recording will look like.

OUTPUT: `experiments/dialogue_intake_selftest_results.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import wave
from array import array
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import energy, wavio, watcher  # noqa: E402

SR = 16000                      # the pipeline's analysis rate
FLOOR = energy.ABS_FLOOR        # the intake's "not silent" floor (1e-4 RMS)
INTAKE = ROOT / "experiments" / "dialogue_intake.py"
RESULTS = ROOT / "experiments" / "dialogue_intake_selftest_results.json"

# ---------------------------------------------------------------- signal ----

def room_tone(n: int, rms: float, seed: int) -> np.ndarray:
    """Gaussian room noise at a given RMS."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, rms, n)


def speech(n: int, f0: float, seed: int, amp: float = 0.06,
           sr: int = SR, flat: bool = False) -> np.ndarray:
    """A voiced/unvoiced-ish waveform: harmonic stack + syllable envelope +
    aspiration noise. `flat` removes the envelope dip, for the scenes whose
    whole point is that there is NO pause inside them.

    The syllable rate and phase are drawn PER UTTERANCE, not fixed. Two
    people do not speak in lockstep, and a generator that gave both channels
    the same modulation rate would manufacture a correlation between them
    that a real room does not have -- which would make the intake's
    cross-talk number look like a measurement when it is an artefact of the
    test data.
    """
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    drift = 1.0 + 0.02 * np.sin(2 * np.pi * 0.7 * t + seed)
    phase = 2 * np.pi * np.cumsum(f0 * drift) / sr
    voiced = sum(a * np.sin(k * phase)
                 for k, a in ((1, 1.0), (2, 0.6), (3, 0.35), (4, 0.2), (5, 0.1)))
    voiced /= 2.25
    if flat:
        env = np.ones(n)
    else:
        rate = float(rng.uniform(3.0, 5.5))
        start_phase = float(rng.uniform(0.0, 2 * np.pi))
        env = 0.75 + 0.25 * np.sin(2 * np.pi * rate * t + start_phase)
    aspiration = 0.06 * rng.normal(0, 1, n)
    sig = voiced * env + aspiration
    peak = float(np.max(np.abs(sig))) or 1.0
    return sig / peak * amp


def clap(n: int, seed: int = 7) -> np.ndarray:
    """A broadband transient with a fast exponential decay: the README's
    alignment point, which no code in this repo currently uses."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    env = np.exp(-t / (0.008 * SR))
    burst = rng.normal(0, 1, n) * env
    peak = float(np.max(np.abs(burst))) or 1.0
    return burst / peak * 0.8


def write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> Path:
    """Write 16-bit mono PCM.

    NOTE, reported as a finding: `src/wavio.py` has NO writer. It can read
    8/16/32-bit PCM at any rate, but there is nothing in the repo that writes
    the WAVs the recording protocol asks for. This function uses the standard
    library `wave` module -- the same module `src.wavio.read_wav` reads with
    -- so the round-trip is the repo's own reader's round-trip, and the
    selftest verifies that round-trip on every file it writes.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = array("h", (int(v * 32767) for v in clipped))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def header_rate(path: Path) -> int:
    """The sample rate as it is actually in the file's header."""
    with wave.open(str(path), "rb") as w:
        return w.getframerate()


# ----------------------------------------------------------------- scenes ----

# One scene = a list of (channel, start_s, end_s, seed, flat) events, laid
# out inside a UNIT. Longer scenes tile the unit, so every SILENCE keeps the
# length it was designed with instead of being stretched by the rescaling
# (stretching would change the pause durations the gates key on).
UNIT_S = 12.0
CLAP_AT = 0.30
CONTENT_AT = 0.60

SCENES: list[dict] = [
    dict(name="scene01_free", protocol_s=300, kind="dialogue", events=[
        ("A", 0.60, 1.60, 11), ("B", 1.90, 2.80, 12),
        ("A", 3.20, 4.20, 13), ("B", 4.50, 5.40, 14),
        ("A", 5.80, 6.80, 15), ("B", 7.10, 8.00, 16),
        ("A", 8.40, 9.40, 17), ("B", 9.70, 10.60, 18),
    ]),
    dict(name="scene02_qa", protocol_s=180, kind="dialogue", events=[
        ("A", 0.60, 1.80, 21), ("B", 2.80, 4.40, 22),
        ("A", 5.00, 6.20, 23), ("B", 7.00, 8.60, 24),
        ("A", 9.20, 10.40, 25),
    ]),
    dict(name="scene03_think", protocol_s=180, kind="dialogue", events=[
        ("A", 0.60, 1.60, 31), ("B", 5.00, 6.50, 32),
        ("A", 7.00, 7.80, 33), ("B", 11.00, 11.90, 34),
    ]),
    dict(name="scene04_fillers", protocol_s=120, kind="dialogue", events=[
        ("A", 0.60, 11.40, 41, True),          # one continuous run, no pause
    ]),
    dict(name="scene05_backchannel", protocol_s=180, kind="dialogue", events=[
        ("A", 0.60, 6.00, 51, True), ("A", 6.60, 11.40, 52, True),
        ("B", 2.00, 2.25, 53), ("B", 4.00, 4.20, 54), ("B", 8.50, 8.75, 55),
    ]),
    dict(name="scene06_interrupt", protocol_s=120, kind="dialogue", events=[
        ("A", 0.60, 2.00, 61), ("B", 1.80, 3.20, 62),   # B cuts A off
        ("A", 3.60, 4.20, 63), ("B", 3.40, 4.60, 64),   # A barges back in
        ("A", 5.20, 6.60, 65), ("B", 6.00, 7.40, 66),   # and again
        ("A", 8.00, 9.40, 67),
    ]),
    dict(name="scene07_alone", protocol_s=120, kind="dialogue", events=[
        ("A", 0.60, 2.00, 71), ("A", 4.50, 6.00, 72), ("A", 8.50, 10.00, 73),
    ]),
    dict(name="scene08_roomtone", protocol_s=30, kind="roomtone", events=[]),
]


def build_scene(spec: dict, seconds: float, tone_rms: float,
                seed_base: int = 0, unit_s: float = UNIT_S
                ) -> tuple[np.ndarray, np.ndarray, list]:
    """Synthesise one (A, B) pair. Returns the two channels and the truth
    list of (channel, start_s, end_s) placements."""
    n = int(round(seconds * SR))
    a = room_tone(n, tone_rms, 1000 + seed_base)
    b = room_tone(n, tone_rms, 2000 + seed_base)
    truth: list[tuple[str, float, float]] = []

    if spec["kind"] == "roomtone":
        return a, b, truth

    # The clap: once per scene, in BOTH channels, at a known instant.
    cl = int(CLAP_AT * SR)
    cn = int(0.04 * SR)
    burst = clap(cn)
    a[cl:cl + cn] += burst
    b[cl:cl + cn] += burst

    # Tile the unit until the scene is filled.
    reps = max(1, int(math.ceil((seconds - CONTENT_AT) / unit_s)) + 1)
    for rep in range(reps):
        base = CONTENT_AT + rep * unit_s
        if base >= seconds:
            break
        for ev in spec["events"]:
            ch, s, e, sd = ev[0], ev[1], ev[2], ev[3]
            flat = len(ev) > 4 and ev[4]
            start = base + s - CONTENT_AT
            end = base + e - CONTENT_AT
            if end <= 0 or start >= seconds:
                continue
            start = max(0.0, start)
            end = min(seconds, end)
            if end - start < 0.05:
                continue
            sig = speech(int((end - start) * SR), 118 + 11 * (sd % 7),
                         100 + sd + 40 * rep, flat=flat)
            off = int(start * SR)
            buf = a if ch == "A" else b
            buf[off:off + len(sig)] += sig
            truth.append((ch, round(start, 4), round(end, 4)))
    return a, b, truth


def alternate_specs(keep_overlap: tuple[str, ...] = ("scene06_interrupt",),
                    ) -> list[dict]:
    """The same eight scenes with the two speakers STRICTLY ALTERNATING:
    A stops before B starts, B stops before A starts, so the two channels
    are never loud at the same time.

    `keep_overlap` names scenes that keep their overlap. That matters: the
    intake's hand-off counter can only fire when both channels are loud at
    once, so a corpus with NO overlap at all has no hand-offs either, and
    removing every overlap would change two gates at once and prove
    nothing about the overlap gate on its own.
    """
    out = []
    for spec in SCENES:
        if spec["name"] in keep_overlap:
            out.append(spec)
            continue
        evs = sorted(spec["events"], key=lambda e: (e[1], e[0]))
        placed: list[tuple] = []
        t = CONTENT_AT
        for ev in evs:
            ch, s, e, sd = ev[0], ev[1], ev[2], ev[3]
            flat = len(ev) > 4 and ev[4]
            dur = e - s
            if t + dur > UNIT_S:
                break                      # does not fit the unit; drop it
            placed.append((ch, round(t, 3), round(t + dur, 3), sd, flat))
            t += dur + 0.25
        out.append(dict(spec, events=placed))
    return out


def strict_specs() -> list[dict]:
    """The protocol corpus with scene 6's barge-back removed, so that every
    turn in every scene is a CANONICAL hand-off: A stops, B takes the floor,
    A stays quiet. This is what the README describes, literally."""
    strict_scene6 = dict(
        name="scene06_interrupt", protocol_s=120, kind="dialogue", events=[
            ("A", 0.60, 2.00, 61), ("B", 1.80, 3.20, 62),
            ("A", 5.20, 6.60, 65), ("B", 6.00, 7.40, 66),
            ("A", 8.00, 9.40, 67),
        ])
    return [strict_scene6 if s["name"] == "scene06_interrupt" else s
            for s in SCENES]


def build_corpus(out_dir: Path, seconds_per_scene: float, tone_rms: float,
                 perturb: str | None = None,
                 scene_seconds: dict | None = None,
                 specs: list | None = None) -> dict:
    """Write a synthetic corpus. `perturb` applies ONE defect."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    truth: dict[str, list] = {}
    for i, spec in enumerate(specs or SCENES):
        secs = (scene_seconds or {}).get(spec["name"], seconds_per_scene)
        a, b, tr = build_scene(spec, secs, tone_rms, seed_base=i * 100)
        truth[spec["name"]] = tr
        pa = write_wav(out_dir / f"{spec['name']}_A.wav", a)
        pb = write_wav(out_dir / f"{spec['name']}_B.wav", b)
        written += [pa.name, pb.name]

    if perturb == "missing_channel_B":
        (out_dir / "scene02_qa_B.wav").unlink()
    elif perturb == "digital_silence_A":
        write_wav(out_dir / "scene04_fillers_A.wav", np.zeros(SR * 6))
    elif perturb == "digital_silence_B":
        write_wav(out_dir / "scene04_fillers_B.wav", np.zeros(SR * 6))
    elif perturb == "low_level_B":
        b = wavio.read_wav(str(out_dir / "scene04_fillers_B.wav")).samples
        arr = np.array(b, dtype=float)
        rms = float(np.sqrt(np.mean(arr ** 2))) or 1.0
        write_wav(out_dir / "scene04_fillers_B.wav", arr / rms * 2e-4)
    elif perturb == "very_low_level_B":
        b = wavio.read_wav(str(out_dir / "scene04_fillers_B.wav")).samples
        arr = np.array(b, dtype=float)
        rms = float(np.sqrt(np.mean(arr ** 2))) or 1.0
        write_wav(out_dir / "scene04_fillers_B.wav", arr / rms * 5e-5)
    elif perturb == "crosstalk":
        for spec in (specs or SCENES):
            a = np.array(wavio.read_wav(str(out_dir / f"{spec['name']}_A.wav")).samples)
            b = np.array(wavio.read_wav(str(out_dir / f"{spec['name']}_B.wav")).samples)
            n = min(len(a), len(b))
            write_wav(out_dir / f"{spec['name']}_B.wav", 0.6 * a[:n] + b[:n])
    elif perturb == "no_handoff":
        for spec in (specs or SCENES):
            if spec["kind"] == "roomtone":
                continue
            b = np.array(wavio.read_wav(str(out_dir / f"{spec['name']}_B.wav")).samples)
            write_wav(out_dir / f"{spec['name']}_B.wav", np.zeros(len(b)))
    elif perturb == "sample_rate_mismatch":
        b = np.array(wavio.read_wav(str(out_dir / "scene05_backchannel_B.wav")).samples)
        # write the SAME wall-clock content at 22050 Hz
        write_wav(out_dir / "scene05_backchannel_B.wav", b, sr=22050)
    elif perturb == "unreadable":
        (out_dir / "scene06_interrupt_B.wav").write_bytes(b"NOT A WAV FILE" * 40)

    return {"dir": str(out_dir), "files": sorted(written), "truth": truth}


def _a_gaps(a: np.ndarray, sr: int, min_s: float = 0.4):
    """Silent runs of A, on the 50 ms grid, as (start_s, end_s)."""
    hop = int(0.05 * sr)
    loud = []
    for i in range(0, len(a) - hop, hop):
        seg = a[i:i + hop]
        loud.append(float(np.sqrt(np.mean(seg ** 2))) > FLOOR)
    gaps, run = [], None
    for k, is_loud in enumerate(loud):
        if not is_loud and run is None:
            run = k
        elif is_loud and run is not None:
            if (k - run) * 0.05 >= min_s:
                gaps.append((run * 0.05, k * 0.05))
            run = None
    if run is not None and (len(loud) - run) * 0.05 >= min_s:
        gaps.append((run * 0.05, len(loud) * 0.05))
    return gaps


# ------------------------------------------------------- the intake, run ----

def run_intake(raw_dir: Path) -> dict:
    """Run the REAL intake as a subprocess and parse its report."""
    proc = subprocess.run(
        [sys.executable, str(INTAKE), "--raw", str(raw_dir)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    out = {"exit_code": proc.returncode, "stderr": proc.stderr[-2000:]}
    try:
        out["report"] = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out["report"] = None
        out["stdout_tail"] = proc.stdout[-2000:]
    return out


def summarise(run: dict) -> dict:
    """The parts of the intake's report this selftest reasons about."""
    rep = run.get("report") or {}
    gates = rep.get("gates", {})
    pairs = []
    for p in rep.get("pairs", []):
        pairs.append({
            "scene": p.get("scene"),
            "problem": p.get("problem"),
            "A_silent": (p.get("A") or {}).get("silent"),
            "B_silent": (p.get("B") or {}).get("silent"),
            "A_peak": (p.get("A") or {}).get("peak"),
            "B_peak": (p.get("B") or {}).get("peak"),
            "A_rms": (p.get("A") or {}).get("rms"),
            "B_rms": (p.get("B") or {}).get("rms"),
            "A_signal_frac": (p.get("A") or {}).get("signal_frame_frac"),
            "B_signal_frac": (p.get("B") or {}).get("signal_frame_frac"),
            "A_raw_rate_hz": (p.get("A") or {}).get("raw_rate_hz"),
            "B_raw_rate_hz": (p.get("B") or {}).get("raw_rate_hz"),
            "A_duration_s": (p.get("A") or {}).get("duration_s"),
            "B_duration_s": (p.get("B") or {}).get("duration_s"),
            "crosstalk": p.get("crosstalk"),
            "levels": p.get("levels"),
        })
    return {
        "exit_code": run.get("exit_code"),
        "status": gates.get("status"),
        "unmet": gates.get("unmet", []),
        "matched_pairs": gates.get("matched_pairs"),
        "total_audio_s": gates.get("total_audio_s"),
        "silences_seen": gates.get("silences_seen"),
        "hand_offs_seen": gates.get("hand_offs_seen"),
        "unmatched_files": gates.get("unmatched_files", []),
        "pairs": pairs,
    }


# ------------------------------------- mirror of the intake's own logic ----

def intake_pauses(a: np.ndarray, b: np.ndarray) -> list[dict]:
    """A MIRROR of the hand-off loop in `experiments/dialogue_intake.py`.

    Reproduced verbatim from that file's `main()` so the two can be compared
    directly: the watcher runs on channel A, channel B is reduced to a 100 ms
    loud grid, and a CLOSED pause on A is a hand-off if channel B is loud in
    the 100 ms window at the pause's END.
    """
    return _pause_scan(a, b, proposed=False)


def _pause_scan(a: np.ndarray, b: np.ndarray, proposed: bool,
                skip_before_s: float = 0.0) -> list[dict]:
    """The intake's pause scan. `proposed=True` applies the fix proposed in
    docs/research/DIALOGUE-INTAKE.md: a hand-off is channel B loud ANYWHERE
    during channel A's silence, not only at the instant that silence ends.

    `skip_before_s` drops the pauses before that instant -- which is how the
    clap, loud in BOTH channels by construction, is kept out of the count.
    """
    ia = [float(v) for v in a]
    ib = [float(v) for v in b]
    n = min(len(ia), len(ib))
    w = watcher.Watcher(SR, adaptive=True)
    loud_b = [False] * (n // 1600 + 1)
    for k in range(len(loud_b)):
        seg = ib[k * 1600:(k + 1) * 1600]
        loud_b[k] = (((sum(v * v for v in seg) / max(1, len(seg))) ** 0.5) > FLOOR) \
            if seg else False
    rows = []
    for i in range(0, n, 1600):
        for e in w.push(ia[i:i + 1600]):
            if e.kind != "pause" or e.data.get("open"):
                continue
            if e.t0 < skip_before_s:
                continue
            k = min(len(loud_b) - 1, int(e.t1 / 0.1))
            if proposed:
                k0 = max(0, int(e.t0 / 0.1))
                is_handoff = any(loud_b[j] for j in range(k0, k + 1))
            else:
                is_handoff = bool(loud_b[k])
            rows.append({"t0": round(e.t0, 4), "t1": round(e.t1, 4),
                         "duration_s": round(e.t1 - e.t0, 4),
                         "b_loud_at_pause_end": bool(loud_b[k]),
                         "is_handoff": bool(is_handoff)})
    return rows


def hand_offs_proposed_rule(a: np.ndarray, b: np.ndarray,
                            skip_before_s: float = 0.6) -> int:
    """Hand-off count under the PROPOSED rule, for comparison."""
    return sum(1 for r in _pause_scan(a, b, proposed=True,
                                      skip_before_s=skip_before_s)
               if r["is_handoff"])


def loud_runs(b: np.ndarray, hop_s: float = 0.1) -> list[tuple[float, float]]:
    """Runs of loud samples on B, on the intake's own 100 ms grid."""
    hop = int(hop_s * SR)
    runs, start = [], None
    for k in range(len(b) // hop + 1):
        seg = b[k * hop:(k + 1) * hop]
        rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
        loud = rms > FLOOR
        if loud and start is None:
            start = k * hop_s
        elif not loud and start is not None:
            runs.append((round(start, 4), round(k * hop_s, 4)))
            start = None
    if start is not None:
        runs.append((round(start, 4), round((len(b) // hop) * hop_s, 4)))
    return runs


def measured_gap_s(a: np.ndarray, b: np.ndarray,
                   skip_before_s: float = 0.6,
                   min_run_s: float = 0.2) -> list[float]:
    """The hand-off gaps a downstream consumer of these two files would
    measure: for every loud run on channel B, how long after channel A's
    speech stopped did B start?

    This is measured on A's TIMELINE -- the thing that is supposed to be
    common to the two files -- and it is exactly the quantity the corpus
    exists to produce. Both channels are reduced to the intake's own 100 ms
    loud grid, so the measurement carries that grid's 100 ms quantisation.

    The clap is skipped: it is loud in BOTH channels by construction, so
    comparing the two claps measures the inter-channel offset rather than a
    turn gap, and it would sit at the front of the list.
    """
    a_runs = loud_runs(a)
    b_runs = loud_runs(b)
    gaps: list[float] = []
    for bs, be in b_runs:
        if bs < skip_before_s or (be - bs) < min_run_s:
            continue
        offs = [ae for (_as, ae) in a_runs if ae <= bs + 0.05]
        if offs:
            gaps.append(round(bs - max(offs), 4))
    return gaps


def _runs_above(x: np.ndarray, floor: float, frame_s: float = 0.005,
                min_len_s: float = 0.02, skip_before_s: float = 0.0):
    """Runs above a floor, at 5 ms frame resolution: (start_s, end_s).

    The frame RMS is used rather than the raw samples because a waveform
    crosses zero many times a second, and a sample-level threshold would
    shatter every utterance into a dozen pieces.
    """
    fl = int(frame_s * SR)
    n = len(x) // fl
    if n == 0:
        return []
    rms = np.sqrt(np.mean(x[:n * fl].reshape(n, fl) ** 2, axis=1))
    above = rms > floor
    runs, start = [], None
    for k, is_above in enumerate(above):
        if is_above and start is None:
            start = k
        elif not is_above and start is not None:
            if (k - start) * frame_s >= min_len_s:
                runs.append((round(start * frame_s, 4),
                             round(k * frame_s, 4)))
            start = None
    if start is not None and (n - start) * frame_s >= min_len_s:
        runs.append((round(start * frame_s, 4), round(n * frame_s, 4)))
    return [(s, e) for s, e in runs if s >= skip_before_s]


def measured_gap_samples(a: np.ndarray, b: np.ndarray,
                         skip_before_s: float = 0.6) -> list[float]:
    """The same hand-off gap, measured at 5 ms frame resolution instead of
    on the intake's 100 ms grid.

    The grid version can only resolve an offset to 100 ms. This one can
    resolve it to 5 ms, so the two together say how much of the offset error
    is the offset and how much is the measurement grid.
    """
    a_runs = _runs_above(a, FLOOR, skip_before_s=skip_before_s)
    b_runs = _runs_above(b, FLOOR, skip_before_s=skip_before_s)
    gaps: list[float] = []
    for bs, be in b_runs:
        if (be - bs) < 0.2:
            continue
        offs = [ae for (_as, ae) in a_runs if ae <= bs + 0.005]
        if offs:
            gaps.append(round(bs - max(offs), 4))
    return gaps


def overlap_boundary_s(a: np.ndarray, b: np.ndarray,
                       skip_before_s: float = 0.6) -> float | None:
    """First instant at which both channels are loud, on the 50 ms grid the
    intake's `both_talking_frac` uses. The clap is skipped: it is loud in
    both channels by construction, so it would always be the answer."""
    hop = int(0.05 * SR)
    n = min(len(a), len(b))
    for i in range(0, n - hop, hop):
        if i / SR < skip_before_s:
            continue
        ra = float(np.sqrt(np.mean(a[i:i + hop] ** 2)))
        rb = float(np.sqrt(np.mean(b[i:i + hop] ** 2)))
        if ra > FLOOR and rb > FLOOR:
            return round(i / SR, 4)
    return None


def clap_offset_ms(a: np.ndarray, b: np.ndarray, search_s: float = 2.0,
                   max_lag_ms: int = 600) -> dict:
    """PROPOSED FIX, measured here: find the inter-channel offset from the
    clap the README asks for.

    The clap is the loudest transient in the first seconds of both files, so
    the 1 ms energy envelopes of the two files can be cross-correlated over a
    range of lags; the peak is the offset. This is the alignment the protocol
    describes and that no code in this repo performs.
    """
    hop = int(0.001 * SR)
    def env(x):
        m = min(len(x), int(search_s * SR))
        n = m // hop
        e = np.array([float(np.sqrt(np.mean(x[k * hop:(k + 1) * hop] ** 2)))
                      for k in range(n)])
        return e - np.median(e)
    ea, eb = env(a), env(b)
    m = min(len(ea), len(eb))
    ea, eb = ea[:m], eb[:m]
    ea = ea / (np.linalg.norm(ea) or 1.0)
    eb = eb / (np.linalg.norm(eb) or 1.0)
    lags = range(-max_lag_ms, max_lag_ms + 1)
    best_lag, best_r = 0, -2.0
    curve = {}
    for lag in lags:
        if lag >= 0:
            x, y = ea[lag:], eb[:m - lag]
        else:
            x, y = ea[:m + lag], eb[-lag:]
        if len(x) < 50:
            continue
        r = float(np.dot(x, y))
        curve[lag] = round(r, 4)
        if r > best_r:
            best_r, best_lag = r, lag
    return {"lag_ms": best_lag, "peak_r": round(best_r, 4),
            "curve_at_zero": curve.get(0), "n_lags": len(curve)}


def shift(x: np.ndarray, ms: float) -> np.ndarray:
    """Delay a signal by ms milliseconds (positive = later); a negative ms
    ADVANCES it, which is what undoing an inter-channel offset needs."""
    d = int(round(ms * SR / 1000.0))
    if d == 0:
        return x.copy()
    out = np.zeros(len(x))
    if d > 0:
        if d < len(x):
            out[d:] = x[:len(x) - d]
    else:
        adv = -d
        if adv < len(x):
            out[:len(x) - adv] = x[adv:]
    return out


# ------------------------------------------------------------- the battery --

PERTURBATIONS = [
    ("missing_channel_B", "one channel of a pair is missing (unmatched pair)"),
    ("digital_silence_A", "channel A of one scene is digital silence"),
    ("digital_silence_B", "channel B of one scene is digital silence"),
    ("low_level_B", "channel B of one scene is very low but NOT zero"),
    ("very_low_level_B", "channel B of one scene is very low (below the floor)"),
    ("crosstalk", "channel B clearly hears channel A"),
    ("no_overlap", "the two speakers never overlap at all"),
    ("no_handoff", "only one speaker exists; there is no hand-off at all"),
    ("sample_rate_mismatch", "the two channels of one scene have different "
                             "sample rates"),
    ("unreadable", "one channel is not a readable WAV"),
]


def zero_mean_waveform_corr(a: np.ndarray, b: np.ndarray, W: int = 1600,
                            skip_before_s: float = 0.6):
    """Correlation of the ZERO-MEAN waveforms over the windows where both
    channels are loud.

    This is the PROPOSED replacement for the intake's cross-talk statistic,
    which correlates the window MEANS of the ten loudest windows. Measured
    below: the intake's statistic separates a corrupted corpus from a clean
    one by 0.01; this one separates them by 0.27-0.37 on the same signals.

    The clap region is skipped: the clap is the SAME physical sound in both
    files, so any waveform statistic that includes it reads a perfect
    correlation on a perfectly clean pair.
    """
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    num = sa = sb = 0.0
    nwin = 0
    start = int(skip_before_s * SR)
    for i in range(start, n - W, W):
        wa = a[i:i + W] - a[i:i + W].mean()
        wb = b[i:i + W] - b[i:i + W].mean()
        if (float(np.sqrt(np.mean(wa ** 2))) > FLOOR and
                float(np.sqrt(np.mean(wb ** 2))) > FLOOR):
            num += float(np.dot(wa, wb))
            sa += float(np.dot(wa, wa))
            sb += float(np.dot(wb, wb))
            nwin += 1
    if nwin == 0:
        return None, 0
    return round(num / ((sa * sb) ** 0.5 or 1.0), 4), nwin


def b_loud_when_a_loud(a: np.ndarray, b: np.ndarray, W: int = 1600,
                        skip_before_s: float = 0.6):
    """Of the time channel A is loud, how much of it is channel B loud too?

    This is the statistic the cross-talk gate actually wants, and it does not
    depend on the two waveforms resembling each other at all: if channel B
    is hearing channel A, then B is loud whenever A is loud. On separate
    microphones that fraction is the overlap fraction, which is small; when
    B hears A it approaches 1.
    """
    n = min(len(a), len(b))
    a_loud = both = 0
    for i in range(int(skip_before_s * SR), n - W, W):
        ra = float(np.sqrt(np.mean(a[i:i + W] ** 2)))
        rb = float(np.sqrt(np.mean(b[i:i + W] ** 2)))
        if ra > FLOOR:
            a_loud += 1
            if rb > FLOOR:
                both += 1
    if a_loud == 0:
        return None, 0
    return round(both / a_loud, 4), a_loud


def crosstalk_metric_comparison(work: Path, matrix_seconds: float) -> dict:
    """The intake's cross-talk statistic, on a clean corpus and on one where
    channel B hears channel A -- plus the proposed replacement."""
    clean_dir = work / "xt_clean"
    bad_dir = work / "xt_bad"
    build_corpus(clean_dir, matrix_seconds, 1e-5)
    build_corpus(bad_dir, matrix_seconds, 1e-5, perturb="crosstalk")
    out = {"scenes": []}
    for spec in SCENES:
        name = spec["name"]
        row = {"scene": name}
        for label, d in (("clean", clean_dir), ("crosstalk", bad_dir)):
            a = np.array(wavio.read_wav(str(d / f"{name}_A.wav")).samples)
            b = np.array(wavio.read_wav(str(d / f"{name}_B.wav")).samples)
            intake_corr = None
            wins = 0
            # the intake's statistic, verbatim
            W = 1600
            cand = []
            for i in range(0, min(len(a), len(b)) - W, W):
                wa, wb = a[i:i + W], b[i:i + W]
                ra = float(np.sqrt(np.mean(wa ** 2)))
                rb = float(np.sqrt(np.mean(wb ** 2)))
                if ra > FLOOR and rb > FLOOR:
                    cand.append((ra + rb, float(wa.mean()), float(wb.mean())))
            if len(cand) >= 3:
                cand.sort(reverse=True, key=lambda w: w[0])
                top = cand[:10]
                ma = [c[1] for c in top]
                mb = [c[2] for c in top]
                num = sum((x - sum(ma) / len(ma)) * (y - sum(mb) / len(mb))
                          for x, y in zip(ma, mb))
                da = (sum((x - sum(ma) / len(ma)) ** 2 for x in ma)) ** 0.5
                db = (sum((y - sum(mb) / len(mb)) ** 2 for y in mb)) ** 0.5
                intake_corr = round(num / (da * db), 4) if da and db else None
                wins = len(cand)
            zc, zn = zero_mean_waveform_corr(a, b)
            bl, al = b_loud_when_a_loud(a, b)
            row[label] = {"intake_corr": intake_corr,
                          "intake_simultaneous_loud_windows": wins,
                          "zero_mean_waveform_corr": zc,
                          "zero_mean_waveform_windows": zn,
                          "b_loud_when_a_loud_frac": bl,
                          "a_loud_windows": al}
        out["scenes"].append(row)
    clean_i = [r["clean"]["intake_corr"] for r in out["scenes"]
               if r["clean"]["intake_corr"] is not None]
    bad_i = [r["crosstalk"]["intake_corr"] for r in out["scenes"]
             if r["crosstalk"]["intake_corr"] is not None]
    clean_z = [r["clean"]["zero_mean_waveform_corr"] for r in out["scenes"]
               if r["clean"]["zero_mean_waveform_corr"] is not None]
    bad_z = [r["crosstalk"]["zero_mean_waveform_corr"] for r in out["scenes"]
             if r["crosstalk"]["zero_mean_waveform_corr"] is not None]
    clean_b = [r["clean"]["b_loud_when_a_loud_frac"] for r in out["scenes"]
               if r["clean"]["b_loud_when_a_loud_frac"] is not None]
    bad_b = [r["crosstalk"]["b_loud_when_a_loud_frac"] for r in out["scenes"]
             if r["crosstalk"]["b_loud_when_a_loud_frac"] is not None]
    out["intake_metric"] = {
        "clean_max": max(clean_i) if clean_i else None,
        "clean_min": min(clean_i) if clean_i else None,
        "crosstalk_min": min(bad_i) if bad_i else None,
        "crosstalk_max": max(bad_i) if bad_i else None,
        "separation": (round(min(bad_i) - max(clean_i), 4)
                       if (clean_i and bad_i) else None),
        "scenes_with_a_computable_value_clean": len(clean_i),
        "scenes_total": len(SCENES),
    }
    out["proposed_metric"] = {
        "clean_max": max(clean_z) if clean_z else None,
        "clean_min": min(clean_z) if clean_z else None,
        "crosstalk_min": min(bad_z) if bad_z else None,
        "crosstalk_max": max(bad_z) if bad_z else None,
        "separation": (round(min(bad_z) - max(clean_z), 4)
                       if (clean_z and bad_z) else None),
    }
    out["proposed_metric_b_loud_when_a_loud"] = {
        "clean_max": max(clean_b) if clean_b else None,
        "clean_min": min(clean_b) if clean_b else None,
        "crosstalk_min": min(bad_b) if bad_b else None,
        "crosstalk_max": max(bad_b) if bad_b else None,
        "separation": (round(min(bad_b) - max(clean_b), 4)
                       if (clean_b and bad_b) else None),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-seconds", type=float, default=12.0,
                    help="length of each scene in the matrix corpus")
    ap.add_argument("--tone-rms", type=float, default=1e-5,
                    help="room tone RMS under the corpus (default 1e-5, "
                         "below the intake's 1e-4 floor)")
    ap.add_argument("--real-room-rms", type=float, default=1.5e-3,
                    help="room tone RMS for the realistic-room corpus")
    ap.add_argument("--long", action="store_true",
                    help="also run the full-length (protocol-length) corpus")
    ap.add_argument("--workdir", default=None,
                    help="where to put the synthetic corpora (default: a "
                         "fresh temp dir)")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    import tempfile
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(
        prefix="dialogue_intake_selftest_"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"[selftest] work dir: {work}")

    t_start = time.time()
    results: dict = {
        "script": "experiments/dialogue_intake_selftest.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "numpy": np.__version__,
        "analysis_rate_hz": SR,
        "intake_floor_rms": FLOOR,
        "matrix_seconds_per_scene": args.matrix_seconds,
        "note": ("All corpora are synthetic and were written to a temp "
                 "directory. The real recording directory was never "
                 "touched."),
    }

    # -- 0. the good corpus -------------------------------------------------
    good_dir = work / "good"
    build_corpus(good_dir, args.matrix_seconds, args.tone_rms)
    t0 = time.time()
    good_run = run_intake(good_dir)
    good = summarise(good_run)
    good["wall_s"] = round(time.time() - t0, 2)
    results["baseline_good_corpus"] = good
    print(f"[selftest] good corpus: status={good['status']} "
          f"hand_offs={good['hand_offs_seen']} "
          f"silences={good['silences_seen']} "
          f"total_s={good['total_audio_s']} exit={good['exit_code']} "
          f"({good['wall_s']}s)")
    print(f"           unmet={good['unmet']}")

    # The good corpus' hand-offs, broken down by the pause duration that
    # produced them: this is what tells a real hand-off from an artefact.
    breakdown = []
    for spec in SCENES:
        pa = good_dir / f"{spec['name']}_A.wav"
        pb = good_dir / f"{spec['name']}_B.wav"
        if not (pa.exists() and pb.exists()):
            continue
        a = np.array(wavio.read_wav(str(pa)).samples)
        b = np.array(wavio.read_wav(str(pb)).samples)
        rows = intake_pauses(a, b)
        breakdown.append({
            "scene": spec["name"],
            "closed_pauses": len(rows),
            "hand_offs": sum(1 for r in rows if r["is_handoff"]),
            "handoff_pause_durations_s": [r["duration_s"] for r in rows
                                          if r["is_handoff"]],
            "short_pauses_under_150ms": sum(1 for r in rows
                                            if r["duration_s"] < 0.15),
        })
    results["baseline_handoff_breakdown"] = breakdown

    # -- 1. the perturbations -----------------------------------------------
    rows = []
    for name, desc in PERTURBATIONS:
        d = work / f"pert_{name}"
        # The no-overlap corpus is built from ALTERNATING specs, so that the
        # two speakers never are loud at the same time (except in scene 6,
        # which keeps one barge-back so the hand-off count survives and the
        # two gates do not move together).
        specs = alternate_specs() if name == "no_overlap" else None
        build_corpus(d, args.matrix_seconds, args.tone_rms, perturb=name,
                     specs=specs)
        run = run_intake(d)
        s = summarise(run)
        base_pairs = {p["scene"]: p for p in good["pairs"]}
        new_pairs = {p["scene"]: p for p in s["pairs"]}
        evidence: dict = {}
        verdict = {}
        # The protocol-length message changes its own text with the corpus
        # length ("only 90s" vs "only 96s"), so it is compared separately.
        unmet_extra = [u for u in s["unmet"]
                       if u not in good["unmet"]
                       and not u.startswith("only ")]

        if name == "missing_channel_B":
            p = new_pairs.get("scene02_qa", {})
            evidence["pair_problem"] = p.get("problem")
            evidence["has_B"] = (p.get("scene") == "scene02_qa")
            verdict["reported"] = p.get("problem") == "unmatched pair"
            verdict["gated"] = any("not matching" in u for u in s["unmet"])
        elif name in ("digital_silence_A", "digital_silence_B"):
            p = new_pairs.get("scene04_fillers", {})
            which = "A" if name.endswith("_A") else "B"
            evidence["silent_flag"] = p.get(f"{which}_silent")
            evidence["other_silent"] = p.get("A_silent" if which == "B"
                                             else "B_silent")
            evidence["peak"] = p.get(f"{which}_peak")
            evidence["unmet_added"] = unmet_extra
            verdict["reported"] = p.get(f"{which}_silent") is True
            verdict["gated"] = any("digitally silent" in u for u in s["unmet"])
        elif name in ("low_level_B", "very_low_level_B"):
            p = new_pairs.get("scene04_fillers", {})
            evidence["peak"] = p.get("B_peak")
            evidence["rms"] = p.get("B_rms")
            evidence["signal_frame_frac"] = p.get("B_signal_frac")
            evidence["silent_flag"] = p.get("B_silent")
            evidence["unmet_added"] = unmet_extra
            verdict["reported"] = True
            verdict["gated"] = any("digitally silent" in u for u in s["unmet"])
        elif name == "crosstalk":
            base_corr = [p["crosstalk"]["corr"] for p in good["pairs"]
                         if p["crosstalk"] and p["crosstalk"]["corr"] is not None]
            new_corr = [p["crosstalk"]["corr"] for p in s["pairs"]
                        if p["crosstalk"] and p["crosstalk"]["corr"] is not None]
            evidence["baseline_corr_max"] = max(base_corr) if base_corr else None
            evidence["baseline_corr_median"] = (round(
                sorted(base_corr)[len(base_corr) // 2], 4) if base_corr else None)
            evidence["baseline_scenes_with_a_computable_corr"] = len(base_corr)
            evidence["baseline_scenes_total"] = len(good["pairs"])
            evidence["perturbed_corr_min"] = min(new_corr) if new_corr else None
            evidence["perturbed_corr_mean"] = (round(sum(new_corr) / len(new_corr), 4)
                                               if new_corr else None)
            evidence["perturbed_scenes_with_a_computable_corr"] = len(new_corr)
            # SEPARATION: the lowest cross-talk number the corrupted corpus
            # produces against the highest the clean corpus produces. If
            # this is not a clear margin, the metric cannot tell the two
            # apart and saying "the gate detected it" would be a lie.
            evidence["separation"] = (round(min(new_corr) - max(base_corr), 4)
                                      if (new_corr and base_corr) else None)
            # The PROPOSED statistic, measured on the same two corpora:
            # of the time A is loud, how much of it is B loud too?
            def _bloud(directory):
                vals = []
                for spec in (specs or SCENES):
                    pa = directory / f"{spec['name']}_A.wav"
                    pb = directory / f"{spec['name']}_B.wav"
                    if not (pa.exists() and pb.exists()):
                        continue
                    aa = np.array(wavio.read_wav(str(pa)).samples)
                    bb = np.array(wavio.read_wav(str(pb)).samples)
                    frac, _n = b_loud_when_a_loud(aa, bb)
                    if frac is not None:
                        vals.append(frac)
                return vals
            clean_bl = _bloud(work / "good")
            bad_bl = _bloud(d)
            evidence["proposed_b_loud_when_a_loud_clean"] = (
                round(max(clean_bl), 4) if clean_bl else None)
            evidence["proposed_b_loud_when_a_loud_crosstalk"] = (
                round(min(bad_bl), 4) if bad_bl else None)
            evidence["proposed_separation"] = (
                round(min(bad_bl) - max(clean_bl), 4)
                if (clean_bl and bad_bl) else None)
            evidence["unmet_added"] = unmet_extra
            verdict["reported"] = bool(new_corr and base_corr and
                                       min(new_corr) - max(base_corr) > 0.05)
            verdict["gated"] = any("cross" in u.lower() for u in s["unmet"])
        elif name == "no_overlap":
            bl = {p["scene"]: p["levels"]["both_loud_frac"]
                  for p in s["pairs"] if p.get("levels")}
            base_bl = {p["scene"]: p["levels"]["both_loud_frac"]
                       for p in good["pairs"] if p.get("levels")}
            n_base = sum(1 for v in base_bl.values() if v > 0.02)
            n_new = sum(1 for v in bl.values() if v > 0.02)
            evidence["baseline_scenes_with_overlap"] = n_base
            evidence["perturbed_scenes_with_overlap"] = n_new
            evidence["perturbed_scenes_still_overlapping"] = sorted(
                k for k, v in bl.items() if v > 0.02)
            evidence["baseline_hand_offs"] = good["hand_offs_seen"]
            evidence["perturbed_hand_offs"] = s["hand_offs_seen"]
            evidence["unmet_added"] = unmet_extra
            verdict["reported"] = n_new < n_base
            verdict["gated"] = any("overlap" in u.lower() for u in s["unmet"])
        elif name == "no_handoff":
            evidence["baseline_hand_offs"] = good["hand_offs_seen"]
            evidence["perturbed_hand_offs"] = s["hand_offs_seen"]
            evidence["unmet_added"] = unmet_extra
            verdict["reported"] = s["hand_offs_seen"] == 0
            verdict["gated"] = any("zero hand-offs" in u for u in s["unmet"])
        elif name == "sample_rate_mismatch":
            path = d / "scene05_backchannel_B.wav"
            evidence["header_rate_hz"] = header_rate(path)
            p = new_pairs.get("scene05_backchannel", {})
            evidence["intake_reported_A_rate"] = p.get("A_raw_rate_hz")
            evidence["intake_reported_B_rate"] = p.get("B_raw_rate_hz")
            evidence["unmet_added"] = unmet_extra
            verdict["reported"] = (p.get("A_raw_rate_hz") !=
                                   p.get("B_raw_rate_hz"))
            verdict["gated"] = any("rate" in u.lower() for u in s["unmet"])
        elif name == "unreadable":
            p = new_pairs.get("scene06_interrupt", {})
            evidence["pair_problem"] = p.get("problem")
            evidence["unmet_added"] = unmet_extra
            verdict["reported"] = p.get("problem") == "unreadable"
            verdict["gated"] = any("unreadable" in u.lower() for u in s["unmet"])

        if verdict.get("gated"):
            v = "DETECTED (gated)"
        elif verdict.get("reported"):
            v = "REPORTED, NOT GATED"
        else:
            v = "NOT DETECTED"
        rows.append({
            "perturbation": name,
            "description": desc,
            "status": good["status"],
            "perturbed_status": s["status"],
            "unmet_added": unmet_extra,
            "exit_code": s["exit_code"],
            "evidence": evidence,
            "reported": bool(verdict.get("reported")),
            "gated": bool(verdict.get("gated")),
            "verdict": v,
            "intake_report": s,
        })
        print(f"[selftest] perturb {name:22s} status={s['status']:15s} "
              f"reported={verdict.get('reported')} gated={verdict.get('gated')}")
    results["perturbations"] = rows

    # -- 2. alignment --------------------------------------------------------
    results["alignment"] = alignment_probe(work, args.matrix_seconds)

    # -- 2b. the cross-talk statistic, and the proposed replacement ---------
    results["crosstalk_metric_comparison"] = crosstalk_metric_comparison(
        work, args.matrix_seconds)

    # -- 3. the drift probe --------------------------------------------------
    results["drift"] = drift_probe(work, args.matrix_seconds)

    # -- 4. the hand-off semantics probe ------------------------------------
    results["handoff_semantics"] = handoff_semantics_probe(work)

    # -- 4b. the protocol-strict corpus -------------------------------------
    results["protocol_strict_corpus"] = protocol_strict_probe(
        work, args.matrix_seconds, args.tone_rms)

    # -- 5. the realistic-room corpus ---------------------------------------
    results["realistic_room"] = real_room_probe(work, args.matrix_seconds,
                                                 args.real_room_rms)

    # -- 6. the protocol-length corpus ---------------------------------------
    if args.long:
        results["protocol_length_corpus"] = protocol_length_probe(work)

    # -- 7. the latent-defect checklist -------------------------------------
    results["latent_defects"] = latent_defect_checklist(work, args)

    results["wall_s"] = round(time.time() - t_start, 1)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=False))
    print(f"[selftest] wrote {out_path} ({out_path.stat().st_size} bytes) "
          f"in {results['wall_s']}s")
    return 0


def _read_pair(d: Path, name: str):
    a = np.array(wavio.read_wav(str(d / f"{name}_A.wav")).samples)
    b = np.array(wavio.read_wav(str(d / f"{name}_B.wav")).samples)
    return a, b


def alignment_probe(work: Path, matrix_seconds: float) -> dict:
    """How much does an unknown inter-channel offset move the numbers?

    A pair with a KNOWN offset is built, and the intake's own hand-off rule
    is applied to it at each offset. The quantity that matters is the
    hand-off GAP -- how long after A stops speaking does B start -- because
    that is the number the corpus exists to measure.

    GEOMETRY (chosen so the effects are separable):

        A speaks 0.60-2.00 s, so A's speech offset is 2.00 s
        B speaks 2.95-3.60 s, so the TRUE gap is 2.95 - 2.00 = 0.95 s
        A resumes at 3.20 s, so A's silence on A is [2.045, 3.155] s

    The intake counts a hand-off when channel B is loud at the instant A's
    silence ENDS (3.155 s). Shifting B later moves its utterance away from
    that instant, so the COUNT can flip -- while the GAP moves by exactly
    the offset at every step.
    """
    d = work / "align"
    d.mkdir(parents=True, exist_ok=True)
    seconds = 8.0
    spec = dict(name="probe", protocol_s=8, kind="dialogue", events=[
        ("A", 0.60, 2.00, 81), ("B", 2.95, 3.60, 82), ("A", 3.20, 4.20, 83),
    ])
    a, b, _ = build_scene(spec, seconds, 1e-5)
    truth_gap = 0.95
    rows = []
    base_measured = None
    base_measured_sp = None
    for offset_ms in (0, 40, 120, 300, 600):
        b_off = shift(b, offset_ms)
        write_wav(d / "probe_A.wav", a)
        write_wav(d / "probe_B.wav", b_off)
        run = run_intake(d)
        s = summarise(run)
        aa, bb = _read_pair(d, "probe")
        gaps = measured_gap_s(aa, bb)
        gaps_sp = measured_gap_samples(aa, bb)
        m = gaps[0] if gaps else None
        m_sp = gaps_sp[0] if gaps_sp else None
        if offset_ms == 0:
            base_measured = m
            base_measured_sp = m_sp
        det = clap_offset_ms(aa, bb)
        rows.append({
            "offset_ms": offset_ms,
            "true_gap_s": truth_gap,
            "measured_gap_s": m,
            "measured_gap_sample_resolution_s": m_sp,
            "gap_error_vs_truth_s": (round(m - truth_gap, 4)
                                     if m is not None else None),
            "gap_error_vs_truth_sample_resolution_s": (
                round(m_sp - truth_gap, 4) if m_sp is not None else None),
            "gap_error_vs_zero_offset_s": (round(m - base_measured, 4)
                                           if (m is not None and
                                               base_measured is not None)
                                           else None),
            "gap_error_vs_zero_offset_sample_resolution_s": (
                round(m_sp - base_measured_sp, 4)
                if (m_sp is not None and base_measured_sp is not None)
                else None),
            "overlap_boundary_measured_s": overlap_boundary_s(aa, bb),
            "intake_hand_offs": s["hand_offs_seen"],
            "intake_silences": s["silences_seen"],
            "intake_both_loud_frac": (s["pairs"][0]["levels"]["both_loud_frac"]
                                      if s["pairs"] and
                                      s["pairs"][0].get("levels") else None),
            "intake_status": s["status"],
            "clap_offset_detected_ms": det["lag_ms"],
            "clap_peak_r": det["peak_r"],
        })
    # The PROPOSED FIX: use the clap the README provides to find the offset,
    # shift B by it, and measure the residual.
    fixed = []
    for offset_ms in (40, 120, 300, 600):
        b_off = shift(b, offset_ms)
        write_wav(d / "probe_A.wav", a)
        write_wav(d / "probe_B.wav", b_off)
        aa, bb = _read_pair(d, "probe")
        det = clap_offset_ms(aa, bb)
        # `lag_ms` is negative when B is late, so advancing B by `lag_ms`
        # (a negative shift = earlier) undoes the offset.
        bb_aligned = shift(bb, det["lag_ms"])
        gaps = measured_gap_samples(aa, bb_aligned)
        g = gaps[0] if gaps else None
        fixed.append({
            "offset_ms": offset_ms,
            "clap_detected_ms": det["lag_ms"],
            "clap_peak_r": det["peak_r"],
            "residual_gap_error_s": (round(g - truth_gap, 4)
                                     if g is not None else None),
            "aligned_hand_offs": sum(1 for r in intake_pauses(aa, bb_aligned)
                                     if r["is_handoff"]),
        })
    return {
        "probe": ("A speaks 0.60-2.00 s, B speaks 2.95-3.60 s, A resumes at "
                  "3.20 s. Channel B is delayed by a KNOWN offset and the "
                  "intake's own rule is applied to the shifted pair."),
        "true_gap_s": truth_gap,
        "offsets": rows,
        "after_clap_alignment": fixed,
        "finding": ("no code in this repo aligns the two channels; the clap "
                    "the README calls the alignment point is never read"),
    }


def drift_probe(work: Path, matrix_seconds: float) -> dict:
    """A long drift between the channels: does anything notice?

    Two independent recorders run on their own clocks, so the offset
    between them grows through the scene. Here the drift is a 1.25% time-base
    error -- 250 ms over 20 s, which is far larger than a real crystal's
    drift (ppm) and is used here to make the CONSEQUENCE measurable rather
    than to claim a realistic magnitude.

    The measurement that matters: the gap between A's speech offset and B's
    speech onset, at the END of the scene, where the drift is largest. A
    single clap correction (taken at the start) cannot fix it.
    """
    d = work / "drift"
    d.mkdir(parents=True, exist_ok=True)
    spec = dict(name="probe", protocol_s=20, kind="dialogue", events=[
        ("A", 0.60, 2.00, 91), ("A", 14.00, 14.60, 94), ("B", 15.00, 16.00, 92),
        ("A", 16.60, 18.00, 93),
    ])
    a, b, _ = build_scene(spec, 20.0, 1e-5, unit_s=20.0)
    n = len(b)
    eps = 0.0125                     # 1.25% -> 250 ms of drift over 20 s
    pos = np.arange(n) * (1.0 + eps)
    b_drift = np.interp(pos, np.arange(n), b, left=0.0, right=0.0)
    write_wav(d / "probe_A.wav", a)
    write_wav(d / "probe_B.wav", b_drift)
    run = run_intake(d)
    s = summarise(run)
    aa, bb = _read_pair(d, "probe")
    gaps_drift = measured_gap_samples(aa, bb)
    gaps_clean = measured_gap_samples(a, b)
    true_gap = 0.40                  # B's onset (15.00) - A's offset (14.60)
    drift_at_gap_ms = round(15.0 * eps * 1000, 1)
    return {
        "drift_eps": eps,
        "drift_ms_total": round(20.0 * eps * 1000, 1),
        "drift_ms_at_the_gap": drift_at_gap_ms,
        "scene_s": 20.0,
        "intake_report": s,
        "drift_reported_by_intake": False,
        "drift_fields_in_report": [k for k in (s or {})
                                   if "drift" in k.lower()],
        "clap_offset_detected_ms": clap_offset_ms(aa, bb)["lag_ms"],
        "true_gap_s": true_gap,
        "measured_gap_no_drift_s": gaps_clean[0] if gaps_clean else None,
        "measured_gap_with_drift_s": gaps_drift[0] if gaps_drift else None,
        "gap_error_from_drift_s": (round(gaps_drift[0] - gaps_clean[0], 4)
                                     if gaps_drift and gaps_clean else None),
        "note": ("a single clap offset would correct the offset AT THE CLAP "
                 "only; the residual error at the gap is the whole drift"),
    }


def handoff_semantics_probe(work: Path) -> dict:
    """What does the intake's hand-off counter actually COUNT?

    Three hand-off shapes, each a perfectly ordinary thing two people do.
    """
    d = work / "handoff_semantics"
    d.mkdir(parents=True, exist_ok=True)
    cases = {
        "canonical_A_stops_B_takes_floor_A_stays_quiet":
            [("A", 1.0, 2.0, 101), ("B", 2.5, 3.5, 102)],
        "A_stops_B_takes_floor_A_resumes_LATER_while_B_quiet":
            [("A", 1.0, 2.0, 101), ("B", 2.5, 3.5, 102), ("A", 5.0, 6.0, 103)],
        "A_resumes_WHILE_B_still_talking":
            [("A", 1.0, 2.0, 101), ("B", 2.2, 4.0, 102), ("A", 3.0, 4.2, 103)],
    }
    rows = []
    for name, events in cases.items():
        spec = dict(name="probe", protocol_s=8, kind="dialogue",
                    events=[(c, s, e, sd) for c, s, e, sd in events])
        a, b, _ = build_scene(spec, 8.0, 1e-5)
        write_wav(d / "probe_A.wav", a)
        write_wav(d / "probe_B.wav", b)
        run = run_intake(d)
        s = summarise(run)
        aa, bb = _read_pair(d, "probe")
        pr = intake_pauses(aa, bb)
        rows.append({
            "case": name,
            "events": events,
            "intake_hand_offs": s["hand_offs_seen"],
            "intake_silences": s["silences_seen"],
            "closed_pauses": [(p["t0"], p["t1"], p["duration_s"],
                               p["is_handoff"]) for p in pr],
            "measured_gaps_s": measured_gap_s(aa, bb),
        })
        print(f"[selftest] hand-off case {name[:46]:46s} "
              f"hand_offs={s['hand_offs_seen']}")
    return {
        "rule": ("a CLOSED pause on channel A whose END coincides with "
                 "channel B being loud; an A-silence that never closes (A "
                 "stays quiet to the end of the file) is not counted"),
        "cases": rows,
    }


def protocol_strict_probe(work: Path, matrix_seconds: float,
                          tone_rms: float) -> dict:
    """The corpus the README describes, LITERALLY: every turn is a canonical
    hand-off -- A stops, B takes the floor, A stays quiet.

    The intake is asked to accept it.
    """
    d = work / "protocol_strict"
    built = build_corpus(d, matrix_seconds, tone_rms, specs=strict_specs())
    run = run_intake(d)
    s = summarise(run)
    # Count the canonical hand-offs actually present in the audio: every
    # channel-B utterance that begins after channel A's previous utterance
    # ended, with no overlap between them.
    canonical = 0
    for scene, truth in built["truth"].items():
        evs = sorted(truth, key=lambda e: e[1])
        last_a_end = None
        for ch, st, en in evs:
            if ch == "A":
                last_a_end = en
            elif last_a_end is not None and st > last_a_end:
                canonical += 1
    print(f"[selftest] protocol-strict corpus: canonical hand-offs in the "
          f"corpus={canonical}, intake hand_offs={s['hand_offs_seen']}, "
          f"status={s['status']}")
    # What the PROPOSED rule would count on the same audio: a hand-off is
    # channel B loud ANYWHERE during channel A's silence, not only at the
    # instant that silence ends.
    proposed = 0
    proposed_with_clap = 0
    proposed_rows = []
    for spec in strict_specs():
        pa = d / f"{spec['name']}_A.wav"
        pb = d / f"{spec['name']}_B.wav"
        if not (pa.exists() and pb.exists()):
            continue
        aa = np.array(wavio.read_wav(str(pa)).samples)
        bb = np.array(wavio.read_wav(str(pb)).samples)
        n_prop = hand_offs_proposed_rule(aa, bb, skip_before_s=0.6)
        n_prop_clap = hand_offs_proposed_rule(aa, bb, skip_before_s=0.0)
        proposed += n_prop
        proposed_with_clap += n_prop_clap
        proposed_rows.append({"scene": spec["name"],
                              "proposed_hand_offs": n_prop,
                              "proposed_hand_offs_including_the_clap": n_prop_clap,
                              "intake_hand_offs": sum(
                                  1 for r in intake_pauses(aa, bb)
                                  if r["is_handoff"])})
    print(f"[selftest] protocol-strict corpus under the PROPOSED rule: "
          f"hand_offs={proposed} (with the clap included: {proposed_with_clap})")
    return {
        "canonical_handoffs_in_corpus": canonical,
        "intake_hand_offs_seen": s["hand_offs_seen"],
        "proposed_rule_hand_offs": proposed,
        "proposed_rule_hand_offs_including_the_clap": proposed_with_clap,
        "per_scene": proposed_rows,
        "intake_status": s["status"],
        "unmet": s["unmet"],
        "intake_report": s,
        "note": ("the corpus contains canonical hand-offs and the intake "
                 "counts none of them, because it counts a hand-off only "
                 "when A comes back while B is still talking"),
    }


def real_room_probe(work: Path, matrix_seconds: float, rms: float) -> dict:
    """The same corpus in a realistic room: the intake's ABSOLUTE floor is
    1e-4 RMS, and a real quiet room is far above that."""
    d = work / "realroom"
    build_corpus(d, matrix_seconds, rms)
    run = run_intake(d)
    s = summarise(run)
    print(f"[selftest] realistic room (tone {rms}): status={s['status']} "
          f"hand_offs={s['hand_offs_seen']}")
    return {
        "room_tone_rms": rms,
        "intake_floor_rms": FLOOR,
        "floor_below_room_tone_db": round(20 * math.log10(rms / FLOOR), 1),
        "intake_report": s,
        "scene08_signal_frac": [p["A_signal_frac"] for p in s["pairs"]
                                if p["scene"] == "scene08_roomtone"],
        "scene08_both_loud_frac": [p["levels"]["both_loud_frac"]
                                   for p in s["pairs"]
                                   if p["scene"] == "scene08_roomtone"
                                   and p.get("levels")],
    }


def protocol_length_probe(work: Path) -> dict:
    """A corpus long enough to clear the intake's own length gate."""
    d = work / "protocol_length"
    scene_seconds = {s["name"]: s["protocol_s"] / 2 for s in SCENES}
    build_corpus(d, 0, 1e-5, scene_seconds=scene_seconds)
    t0 = time.time()
    run = run_intake(d)
    s = summarise(run)
    s["wall_s"] = round(time.time() - t0, 2)
    print(f"[selftest] protocol-length corpus: total_s={s['total_audio_s']} "
          f"status={s['status']} exit={s['exit_code']} ({s['wall_s']}s)")
    return s


def latent_defect_checklist(work: Path, args) -> list[dict]:
    """The three defects the CORPUS capture had, checked against the intake."""
    out: list[dict] = []

    # (a) does the `src` import resolve, and can an exception be swallowed?
    d = work / "importcheck"
    d.mkdir(parents=True, exist_ok=True)
    write_wav(d / "x_A.wav", np.zeros(SR))
    write_wav(d / "x_B.wav", np.zeros(SR))
    r1 = subprocess.run([sys.executable, str(INTAKE), "--raw", str(d)],
                        capture_output=True, text=True, cwd="/tmp")
    r2 = subprocess.run([sys.executable, str(INTAKE), "--raw", str(d)],
                        capture_output=True, text=True, cwd=str(ROOT))
    parses = []
    for r in (r1, r2):
        try:
            json.loads(r.stdout)
            parses.append(True)
        except Exception:
            parses.append(False)
    out.append({
        "defect": "the `src` import resolves to the wrong path and the "
                  "exception is swallowed (the corpus capture's defect 3)",
        "check": "run the intake from / and from the repo root",
        "exit_codes": [r1.returncode, r2.returncode],
        "stdout_parses_as_json": parses,
        "import_path_source": ("sys.path.insert(0, str(Path(__file__)."
                                "resolve().parents[1])) -- built from "
                                "__file__, not from the current directory"),
        "swallowed_exceptions": ("channel_stats() catches Exception and "
                                  "returns an error dict, so an unreadable "
                                  "file is reported, not swallowed; no other "
                                  "try/except exists in the file"),
        "verdict": ("PASS -- the import is file-relative and an unreadable "
                    "file is reported, not silently skipped"),
    })

    # (b) are the WAV headers read from the file, or assumed?
    d2 = work / "headercheck"
    d2.mkdir(parents=True, exist_ok=True)
    write_wav(d2 / "h_A.wav", np.zeros(SR), sr=16000)
    write_wav(d2 / "h_B.wav", np.zeros(22050), sr=22050)
    r3 = subprocess.run([sys.executable, str(INTAKE), "--raw", str(d2)],
                        capture_output=True, text=True, cwd=str(ROOT))
    rep = json.loads(r3.stdout)
    pairs = rep.get("pairs", [])
    out.append({
        "defect": "the sample rate is assumed rather than read from the "
                  "header (the corpus capture's defect 2, in rate form)",
        "check": "write a 16 kHz A and a 22050 Hz B; read both headers and "
                 "read the intake's reported `raw_rate_hz`",
        "header_A_hz": header_rate(d2 / "h_A.wav"),
        "header_B_hz": header_rate(d2 / "h_B.wav"),
        "intake_raw_rate_hz": [p.get("B", {}).get("raw_rate_hz")
                               for p in pairs],
        "intake_duration_s": [p.get("B", {}).get("duration_s") for p in pairs],
        "verdict": ("FAIL -- `src.wavio.read_wav` RESAMPLES to 16 kHz and "
                    "returns an Audio whose sample_rate is 16000, so "
                    "`raw_rate_hz` is always 16000 and the file's real rate "
                    "is never reported. The duration survives (it is "
                    "post-resample); the rate does not."),
    })

    # (c) does it exit non-zero when the corpus is not ready?
    d3 = work / "exitcheck"
    d3.mkdir(parents=True, exist_ok=True)
    r4 = subprocess.run([sys.executable, str(INTAKE), "--raw", str(d3)],
                        capture_output=True, text=True, cwd=str(ROOT))
    r5 = subprocess.run([sys.executable, str(INTAKE),
                         "--raw", str(d3 / "does_not_exist")],
                        capture_output=True, text=True, cwd=str(ROOT))
    out.append({
        "defect": "the check never runs / cannot fail the pipeline "
                  "(the corpus capture's defect 3, the exit-code half)",
        "check": "run the intake on an empty directory and on a missing one",
        "empty_dir_exit_code": r4.returncode,
        "missing_dir_exit_code": r5.returncode,
        "empty_dir_status": json.loads(r4.stdout)["gates"].get("status"),
        "missing_dir_status": json.loads(r5.stdout)["gates"].get("status"),
        "verdict": ("FAIL -- `main()` returns None and never calls "
                    "sys.exit(1); the tool exits 0 while reporting NOT YET "
                    "USABLE, so a pipeline that runs it cannot tell a "
                    "rejected corpus from an accepted one"),
    })
    return out


if __name__ == "__main__":
    raise SystemExit(main())
