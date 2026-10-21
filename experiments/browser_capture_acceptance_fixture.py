#!/usr/bin/env python3
"""
browser_capture_acceptance_fixture.py -- build a session from the DERIVED
expectations, to validate the acceptance checker itself.

WHY THIS FILE EXISTS.

`experiments/browser_capture_acceptance.py` is a referee: it is supposed to
say PASS when a session is right and FAIL when it is wrong. A checker that
only ever fails is worthless, and a checker that has been shaped around one
producer's layout is not a referee at all.

So this builds a session from `docs/research/ACCEPTANCE-EXPECTED.md` ALONE --
the reference implementation's own functions, the reference's own schemas, the
addendum's own contents -- and the checker must pass it.

It is deliberately adversarial about the things a checker can get wrong:

  * the change maps and the drift are computed by the REFERENCE's own
    functions (`experiments/corpus_capture_win.py`), so a checker that
    re-derives them must agree;
  * `speech_frac` is MEASURED with the reference's own Silero gate, so an
    independent re-measurement must agree;
  * one take is labelled speak=True but is silent, so VOICE_MISSING must be
    flagged, and one is labelled speak=False but is loud, so NOT_SILENT must
    be flagged -- the checker must require the flag in BOTH directions;
  * the WebM is a REAL container, with a video track and an audio track, so a
    container parser that only ever sees the truth cannot hide behind it;
  * `session.json` sits AT THE OUT ROOT, exactly as the addendum asks, which
    is what makes the gate individuate it as a take -- so the checker must
    report the gate's verdict on the directory and on the takes separately.

USAGE:

    python experiments/browser_capture_acceptance_fixture.py /tmp/fixture
    python experiments/browser_capture_acceptance.py /tmp/fixture

The fixture is written wherever you point it. Nothing here touches the real
corpus.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import wavio  # noqa: E402

W, H = 160, 96
MAP_W, MAP_H = 80, 48
RATE = 16000
FPS = 30
REPEATS = 3
BLK = RATE // FPS

PLAN = []
for _rep in range(1, REPEATS + 1):
    for _base, _kind, _speak, _pose, _secs in (
            ("A", "condition", True, "facing", 50),
            ("B", "condition", True, "turned", 50),
            ("C", "condition", False, "facing", 30),
            ("D", "condition", False, "turned", 30),
            ("F", "free", None, "any", 60)):
        PLAN.append({"name": f"{_base}{_rep}", "base": _base, "kind": _kind,
                     "speak": _speak, "pose": _pose, "seconds": _secs,
                     "rep": _rep})
PLAN.append({"name": "FREE", "base": "FREE", "kind": "free", "speak": None,
             "pose": "any", "seconds": 300, "rep": 1})


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


REF = _load("_fix_ref", ROOT / "experiments" / "corpus_capture_win.py")


# -- the speech measurement, with the reference's own gate -------------------

def _silero():
    import os
    for cand in (Path(os.environ.get("SILERO_VAD_MODEL", "")),
                 ROOT / "models" / "silero_vad.onnx"):
        if cand.is_file():
            os.environ["SILERO_VAD_MODEL"] = str(cand)
            break
    base = ROOT.name.split("-")[0]
    try:
        for sib in ROOT.parent.iterdir():
            cand = sib / "models" / "silero_vad.onnx"
            if sib.is_dir() and sib.name.startswith(base) and cand.is_file():
                os.environ["SILERO_VAD_MODEL"] = str(cand)
                break
    except OSError:
        pass
    from src.vad import SileroGate
    return SileroGate()


def measure_speech(wav_path: Path) -> float:
    audio = wavio.read_wav(str(wav_path))
    probs = _silero().push(list(audio.samples))
    if not probs:
        return 0.0
    return sum(1 for v in probs if v >= 0.5) / len(probs)


# -- the synthetic video and audio -------------------------------------------

def synth_luma(n: int, seed: int) -> np.ndarray:
    """n frames of 160x96 luma with real frame-to-frame motion."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    frames = np.empty((n, H, W), dtype=np.uint8)
    for t in range(n):
        f = base.astype(np.float64) + rng.normal(0, 1.2, size=(H, W))
        bx = (t * 5) % (W - 20)
        f[30:50, bx:bx + 20] = 210
        frames[t] = np.clip(f, 0, 255).astype(np.uint8)
    return frames


REAL_SPEECH = (ROOT / "examples" / "realaudio" / "corpus" /
               "poetry_weeds_by_the_wall_1702_librivox.wav")

_SPEECH_PCM: list[int] | None = None


def _real_speech_pcm() -> list[int]:
    """Real speech from the repo's own corpus, as int16 samples at 16 kHz.

    A synthesised tone is NOT speech to the Silero gate, and the reference's
    gate is what computes `speech_frac`. So a take that claims to be speaking
    has to hold real speech, or the fixture would be asserting a speech
    fraction that the gate -- and therefore the pipeline -- would not
    reproduce.
    """
    global _SPEECH_PCM
    if _SPEECH_PCM is None:
        with wave.open(str(REAL_SPEECH), "rb") as w:
            raw = w.readframes(w.getnframes())
        vals = struct.unpack("<%dh" % (len(raw) // 2), raw)
        _SPEECH_PCM = list(vals)
    return _SPEECH_PCM


def synth_wav(n_samples: int, mode: str) -> bytes:
    """A mono 16-bit WAV: real speech, a loud tone, or near-silence."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        frames = bytearray()
        if mode == "speech":
            speech = _real_speech_pcm()
            n = len(speech)
            for i in range(n_samples):
                frames += struct.pack("<h", speech[i % n])
        else:
            for i in range(n_samples):
                if mode == "loud":
                    v = int(math.sin(2 * math.pi * 220 * i / RATE) * 20000)
                else:  # near-silence: a whisper of noise
                    v = int((i % 7) - 3)
                frames += struct.pack("<h", max(-32768, min(32767, v)))
        f.writeframes(bytes(frames))
    return buf.getvalue()


# -- a REAL WebM, with a video track and an audio track ---------------------
#
# The addendum asks for a WebM "WITH the mic audio track muxed", so the
# fixture builds a container that actually has both, and the checker's
# container parser has to find them.

def _vint_size(n: int) -> bytes:
    for length in range(1, 9):
        if n < (1 << (7 * length)) - 1:
            return (n | (1 << (7 * length))).to_bytes(length, "big")
    raise ValueError("too large")


def _elem(eid: bytes, payload: bytes) -> bytes:
    return eid + _vint_size(len(payload)) + payload


def _uint(eid: bytes, value: int) -> bytes:
    length = max(1, (value.bit_length() + 7) // 8)
    return _elem(eid, value.to_bytes(length, "big"))


def _float(eid: bytes, value: float) -> bytes:
    return _elem(eid, struct.pack("<d", value))


def build_webm(width: int = 640, height: int = 480) -> bytes:
    ebml = _elem(b"\x1a\x45\xdf\xa3", b"".join((
        _uint(b"\x42\x86", 1),          # EBMLVersion
        _uint(b"\x42\xf7", 1),          # EBMLReadVersion
        _uint(b"\x42\xf2", 4),          # EBMLMaxIDLength
        _uint(b"\x42\xf3", 8),          # EBMLMaxSizeLength
        _elem(b"\x42\x82", b"webm"),    # DocType
        _uint(b"\x42\x87", 2),          # DocTypeVersion
        _uint(b"\x42\x85", 2),          # DocTypeReadVersion
    )))
    video = _elem(b"\xe0", b"".join((
        _uint(b"\xb0", width),          # PixelWidth
        _uint(b"\xba", height),         # PixelHeight
    )))
    audio = _elem(b"\xe1", b"".join((
        _float(b"\xb5", 48000.0),       # SamplingFrequency
        _uint(b"\x9f", 2),              # Channels
    )))
    track_video = _elem(b"\xae", b"".join((
        _uint(b"\xd7", 1),              # TrackNumber
        _uint(b"\x73\xc5", 1),          # TrackUID
        _uint(b"\x83", 1),              # TrackType: video
        _elem(b"\x86", b"V_VP8"),       # CodecID
        video,
    )))
    track_audio = _elem(b"\xae", b"".join((
        _uint(b"\xd7", 2),
        _uint(b"\x73\xc5", 2),
        _uint(b"\x83", 2),              # TrackType: audio
        _elem(b"\x86", b"A_OPUS"),      # CodecID
        audio,
    )))
    tracks = _elem(b"\x16\x54\xae\x6b", track_video + track_audio)
    block = _elem(b"\xa3", b"\x81\x00\x00\x80" + b"\x00" * 16)
    cluster = _elem(b"\x1f\x43\xb6\x75", block)
    segment = _elem(b"\x18\x53\x80\x67", tracks + cluster)
    return ebml + segment


# -- the session --------------------------------------------------------------

def build(out: Path, seed: int = 7) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rich_dir = out / "rich"
    rich_dir.mkdir(exist_ok=True)
    entries = []
    for i, spec in enumerate(PLAN):
        name = spec["name"]
        # A short take per condition: the fixture checks the FORMAT, not the
        # duration. 90 frames is 3 s at 30 fps.
        n = 90 if spec["name"] != "FREE" else 150
        stamp = 1758000000 + i * 37
        stem = f"{name}_{stamp}"

        # -- the adversarial cases ---------------------------------------
        # A2 is labelled speak=True but is silent: VOICE_MISSING must be
        # flagged. C2 is labelled speak=False but is loud: NOT_SILENT must
        # be flagged.
        if name == "A2":
            mode, want_flag = "silent", "VOICE_MISSING"
        elif name == "C2":
            mode, want_flag = "loud", "NOT_SILENT"
        elif spec["speak"] is True:
            mode, want_flag = "speech", None
        else:
            mode, want_flag = "silent", None

        luma = synth_luma(n, seed + i)
        Yf = luma.astype(np.float64)
        Yr = REF.row_band_remove(Yf)
        maps = REF.change_maps(Yr)
        q, scale = REF.quantise(maps)
        maps_raw = REF.change_maps(Yf)
        q_raw, scale_raw = REF.quantise(maps_raw)
        drift_raw = float(np.abs(np.diff(Yf, axis=0)).mean())
        drift_row = float(np.abs(np.diff(Yr, axis=0)).mean())

        n_samples = n * BLK
        wav_bytes = synth_wav(n_samples, mode)
        (out / f"{stem}.wav").write_bytes(wav_bytes)
        speech_frac = measure_speech(out / f"{stem}.wav")

        np.savez_compressed(out / f"{stem}.npz", maps=q,
                            scale=np.float32(scale), maps_raw=q_raw,
                            scale_raw=np.float32(scale_raw), luma=luma)

        timing = {
            "condition": name, "n_frames": n,
            "frame_offsets_s": [round(i / FPS, 6) for i in range(n)],
            "assumed_fps": FPS, "achieved_fps": 30.0,
            "video_span_s": round(n / FPS, 3),
            "audio_span_s": round(n_samples / RATE, 3),
            "audio_sample_rate": RATE, "requested_sample_rate": RATE,
            "mic": "fixture-mic", "capture_wallclock_s": 0.25,
        }
        (out / f"{stem}_timing.json").write_text(json.dumps(timing))

        meta = {
            "condition": name, "kind": spec["kind"],
            "note": f"{spec['pose']} {spec['kind']} take",
            "pose": spec["pose"], "speak": spec["speak"],
            "stimulus": "invictus", "stimulus_title": "Invictus",
            "stimulus_estimated_s": 40.0, "speech_frac": speech_frac,
            "n_frames": n, "seconds": round(n / FPS, 3),
            "achieved_fps": 30.0, "audio_sample_rate": RATE,
            "audio_seconds": round(n_samples / RATE, 3),
            "map_shape": [MAP_H, MAP_W], "map_scale": scale,
            "map_scale_raw": scale_raw, "drift_raw": drift_raw,
            "drift_after_row_removal": drift_row, "timestamp": stamp,
            "rep": spec["rep"], "filter": "row",
            "filters": {"maps": "row", "maps_raw": "none", "luma": "none"},
            "arrays": ["maps", "maps_raw", "luma"],
        }
        (out / f"{stem}.json").write_text(json.dumps(meta, indent=2))

        rich = {
            "condition": name, "instruction_text": f"TAKE {name}",
            "stimulus": "invictus",
            "wall_clock_start": "2026-10-20T12:00:00.000Z",
            "wall_clock_end": "2026-10-20T12:00:05.000Z",
            "achieved_fps": 30.0, "audio_sample_rate": RATE,
            "frame_timestamps_file": f"{stem}_timing.json",
            "camera_device": "fixture-camera", "mic_device": "fixture-mic",
            "browser": "fixture", "notes": "",
        }
        (rich_dir / f"{stem}_rich.json").write_text(json.dumps(rich, indent=2))

        webm_name = f"{stem}_full.webm"
        (out / webm_name).write_bytes(build_webm())

        flags = []
        if want_flag:
            flags.append(want_flag)
        entries.append({
            "label": name, "stamp": stamp, "kind": spec["kind"],
            "pose": spec["pose"], "speak": spec["speak"],
            "instruction_text": rich["instruction_text"],
            "stimulus": "invictus", "flags": flags, "notes": "",
            "files": {
                "npz": f"{stem}.npz", "wav": f"{stem}.wav",
                "timing": f"{stem}_timing.json",
                "rich": f"rich/{stem}_rich.json",
                "full_video": webm_name,
            },
        })
        print(f"  {stem}: {n} frames, speech_frac {speech_frac:.3f}, "
              f"drift {drift_row:.3f}, flags {flags or '-'}")

    # The manifest AT THE OUT ROOT, exactly as the addendum asks. This is
    # what makes the gate individuate it as a take, which is the conflict
    # the checker is expected to report rather than hide.
    session = {
        "manifest": "every take in this session, for external ingest",
        "generated_at": "2026-10-20T12:00:00Z",
        "out": str(out), "n_takes": len(entries), "takes": entries,
    }
    (out / "session.json").write_text(json.dumps(session, indent=2))
    return {"out": str(out), "n_takes": len(entries)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="where to build the fixture session")
    a = ap.parse_args(argv)
    info = build(Path(a.out))
    print(f"built {info['n_takes']} takes in {info['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
