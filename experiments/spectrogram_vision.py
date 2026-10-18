#!/usr/bin/env python
"""
Does Jev 1.14 SEE a spectrogram?

Images are documented as supported on jev-1.14 ("Jev 1.13 and Jev 1.14 both
accept images. Box questions need Jev 1.14."), in three places: the state, the
question instructions, and the question criteria. The SDK carries them as
`Image` objects whose reference string `<image:...>` can be embedded inline
anywhere text goes.

This matters here more than it usually would, because everything upstream of
this point has been text. Jev reads text only -- no audio, no video -- so the
measured acoustic tracks have been reaching it as already-reduced numbers
(PITCH, TEMPO, INFLECTION, SILENCE) rather than as the signal itself. If Jev
can read a spectrogram, the reduction happens somewhere we chose instead of
somewhere the model chose, and we get to find out which features it uses.

The docs publish NO accuracy numbers for images, so there is no baseline. Every
claim below is a prediction made before the run.

  EXP 1 -- pitch discrimination from the image alone, with NO numeric hints in
           the state. Three known tones (180/240/300 Hz). If the model can
           see, accuracy is well above the 1/3 random floor.
  EXP 2 -- the same question with the numeric measurements INCLUDED in the
           state as text. If accuracy does not move, the image is being
           ignored and the text is doing the work.
  EXP 3 -- image ablation, jev-xray style: same question, image replaced by
           blank noise of identical dimensions. The delta is what the image
           contributed.
  EXP 4 -- Box (1.14-only) asking WHERE the harmonic content sits. This is the
           probe for whether the model localises time-frequency structure or
           only reacts holistically.

No Pillow in this environment, so the PNG is written by hand: grayscale,
8-bit, one IDAT. ~25 lines, and it keeps the experiment dependency-free.

    set -a && . /home/proart/projects/glasses/.env && set +a
    /home/proart/projects/glasses/.venv/bin/python experiments/spectrogram_vision.py
"""

from __future__ import annotations

import base64
import json
import math
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = os.environ.get("JEV_MODEL", "jev-1.14.0")
SR = 16000

# Both sides must be 280..1568 px; anything narrower is rejected outright.
W, H = 640, 480


# -- a minimal PNG writer ----------------------------------------------------

def write_png_gray(path: str, arr: np.ndarray) -> None:
    """8-bit grayscale PNG. Filter type 0 on every scanline."""
    h, w = arr.shape
    raw = bytearray()
    for row in arr:
        raw.append(0)                      # filter: None
        raw.extend(row.astype(np.uint8).tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)


# -- spectrogram -------------------------------------------------------------

def spectrogram(samples, n_fft=512, hop=128, fmax=600.0):
    """Magnitude spectrogram in dB, shaped (H, W) for a greyscale image."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = np.stack([x, np.zeros_like(x)], axis=1)   # mono -> (n, 2) rows
    win = np.hanning(n_fft)[:, None]      # broadcast across channels
    frames = 1 + (len(x) - n_fft) // hop
    spec = np.zeros((frames, n_fft // 2 + 1), dtype=np.float64)
    for i in range(frames):
        seg = x[i * hop:i * hop + n_fft]
        spec[i] = np.abs(np.fft.rfft(seg * win, axis=0)).sum(axis=1)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    keep = freqs <= fmax
    spec = spec[:, keep].T
    db = 20.0 * np.log10(np.maximum(spec, 1e-9))
    db -= db.max()
    db = np.clip(db, -60.0, 0.0)
    return ((db + 60.0) / 60.0 * 255.0).astype(np.uint8)


def render(spec: np.ndarray, path: str):
    """Fit a spectrogram into the legal WxH box without distortion."""
    img = np.flipud(spec)                  # low frequencies at the bottom
    ys = (np.linspace(0, img.shape[0] - 1, H)).astype(int)
    xs = (np.linspace(0, img.shape[1] - 1, W)).astype(int)
    out = img[np.ix_(ys, xs)]
    write_png_gray(path, out)
    return out


def tone(hz, seconds, amp=0.3, sr=SR):
    n = int(seconds * sr)
    a = amp * np.sin(2 * math.pi * hz * np.arange(n) / sr)
    return np.stack([a, a], axis=1)


def noise(seconds, rms, sr=SR, seed=3):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    a = rng.normal(0.0, rms, n)
    return np.stack([a, a], axis=1)


IMAGE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}


def client():
    """
    The opt-in header is mandatory and the SDK will not raise if you forget it.

    First run of this experiment produced four clean, plausible, COMPLETELY
    MEANINGLESS results: 1/3 accuracy on pitch-from-image, and an ablation
    delta of +0.0067. The cause was visible only in experiment 4, which 400'd
    with "Images are experimental. Send the header
    TypeSafe-Danger-Zone-Images: true". Without the header the SDK serialises
    the `Image` to its reference string and simply does not attach the bytes,
    so Jev received a literal `<image:1a2b...>` token and answered from the
    text of an empty utterance.

    That is worth recording: the failure mode was not an error, it was a
    plausible number. `preflight` below is the guard against it.
    """
    from typesafe_sdk import TypeSafeClient
    return TypeSafeClient(headers=IMAGE_HEADERS)


def preflight(cli, path):
    """
    PROVE the image is on the wire before believing any answer.

    An attached image changes `usage.input_tokens`; an image that was silently
    dropped does not. If the two match, every number this experiment produces
    is about the absence of an image, and we stop rather than report it.
    """
    from typesafe_sdk import Choice, Image

    def tokens(state):
        r = cli.system_one(state=state,
                           questions={"p": Choice(instructions="Is there an "
                                                            "image here?",
                                                  criteria={"yes": None,
                                                            "no": None})},
                           model=MODEL)
        return r.usage.input_tokens, r.choices["p"].choice

    with_img, _ = tokens(f"a spectrogram: {Image(path)}")
    without, _ = tokens("a spectrogram: (none)")
    print(f"   preflight: input_tokens with image {with_img}, "
          f"without {without} -> "
          f"{'IMAGE IS BEING SENT' if with_img > without else 'IMAGE NOT SENT'}")
    if with_img <= without:
        raise SystemExit("aborting: the image is not reaching the API, so "
                         "every result below would be about its absence")
    return with_img, without


# -- experiments -------------------------------------------------------------

PITCH_OPTIONS = {
    "low": "A low-pitched voice, with its energy concentrated near the "
           "bottom of the frequency range.",
    "mid": "A mid-pitched voice, with its energy around the middle of the "
           "frequency range.",
    "high": "A high-pitched voice, with its energy concentrated near the top "
            "of the frequency range.",
}

PITCH_Q = ("Looking only at the spectrogram, which pitch band is this "
           "utterance in? Judge where the visible energy is concentrated "
           "along the frequency axis.")


def exp1_pitch_from_image(cli, cases):
    """PREDICTION: if the model reads the image at all, accuracy is well above
    the 1/3 random floor."""
    from typesafe_sdk import Choice, Image

    rows = []
    for label, path, truth in cases:
        img = Image(path)
        a = cli.system_one(
            state=f"The spectrogram of one spoken utterance: {img}",
            questions={"p": Choice(instructions=PITCH_Q,
                                   criteria=PITCH_OPTIONS)},
            model=MODEL).answers["p"]
        rows.append({"case": label, "truth": truth, "picked": a.choice,
                     "confidence": round(a.confidence, 4),
                     "probs": {k: round(v, 3)
                               for k, v in a.probabilities.items()}})
    correct = sum(1 for r in rows if r["picked"] == r["truth"])
    return {"rows": rows, "accuracy": f"{correct}/{len(rows)}",
            "random_floor": f"1/{len(PITCH_OPTIONS)}"}


def exp2_pitch_with_numbers(cli, cases, measured):
    """PREDICTION: if the image matters, accuracy is no worse with the numeric
    measurements in the state too. If accuracy is the same as image-only, the
    image is redundant and we are reading our own summary back to ourselves."""
    from typesafe_sdk import Choice, Image

    rows = []
    for label, path, truth in cases:
        img = Image(path)
        a = cli.system_one(
            state=(f"The spectrogram of one spoken utterance: {img}\n"
                   f"Measured pitch: {measured[label]} Hz. "
                   f"Measured duration: 0.50 s."),
            questions={"p": Choice(instructions=PITCH_Q,
                                   criteria=PITCH_OPTIONS)},
            model=MODEL).answers["p"]
        rows.append({"case": label, "truth": truth, "picked": a.choice,
                     "confidence": round(a.confidence, 4)})
    correct = sum(1 for r in rows if r["picked"] == r["truth"])
    return {"rows": rows, "accuracy": f"{correct}/{len(rows)}"}


def exp3_ablation(cli, cases, blank):
    """jev-xray's instrument, on the image channel: swap the spectrogram for a
    blank image of IDENTICAL dimensions and measure how far the answer moved.
    Anything smaller than the movement between two identical requests is noise,
    not an effect."""
    from typesafe_sdk import Choice, Image

    real = {label: cli.system_one(
        state=f"The spectrogram of one spoken utterance: {Image(path)}",
        questions={"p": Choice(instructions=PITCH_Q, criteria=PITCH_OPTIONS)},
        model=MODEL).answers["p"].probabilities for label, path, _ in cases}
    fake = {label: cli.system_one(
        state=f"The spectrogram of one spoken utterance: {Image(blank)}",
        questions={"p": Choice(instructions=PITCH_Q, criteria=PITCH_OPTIONS)},
        model=MODEL).answers["p"].probabilities for label, _, _ in cases}

    rows = []
    for label, path, truth in cases:
        before = real[label][truth]
        after = fake[label][truth]
        rows.append({"case": label, "truth": truth,
                     "p_truth_with_image": round(before, 4),
                     "p_truth_blank": round(after, 4),
                     "delta": round(before - after, 4)})
    return {"rows": rows,
            "mean_delta": round(float(np.mean([r["delta"] for r in rows])), 4)}


def exp4_box(cli, case_path):
    """Box is 1.14-only. PREDICTION: if the model localises time-frequency
    structure, the returned box covers the visible harmonic band. If it is
    guessing, the box will be near the centre and its width will be arbitrary.
    """
    from typesafe_sdk import Box, Image

    try:
        a = cli.system_one(
            state=f"The spectrogram of one spoken utterance: {Image(case_path)}",
            questions={"harmonic": Box(
                instructions=("The box around the bright horizontal band "
                              "where the speech energy is concentrated, in "
                              "frequency."))},
            model=MODEL).answers["harmonic"]
        return {"supported": True, "answer": str(getattr(a, "box", a))}
    except Exception as exc:                      # noqa: BLE001
        return {"supported": False, "error": f"{type(exc).__name__}: {exc}"}


# -- run ---------------------------------------------------------------------

def main() -> int:
    if not (os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFEAI_KEY")):
        print("no TYPESAFE_API_KEY")
        return 2
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_spec")
    os.makedirs(tmp, exist_ok=True)

    # THE FIRST VERSION OF THIS EXPERIMENT WAS WRONG AND ALMOST REPORTED A
    # FALSE NEGATIVE. It used fmax=4000 with tones at 180/240/300 Hz, which
    # puts all three inside the bottom 7.5% of the frequency axis -- so "low",
    # "mid" and "high" were all the same picture. Jev answered "low" three
    # times out of three and the harness printed "accuracy 1/3", which reads as
    # the model failing when it had in fact scored 3/3.
    #
    # fmax is now 600 Hz and the tones are 150/300/450, which land at roughly
    # 25% / 50% / 75% of the visible axis. A wide fmax is the right default for
    # a real spectrogram; it is the wrong default for a test whose labels
    # depend on the axis being used.
    cases = []
    for label, hz, truth in (("low_150hz", 150.0, "low"),
                             ("mid_300hz", 300.0, "mid"),
                             ("high_450hz", 450.0, "high")):
        path = os.path.join(tmp, f"{label}.png")
        render(spectrogram(tone(hz, 0.5)), path)
        cases.append((label, path, truth))

    # A blank control of identical dimensions: same bytes count, same header,
    # no structure at all. If the model answers the same way here, the image
    # channel is not carrying the decision.
    blank = os.path.join(tmp, "blank.png")
    write_png_gray(blank, np.full((H, W), 128, dtype=np.uint8))

    print(f"model {MODEL}   images {W}x{H}")
    for label, path, _ in cases:
        print(f"  {label:12s} {os.path.getsize(path):6d} bytes")

    cli = client()
    report = {}

    print("\n== 0. Preflight: is the image actually on the wire? ==")
    report["preflight_tokens"] = list(preflight(cli, cases[0][1]))

    print("\n== 1. Pitch band from the image alone ==")
    print("   prediction: well above the 1/3 floor, or the model cannot see")
    report["image_only"] = exp1_pitch_from_image(cli, cases)
    for r in report["image_only"]["rows"]:
        print(f"   {r['case']:12s} truth={r['truth']:5s} picked={r['picked']:5s} "
              f"conf={r['confidence']:.3f}  {r['probs']}")
    print(f"   accuracy {report['image_only']['accuracy']} "
          f"(random floor {report['image_only']['random_floor']})")

    print("\n== 2. Same question WITH the measured pitch in the state ==")
    print("   prediction: unchanged if the image is already being used")
    report["image_plus_numbers"] = exp2_pitch_with_numbers(
        cli, cases, {"low_150hz": "150.0", "mid_300hz": "300.0",
                     "high_450hz": "450.0"})
    for r in report["image_plus_numbers"]["rows"]:
        print(f"   {r['case']:12s} truth={r['truth']:5s} picked={r['picked']:5s} "
              f"conf={r['confidence']:.3f}")
    print(f"   accuracy {report['image_plus_numbers']['accuracy']}")

    print("\n== 3. Ablation: swap the spectrogram for a blank image ==")
    print("   prediction: a real delta means the pixels carried the decision")
    report["ablation"] = exp3_ablation(cli, cases, blank)
    for r in report["ablation"]["rows"]:
        print(f"   {r['case']:12s} p(true|real)={r['p_truth_with_image']:.3f} "
              f"p(true|blank)={r['p_truth_blank']:.3f} "
              f"delta={r['delta']:+.3f}")
    print(f"   mean delta {report['ablation']['mean_delta']:+.4f}")

    print("\n== 4. Box: does it localise the band? (1.14-only) ==")
    report["box"] = exp4_box(cli, cases[1][1])
    print(f"   {report['box']}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "spectrogram_vision_results.json")
    with open(out, "w") as fh:
        json.dump({"model": MODEL, "size": [W, H], "results": report}, fh,
                  indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())