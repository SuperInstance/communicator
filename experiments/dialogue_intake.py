"""
dialogue_intake.py -- check whether the real-dialogue corpus is actually usable.

WHY THIS EXISTS.

`README.md` in this directory asks for eight specific scenes, because each one
answers a question that no existing corpus can (`docs/research/TWO-SPEAKER-TURNS.md`
records the three limits of the AMI reconstruction). The failure mode for a
hand-built corpus is arriving full of files that look like the thing asked for
and are not -- two channels recorded separately, so they drift; a "conversation"
that is two people taking turns for twenty minutes, which contains no overlap at
all; a scene recorded at a level that will be read as silence.

So this file does not count files. It measures the corpus and states plainly
which requirements are UNMET, because a check that cannot say "this is not
enough yet" is decoration.

WHAT IT MEASURES, AND WHY EACH ITEM IS A GATE RATHER THAN A STATISTIC.

    matched pairs   two channels from the same take, or the speaker labels are
                    fiction and every downstream number inherits the error
    level           a channel that is digitally quiet is not a recording, and a
                    silence that comes from gain rather than from a room is
                    indistinguishable from a real pause
    cross-talk      if channel B hears channel A clearly the "separate
                    microphones" design failed, and the overlap scenes are
                    contaminated at source
    signal          the fraction of audio above the repo's own absolute floor,
                    so "silence" in these files can be trusted to be silence
    overlap         the one thing the AMI far-field channel cannot represent --
                    reported as a count because if it is zero the barge-in
                    scenes did not happen
    hand-offs       silences followed by the other channel speaking; this is
                    the positive class that was unrepresentable until now

WHAT IT CANNOT DO.

It cannot judge the conversation, only the signal. A take can pass everything
here and still be two people politely alternating with no hand-off worth
measuring; that shows up as `hand_offs == 0` and is reported as such. It cannot
align channels beyond the clap described in the README, and it does not attempt
to -- alignment that silently drifts by 200 ms would corrupt every gap.
"""

from __future__ import annotations

import json
import re
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import energy, wavio, watcher  # noqa: E402

SR = 16000
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "examples/realdialogue/raw"

# energy.ABS_FLOOR is the repo's own definition of "not silent"; anything under
# it is not evidence of a pause, it is below measurement.
SPEECH_FLOOR = energy.ABS_FLOOR
PAIR_RE = re.compile(r"^(?P<scene>.+)_(?P<who>[AB])\.wav$", re.IGNORECASE)


def channel_stats(path):
    """Level and signal fraction for one file, at the pipeline's own rate."""
    try:
        wav = wavio.read_wav(str(path))
        a = wav.samples
    except Exception as exc:                      # unreadable is a hard fail
        return {"error": f"{type(exc).__name__}: {exc}"}
    n = len(a)
    if n == 0:
        return {"error": "empty"}
    peak = max(abs(v) for v in a)
    rms = (sum(v * v for v in a) / n) ** 0.5
    # Fraction of 25 ms frames carrying anything above the repo's floor.
    fr = 400
    frames = n // fr
    above = 0
    for i in range(frames):
        s = a[i * fr:(i + 1) * fr]
        if (sum(v * v for v in s) / fr) ** 0.5 > SPEECH_FLOOR:
            above += 1
    return {"duration_s": round(n / SR, 2),
            "raw_rate_hz": getattr(wav, "sample_rate", None),
            "peak": round(peak, 5), "rms": round(rms, 6),
            "signal_frame_frac": round(above / frames, 4) if frames else 0.0,
            "silent": peak == 0.0}


def crosstalk(a_path, b_path):
    """How much of B is also present in A.

    A crude but decisive correlation over the 10 loudest 100 ms windows: if two
    separate microphones in one room are not picking up each other up, the
    "separate channel" premise has failed and overlap scenes are unusable.
    """
    a = wavio.read_wav(str(a_path)).samples
    b = wavio.read_wav(str(b_path)).samples
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    W = 1600
    wins = []
    for i in range(0, n - W, W):
        wa, wb = a[i:i + W], b[i:i + W]
        ra = (sum(v * v for v in wa) / W) ** 0.5
        rb = (sum(v * v for v in wb) / W) ** 0.5
        if ra > SPEECH_FLOOR and rb > SPEECH_FLOOR:
            wins.append((ra + rb, wa, wb))
    if len(wins) < 3:
        return {"windows": len(wins), "corr": None,
                "note": "too few simultaneous-loud windows to judge"}
    wins.sort(reverse=True, key=lambda w: w[0])
    top = wins[:10]
    ma = [sum(v for v in w[1]) / W for w in top]
    mb = [sum(v for v in w[2]) / W for w in top]
    num = sum((x - sum(ma) / len(ma)) * (y - sum(mb) / len(mb)) for x, y in zip(ma, mb))
    da = (sum((x - sum(ma) / len(ma)) ** 2 for x in ma)) ** 0.5
    db = (sum((y - sum(mb) / len(mb)) ** 2 for y in mb)) ** 0.5
    corr = round(num / (da * db), 4) if da and db else None
    return {"windows": len(wins), "corr": corr}


def both_talking_frac(a_path, b_path):
    """Fraction of time both channels are loud: the overlap budget."""
    a = wavio.read_wav(str(a_path)).samples
    b = wavio.read_wav(str(b_path)).samples
    n = min(len(a), len(b))
    fr = 800                      # 50 ms
    both = either = frames = 0
    for i in range(0, n - fr, fr):
        sa = (sum(v * v for v in a[i:i + fr]) / fr) ** 0.5
        sb = (sum(v * v for v in b[i:i + fr]) / fr) ** 0.5
        la, lb = sa > SPEECH_FLOOR, sb > SPEECH_FLOOR
        frames += 1
        both += la and lb
        either += la or lb
    return {"frames": frames,
            "both_loud_frac": round(both / frames, 4) if frames else 0.0,
            "either_loud_frac": round(either / frames, 4) if frames else 0.0}


def main():
    # `--raw` exists so the checker itself can be tested. A gate that has only
    # ever been observed rejecting things is not known to work: it could be
    # rejecting everything. `tests/test_dialogue_intake.py` builds synthetic
    # corpora that must PASS and synthetic ones that must fail, and this
    # argument is what lets it point at either.
    argv = sys.argv[1:]
    raw = RAW
    if "--raw" in argv:
        raw = Path(argv[argv.index("--raw") + 1])
    report = {"raw_dir": str(raw), "exists": raw.is_dir(), "pairs": [],
              "unmatched": [], "gates": {}}

    if not raw.is_dir():
        report["gates"]["status"] = "NOTHING RECORDED YET"
        report["gates"]["detail"] = (
            f"{raw} does not exist. See examples/realdialogue/README.md for the "
            "eight scenes and the two-microphone rule.")
        print(json.dumps(report, indent=2))
        return

    found = sorted(p for p in raw.glob("*.wav"))
    scenes = {}
    for p in found:
        m = PAIR_RE.match(p.name)
        if not m:
            report["unmatched"].append(p.name)
            continue
        scenes.setdefault(m.group("scene"), {})[m.group("who").upper()] = p

    total_s = 0.0
    hand_offs = 0
    silences = 0
    for scene, sides in sorted(scenes.items()):
        a, b = sides.get("A"), sides.get("B")
        entry = {"scene": scene, "has_A": a is not None, "has_B": b is not None}
        if not a or not b:
            entry["problem"] = "unmatched pair"
            report["pairs"].append(entry)
            continue
        sa, sb = channel_stats(a), channel_stats(b)
        entry["A"], entry["B"] = sa, sb
        if "error" in sa or "error" in sb:
            entry["problem"] = "unreadable"
            report["pairs"].append(entry)
            continue
        total_s += min(sa["duration_s"], sb["duration_s"])
        entry["crosstalk"] = crosstalk(a, b)
        entry["levels"] = both_talking_frac(a, b)

        # Hand-offs and silences, from the watcher on channel A, labelled by
        # whether channel B is speaking when A's silence ends.
        ia = wavio.read_wav(str(a)).samples
        ib = wavio.read_wav(str(b)).samples
        n = min(len(ia), len(ib))
        w = watcher.Watcher(SR, adaptive=True)
        loud_b = [False] * (n // 1600 + 1)
        for k in range(len(loud_b)):
            seg = ib[k * 1600:(k + 1) * 1600]
            loud_b[k] = ((sum(v * v for v in seg) / max(1, len(seg))) ** 0.5
                         > SPEECH_FLOOR) if seg else False
        for i in range(0, n, 1600):
            for e in w.push(ia[i:i + 1600]):
                if e.kind != "pause" or e.data.get("open"):
                    continue
                silences += 1
                k = min(len(loud_b) - 1, int(e.t1 / 0.1))
                if loud_b[k]:
                    hand_offs += 1
        report["pairs"].append(entry)

    ok = all(p.get("has_A") and p.get("has_B") for p in report["pairs"]) \
         and not report["unmatched"] and report["pairs"]
    gates = report["gates"]
    gates["matched_pairs"] = len(report["pairs"])
    gates["total_audio_s"] = round(total_s, 1)
    gates["silences_seen"] = silences
    gates["hand_offs_seen"] = hand_offs
    gates["unmatched_files"] = report["unmatched"]

    unmet = []
    if not report["pairs"]:
        unmet.append("no matched _A/_B pairs yet")
    if report["unmatched"]:
        unmet.append(f"{len(report['unmatched'])} file(s) not matching NAME_A.wav")
    if total_s < 600:
        unmet.append(f"only {total_s:.0f}s of audio; the protocol asks for ~25 min")
    if report["pairs"] and hand_offs == 0:
        unmet.append("zero hand-offs: no silence is followed by the other "
                     "speaker, so the positive class is still absent")
    quiet = [p["scene"] for p in report["pairs"]
             if "A" in p and p["A"].get("silent")]
    if quiet:
        unmet.append(f"digitally silent channel(s): {quiet}")
    gates["unmet"] = unmet
    gates["status"] = "USABLE" if not unmet else "NOT YET USABLE"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()