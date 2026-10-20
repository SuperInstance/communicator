"""
dialogue_intake_fixes.py -- measure what the intake fixes actually bought.

WHY THIS FILE EXISTS.

`docs/research/DIALOGUE-INTAKE.md` found seven defects in
`experiments/dialogue_intake.py`. Fixing a gate without measuring what the
fix changed is how a gate becomes decoration with extra steps, so every fix
here is measured on a signal whose truth is known, against the behaviour it
replaces, and the numbers are written to
`experiments/dialogue_intake_fixes_results.json`.

WHAT IS MEASURED.

    hand_off_rule     the old rule and the new one, on three ordinary shapes
                      of conversation and on a protocol-strict corpus whose
                      hand-off count is known by construction
    cross_talk        the old statistic and the new one, on a clean corpus and
                      on one where channel B hears channel A
    level_floor       the same corpus in a quiet room and in an ordinary one,
                      under the old absolute floor and under the new
                      measured one
    alignment         a pair with a KNOWN offset, before and after the clap
                      correction
    drift             what the single alignment point can and cannot say
    perturbations     every defect the audit listed, and whether the gate
                      now catches it
    exit_codes        what a pipeline running this tool can now tell apart

WHAT IT IS NOT.

It is not a verdict on a real recording. Everything here is synthetic, with
known truth, exercising the gates. The intake has still never seen a real
file, and nothing in this file changes that.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src import energy, wavio                       # noqa: E402
import dialogue_intake as di                       # noqa: E402
from dialogue_intake_selftest import (            # noqa: E402
    SCENES, build_corpus, build_scene, shift,
)

SR = 16000


# ------------------------------------------------------------ the old rule --

def old_hand_offs(a, b, sr=SR):
    """The OLD hand-off rule, verbatim from the pre-fix intake.

    A CLOSED pause on channel A whose END coincides with channel B being
    loud. Reproduced here so the comparison is against the code that was
    there, not against a memory of it.
    """
    from src import watcher
    w = watcher.Watcher(sr, adaptive=True)
    n = min(len(a), len(b))
    loud_b = [False] * (n // 1600 + 1)
    for k in range(len(loud_b)):
        seg = b[k * 1600:(k + 1) * 1600]
        loud_b[k] = ((sum(v * v for v in seg) / max(1, len(seg))) ** 0.5
                     > energy.ABS_FLOOR) if len(seg) else False
    count = 0
    for i in range(0, n, 1600):
        for e in w.push(a[i:i + 1600]):
            if e.kind != "pause" or e.data.get("open"):
                continue
            k = min(len(loud_b) - 1, int(e.t1 / 0.1))
            if loud_b[k]:
                count += 1
    return count


def old_level_measures(a, b):
    """The OLD level measures, at the OLD absolute floor."""
    floor = energy.ABS_FLOOR
    fr = 400
    n = min(len(a), len(b))

    def sig_frac(x):
        frames = len(x) // fr
        above = 0
        for i in range(frames):
            s = x[i * fr:(i + 1) * fr]
            if (sum(v * v for v in s) / fr) ** 0.5 > floor:
                above += 1
        return above / frames if frames else 0.0

    both = either = frames = 0
    for i in range(0, n - fr, fr):
        sa = (sum(v * v for v in a[i:i + fr]) / fr) ** 0.5
        sb = (sum(v * v for v in b[i:i + fr]) / fr) ** 0.5
        la, lb = sa > floor, sb > floor
        frames += 1
        both += la and lb
        either += la or lb
    return {"signal_frame_frac_A": round(sig_frac(a), 4),
            "signal_frame_frac_B": round(sig_frac(b), 4),
            "both_loud_frac": round(both / frames, 4) if frames else 0.0,
            "either_loud_frac": round(either / frames, 4) if frames else 0.0}


# ---------------------------------------------------------- the new intake --

def run_intake(raw_dir: Path) -> dict:
    """Run the intake as a SUBPROCESS, and keep the exit code.

    The exit code is the thing this whole exercise is about, so it is
    observed from outside the process, not by calling a function.
    """
    proc = subprocess.run(
        [sys.executable, str(ROOT / "experiments/dialogue_intake.py"),
         "--raw", str(raw_dir)],
        capture_output=True, text=True)
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"exit_code": proc.returncode, "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:], "parse_error": True}
    report["exit_code"] = proc.returncode
    return report


def new_hand_offs(raw_dir: Path, scene: str) -> dict:
    report = run_intake(raw_dir)
    for p in report.get("pairs", []):
        if p.get("scene") == scene:
            return {"hand_offs": p.get("hand_offs"),
                    "silences": p.get("silences"),
                    "alignment": p.get("alignment"),
                    "crosstalk": p.get("crosstalk"),
                    "levels": p.get("levels"),
                    "status": report.get("gates", {}).get("status"),
                    "unmet": report.get("gates", {}).get("unmet"),
                    "exit_code": report.get("exit_code")}
    return {"error": "scene not found", "report": report}


# ------------------------------------------------------------- the battery --

def hand_off_semantics(work: Path) -> dict:
    """Three ordinary shapes of conversation, old rule vs new rule.

        canonical       A stops, B takes the floor, A stays quiet
        canonical+ret  A stops, B answers, A comes back much later
        barge_back      A stops, B takes the floor, A comes back WHILE B is
                        still talking
    """
    d = work / "handoff_shapes"
    d.mkdir(parents=True, exist_ok=True)
    shapes = {
        "canonical": [("A", 0.60, 2.00, 11), ("B", 2.95, 3.60, 12)],
        "canonical_then_return": [("A", 0.60, 2.00, 11), ("B", 2.95, 3.60, 12),
                                  ("A", 6.00, 7.00, 13)],
        "barge_back": [("A", 0.60, 2.00, 11), ("B", 2.95, 3.60, 12),
                       ("A", 3.20, 4.20, 13)],
    }
    out = {}
    for name, events in shapes.items():
        spec = dict(name="probe", protocol_s=8, kind="dialogue", events=events)
        a, b, _ = build_scene(spec, 8.0, 1e-5)
        from src import wavio as _w
        _w.write_wav(d / "probe_A.wav", a)
        _w.write_wav(d / "probe_B.wav", b)
        new = new_hand_offs(d, "probe")
        out[name] = {
            "old_rule_hand_offs": old_hand_offs(a, b),
            "new_rule_hand_offs": new.get("hand_offs"),
            "silences": new.get("silences"),
            "alignment_applied": (new.get("alignment") or {}).get("applied"),
        }
    out["expected"] = {
        "canonical": 1,
        "canonical_then_return": 1,
        "barge_back": 1,
        "note": ("the old rule scored the canonical shapes ZERO and the "
                 "barge-back shape 2 -- it was counting the rare shape and "
                 "missing the common one"),
    }
    return out


def protocol_strict(work: Path) -> dict:
    """A corpus where every turn is canonical: the hand-off count is known.

    The old intake reported "NOT YET USABLE / zero hand-offs" on a corpus
    built to the protocol. This is that corpus, and this is the count.
    """
    d = work / "strict"
    build_corpus(d, 12.0, 1e-5)
    report = run_intake(d)
    truth = _canonical_hand_off_truth(d)
    old = 0
    for p in sorted(d.glob("*_A.wav")):
        a = np.array(wavio.read_wav(str(p)).samples)
        b = np.array(wavio.read_wav(str(d / (p.name[:-6] + "_B.wav"))).samples)
        old += old_hand_offs(a, b)
    new = sum(p.get("hand_offs", 0) for p in report.get("pairs", []))
    return {
        "hand_offs_in_the_corpus": truth,
        "old_rule_hand_offs": old,
        "new_rule_hand_offs": new,
        "status": report.get("gates", {}).get("status"),
        "exit_code": report.get("exit_code"),
        "unmet": report.get("gates", {}).get("unmet"),
        "per_scene": [{"scene": p.get("scene"), "hand_offs": p.get("hand_offs"),
                       "silences": p.get("silences")}
                      for p in report.get("pairs", [])],
    }


def _canonical_hand_off_truth(d: Path) -> dict:
    """How many hand-offs the corpus was built with, counted from the SCENE
    DEFINITIONS rather than from the signal.

    A hand-off, as the protocol's own scene descriptions use the word, is a
    silence on channel A during which channel B took the floor. Two kinds:

      canonical   B's utterance BEGINS inside A's silence -- A stopped, and
                  B answered. This is the positive class the corpus exists
                  for.
      barge_in    B's utterance BEGAN before A's silence -- B cut A off and
                  was still talking when A stopped.

    Both are floor-takes, and the intake's rule ("was B loud during A's
    silence?") counts both, which is correct for the question the intake
    asks. Counting them separately is what makes the comparison meaningful:
    a rule that only counted the barge-in would look busy while missing the
    common case.
    """
    canonical = barge_in = 0
    per_scene = {}
    for spec in SCENES:
        if spec["kind"] == "roomtone":
            continue
        a_spans = sorted((ev[1], ev[2]) for ev in spec["events"]
                          if ev[0] == "A")
        b_spans = sorted((ev[1], ev[2]) for ev in spec["events"]
                          if ev[0] == "B")
        if not a_spans or not b_spans:
            continue
        # A's silence runs: before the first utterance, between them, and
        # after the last one.
        runs = []
        prev_end = 0.0
        for s, e in a_spans:
            runs.append((prev_end, s))
            prev_end = e
        runs.append((prev_end, float("inf")))
        c = b = 0
        for s0, s1 in runs:
            if s1 - s0 < 0.15:
                continue
            loud = [bs for bs, be in b_spans if bs < s1 and s0 < be]
            if not loud:
                continue
            onset = min(loud)
            if onset >= s0:
                c += 1          # B began inside the silence
            else:
                b += 1          # B was already talking: a barge-in
        canonical += c
        barge_in += b
        if c or b:
            per_scene[spec["name"]] = {"canonical": c, "barge_in": b}
    return {"canonical": canonical, "barge_in": barge_in,
            "total": canonical + barge_in, "per_scene": per_scene}


def cross_talk_separation(work: Path) -> dict:
    """The old statistic and the new one, clean vs contaminated."""
    clean = work / "xt_clean"
    bad = work / "xt_bad"
    build_corpus(clean, 12.0, 1e-5)
    build_corpus(bad, 12.0, 1e-5, perturb="crosstalk")
    out = {"scenes": []}
    for spec in SCENES:
        name = spec["name"]
        row = {"scene": name}
        for label, d in (("clean", clean), ("crosstalk", bad)):
            a = np.array(wavio.read_wav(str(d / f"{name}_A.wav")).samples)
            b = np.array(wavio.read_wav(str(d / f"{name}_B.wav")).samples)
            report = run_intake(d)
            entry = None
            for p in report.get("pairs", []):
                if p["scene"] == name:
                    entry = p
            row[label] = {
                "new_b_loud_when_a_loud": (entry or {}).get("crosstalk", {})
                                        .get("b_loud_when_a_loud"),
                "legacy_corr": (entry or {}).get("crosstalk", {})
                               .get("legacy_statistic", {}).get("corr"),
            }
        out["scenes"].append(row)
    new_clean = [r["clean"]["new_b_loud_when_a_loud"] for r in out["scenes"]
                 if r["clean"]["new_b_loud_when_a_loud"] is not None]
    new_bad = [r["crosstalk"]["new_b_loud_when_a_loud"] for r in out["scenes"]
               if r["crosstalk"]["new_b_loud_when_a_loud"] is not None]
    leg_clean = [r["clean"]["legacy_corr"] for r in out["scenes"]
                 if r["clean"]["legacy_corr"] is not None]
    leg_bad = [r["crosstalk"]["legacy_corr"] for r in out["scenes"]
               if r["crosstalk"]["legacy_corr"] is not None]
    out["new_statistic"] = {
        "clean_min": min(new_clean) if new_clean else None,
        "clean_max": max(new_clean) if new_clean else None,
        "crosstalk_min": min(new_bad) if new_bad else None,
        "crosstalk_max": max(new_bad) if new_bad else None,
        "separation": (round(min(new_bad) - max(new_clean), 4)
                       if (new_clean and new_bad) else None),
        "gate": di.CROSSTALK_MAX,
    }
    out["legacy_statistic"] = {
        "clean_min": min(leg_clean) if leg_clean else None,
        "clean_max": max(leg_clean) if leg_clean else None,
        "crosstalk_min": min(leg_bad) if leg_bad else None,
        "crosstalk_max": max(leg_bad) if leg_bad else None,
        "separation": (round(min(leg_bad) - max(leg_clean), 4)
                       if (leg_clean and leg_bad) else None),
    }
    return out


def level_floor(work: Path) -> dict:
    """The same corpus in a quiet room and in an ordinary one."""
    out = {"rooms": {}}
    for label, tone in (("quiet_room_1e-5", 1e-5), ("ordinary_room_1.5e-3", 1.5e-3)):
        d = work / ("floor_" + label)
        build_corpus(d, 12.0, tone)
        report = run_intake(d)
        rows = []
        for p in report.get("pairs", []):
            a = np.array(wavio.read_wav(str(d / f"{p['scene']}_A.wav")).samples)
            b = np.array(wavio.read_wav(str(d / f"{p['scene']}_B.wav")).samples)
            old = old_level_measures(a, b)
            rows.append({
                "scene": p["scene"],
                "old": old,
                "new": {
                    "signal_frame_frac_A": p["A"]["signal_frame_frac"],
                    "signal_frame_frac_B": p["B"]["signal_frame_frac"],
                    "both_loud_frac": p["levels"]["both_loud_frac"],
                },
            })
        out["rooms"][label] = {
            "tone_rms": tone,
            "rows": rows,
            "roomtone_scene_new": next(
                (r["new"] for r in rows if r["scene"] == "scene08_roomtone"),
                None),
            "roomtone_scene_old": next(
                (r["old"] for r in rows if r["scene"] == "scene08_roomtone"),
                None),
        }
    q = out["rooms"]["quiet_room_1e-5"]["roomtone_scene_new"]
    o = out["rooms"]["ordinary_room_1.5e-3"]["roomtone_scene_new"]
    qo = out["rooms"]["quiet_room_1e-5"]["roomtone_scene_old"]
    oo = out["rooms"]["ordinary_room_1.5e-3"]["roomtone_scene_old"]
    out["finding"] = {
        "old_floor_reads_an_empty_room_as_full": (
            None if (qo is None or oo is None) else
            {"quiet_room_both_loud_frac": qo["both_loud_frac"],
             "ordinary_room_both_loud_frac": oo["both_loud_frac"]}),
        "new_floor_reads_an_empty_room_as_empty": (
            None if (q is None or o is None) else
            {"quiet_room_both_loud_frac": q["both_loud_frac"],
             "ordinary_room_both_loud_frac": o["both_loud_frac"]}),
        "new_floor_is_room_independent": (
            None if (q is None or o is None) else
            q["both_loud_frac"] == o["both_loud_frac"]),
    }
    return out


def alignment(work: Path) -> dict:
    """A pair with a KNOWN offset, before and after the clap correction.

    Geometry: A speaks 0.60-2.00 s, B speaks 2.95-3.60 s, A resumes at
    3.20 s. The TRUE hand-off gap is 2.95 - 2.00 = 0.95 s.
    """
    d = work / "align"
    d.mkdir(parents=True, exist_ok=True)
    spec = dict(name="probe", protocol_s=8, kind="dialogue", events=[
        ("A", 0.60, 2.00, 81), ("B", 2.95, 3.60, 82), ("A", 3.20, 4.20, 83),
    ])
    a, b, _ = build_scene(spec, 8.0, 1e-5)
    rows = []
    for offset_ms in (0, 40, 120, 300, 600):
        b_off = shift(b, offset_ms)
        wavio.write_wav(d / "probe_A.wav", a)
        wavio.write_wav(d / "probe_B.wav", b_off)
        report = run_intake(d)
        entry = next((p for p in report.get("pairs", [])
                      if p["scene"] == "probe"), {})
        rows.append({
            "true_offset_ms": offset_ms,
            "detected_offset_ms": (entry.get("alignment") or {}).get("clap_offset_ms"),
            "peak_r": (entry.get("alignment") or {}).get("peak_r"),
            "alignment_applied": (entry.get("alignment") or {}).get("applied"),
            "old_rule_hand_offs": old_hand_offs(a, b_off),
            "new_rule_hand_offs": entry.get("hand_offs"),
            "true_hand_offs": 1,
        })
    return {
        "rows": rows,
        "old_rule_flip": [r["true_offset_ms"] for r in rows
                          if r["old_rule_hand_offs"] != 1],
        "new_rule_flip": [r["true_offset_ms"] for r in rows
                          if r["new_rule_hand_offs"] != 1],
    }


def drift(work: Path) -> dict:
    """What one alignment point can and cannot say about clock drift."""
    d = work / "drift"
    d.mkdir(parents=True, exist_ok=True)
    spec = dict(name="probe", protocol_s=20, kind="dialogue", events=[
        ("A", 0.60, 2.00, 91), ("A", 14.00, 14.60, 94), ("B", 15.00, 16.00, 92),
        ("A", 16.60, 18.00, 93),
    ])
    a, b, _ = build_scene(spec, 20.0, 1e-5)
    eps = 0.0125                       # 1.25%: 250 ms over 20 s
    n = len(b)
    b_drift = np.interp(np.arange(n) * (1.0 + eps), np.arange(n), b,
                        left=0.0, right=0.0)
    wavio.write_wav(d / "probe_A.wav", a)
    wavio.write_wav(d / "probe_B.wav", b_drift)
    report = run_intake(d)
    entry = next((p for p in report.get("pairs", [])
                  if p["scene"] == "probe"), {})
    dr = entry.get("drift", {})
    return {
        "drift_eps": eps,
        "drift_ms_total": round(20.0 * eps * 1000, 1),
        "drift_ms_at_the_gap": round(15.0 * eps * 1000, 1),
        "reported_by_the_intake": dr,
        "measurable": dr.get("measured_ms") is not None,
        "finding": ("a single alignment point cannot measure drift; the "
                    "report carries a bound and says so"),
    }


def perturbations(work: Path) -> list:
    """Every defect the audit listed, and whether the gate now catches it.

    NOT every perturbation is a defect. Two of them are legitimate
    variations that the intake must NOT reject: a corpus with no overlap is
    a real corpus with no barge-in, and the protocol explicitly allows the
    two channels to be at different sample rates ("Rate: anything 16 kHz and
    up"). Both are in the battery to prove the tool does not false-positive
    on them.

    The LENGTH gate fires on every one of these corpora, because they are
    96 s and the protocol asks for 1230 s. So the verdict here is about the
    RELEVANT gate, not about the exit code alone.
    """
    out = []
    cases = [
        ("missing_channel_B", "one channel of a pair is missing",
         True, "unmatched pair"),
        ("digital_silence_A", "channel A of one scene is digital silence",
         True, "digitally silent"),
        ("digital_silence_B", "channel B of one scene is digital silence",
         True, "digitally silent"),
        ("low_level_B", "channel B of one scene is very low but NOT zero",
         True, "differ by"),
        ("very_low_level_B", "channel B of one scene is below the floor",
         True, "differ by"),
        ("crosstalk", "channel B clearly hears channel A",
         True, "hears channel A"),
        ("no_overlap", "the two speakers never overlap at all",
         False, None),
        ("no_handoff", "only one speaker exists",
         True, "silent"),
        ("sample_rate_mismatch", "the two channels have different rates",
         False, None),
        ("unreadable", "one channel is not a readable WAV",
         True, "unreadable"),
    ]
    for perturb, description, must_be_caught, relevant in cases:
        d = work / ("pert_" + perturb)
        build_corpus(d, 12.0, 1e-5, perturb=perturb)
        report = run_intake(d)
        gates = report.get("gates", {})
        unmet = gates.get("unmet", [])
        other = [u for u in unmet
                 if "the protocol's eight scenes are" not in u]
        caught = any((relevant in u) for u in other) if relevant else False
        row = {
            "perturbation": perturb,
            "description": description,
            "must_be_caught": must_be_caught,
            "caught_by_the_relevant_gate": caught,
            "correct": caught == must_be_caught,
            "exit_code": report.get("exit_code"),
            "status": gates.get("status"),
            "unmet_excluding_the_length_gate": other,
            "unmet": unmet,
        }
        if perturb == "sample_rate_mismatch":
            row["source_rates_hz"] = next(
                (p.get("source_rates_hz") for p in report.get("pairs", [])
                 if p.get("scene") == "scene05_backchannel"), None)
        out.append(row)
    return out


def exit_codes(work: Path) -> dict:
    """What a pipeline running this tool can now tell apart."""
    missing = work / "does_not_exist"
    empty = work / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    good = work / "strict"
    if not good.exists():
        build_corpus(good, 12.0, 1e-5)
    return {
        "nothing_recorded": {
            "exit_code": run_intake(missing).get("exit_code"),
            "expected": 2,
        },
        "empty_directory": {
            "exit_code": run_intake(empty).get("exit_code"),
            "expected": 1,
        },
        "short_corpus": {
            "exit_code": run_intake(good).get("exit_code"),
            "expected": 1,
            "note": ("the synthetic corpus is 96 s and the protocol asks for "
                     "1230 s, so this one is rejected for length -- which is "
                     "the gate working, not a failure of it"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default="/tmp/dialogue_intake_fixes")
    ap.add_argument("--out", default=str(ROOT / "experiments" /
                                         "dialogue_intake_fixes_results.json"))
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    results = {
        "what_changed": "experiments/dialogue_intake.py, 2026-10-19",
        "hand_off_rule": hand_off_semantics(work),
        "protocol_strict": protocol_strict(work),
        "cross_talk": cross_talk_separation(work),
        "level_floor": level_floor(work),
        "alignment": alignment(work),
        "drift": drift(work),
        "perturbations": perturbations(work),
        "exit_codes": exit_codes(work),
        "constants": {
            "crosstalk_max": di.CROSSTALK_MAX,
            "noise_multiple": di.NOISE_MULTIPLE,
            "protocol_seconds": di.PROTOCOL_SECONDS,
            "min_silence_s": di.MIN_SILENCE_S,
            "align_min_r": di.ALIGN_MIN_R,
            "level_balance_db": di.LEVEL_BALANCE_DB,
            "min_signal_frac": di.MIN_SIGNAL_FRAC,
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
