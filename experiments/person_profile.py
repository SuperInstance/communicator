"""The person profile: a living document that grows with every session.

Casey's framing: the corpus is a portrait of ONE speaker, deepening over
time. Not a summary of what was said -- a profile of how this person
speaks, measured the same way every time, so that the difference between
one session and the next is visible.

The design law says to start descriptive and earn inference slowly. So
this tool measures and it does not interpret. It reports contour
statistics, the vocabulary that is growing, the drift against the
running baseline, and nothing else. Where a drift is large it says the
drift is large; it does not say the speaker was tired, or excited, or
lying.

Three things accumulate, all from the score bundles the other tools
produce:

    CONTOUR STATISTICS   how the voice moved: centre, range, slope, rate,
                         note durations, step sizes, jitter
    VOCABULARY           the words, and which of them are new
    MOTIF VOCABULARY     the shapes of the contour -- rising, falling,
                         arched, dipped -- and how often each is used,
                         plus any intent labels the drill mode supplies

The drift is measured in standard deviations against the running
baseline of all the sessions seen so far, which is why a profile with one
session reports no drift at all: there is nothing to drift from. The
first session establishes the baseline and the second one starts the
comparison.

    python experiments/person_profile.py --bundle <bundle>/alignment.json
    python experiments/person_profile.py --session <dir> --out <dir>
    python experiments/person_profile.py --report
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: The statistics the profile carries. They are the same set the speech
#: tool measures, so a session measured by one tool is comparable with a
#: session measured by the other.
CONTOUR_STATS = [
    "centre_midi",       # the melody's centre
    "range_semitones",   # how wide the melody got
    "slope_semitones_per_s",   # which way it went overall
    "notes_per_second",
    "median_note_s",
    "note_duration_std",
    "step_abs_mean",     # the typical jump between notes
    "step_std",
    "jitter_semitones",  # how far each note sat from the local trend
    "voiced_fraction",
]

#: The coarse shapes the contour vocabulary holds.
MOTIFS = ["rising", "falling", "level", "arched", "dipped", "mixed"]


# ============================================================================
# 1. READING A BUNDLE
# ============================================================================

def contour_stats(notes: list) -> dict:
    """The contour statistics of one bundle's notes.

    Measured the same way for every session, from the note events the
    score pipeline produced. Nothing here is a judgement.
    """
    if not notes:
        return {k: None for k in CONTOUR_STATS}

    midis = [float(n["midi"]) for n in notes]
    starts = [float(n["start_s"]) for n in notes]
    ends = [float(n["end_s"]) for n in notes]
    durs = [max(1e-3, e - s) for s, e in zip(starts, ends)]
    steps = [midis[i + 1] - midis[i] for i in range(len(midis) - 1)]

    span = max(1e-6, ends[-1] - starts[0])
    # the slope, by the least squares line through the notes
    n = len(midis)
    mx = sum(starts) / n
    my = sum(midis) / n
    num = sum((x - mx) * (y - my) for x, y in zip(starts, midis))
    den = sum((x - mx) ** 2 for x in starts)
    slope = (num / den) if den > 1e-9 else 0.0

    # the jitter: how far each note sat from the trend
    residuals = []
    for x, y in zip(starts, midis):
        residuals.append(y - (my + slope * (x - mx)))
    jitter = statistics.pstdev(residuals) if len(residuals) > 1 else 0.0

    return {
        "centre_midi": round(statistics.median(midis), 3),
        "range_semitones": round(max(midis) - min(midis), 3),
        "slope_semitones_per_s": round(slope, 3),
        "notes_per_second": round(n / span, 3),
        "median_note_s": round(statistics.median(durs), 4),
        "note_duration_std": round(statistics.pstdev(durs), 4) if n > 1 else 0.0,
        "step_abs_mean": round(sum(abs(s) for s in steps) / len(steps), 3)
                         if steps else 0.0,
        "step_std": round(statistics.pstdev(steps), 3) if len(steps) > 1 else 0.0,
        "jitter_semitones": round(jitter, 3),
        "voiced_fraction": None,   # filled in by the caller, which has the audio
    }


def motif_of(notes: list) -> str:
    """The coarse shape of a phrase.

    A vocabulary of shapes rather than of meanings: the intent labels
    need labelled data, and a labelled-data shape vocabulary is what can
    honestly be built from an unlabelled session.
    """
    if len(notes) < 2:
        return "level"
    midis = [float(n["midi"]) for n in notes]
    first, last = midis[0], midis[-1]
    net = last - first
    lo, hi = min(midis), max(midis)
    interior_max = max(midis[1:-1]) if len(midis) > 2 else first
    interior_min = min(midis[1:-1]) if len(midis) > 2 else first
    if net >= 3:
        return "rising"
    if net <= -3:
        return "falling"
    if interior_max - lo >= 4 and interior_max > first and interior_max > last:
        return "arched"
    if hi - interior_min >= 4 and interior_min < first and interior_min < last:
        return "dipped"
    if abs(net) <= 2 and hi - lo <= 3:
        return "level"
    return "mixed"


def read_bundle(path: Path) -> dict:
    """One score bundle, in the profile's terms."""
    doc = json.loads(Path(path).read_text())
    notes = doc.get("notes", [])
    words = doc.get("transcribe", {}).get("n_words", 0)
    alignment = doc.get("alignment", {})
    rows = alignment.get("rows", [])

    text_words = []
    for r in rows:
        w = (r.get("word") or "").strip().lower()
        if w and w.isalpha():
            text_words.append(w)
    if not text_words:
        tpath = Path(path).parent / "transcript.json"
        if tpath.is_file():
            tdoc = json.loads(tpath.read_text())
            text_words = [(w.get("text") or "").strip().lower()
                          for w in tdoc.get("words", [])]
            text_words = [w for w in text_words if w and w.isalpha()]

    stats = contour_stats(notes)
    pitch = doc.get("pitch", {})
    if stats.get("voiced_fraction") is None and pitch.get("n_frames"):
        stats["voiced_fraction"] = round(
            float(pitch.get("n_voiced", 0)) / float(pitch["n_frames"]), 4)

    intents = []
    for r in rows:
        lab = r.get("intent")
        if lab and lab not in intents:
            intents.append(lab)

    return {
        "bundle": str(path),
        "session": doc.get("session") or doc.get("bundle_session"),
        "segment_start_s": doc.get("segment_start_s"),
        "segment_end_s": doc.get("segment_end_s"),
        "duration_s": doc.get("duration_s"),
        "n_words": words,
        "n_syllables": alignment.get("n_syllables", 0),
        "n_notes": len(notes),
        "stats": stats,
        "words": text_words,
        "motif": motif_of(notes),
        "intents": intents,
        "digest": (doc.get("digest") or {}).get("text"),
    }


# ============================================================================
# 2. THE PROFILE
# ============================================================================

def empty_profile() -> dict:
    return {
        "tool": "experiments/person_profile.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at": None,
        "n_sessions": 0,
        "sessions": [],
        "vocabulary": {"size": 0, "words": []},
        "motif_vocabulary": {m: 0 for m in MOTIFS},
        "intent_vocabulary": {},
        "baseline": {},
        "note": ("descriptive only. These are measurements of how the voice "
                 "moved, and a drift says a thing moved; it does not say "
                 "what the speaker was doing. The interpretation is earned "
                 "slowly or not at all."),
    }


def update_profile(profile: dict, bundles: list, label: str | None = None) -> dict:
    """Fold a session's bundles into the profile."""
    if not bundles:
        return profile
    session = {
        "label": label or bundles[0].get("session") or bundles[0]["bundle"],
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_bundles": len(bundles),
        "duration_s": round(sum(b.get("duration_s") or 0.0 for b in bundles), 3),
        "n_words": sum(b.get("n_words") or 0 for b in bundles),
        "n_syllables": sum(b.get("n_syllables") or 0 for b in bundles),
        "n_notes": sum(b.get("n_notes") or 0 for b in bundles),
        "stats": {},
        "motifs": {},
        "intents": [],
        "new_words": [],
        "drift": {},
    }

    # the session's statistics, averaged over its bundles
    for key in CONTOUR_STATS:
        vals = [b["stats"].get(key) for b in bundles
                if b["stats"].get(key) is not None]
        session["stats"][key] = round(sum(vals) / len(vals), 4) if vals else None

    for b in bundles:
        session["motifs"][b["motif"]] = session["motifs"].get(b["motif"], 0) + 1
        for lab in b["intents"]:
            if lab not in session["intents"]:
                session["intents"].append(lab)

    # the vocabulary
    known = set(profile["vocabulary"]["words"])
    seen = set()
    for b in bundles:
        seen.update(b["words"])
    session["new_words"] = sorted(w for w in seen if w not in known)
    profile["vocabulary"]["words"] = sorted(known | seen)
    profile["vocabulary"]["size"] = len(profile["vocabulary"]["words"])

    # the motif vocabulary
    for b in bundles:
        profile["motif_vocabulary"][b["motif"]] = \
            profile["motif_vocabulary"].get(b["motif"], 0) + 1
    for lab in session["intents"]:
        profile["intent_vocabulary"][lab] = \
            profile["intent_vocabulary"].get(lab, 0) + 1

    # the drift, against the baseline of every session so far
    prev = profile.get("baseline") or {}
    for key in CONTOUR_STATS:
        v = session["stats"].get(key)
        base = prev.get(key)
        if v is None or base is None or base.get("n", 0) < 1:
            continue
        sd = base.get("sd", 0.0)
        session["drift"][key] = {
            "value": v,
            "baseline_mean": base.get("mean"),
            "delta": round(v - base["mean"], 4),
            # a baseline of one point has no spread to measure against, so
            # the first comparison carries a delta and no standard
            # deviations, and the standard deviations arrive with the
            # next session that has something to be compared against
            "sd_from_baseline": (round((v - base["mean"]) / sd, 3)
                                 if sd > 1e-9 else None),
        }

    profile["sessions"].append(session)
    profile["n_sessions"] = len(profile["sessions"])
    profile["updated_at"] = session["at"]

    # the baseline is recomputed over every session seen so far
    base = {}
    for key in CONTOUR_STATS:
        vals = [s["stats"].get(key) for s in profile["sessions"]
                if s["stats"].get(key) is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        base[key] = {"mean": round(mean, 4), "sd": round(sd, 4),
                     "n": len(vals), "min": round(min(vals), 4),
                     "max": round(max(vals), 4)}
    profile["baseline"] = base
    return profile


# ============================================================================
# 3. THE LIVING DOCUMENT
# ============================================================================

def render_markdown(profile: dict) -> str:
    """The profile as a document that a person can read.

    It is regenerated whole each time, so it never has a stale paragraph
    in it, and it carries the same honesty the JSON does.
    """
    L = []
    L.append("# THE PERSON PROFILE — one speaker, deepening")
    L.append("")
    L.append(f"**Sessions folded in:** {profile['n_sessions']}  ")
    L.append(f"**First session:** {profile['sessions'][0]['at'] if profile['sessions'] else '—'}  ")
    L.append(f"**Last update:** {profile['updated_at'] or '—'}")
    L.append("")
    L.append(profile["note"])
    L.append("")
    L.append("---")
    L.append("")

    L.append("## 1. THE CONTOUR BASELINE")
    L.append("")
    base = profile.get("baseline") or {}
    if not base:
        L.append("No sessions yet.")
    else:
        L.append("What this voice does, averaged over every session folded in so far.")
        L.append("")
        L.append("| statistic | mean | sd | min | max | n |")
        L.append("|---|---|---|---|---|---|")
        for key in CONTOUR_STATS:
            b = base.get(key)
            if not b:
                continue
            L.append(f"| {key} | {b['mean']} | {b['sd']} | {b['min']} | "
                     f"{b['max']} | {b['n']} |")
    L.append("")

    L.append("## 2. THE DRIFT, SESSION BY SESSION")
    L.append("")
    L.append("How far each session sat from the baseline of everything before it.")
    L.append("The first session has no drift: it is the baseline. The second")
    L.append("carries a delta but no standard deviations, because a baseline of")
    L.append("one point has no spread to measure against; from the third on the")
    L.append("comparison is in standard deviations.")
    L.append("")
    for s in profile["sessions"]:
        L.append(f"### {s['label']}")
        L.append("")
        L.append(f"{s['at']} — {s['n_bundles']} segment(s), {s['duration_s']} s, "
                 f"{s['n_words']} words, {s['n_syllables']} syllables, "
                 f"{s['n_notes']} notes.")
        L.append("")
        if not s["drift"]:
            L.append("No drift yet — this is the baseline, or there is not "
                     "enough history to measure against.")
            L.append("")
            continue
        L.append("| statistic | value | baseline | delta | sd out |")
        L.append("|---|---|---|---|---|")
        for key, d in s["drift"].items():
            sd = d.get("sd_from_baseline")
            L.append(f"| {key} | {d['value']} | {d['baseline_mean']} | "
                     f"{d['delta']} | {sd if sd is not None else '—'} |")
        L.append("")

    L.append("## 3. THE VOCABULARY")
    L.append("")
    L.append(f"**Distinct words seen:** {profile['vocabulary']['size']}")
    L.append("")
    for s in profile["sessions"]:
        new = s["new_words"]
        if new:
            shown = ", ".join(new[:40])
            more = f" (+{len(new) - 40} more)" if len(new) > 40 else ""
            L.append(f"- **{s['label']}** — {len(new)} new: {shown}{more}")
        else:
            L.append(f"- **{s['label']}** — no new words")
    L.append("")

    L.append("## 4. THE CONTOUR MOTIF VOCABULARY")
    L.append("")
    L.append("The shapes of the melody, and how often each one is used. This is a")
    L.append("vocabulary of shapes and not of meanings; the intent labels need")
    L.append("labelled data, and the drill supplies that when it lands.")
    L.append("")
    L.append("| motif | count |")
    L.append("|---|---|")
    for m in MOTIFS:
        L.append(f"| {m} | {profile['motif_vocabulary'].get(m, 0)} |")
    L.append("")
    if profile["intent_vocabulary"]:
        L.append("Labelled intents seen so far:")
        L.append("")
        L.append("| intent | count |")
        L.append("|---|---|")
        for k, v in sorted(profile["intent_vocabulary"].items()):
            L.append(f"| {k} | {v} |")
        L.append("")

    L.append("## 5. WHAT THIS PROFILE DOES NOT SAY")
    L.append("")
    L.append("It does not say what the speaker was feeling, and it does not")
    L.append("label the drift with an emotion. A large drift is a measurement,")
    L.append("not a diagnosis. The interpretation is earned slowly: it needs")
    L.append("sessions, and it needs a second opinion, and Casey's own drill")
    L.append("deliveries are the ground truth that would make it mean anything.")
    L.append("")
    return "\n".join(L) + "\n"


# ============================================================================
# 4. THE DRIVER
# ============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", action="append", default=None,
                    help="a score bundle (alignment.json); repeat for more")
    ap.add_argument("--session", default=None,
                    help="a folder of score bundles, one per segment")
    ap.add_argument("--profile", default=str(REPO / "experiments" /
                                             "person_profile.json"),
                    help="where the profile is kept")
    ap.add_argument("--markdown", default=str(REPO / "docs" / "research" /
                                              "PERSON-PROFILE.md"),
                    help="where the living document is written")
    ap.add_argument("--report", action="store_true",
                    help="print the profile and change nothing")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    profile_path = Path(a.profile)
    if profile_path.is_file():
        profile = json.loads(profile_path.read_text())
    else:
        profile = empty_profile()

    if a.report:
        print(render_markdown(profile))
        return 0

    bundles = []
    for b in (a.bundle or []):
        bundles.append(read_bundle(Path(b)))
    if a.session:
        sdir = Path(a.session)
        for p in sorted(sdir.glob("**/alignment.json")):
            bundles.append(read_bundle(Path(p)))
    if not bundles:
        print("give --bundle, --session or --report")
        return 2

    # the profile accumulates PER SESSION, so the bundles are grouped by
    # the session they came from. A bundle that does not say which
    # session it is falls back to its own folder, and a group that is
    # still not identifiable becomes a session of its own rather than
    # being merged into somebody else's.
    groups: dict = {}
    for b in bundles:
        key = b.get("session") or str(Path(b["bundle"]).parent.parent)
        groups.setdefault(key, []).append(b)

    for key in sorted(groups):
        profile = update_profile(profile, groups[key], label=key)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2))
    md_path = Path(a.markdown)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(profile))
    print(f"folded in {len(bundles)} bundle(s) across {len(groups)} session(s)")
    print(f"profile   {profile_path}  ({profile['n_sessions']} sessions)")
    print(f"document  {md_path}")
    print(f"vocabulary {profile['vocabulary']['size']} words, "
          f"{sum(profile['motif_vocabulary'].values())} motifs")
    return 0


# ============================================================================
# 5. THE SELFTEST
# ============================================================================

def selftest() -> int:
    """The checks that need no session and no bundle."""
    import tempfile

    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  [{'ok' if got == want else 'FAIL'}] {name}: {got!r}")

    print("SELFTEST -- the statistics, the motifs, the drift")
    print("-" * 60)

    def note(t, d, m):
        return {"start_s": t, "end_s": t + d, "midi": m, "hz": 440.0,
                "duration_s": d}

    # 1. THE STATISTICS. A rising line of four notes.
    st = contour_stats([note(0.0, 0.2, 60), note(0.3, 0.2, 62),
                        note(0.6, 0.2, 64), note(0.9, 0.2, 66)])
    check("the centre is the median", st["centre_midi"], 63.0)
    check("the range is the span", st["range_semitones"], 6.0)
    check("the slope is positive", st["slope_semitones_per_s"] > 0, True)
    check("and the jitter is small", st["jitter_semitones"] < 1.0, True)
    check("an empty contour has no statistics",
          contour_stats([])["centre_midi"], None)

    # 2. THE MOTIFS.
    check("a rising phrase is rising",
          motif_of([note(0, 0.2, 60), note(0.3, 0.2, 64), note(0.6, 0.2, 68)]),
          "rising")
    check("a falling phrase is falling",
          motif_of([note(0, 0.2, 68), note(0.3, 0.2, 64), note(0.6, 0.2, 60)]),
          "falling")
    check("a level phrase is level",
          motif_of([note(0, 0.2, 62), note(0.3, 0.2, 62), note(0.6, 0.2, 62)]),
          "level")
    check("an arch is arched",
          motif_of([note(0, 0.2, 60), note(0.3, 0.2, 68), note(0.6, 0.2, 62)]),
          "arched")
    check("a dip is dipped",
          motif_of([note(0, 0.2, 68), note(0.3, 0.2, 60), note(0.6, 0.2, 66)]),
          "dipped")

    # 3. THE PROFILE. One session is a baseline and has no drift; the
    # second one does.
    p = empty_profile()
    b1 = {"bundle": "b1", "session": "s1", "duration_s": 10.0, "n_words": 20,
          "n_syllables": 30, "n_notes": 12, "stats": contour_stats(
              [note(0, 0.2, 60), note(0.3, 0.2, 62), note(0.6, 0.2, 64)]),
          "words": ["hello", "there"], "motif": "rising", "intents": []}
    b2 = dict(b1)
    b2 = {"bundle": "b2", "session": "s2", "duration_s": 12.0, "n_words": 25,
          "n_syllables": 36, "n_notes": 13, "stats": contour_stats(
              [note(0, 0.2, 66), note(0.3, 0.2, 68), note(0.6, 0.2, 70)]),
          "words": ["hello", "world"], "motif": "falling", "intents": []}

    p = update_profile(p, [b1])
    check("the first session is in the profile", p["n_sessions"], 1)
    check("and it has no drift", p["sessions"][0]["drift"], {})
    check("the vocabulary grew", p["vocabulary"]["size"], 2)

    p = update_profile(p, [b2])
    check("the second session is in the profile", p["n_sessions"], 2)
    check("and it has drift", bool(p["sessions"][1]["drift"]), True)
    check("the new word is recorded", p["sessions"][1]["new_words"], ["world"])
    check("the vocabulary grew again", p["vocabulary"]["size"], 3)
    check("the motif vocabulary counts both",
          p["motif_vocabulary"]["rising"], 1)
    check("the baseline is over both sessions",
          p["baseline"]["centre_midi"]["n"], 2)

    # 4. THE DOCUMENT. It renders, and it carries the honesty.
    md = render_markdown(p)
    check("the document mentions the sessions", "## 2. THE DRIFT" in md, True)
    check("and the honesty", "does not say what the speaker was feeling"
          in md, True)

    print("-" * 60)
    if failures:
        print(f"SELFTEST FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
