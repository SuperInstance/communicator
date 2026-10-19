"""
turn_eval.py -- now that two speakers exist, does silence separate a
speaker change from a continuation?

WHY THIS FILE HAD TO WAIT.

Commit `17ae46a` measured the thing that blocked everything else: across the
entire corpus, after any silence of 0.6 s or more, the speaker resumed 98.1% to
100% of the time. Every file is one person reading alone, so the positive
class -- a speaker yielding the floor and someone else taking it -- does not
occur in the data at all. Any turn-taking accuracy computed on that corpus is
a measurement of the corpus, not of the system.

`examples/realtwospeaker/` supplies what is missing: AMI Meeting Corpus,
far-field (SDM) channel, spontaneous four-party meetings, with every turn
labelled by `speaker_id`. That makes the question falsifiable for the first
time.

THE QUESTION, STATED SO IT CAN COME OUT WRONG.

Does the length of a silence tell you whether the next voice is a different
person? pipecat's operational numbers imply yes at ~200 ms to trigger and 3 s
to give up. This file measures the actual separation and, more importantly, the
cost of being wrong in each direction.

WHAT IS MEASURED.

Adjacent labelled turn pairs, each scored as

    SAME    the same speaker_id continues -- a pause inside one turn
    CHANGE  a different speaker_id speaks next -- a real hand-off

    gap               seconds from the end of A to the start of B
    overlap           a NEGATIVE gap: B starts before A finishes
    silence_runs      measured from the AUDIO through the watcher, not from
                      the annotations, so it is the number the system
                      actually sees -- and each is labelled by whether the
                      speaker after it changed

Reporting both views matters. Annotation gaps are ground truth but they are
not what the pipeline sees; watcher silence runs are what the pipeline sees but
they inherit the watcher's segmentation error. Where they disagree, the
disagreement is the interesting quantity.

THE BAR THIS FILE REFUSES TO CROSS.

An accuracy number here would be meaningless without a baseline, and the
honest baseline is the one the system already implies: **every silence is a
turn end**. That is what a fixed threshold does, and on a single-speaker corpus
it scores ~100%. It is reported here as `always_predict_change`. Any method
that cannot beat that number has learned nothing -- because the corpus, not the
method, would be supplying the accuracy.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import wavio, watcher  # noqa: E402

SR = 16000
ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples/realtwospeaker"


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return round(s[i], 3)


def describe(xs):
    if not xs:
        return {"n": 0}
    return {"n": len(xs),
            "min": round(min(xs), 3), "p10": pct(xs, .10),
            "p25": pct(xs, .25), "median": round(statistics.median(xs), 3),
            "p75": pct(xs, .75), "p90": pct(xs, .90),
            "max": round(max(xs), 3)}


def auc(pos, neg):
    """Probability a random CHANGE gap exceeds a random SAME gap.

    Computed by counting pairwise wins with ties at 0.5, so 0.5 means the
    signal carries no information and 1.0 means perfect. This is the single
    number that decides whether silence can classify a hand-off at all.
    """
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return round(wins / (len(pos) * len(neg)), 4)


def load(stem):
    wav = CORPUS / f"{stem}.wav"
    seg = CORPUS / f"{stem}.segments.json"
    if not wav.is_file() or not seg.is_file():
        return None
    meta = json.loads(seg.read_text())
    return wav, meta


def pairs_from_annotations(meta):
    """Adjacent labelled turns with the hand-off label attached."""
    segs = sorted(meta["segments"], key=lambda s: s["begin_time"])
    start = meta.get("meeting_window_start_s", 0.0)
    out = []
    for a, b in zip(segs, segs[1:]):
        # Times are meeting-absolute; the wav begins at the window start.
        gap = (b["begin_time"] - start) - (a["end_time"] - start)
        out.append({"gap": gap,
                    "label": "CHANGE" if a["speaker_id"] != b["speaker_id"]
                             else "SAME",
                    "from": a["speaker_id"], "to": b["speaker_id"]})
    return out


def pairs_from_audio(meta):
    """Silence runs the WATCHER reports, labelled by what happens next.

    This is the view that matters operationally: it is the number the turn
    stage would actually consume. Each pause the watcher emits is matched to
    the annotation whose turn follows it, so the label comes from ground truth
    and only the gap comes from our own pipeline.
    """
    wav = CORPUS / f"{meta['file']}"
    samples = wavio.read_wav(str(wav)).samples
    w = watcher.Watcher(SR, adaptive=True)
    pauses, utts = [], []
    for i in range(0, len(samples), 1600):
        for e in w.push(samples[i:i + 1600]):
            if e.kind == "utterance":
                utts.append((e.t0, e.t1))
            elif e.kind == "pause":
                pauses.append(e)

    segs = sorted(meta["segments"], key=lambda s: s["begin_time"])
    start = meta.get("meeting_window_start_s", 0.0)
    rel = [(s["begin_time"] - start, s["end_time"] - start, s["speaker_id"])
           for s in segs]

    out = []
    for p in pauses:
        # ONLY completed runs. An open pause re-reports itself on a 0.25 s
        # cadence, so a 420 s meeting yields 2037 pause events of which 1350
        # are checkpoints of a run already counted -- 1345 carry
        # data['checkpoint'] True. Counting those would weight long silences
        # by their length and make every duration statistic an artefact of the
        # reporting cadence rather than of the audio.
        if p.data.get("open"):
            continue
        # The next annotated turn that begins after this pause ended.
        nxt = next((r for r in rel if r[0] >= p.t1 - 0.15), None)
        prev = None
        for r in rel:
            if r[1] <= p.t0 + 0.15:
                prev = r
            else:
                break
        if not nxt or not prev:
            continue
        out.append({"gap": p.t1 - p.t0,
                    "label": "CHANGE" if prev[2] != nxt[2] else "SAME",
                    "from": prev[2], "to": nxt[2]})
    return out, len(utts), len(samples) / SR


def annotation_structure(meta):
    """How AMI's two sets of turn bounds relate, because they disagree.

    An earlier version of this file measured 52.7% of adjacent turn pairs as
    OVERLAPPING, and read that as barge-in -- the opposite of silence as the
    hand-off signal. That was wrong, and it is wrong in a way worth keeping.

    AMI pads each turn on its LEADING edge: `begin_time` runs 0.00-0.70 s
    ahead of `audible_start_s`, while `audible_end_s == end_time` with no
    trailing pad. Two consecutive turns from different speakers therefore
    "overlap" on paper purely because each one's lead-in sits inside the
    previous one's span. Measured:

        adjacent pairs                        313 / 229
        overlap on begin_time/end_time      52.7% / 40.2%
        overlap on audible_start/end          0.0% /  0.0%

    The audible bounds do not merely remove the overlap -- they make it
    exactly zero, because consecutive AUDIBLE turns TILE: they share a
    boundary to the millisecond. So on this corpus the annotation gap is
    structurally 0.000 s and carries no information at all.

    That is a hard limit on what this corpus can prove, stated here so the
    sweep below is not read as a general claim about conversation. What
    survives is the base rate: the LABELS come from AMI and are solid, and
    the gap the watcher sees is real audio with real silences in it (83% of
    frames carry signal; the remaining 11% digital holes are placed by the
    annotation timeline, so their POSITION is conventional, not acoustic).
    """
    segs = sorted(meta["segments"], key=lambda s: s["begin_time"])
    raw = aud = n = na = 0
    raw_gaps, aud_gaps = [], []
    for a, b in zip(segs, segs[1:]):
        n += 1
        g = b["begin_time"] - a["end_time"]
        raw_gaps.append(g)
        if g < 0:
            raw += 1
        if a["audible_end_s"] is None or b["audible_start_s"] is None:
            continue
        na += 1
        ga = b["audible_start_s"] - a["audible_end_s"]
        aud_gaps.append(ga)
        if ga < 0:
            aud += 1
    return {"n_pairs": n, "n_with_audible_bounds": na,
            "raw_overlap_frac": round(raw / n, 4) if n else None,
            "audible_overlap_frac": round(aud / na, 4) if na else None,
            "raw_median_gap": round(statistics.median(raw_gaps), 4) if raw_gaps else None,
            "audible_median_gap": round(statistics.median(aud_gaps), 4) if aud_gaps else None,
            "audible_segments": sum(1 for s in segs if s["audible_start_s"] is not None),
            "segments_total": len(segs)}


def sweep(pairs, key="gap"):
    """Accuracy / precision / recall as the 'silence means hand-off' bar moves.

    `always_predict_change` is the corpus baseline: claim a hand-off at every
    pause. On the old single-speaker corpus that scored ~1.00, which is the
    whole reason that corpus could not test anything.
    """
    rows = []
    for bar in (0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0):
        tp = fp = fn = tn = 0
        for p in pairs:
            pred = p[key] >= bar
            actual = p["label"] == "CHANGE"
            if pred and actual:
                tp += 1
            elif pred and not actual:
                fp += 1
            elif not pred and actual:
                fn += 1
            else:
                tn += 1
        n = tp + fp + fn + tn
        rows.append({
            "bar_s": bar,
            "n": n,
            "accuracy": round((tp + tn) / n, 4) if n else None,
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "false_interrupt": fp,
        })
    n = len(pairs)
    always = sum(1 for p in pairs if p["label"] == "CHANGE")
    return rows, {"n": n,
                  "base_rate_change": round(always / n, 4) if n else None,
                  "always_predict_change_accuracy": round(always / n, 4) if n else None}


def main():
    # `Path.stem` strips only the final suffix, so a file called
    # `x.segments.json` yields the stem `x.segments` and every lookup misses by
    # one extension. Removesuffix takes the whole compound suffix.
    stems = sorted(p.name[:-len(".segments.json")]
                   for p in CORPUS.glob("*.segments.json"))
    if not stems:
        print(json.dumps({"error": f"no *.segments.json under {CORPUS}"}))
        return
    all_pairs, report = [], {"meetings": {}}
    for stem in stems:
        got = load(stem)
        if got is None:
            continue
        wav, meta = got
        ann = pairs_from_annotations(meta)
        aud, n_utts, dur = pairs_from_audio(meta)
        rows, base = sweep(aud)
        report["meetings"][stem] = {
            "speakers": meta.get("speakers"),
            "config": meta.get("config"),
            "audio_s": round(dur, 1),
            "n_annotated_turns": len(meta["segments"]),
            "n_watcher_utterances": n_utts,
            "annotation_pairs": base,
            "annotation_gap": {
                "SAME": describe([p["gap"] for p in ann if p["label"] == "SAME"]),
                "CHANGE": describe([p["gap"] for p in ann if p["label"] == "CHANGE"]),
            },
            "annotation_auc": auc([p["gap"] for p in ann if p["label"] == "CHANGE"],
                                  [p["gap"] for p in ann if p["label"] == "SAME"]),
            "overlapping_turns": sum(1 for p in ann if p["gap"] < 0),
            "annotation_structure": annotation_structure(meta),
            "watcher_pairs": base,
            "watcher_auc": auc([p["gap"] for p in aud if p["label"] == "CHANGE"],
                               [p["gap"] for p in aud if p["label"] == "SAME"]),
            "threshold_sweep": rows,
        }
        all_pairs.extend(aud)

    rows, base = sweep(all_pairs)
    report["pooled"] = {
        "watcher_pairs": base,
        "auc": auc([p["gap"] for p in all_pairs if p["label"] == "CHANGE"],
                   [p["gap"] for p in all_pairs if p["label"] == "SAME"]),
        "threshold_sweep": rows,
    }
    report["note"] = ("Ground truth is AMI speaker_id. Gap is measured from the "
                      "watcher's own events, so this is the number the turn "
                      "stage would consume. `always_predict_change_accuracy` is "
                      "the baseline any method must beat; on the old "
                      "single-speaker corpus it was ~1.00, which is why that "
                      "corpus could not falsify anything.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()