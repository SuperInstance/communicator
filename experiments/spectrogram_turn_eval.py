#!/usr/bin/env python
"""
Can the vision model classify a speaker hand-off from a spectrogram IMAGE?

THE QUESTION.  Silence duration is a measured dead end for detecting that the
floor is being handed over (pooled AUC 0.529 over 1396 pauses, below the
trivial always-predict-change baseline of 0.6225 -- `experiments/turn_eval.py`),
and pitch separates voice CLASSES, not speakers within one
(`docs/research/F0-SPEAKER-CHANGE.md`).  The remaining cheap route this lane
tests: render the turn boundary as a spectrogram image and let a vision model
decide SAME vs CHANGE.

WHAT IS AVAILABLE HERE (measured before designing, per the brief):

  torch / transformers / timm / open_clip / PIL / scipy : MISSING
  ollama : present, but every local model on it (gemma2:2b, llama3.2:1b,
           qwen3:0.6b, qwen2.5:0.5b) rejects image input with HTTP 400 --
           all four are text-only.  NO local vision model exists.
  typesafe_sdk + TYPESAFE_API_KEY : present.  The cloud model jev-1.14.0
           accepts images with the opt-in header (see below) -- the ONLY
           vision model reachable from this machine.

So "vision model" here means exactly one system, the same one
`experiments/spectrogram_vision.py` already measured.  That file's prior
result matters and is restated in the writeup: pitch-band discrimination from
a spectrogram image scored 1/3 -- the random floor -- even though the three
tones were rendered at 25% / 50% / 75% of the visible frequency axis.  This
experiment therefore also carries a positive control (gross visible pitch
differences) so a null result can be attributed correctly: to the model's
inability to read spectrograms at all, or to the representation lacking the
speaker cue.  Those are different failures and the doc must not conflate them.

PRIOR ART'S TRAP, INHERITED.  Without the `TypeSafe-Danger-Zone-Images: true`
header the SDK silently drops the image bytes and every number comes back
plausible and meaningless.  The preflight below (input tokens with vs without
an image) is the guard; each per-example call also records its token count so
a mid-run drop is visible.

DESIGN.

  Population   the same one behind 0.6225: watcher pause events on the two
               far-field AMI meetings, each matched to the annotated turns
               before and after it (`pairs_from_audio` logic in
               `turn_eval.py`, re-implemented locally so the pause TIMES
               survive -- they are needed to centre the image).
  Sample       balanced by design, N per class (default 60+60), seeded, no
               two windows overlapping by more than half a window.  Balanced
               sampling makes the on-sample trivial baseline exactly 0.50;
               the pooled corpus baseline 0.6225 and a reweighted accuracy
               are reported alongside so nothing is hidden.
  Image        fixed 4.0 s window centred on the pause midpoint -- geometry
               identical for both classes, so no shape/duration cue exists.
               Grayscale magnitude spectrogram, 0-4000 Hz (formants live
               there), dB-scaled per window (removes absolute level as a
               trivial cue), 640x480, hand-rolled PNG writer, one 2 px
               bright column at the centre marking the boundary.
  Question     a Choice with probabilities: same speaker continues vs a
               different speaker takes over.  Score = P(different); AUC is
               threshold-free so no operating point is smuggled in.
  Controls     (a) NO-INFORMATION: identical question over a uniform gray
               image of identical dimensions, same ground-truth labels.
               Must score near chance or the run is void.  (b) POSITIVE:
               the three synthetic tones at 150/300/450 Hz rendered at 25/
               50/75% of a 600 Hz axis -- a signal that IS in the image; a
               miss here means the model cannot read even gross spectrogram
               structure, which reframes (but does not rescue) the main null.
  Weaknesses   stated in docs/research/SPECTROGRAM-TURN.md, not hidden: one
               far-field microphone, real overlap (extra voices inside the
               window), ~11% digital-silence holes placed by the annotation
               timeline, 8-bit grayscale dynamic range, phase discarded by
               construction -- a negative result is consistent with "phase
               was the missing ingredient" and this experiment cannot rule
               that in or out.

USAGE

    set -a && . /home/proart/projects/glasses/.env && set +a
    /home/proart/projects/glasses/.venv/bin/python \\
        experiments/spectrogram_turn_eval.py [--n-per-class 60] [--seed 11]

Writes experiments/spectrogram_turn_eval_results.json.  Responses are cached
under experiments/_spec/turn/ keyed by image hash + model + question, so a
re-run reproduces the same numbers without re-paying the API; delete that
directory to force a fresh pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.spectrogram_vision import (  # noqa: E402
    H, IMAGE_HEADERS, PITCH_OPTIONS, PITCH_Q, W, render, spectrogram, tone,
    write_png_gray,
)
from experiments.turn_eval import CORPUS, SR, auc  # noqa: E402
from src import wavio, watcher  # noqa: E402

MODEL = os.environ.get("JEV_MODEL", "jev-1.14.0")
WINDOW_S = 4.0          # fixed for every example, both classes
FMAX = 4000.0           # formant range; 8-bit grayscale, per-window dB norm
POOLED_BASELINE = {"n": 1396, "base_rate_change": 0.6225}  # turn_eval.py
F0_GROUPS = {"FEO": ("FEO070", "FEO072"), "MEE": ("MEE071", "MEE073")}
SPEC = ROOT / "experiments" / "_spec" / "turn"

HANDOFF_STATE = (
    "A 4.0-second spectrogram of one far-field meeting microphone. Horizontal "
    "axis: time, 4.0 s total. Vertical axis: frequency, 0 to 4000 Hz, low "
    "frequencies at the bottom; brighter means more energy. The thin bright "
    "vertical line at the centre marks a turn boundary: speech to its left is "
    "the end of one utterance, speech to its right is the start of the next. "
    "{img}"
)
HANDOFF_INSTR = (
    "Decide, from the spectrogram alone, whether the speech on the two sides "
    "of the centre line belongs to the same speaker. Compare pitch, harmonic "
    "spacing and formant structure across the boundary."
)
HANDOFF_CRITERIA = {
    "same": "The voice after the centre line is the SAME person who was "
            "speaking before it: a pause inside one speaker's turn.",
    "different": "The voice after the centre line is a DIFFERENT person "
                 "taking over the floor.",
}


# -- population: watcher pauses with their times retained ---------------------

def pairs_with_times(meta):
    """`turn_eval.pairs_from_audio`, but keeping each pause's t0/t1 so the
    image window can be centred on it.  Matching logic copied verbatim: only
    completed (non-open) pause runs, label from AMI speaker_id of the turns
    around the pause."""
    wav = CORPUS / f"{meta['file']}"
    samples = wavio.read_wav(str(wav)).samples
    w = watcher.Watcher(SR, adaptive=True)
    pauses = []
    for i in range(0, len(samples), 1600):
        for e in w.push(samples[i:i + 1600]):
            if e.kind == "pause" and not e.data.get("open"):
                pauses.append(e)

    segs = sorted(meta["segments"], key=lambda s: s["begin_time"])
    start = meta.get("meeting_window_start_s", 0.0)
    rel = [(s["begin_time"] - start, s["end_time"] - start, s["speaker_id"])
           for s in segs]

    out, n_utts = [], 0
    for p in pauses:
        nxt = next((r for r in rel if r[0] >= p.t1 - 0.15), None)
        prev = None
        for r in rel:
            if r[1] <= p.t0 + 0.15:
                prev = r
            else:
                break
        if not nxt or not prev:
            continue
        out.append({"t0": p.t0, "t1": p.t1,
                    "gap": p.t1 - p.t0,
                    "label": "CHANGE" if prev[2] != nxt[2] else "SAME",
                    "from": prev[2], "to": nxt[2]})
    return out, n_utts, len(samples) / SR


def population():
    """All watcher pause pairs, pooled over both meetings.  Cached because the
    watcher pass costs ~53 s per meeting and is deterministic."""
    SPEC.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.name[:-len(".segments.json")]
                   for p in CORPUS.glob("*.segments.json"))
    pairs = []
    for stem in stems:
        cache = SPEC / f"pairs_{stem}.json"
        if cache.is_file():
            blob = json.loads(cache.read_text())
        else:
            meta = json.loads((CORPUS / f"{stem}.segments.json").read_text())
            got, n_utts, dur = pairs_with_times(meta)
            blob = {"pairs": got, "n_utts": n_utts, "audio_s": dur,
                    "model_of": "watcher(SR, adaptive=True)"}
            cache.write_text(json.dumps(blob))
        for i, p in enumerate(blob["pairs"]):
            p = dict(p, meeting=stem, idx=i, audio_s=blob["audio_s"])
            pairs.append(p)
    return pairs


def eligible(pairs):
    """Windows must fit inside the file; nothing else is filtered -- the
    sample stays natural (micro-pauses included)."""
    out = []
    for p in pairs:
        mid = (p["t0"] + p["t1"]) / 2.0
        if mid - WINDOW_S / 2 < 0 or mid + WINDOW_S / 2 > p["audio_s"]:
            continue
        p = dict(p, mid=mid)
        out.append(p)
    return out


def sample(ex, n_per_class, seed):
    """Balanced, seeded, de-overlapped: no two chosen windows closer than
    half a window, so no near-duplicate image appears twice."""
    rng = np.random.default_rng(seed)
    chosen, taken = [], []
    for label in ("CHANGE", "SAME"):
        pool = [p for p in ex if p["label"] == label]
        rng.shuffle(pool)
        got = 0
        for p in pool:
            if got >= n_per_class:
                break
            if any(abs(p["mid"] - t) < WINDOW_S / 2 for t in taken):
                continue
            taken.append(p["mid"])
            chosen.append(p)
            got += 1
    return chosen


# -- images --------------------------------------------------------------------

def render_window(samples, mid, path):
    """The fixed 4.0 s window centred on the boundary, with the 2 px centre
    marker.  Identical geometry for both classes by construction.  Returns
    the image plus each half's fraction of pixels above -37 dB, so a
    speechless half (previous speaker already faded) is visible in the
    results instead of discovered by hand later."""
    a = int((mid - WINDOW_S / 2) * SR)
    b = int((mid + WINDOW_S / 2) * SR)
    spec = spectrogram(np.asarray(samples[a:b]), fmax=FMAX)
    img = np.flipud(spec)
    ys = (np.linspace(0, img.shape[0] - 1, H)).astype(int)
    xs = (np.linspace(0, img.shape[1] - 1, W)).astype(int)
    out = img[np.ix_(ys, xs)].copy()
    out[:, W // 2 - 1:W // 2 + 1] = 255      # the boundary marker
    write_png_gray(str(path), out)
    left = float((out[:, :W // 2 - 1] > 60).mean())
    right = float((out[:, W // 2 + 1:] > 60).mean())
    return out, left, right


def blank_image(path):
    """NO-INFORMATION control: same dimensions, same 8-bit format, no
    structure at all."""
    write_png_gray(str(path), np.full((H, W), 128, dtype=np.uint8))


# -- one API call, cached ------------------------------------------------------

class Ask:
    def __init__(self, cli, model):
        self.cli, self.model = cli, model
        self.cache_path = SPEC / "response_cache.json"
        self.cache = (json.loads(self.cache_path.read_text())
                      if self.cache_path.is_file() else {})

    def __call__(self, tag, png, state, instructions, criteria):
        key = hashlib.sha256(
            "|".join([tag, self.model, state.replace("<image:", "<"),
                      json.dumps([instructions, criteria], sort_keys=True),
                      hashlib.sha256(Path(png).read_bytes()).hexdigest()])
            .encode()).hexdigest()
        if key in self.cache:
            return dict(self.cache[key], cached=True)
        from typesafe_sdk import Choice, Image
        t0 = time.time()
        try:
            r = self.cli.system_one(
                state=state.format(img=Image(str(png))),
                questions={"q": Choice(instructions=instructions,
                                       criteria=criteria)},
                model=self.model, timeout=180.0)
            a = r.answers["q"]
            row = {"choice": a.choice, "confidence": round(a.confidence, 4),
                   "probabilities": {k: round(v, 4)
                                     for k, v in a.probabilities.items()},
                   "input_tokens": r.usage.input_tokens,
                   "latency_s": round(time.time() - t0, 2), "error": None}
        except Exception as exc:                # noqa: BLE001
            row = {"choice": None, "confidence": None, "probabilities": None,
                   "input_tokens": None, "latency_s": round(time.time() - t0, 2),
                   "error": f"{type(exc).__name__}: {exc}"[:300]}
        self.cache[key] = row
        self.cache_path.write_text(json.dumps(self.cache))
        return dict(row, cached=False)


def preflight(cli):
    """The image must be on the wire.  An attached image costs input tokens;
    a silently dropped one does not (the documented trap in
    spectrogram_vision.py -- without the opt-in header the SDK sends the
    reference string and the model answers from an empty utterance)."""
    from typesafe_sdk import Choice, Image
    png = SPEC / "preflight.png"
    render_window(np.zeros(int(WINDOW_S * SR)), WINDOW_S / 2, png)

    def tokens(state):
        r = cli.system_one(state=state,
                           questions={"p": Choice(
                               instructions="Is there an image here?",
                               criteria={"yes": None, "no": None})},
                           model=MODEL)
        return r.usage.input_tokens

    with_img = tokens(f"a spectrogram: {Image(str(png))}")
    without = tokens("a spectrogram: (none)")
    ok = with_img > without
    return {"input_tokens_with_image": with_img,
            "input_tokens_without": without, "image_on_the_wire": ok}


def positive_control(ask):
    """A signal that IS visible in the image: three tones at 25/50/75% of a
    600 Hz axis.  Prior run (spectrogram_vision.py, jev-1.14.0): 1/3 -- the
    random floor.  If this misses again, a null on the main question is
    dominated by 'the model cannot read spectrograms'."""
    rows = []
    for label, hz, truth in (("low_150hz", 150.0, "low"),
                             ("mid_300hz", 300.0, "mid"),
                             ("high_450hz", 450.0, "high")):
        png = SPEC / f"tone_{label}.png"
        render(spectrogram(tone(hz, 0.5), fmax=600.0), str(png))
        r = ask(f"pitch|{label}", png,
                "The spectrogram of one spoken utterance: {img}",
                PITCH_Q, PITCH_OPTIONS)
        rows.append({"case": label, "truth": truth,
                     "picked": r.get("choice"), "row": r})
    correct = sum(1 for r in rows if r["picked"] == r["truth"])
    return {"rows": rows, "accuracy": f"{correct}/{len(rows)}",
            "random_floor": "1/3"}


# -- metrics -------------------------------------------------------------------

def pdiff(row):
    p = row.get("resp", {}).get("probabilities")
    return None if not p else p.get("different")


def metrics(rows, tag):
    """Accuracy vs both baselines, threshold-free AUC with a bootstrap CI,
    and the degeneracy check that catches a model answering one fixed thing."""
    good = [r for r in rows if r["resp"].get("error") is None]
    pos = [pdiff(r) for r in good if r["label"] == "CHANGE"]
    neg = [pdiff(r) for r in good if r["label"] == "SAME"]
    n_ok, n_err = len(good), len(rows) - len(good)
    tp = sum(1 for r in good
             if r["label"] == "CHANGE" and r["resp"]["choice"] == "different")
    tn = sum(1 for r in good
             if r["label"] == "SAME" and r["resp"]["choice"] == "same")
    tpr = tp / max(1, len(pos))
    tnr = tn / max(1, len(neg))
    acc = (tp + tn) / max(1, n_ok)
    # accuracy reweighted to the POOLED corpus base rate, so the number can
    # be compared with 0.6225 directly even though the sample is balanced
    acc_nat = POOLED_BASELINE["base_rate_change"] * tpr \
        + (1 - POOLED_BASELINE["base_rate_change"]) * tnr
    a = auc(pos, neg)

    rng, boots = np.random.default_rng(0), []
    for _ in range(10000):
        b = [good[i] for i in rng.integers(0, len(good), len(good))]
        p = [pdiff(r) for r in b if r["label"] == "CHANGE"]
        q = [pdiff(r) for r in b if r["label"] == "SAME"]
        if p and q:
            boots.append(auc(p, q))
    lo, hi = (round(float(np.percentile(boots, 2.5)), 4),
              round(float(np.percentile(boots, 97.5)), 4)) if boots else (None, None)

    scores = [pdiff(r) for r in good]
    distinct = sorted({r["resp"]["choice"] for r in good})
    const = (len(distinct) == 1)
    return {"tag": tag, "n": len(rows), "n_scored": n_ok, "n_api_error": n_err,
            "accuracy_balanced_sample": round(acc, 4),
            "on_sample_always_change_baseline": 0.5,
            "pooled_corpus_baseline_1396": POOLED_BASELINE["base_rate_change"],
            "accuracy_reweighted_to_pooled_base_rate": round(acc_nat, 4),
            "tpr_change": round(tpr, 4), "tnr_same": round(tnr, 4),
            "auc_p_different": a,
            "auc_bootstrap95": [lo, hi],
            "distinct_choices": distinct,
            "constant_answer": const,
            "p_diff_mean": round(float(np.mean(scores)), 4) if scores else None,
            "p_diff_std": round(float(np.std(scores)), 4) if scores else None,
            "beats_pooled_baseline": bool(acc_nat > POOLED_BASELINE["base_rate_change"])}


# -- run -----------------------------------------------------------------------

def main() -> int:
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=str(ROOT / "experiments"
                                         / "spectrogram_turn_eval_results.json"))
    args = ap.parse_args()
    MODEL = args.model
    SPEC.mkdir(parents=True, exist_ok=True)

    if not (os.environ.get("TYPESAFE_API_KEY")
            or os.environ.get("TYPESAFEAI_KEY")):
        print("no TYPESAFE_API_KEY (source /home/proart/projects/glasses/.env)")
        return 2

    print("== population (watcher pause pairs; ~53 s per meeting uncached) ==")
    pairs = population()
    n_ch = sum(1 for p in pairs if p["label"] == "CHANGE")
    ex = eligible(pairs)
    ex_ch = sum(1 for p in ex if p["label"] == "CHANGE")
    print(f"   pooled {len(pairs)} pairs, base rate CHANGE "
          f"{n_ch / len(pairs):.4f} (turn_eval.py measured 1396 / 0.6225)")
    print(f"   window fits for {len(ex)} ({ex_ch} CHANGE / {len(ex) - ex_ch} SAME)")

    chosen = sample(ex, args.n_per_class, args.seed)
    got_ch = sum(1 for p in chosen if p["label"] == "CHANGE")
    print(f"   sampled {len(chosen)} ({got_ch} CHANGE / {len(chosen) - got_ch} "
          f"SAME) seed {args.seed}, de-overlapped at {WINDOW_S / 2} s")

    from typesafe_sdk import TypeSafeClient
    cli = TypeSafeClient(headers=IMAGE_HEADERS)
    ask = Ask(cli, MODEL)

    print(f"\n== preflight (model {MODEL}) ==")
    pre = preflight(cli)
    print(f"   {pre}")
    if not pre["image_on_the_wire"]:
        print("   ABORT: image not on the wire; every number would be about "
              "its absence")
        return 3

    print("\n== positive control: gross visible pitch bands ==")
    pc = positive_control(ask)
    print(f"   accuracy {pc['accuracy']} (floor {pc['random_floor']})")

    wavs = {stem: wavio.read_wav(str(CORPUS / f"{stem}.wav")).samples
            for stem in {p["meeting"] for p in chosen}}
    blank = SPEC / "blank.png"
    blank_image(blank)

    print(f"\n== main: {len(chosen)} boundaries -> spectrogram images ==")
    rows = []
    for i, p in enumerate(chosen):
        png = SPEC / f"win_{p['meeting']}_{p['idx']:04d}.png"
        img, lfrac, rfrac = render_window(wavs[p["meeting"]], p["mid"], png)
        r = ask("handoff", png, HANDOFF_STATE, HANDOFF_INSTR,
                HANDOFF_CRITERIA)
        rows.append({"meeting": p["meeting"], "idx": p["idx"],
                     "label": p["label"], "from": p["from"], "to": p["to"],
                     "group_rel": ("cross" if p["from"][:3] != p["to"][:3]
                                   else "within"),
                     "gap_s": round(p["gap"], 3), "mid_s": round(p["mid"], 3),
                     "img_mean_brightness": round(float(img.mean()), 2),
                     "img_left_bright_frac": round(lfrac, 4),
                     "img_right_bright_frac": round(rfrac, 4),
                     "resp": r})
        if (i + 1) % 20 == 0:
            print(f"   {i + 1}/{len(chosen)}")

    print("\n== no-information control: blank image, same labels ==")
    crows = []
    for i, p in enumerate(chosen):
        r = ask("handoff-blank", blank, HANDOFF_STATE, HANDOFF_INSTR,
                HANDOFF_CRITERIA)
        crows.append({"meeting": p["meeting"], "idx": p["idx"],
                      "label": p["label"], "resp": r})
        if (i + 1) % 20 == 0:
            print(f"   {i + 1}/{len(chosen)}")

    # image-token guard: any call whose input tokens fell to the text-only
    # level had its image silently dropped
    tok_only = pre["input_tokens_without"]
    dropped = [r["meeting"] + "#" + str(r["idx"]) for r in rows + crows
               if r["resp"].get("input_tokens") is not None
               and r["resp"]["input_tokens"] <= tok_only + 5]

    main_m = metrics(rows, "main")
    blank_m = metrics(crows, "blank_control")

    # class split: hand-offs across the two F0 groups (spectrally obvious)
    # vs within one group (the measured hard case)
    split = {}
    for rel in ("cross", "within"):
        sub = [r for r in rows if r["group_rel"] == rel
               and r["label"] == "CHANGE" and r["resp"].get("error") is None]
        if sub:
            split[rel] = {"n_change": len(sub),
                          "mean_p_different": round(float(
                              np.mean([pdiff(r) for r in sub])), 4),
                          "frac_predicted_different": round(float(
                              np.mean([r["resp"]["choice"] == "different"
                                       for r in sub])), 4)}
    same_rows = [r for r in rows if r["label"] == "SAME"
                 and r["resp"].get("error") is None]
    if same_rows:
        split["same_class"] = {"n_same": len(same_rows),
                               "mean_p_different": round(float(
                                   np.mean([pdiff(r) for r in same_rows])), 4)}

    # sanity: no trivial image statistic separates the classes, and the
    # sample's gap distribution carries no more than the known 0.529
    br = auc([float(r["img_mean_brightness"]) for r in rows
              if r["label"] == "CHANGE" and r["resp"].get("error") is None],
             [float(r["img_mean_brightness"]) for r in rows
              if r["label"] == "SAME" and r["resp"].get("error") is None])
    gap_s = auc([r["gap_s"] for r in rows if r["label"] == "CHANGE"
                 and r["resp"].get("error") is None],
                [r["gap_s"] for r in rows if r["label"] == "SAME"
                 and r["resp"].get("error") is None])

    # speech-presence audit of the rendered halves: a half with almost no
    # energy above -37 dB cannot inform any reader, and the audit exposes
    # whether that falls on one class
    halves = {}
    for lab in ("CHANGE", "SAME"):
        fr = [min(r["img_left_bright_frac"], r["img_right_bright_frac"])
              for r in rows if r["label"] == lab]
        halves[lab] = {"n": len(fr),
                       "windows_with_side_below_0.05": sum(1 for f in fr if f < 0.05),
                       "min_side_frac": round(min(fr), 4) if fr else None}
    all_sides = ([r["img_left_bright_frac"] for r in rows]
                 + [r["img_right_bright_frac"] for r in rows])

    report = {
        "question": "Can jev classify SAME vs CHANGE speaker at a turn "
                    "boundary from a spectrogram image?",
        "model": MODEL,
        "image": {"size": [W, H], "window_s": WINDOW_S, "fmax_hz": FMAX,
                  "centre_marker_px": 2, "per_window_db_norm": True,
                  "phase": "discarded by construction (magnitude spectrogram)"},
        "population_check": {"pooled_pairs": len(pairs),
                             "pooled_base_rate_change": round(n_ch / len(pairs), 4),
                             "turn_eval_reference": POOLED_BASELINE},
        "sample": {"n": len(chosen), "n_change": got_ch,
                   "n_same": len(chosen) - got_ch, "seed": args.seed,
                   "balanced": True, "min_window_separation_s": WINDOW_S / 2,
                   "group_rel_counts": {
                       rel: sum(1 for r in rows if r["group_rel"] == rel
                                and r["label"] == "CHANGE")
                       for rel in ("cross", "within")}},
        "preflight": pre,
        "positive_control_pitch_bands": pc,
        "main": dict(main_m, rows=rows),
        "no_information_control": dict(blank_m, rows=crows),
        "change_split_by_f0_group": split,
        "sanity": {"auc_mean_brightness": br,
                   "auc_gap_seconds_on_sample": gap_s,
                   "pooled_gap_auc_reference": 0.529,
                   "image_dropped_calls": dropped,
                   "speech_presence_audit": {
                       "per_side_bright_frac_range": [
                           round(min(all_sides), 4), round(max(all_sides), 4)],
                       "by_class": halves,
                       "threshold": "fraction of half-window pixels above -37 dB"}},
        "reproduce": ("set -a && . /home/proart/projects/glasses/.env && "
                      "set +a && /home/proart/projects/glasses/.venv/bin/"
                      f"python experiments/spectrogram_turn_eval.py "
                      f"--n-per-class {args.n_per_class} --seed {args.seed} "
                      f"--model {MODEL}"),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nmain:   {json.dumps({k: v for k, v in main_m.items()})}")
    print(f"blank:  {json.dumps({k: v for k, v in blank_m.items()})}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
