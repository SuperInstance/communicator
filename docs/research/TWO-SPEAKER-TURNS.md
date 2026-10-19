# Two speakers at last: what the turn question actually answers

**Status:** the corpus blocker is broken. The first honest answer is negative.

`17ae46a` established the wall: every corpus file is one person reading alone,
so after any silence the same speaker resumes 98.1–100% of the time, and turn
accuracy is unmeasurable. `examples/realtwospeaker/` supplies what was missing —
AMI Meeting Corpus, far-field (SDM) channel, spontaneous four-party meetings,
every turn labelled by `speaker_id`. CC-BY-4.0, fetched with no token and no
registration.

## The blocker is genuinely broken

| corpus | after a ≥0.6 s silence, a **different** speaker takes the floor |
|---|---|
| LibriVox single-speaker (`17ae46a`) | **1.9–0.0%** (i.e. essentially never) |
| AMI EN2002a | **73.5%** (191 of 260 gaps) |
| AMI EN2002b | **67.5%** (129 of 191 gaps) |

Over 1396 watcher-detected pauses across 840 s of meeting audio, the base rate
of a genuine speaker change is **0.6225**. The class that could not be
represented is now represented, and it is the majority case.

This is the only number in this file that needed no caveat. The labels come
verbatim from AMI and the pauses come from our own pipeline.

## Silence duration does not classify a hand-off

Pooled over both meetings, using the gap the watcher itself reports:

**AUC = 0.529.** Barely distinguishable from a coin.

And every threshold loses to doing nothing clever:

| silence bar | accuracy | precision | recall | false interrupts |
|---:|---:|---:|---:|---:|
| **always predict change** | **0.6225** | — | — | 0 |
| 0.15 s | 0.5709 | 0.647 | 0.685 | 325 |
| 0.20 s | 0.5265 | 0.636 | 0.560 | 279 |
| 0.45 s | 0.4491 | 0.635 | 0.270 | 135 |
| 0.60 s | 0.4291 | 0.646 | 0.183 | 87 |
| 1.00 s | 0.4004 | 0.657 | 0.077 | 35 |
| 3.00 s | 0.3775 | — | 0.000 | 0 |

The best threshold reaches 0.5709 against a trivial baseline of 0.6225. Raising
the bar monotonically destroys it, because in a four-person meeting the
prolonged silences are the ones where **nobody** speaks — agreement, thinking,
waiting — not where someone hands over. pipecat's operational 200 ms/3 s pair
cannot be reproduced here, and on this evidence there is nothing to reproduce.

Note that precision sits near 0.64 across every bar: silence length is a weak
but *stable* filter, and its weakness is entirely in the wrong direction.

## A measurement that was wrong, and why

The first run reported 52.7% of adjacent turn pairs as **overlapping**, with an
annotation AUC of **0.289** — an inversion, reading as if a speaker change were
announced by *barge-in* rather than silence. Tempting, and wrong.

AMI pads each turn on its **leading** edge: `begin_time` runs 0.00–0.70 s ahead
of `audible_start_s`, with no trailing pad. Consecutive turns from different
speakers therefore "overlap" on paper purely because each one's lead-in sits
inside the previous one's span.

| adjacent pairs | on `begin_time`/`end_time` | on `audible_start`/`audible_end` |
|---|---:|---:|
| EN2002a (313 pairs) | 52.7% overlap | **0.0%** |
| EN2002b (229 pairs) | 40.2% overlap | **0.0%** |

The audible bounds make overlap not merely smaller but **exactly zero**,
because consecutive audible turns tile — they share a boundary to the
millisecond. A suspiciously round 0% is a harness signature, and checking it is
what caught this. The inversion was an annotation convention wearing a
psychophysical result's clothes.

## What this corpus cannot prove

Stated plainly, because the sweep above invites over-reading:

1. **The annotation gap is structurally 0.000 s.** Audible turns tile, so the
   annotations carry no gap information at all. Every gap number here comes
   from the watcher on real audio.
2. **That audio is a reconstruction.** Each utterance is placed at its
   annotated `begin_time`; unannotated stretches are digital silence. 83% of
   frames carry real signal, but the 11% of holes are placed by the annotation
   timeline, so their *position* is conventional rather than acoustic. Gaps in
   this corpus are therefore not a faithful sample of meeting-room silences,
   and the AUC of 0.529 is weaker evidence than it looks.
3. **Far-field, one microphone.** 111 of 314 segments are wholly overlapped and
   inaudible. Real barge-in — the phenomenon that actually decides floors in
   conversation — is partly destroyed by the channel before we ever see it.
4. **Two meetings, cherry-picked windows.** 840 s total, selected for turn
   density. Meeting-wide rates (73.5% / 67.5%) are the honest figures; the
   0.6225 slice rate overstates them.

So: the blocker is broken and the base rate is solid. The negative result on
silence duration is *suggestive*, and the way to settle it is a continuous
multi-channel meeting recording where gaps are acoustic rather than
annotation-placed.

## What it does not change

`src/turn.py`'s 3.0 s give-up and 5.0 s yield floor are untouched. This is a
consumer stage by design: it may add information and is never allowed to remove
any. A measurement that says "silence length alone is nearly worthless" is
exactly the input that stage needs — it means the decision must come from
something other than duration.

## Reproduce

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/turn_eval.py
/home/proart/projects/glasses/.venv/bin/python experiments/twospeaker_verify.py
```