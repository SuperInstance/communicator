# Finding: `GATE_FIELD` does not report what it says it reports

> ## ⚠ RETRACTED — there is no bug here, and the "14.5×" is withdrawn
>
> **This document's finding is wrong.** `GATE_FIELD` does report the gate's max
> over the record's own interval, exactly as documented. The code was never
> broken.
>
> Commit `f0a566f` claimed a 14.5× understatement caused by `_trim`. It
> measured that by comparing **the gate's max over the entire signal (0.7293)
> against one record's value at t≥4s (0.0503)** — two different intervals. The
> 0.7293 peak sits at **t = 0.064 s, frame 2 of 500**: an LSTM start-up
> transient that falls inside the *first* record only. Every later record
> genuinely has ~0.05, and `GATE_FIELD` reports it faithfully.
>
> Measured properly — one gate over the whole stream, then each record's
> absolute index range looked up in that single series — every record's ratio
> is **1.000** (worst 1.000). Reproduce with
> `experiments/gate_peak_verdict.py`.
>
> **The control was badly designed, and so was mine.** The original docstring
> dismissed the warm-up hypothesis with "max over the first 0.5 s gives the same
> 0.7293" — but a warm-up transient and real signal at the start answer that
> question identically, so the control could not distinguish them. Worse, a
> re-check that appeared to *confirm* the bug (a flat ratio of 0.07 across all
> records) was itself an artefact: it ran a **fresh `SileroGate()` over each
> record's slice**, and the gate is an LSTM, so every slice re-ran the same
> start-up transient and reported its own 0.7293 regardless of position.
> Slicing is not a control, it is a confound.
>
> This is the third time this project has produced a confident conclusion from
> a statistic summarised over the wrong interval — including the one the
> retracted claim was about. That pattern is the finding worth keeping.
>
> The genuine defect this document accidentally pointed at is real and is fixed
> elsewhere: `_gate_peak` took a max over too long a window, letting one
> start-up transient clear a 2 s window. See `c2319a0` and `GATE_MIN_FRAMES`.

**Status:** retracted. Nothing here is a defect in `GATE_FIELD`.

> ### ⚠ CORRECTION (2026-10-19) — the body below was never amended
>
> The banner above withdraws the finding, but the body it retracts was left
> standing, and a reader who arrives at §"The measurement" finds it asserting
> the withdrawn claim. Four passages are corrected in place below: the **14×
> factor** (two places), the **"Not the LSTM, and not a warm-up artefact"**
> dismissal, and the **"The cause is buffer trimming"** attribution.
>
> What the body should have said, from
> `experiments/gate_peak_verdict_results.json`: `whole_signal_max` =
> **0.7293** at **t = 0.064 s, frame 2 of 500**, and **every per-record ratio is
> 1.000** (median 1.000, worst 1.000) when each record's own absolute index
> range is looked up in one gate series over the whole stream. So the 0.7293
> peak **is** the LSTM start-up transient — exactly what the body rules out —
> and buffer trimming is not the cause, because `GATE_FIELD` does not
> under-report. `NOTES-AUDIT.md` F5.

---

## The claim being made

`src/watcher.py` documents the field it attaches to every utterance record:

> `GATE_FIELD` — The gate's verdict, carried as DATA on every utterance record.
> … Carrying the number on the record means any segment the gate dismissed can
> be found and audited afterwards — a decision that leaves no trace is not
> auditable, and an unauditable decision in the continuous path is exactly the
> kind that gets discovered a week later as a missing baseline.

and `_gate_peak` is documented as:

> The gate's peak probability over a segment.

**Measured: it is the peak over the portion of the segment still inside the
retained buffer, not over the segment.**

## The measurement

A 200 Hz three-harmonic stack at −26 dBFS, fed through `Watcher` exactly as
production does — one continuous gate, 100 ms chunks:

| segment length | true max over segment | `gate_speech_prob` reported | `pitch_skipped` |
|---:|---:|---:|---|
| 0.5 s | 0.7293 | *(no utterance emitted)* | — |
| 1.0 s | 0.7293 | *(no utterance emitted)* | — |
| 2.0 s | 0.7293 | *(no utterance emitted)* | — |
| 4.0 s | **0.7293** | **0.0503** | True |
| 8.0 s | **0.7293** | **0.0539** | True |
| 16.0 s | **0.7293** | **0.0523** | True |

The true peak is constant at 0.7293. The reported value is ~0.05 at every
duration. That is a factor of **14**.

> **WITHDRAWN (2026-10-19).** The factor of 14 is the retracted quantity. It
> divided the gate's max over the **whole signal** (0.7293, at t = 0.064 s)
> by one record's value at t ≥ 4 s (0.0503) — two different intervals. With one
> gate over the whole stream, every record's ratio is **1.000**. No 14× exists.
> `experiments/gate_peak_verdict_results.json`; `NOTES-AUDIT.md` F5.

## Cause

~~Not the LSTM, and not a warm-up artefact.~~ **REVERSED (2026-10-19).** The
peak at **t ≈ 0.064 s** *is* the LSTM start-up transient — the retraction in the
banner says so and this body asserted the opposite. What the measurement does
show is that the maximum is identical over the first 0.5 s, the first 2 s and
all 16 s, which is a property of the signal, not evidence against a warm-up.
The control the original docstring used ("max over the first 0.5 s gives the
same 0.7293") could not exclude a warm-up at all: the transient sits inside the
first 0.5 s, so that control returns the transient's value either way.
`NOTES-AUDIT.md` F5.

Running one continuous gate over the
same signal, the maximum occurs at **t ≈ 0.064 s** and is identical whether you
take the max over the first 0.5 s, the first 2 s, or all 16 s:

```
amp  max 0-0.5s  max 0-1s  max 0-2s  max 0-8s
0.05      0.7293    0.7293    0.7293    0.7293
```

~~The cause is **buffer trimming**.~~ **REJECTED (2026-10-19).** Buffer
trimming is the competing explanation the retraction rejects: `_trim` does
discard probabilities whose frames start before the new buffer origin, but
`GATE_FIELD` reports the gate's max over the record's **own** interval, and
measured that way every record's ratio is **1.000** (worst 1.000). Nothing is
under-reported, so nothing needs a cause. `NOTES-AUDIT.md` F5.

`_trim` discards gate probabilities whose
frames start before the new buffer origin, and by the time a 4-second segment
closes, the buffer holds 6400 samples — 0.4 s. The peak lives at 0.064 s and is
long gone. After 8 s of audio the watcher's `_gate_probs` held **25 entries
with `base=115200`**, i.e. it starts at sample 7.2 s.

## Why the direction matters

Two consequences, and they point in opposite directions.

**In this case the gate did the right thing by accident.** The signal is hum, the
reported probability is low, so YIN is skipped and the CPU saving happens. The
mitigation works.

**But the audit trail is wrong, and that is the stated purpose of the field.**
The design rationale is that a human can find the segments the gate dismissed
and check them. A value that is off by 14× and systematically **under-reports
speech on long segments** cannot support that. Every long-segment dismissal is
recorded as near-silence when the gate's own opinion of that segment was 0.73.

> **WITHDRAWN (2026-10-19).** This paragraph asserts the withdrawn 14× factor
> and the under-reporting it caused. There is no under-reporting: with each
> record's own interval looked up in one gate series, every ratio is **1.000**
> and the field matches its own interval (`field_matches_own_interval: true`).
> The audit trail is not wrong — there is nothing to audit away.
> `experiments/gate_peak_verdict_results.json`; `NOTES-AUDIT.md` F5.

It also means the documented limit and the production number describe different
things. `tests/test_vad.py::TestDeclaredLimits` measures `seg_max` over a
0.5 s window — a fresh gate whose entire lifetime is that window — and reports
0.66–0.97. That reproduces exactly (measured 0.6597, 0.7821, 0.8183; stack
0.9729). But production never sees that regime: it runs one gate over a long
stream, and the peak is trimmed before the verdict is taken.

## A retraction I nearly made, recorded because it nearly happened

My first pass here claimed the declared limit **did not reproduce** and that the
gate scored 0.12 instead of 0.97. It did reproduce, exactly. I had changed three
things at once — duration (8 s vs 0.5 s), harmonic count (12 vs 3), and
statistic (mean vs `seg_max`) — and then read the disagreement as a defect in
someone else's test.

Measured with the test's own helpers imported verbatim, the declared limit holds.
The apparent contradiction was entirely mine.

## The fix, and the decision it needs

`_gate_peak` needs the segment's peak, but the probabilities are gone by the time
the segment closes. Three options, none obviously right:

1. **Accumulate the peak as the segment grows.** Track a running max of gate
   probability over the frames belonging to the current segment, updated per
   push, and use that at close. Correct, cheap, and it needs a reset wherever
   segments are closed.
2. **Stop trimming gate probabilities for the open segment.** Simple, and
   unbounded in the worst case — precisely the continuous-signal case.
3. **Accept the tail semantics and document them.** Cheapest, and honest only if
   `GATE_FIELD` is renamed to something like `gate_tail_prob` and the docstring
   stops claiming segment coverage.

Option 1 is the correct one and is a small change. But it alters a shipped
decision input, and `GATE_SPEECH` comparisons would start behaving differently
on long segments — some currently-skipped segments would start running YIN.
That cost should be measured before it is paid, not assumed.

## Reproduce

`experiments/gate_peak_coverage.py` in this repo prints the table above. The
one-line essence:

```python
# true max over the segment, from a standalone gate on the same signal
g = vad.SileroGate(); p = []
for i in range(0, len(sig), 1600): p.extend(g.push(sig[i:i+1600]))
max(p)                                   # 0.7293

# what the watcher records for that same segment
w = watcher.Watcher(16000, gate=vad.SileroGate())
...                                        # -> GATE_FIELD = 0.0539
```