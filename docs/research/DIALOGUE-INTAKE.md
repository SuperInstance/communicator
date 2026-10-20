# DIALOGUE-INTAKE — can the real-dialogue intake do its job?

**Status: measured 2026-10-19.** Code: `experiments/dialogue_intake_selftest.py`.
Results: `experiments/dialogue_intake_selftest_results.json`.
Tests: `tests/test_dialogue_intake.py` (24 tests).

**The short answer: no, not yet, and not for the reason it would fail.**
The intake's structure is sound — it finds the pairs, it measures them, it
can reject a corpus, and it *did* accept a full-length good corpus in this
test. But three of its gates do not measure what the module docstring says
they measure, and the one that matters most — the hand-off count — counts
**the opposite of a hand-off**. A recording made exactly to the protocol in
`examples/realdialogue/README.md` is **rejected** by this tool with
*"zero hand-offs: no silence is followed by the other speaker, so the
positive class is still absent"*, while the corpus plainly contains eight
canonical hand-offs.

That is the same class of defect as the corpus capture's A/V
non-correspondence: a number that looks like a measurement of the thing and
is not a measurement of the thing at all.

---

## 1. Why this file exists

`examples/realdialogue/raw/` is empty and `experiments/dialogue_intake.py`
has never been run against a single real file. The project has already paid
for that once, on the other corpus (`docs/research/AV-SYNC.md`):

1. the audio and the video did not correspond, and nothing recorded that
   they should;
2. the camera delivered 25 fps where 30 was assumed, silently stretching the
   time base by 1.2×;
3. the achievement check never ran — the script's `src` import resolved to
   the wrong path, the exception was swallowed, and every take was saved
   with *"speech check unavailable"* and no verification at all.

So the question is not "does the intake work". It is: **for each thing the
intake claims to check, can a bad corpus actually make it say so?** A gate
that has only ever been observed accepting things is not known to work — it
might be accepting everything.

## 2. Method

Synthetic matched pairs, built to look like the eight scenes in the
README, written with the standard-library `wave` module (the same module
`src/wavio` reads with) and read back through `src.wavio.read_wav`. Every
number in the results file comes from running the **real**
`experiments/dialogue_intake.py` as a subprocess and parsing its JSON. The
only quantities this lane computes itself are the ones the intake does not
output at all (the measured gaps, the overlap boundary, the clap offset),
and those are computed with a mirror of the intake's own logic, stated in
the code so the two can be compared.

All corpora are written to a temp directory. `examples/realdialogue/raw/`
was never touched — and there is a test that says so.

The speech is a harmonic stack with a syllable-rate envelope and aspiration
noise. It is not speech and does not try to be: this is a test of the
**intake**, which never does speech recognition. What matters is that the
signal has the properties the gates key on — loud and quiet regions, silences
of known length, two independent channels, a clap at a known instant, and an
inter-channel offset known exactly.

### The two corpora

| corpus | what it is | what it is for |
|---|---|---|
| **matrix** | 8 scenes × 12 s = 96 s, room tone at 1e-5 RMS | the perturbation battery; fast |
| **protocol-length** | 8 scenes at the README's scene durations × ½ = **615 s** | whether the intake can say USABLE at all |

The matrix corpus is deliberately short, so the intake's protocol-length
gate ("only 96s of audio; the protocol asks for ~25 min") fires on it. Every
verdict below is about the gates **other than** that one, and the
protocol-length run is the one that clears it.

### What the good corpora produce

| | matrix (96 s) | protocol-length (615 s) |
|---|---:|---:|
| status | NOT YET USABLE | **USABLE** |
| exit code | 0 | 0 |
| matched pairs | 8 | 8 |
| total audio | 96.0 s | 615.0 s |
| silences seen | 38 | 221 |
| hand-offs seen | 1 | 5 |
| `unmet` | length only | **empty** |
| wall time | 10.4 s | 69.4 s |

**The intake does accept a good corpus.** That part of the tool works, and
it is worth saying so plainly before the list of what does not.

## 3. The gates, and which of them were exercised

| gate | where it lives | exercised? | result |
|---|---|---|---|
| matched pairs | `main()` `scenes` dict | yes | reported, **never gated** (see §4) |
| protocol length | `total_s < 600` | yes | **gated** |
| level: digital silence | `p["A"].get("silent")` | yes | **gated for A only** |
| level: low but non-zero | — | yes | **no gate at all** |
| cross-talk | `crosstalk()` | yes | measured, **not gated**, and **not separable** |
| signal fraction | `signal_frame_frac` | yes | reported, not gated |
| overlap | `both_loud_frac` | yes | reported, **not gated** |
| hand-offs | `hand_offs` counter | yes | **gated, and it measures the wrong thing** |
| drift / alignment | — | yes | **does not exist** |
| sample rate | `raw_rate_hz` | yes | **always 16000** |

## 4. The pass/fail table

Every row is a separate corpus, run through the real intake. "Reported" =
the intake's own output contains the evidence. "Gated" = it changes
`gates.status` / `gates.unmet`. A row that is neither is a **finding**, not
a test failure, and the perturbation was **not** weakened to make it pass.

| # | perturbation | reported | gated | verdict |
|---|---|:--:|:--:|---|
| 1 | one channel of a pair missing | yes (`problem: "unmatched pair"`) | **no** | REPORTED, NOT GATED |
| 2 | channel A digitally silent | yes (`silent: true`) | **yes** | DETECTED (gated) |
| 3 | channel B digitally silent | yes (`silent: true`) | **no** | REPORTED, NOT GATED |
| 4 | channel B at RMS 2e-4 (low, non-zero) | yes (numbers only) | **no** | REPORTED, NOT GATED |
| 5 | channel B at RMS 5e-5 (below the floor) | yes (numbers only) | **no** | REPORTED, NOT GATED |
| 6 | cross-talk: B clearly hears A | **no** | **no** | **NOT DETECTED** |
| 7 | no overlap anywhere (7 of 8 scenes) | yes (`both_loud_frac` → 0) | **no** | REPORTED, NOT GATED |
| 8 | no hand-off anywhere | yes (`hand_offs_seen: 0`) | **yes** | DETECTED (gated) |
| 9 | different sample rates on the two channels | **no** | **no** | **NOT DETECTED** |
| 10 | a file named `.wav` that is not one | yes (`problem: "unreadable"`) | **no** | REPORTED, NOT GATED |

Row 10 deserves a footnote: the corpus *is* rejected, but by the hand-off
gate, as a side effect — the unreadable channel was scene 6's, and losing it
takes the corpus's only barge-back with it, so `hand_offs_seen` falls to 0
and *"zero hand-offs"* appears. The unreadable file itself is never the
reason.

### The rows that need explaining

**#1, unmatched pair — reported but never gated.** `main()` computes
`ok = all(p.get("has_A") and p.get("has_B") for p in report["pairs"]) and
not report["unmatched"] and report["pairs"]` and then **never uses it**.
That dead variable is the "matched pairs" gate, unwired. Meanwhile the
broken pair is still appended to `report["pairs"]`, so
`gates.matched_pairs` counts it as matched (8, with a channel missing) and
nothing in `gates.unmet` mentions it. In the measured run the corpus's
`unmet` gains **nothing at all** — its only unmet requirement is the
protocol-length one, which the good corpus has too. The corpus is rejected
for the length and for nothing else.

**#3, silent channel B — reported but never gated.** The silent check reads
`p["A"].get("silent")` only. This is not merely an oversight: scene 7 of
the protocol is *specified* to have a silent channel B, so a blanket "B is
silent ⇒ error" rule would be wrong. The intake cannot distinguish a dead
microphone from a deliberately unused one, and it does not claim to.

**#4, #5, a low channel — no threshold exists.** A channel at RMS 2e-4
reports `silent: false`, `rms: 0.0002`, `signal_frame_frac: 0.0042`; at
RMS 5e-5 it reports `silent: false`, `rms: 5e-05`,
`signal_frame_frac: 0.0021`. Nothing is gated. Note that the reported
`signal_frame_frac` for the low channel (0.0042) is **the same** as for a
digitally silent channel (0.0042) — the report cannot tell "quiet but
alive" from "dead mic" either.

**#6, cross-talk — not detected, and the number is not separable.** The
intake's cross-talk statistic correlates the *window means* of the ten
loudest simultaneous windows. Measured on the same two corpora:

| | clean corpus | B hears A | separation |
|---|---:|---:|---:|
| the intake's statistic | 0.974 – 0.983 (2 of 8 scenes computable) | 0.990 – 0.998 | **0.0067** |
| zero-mean waveform correlation (clap excluded) | −0.027 – 0.006 (2 of 8 computable) | 0.75 – 1.00 | **0.7452** |
| **"B loud when A loud"** | **0.000 – 0.3125** | **1.0000** | **0.6875** |

The intake's number reads 0.97+ on a corpus with **clean separate
microphones**, and 0.99 on a corpus where B hears A clearly. It does not
distinguish them. In the six scenes with no overlap it returns `None` —
*"too few simultaneous-loud windows to judge"* — so cross-talk is
unjudgeable exactly where it would be easiest to miss. The fix is measured,
not guessed: *"of the time A is loud, how much of it is B loud too?"*
separates 0.000–0.3125 (clean) from 1.0000 (corrupted) on the same signals.

**#7, no overlap — measured, never gated.** With the two speakers strictly
alternating in seven of eight scenes, `both_loud_frac` falls to 0.0000 in
those scenes, and the corpus's verdict is **unchanged** from the baseline:
the one kept barge-back preserves the hand-off count, so the hand-off gate
never fires, and `unmet` gains nothing. **The barge-in scenes could be
missing from seven of eight scenes and the intake would not notice.**

**#9, sample-rate mismatch — not detected.** The header of the 22050 Hz
file says 22050. The intake reports `raw_rate_hz: 16000` for **both**
channels, because `src.wavio.read_wav` resamples to the analysis rate and
returns an `Audio` whose `sample_rate` is 16000. The field name claims the
file's rate and never carries it. The *duration* survives, because it is
computed after the resample.

## 5. The alignment measurement — the big one

`examples/realdialogue/README.md` says the clap "gives an unambiguous
alignment point across the two files without needing a shared clock".

**Nothing in this repo aligns the two channels.** A search of every tracked
`.py` file for `align`, `clap`, `xcorr`, `cross-correlation` finds only:
the A/V-sync experiment (video motion against audio — a different thing),
prose uses of "align" ("an audio-only split aligns with the labelled…",
"the envelope stays aligned with frame times"), and the intake's **own
docstring**, which says that it "cannot align channels beyond the clap
described in the README, and it does not attempt to". The word "clap"
appears nowhere in the intake's code — only in that sentence. There is no
code anywhere in the repo that relates two channels of one take **in time**.
The intake's `crosstalk()` correlates *levels*, not *time*. The clap is
never read.

### What an unaligned offset does to the numbers

A probe pair with a known geometry: A speaks 0.60–2.00 s, B speaks
2.95–3.60 s, A resumes at 3.20 s. The **true** hand-off gap — how long
after A stops speaking does B start — is **0.95 s**. Channel B is then
delayed by a known offset and the intake's own rule is applied to the
shifted pair.

| offset | measured gap (5 ms resolution) | error vs 0 ms | measured gap (the intake's 100 ms grid) | error vs 0 ms | intake hand-offs | overlap boundary |
|---:|---:|---:|---:|---:|---:|---:|
| 0 ms | 0.95 s | 0.00 s | 0.90 s | 0.00 s | 1 | 3.2 s |
| 40 ms | 0.99 s | **+0.04 s** | 0.90 s | 0.00 s | 1 | 3.2 s |
| 120 ms | 1.07 s | **+0.12 s** | 1.00 s | +0.10 s | 1 | 3.2 s |
| 300 ms | 1.25 s | **+0.30 s** | 1.20 s | +0.30 s | **0** | 0.6 s |
| 600 ms | 1.55 s | **+0.60 s** | 1.50 s | +0.60 s | **0** | 0.9 s |

**The measured gap moves by the whole offset.** 40 ms → +0.04 s,
120 ms → +0.12 s, 300 ms → +0.30 s, 600 ms → +0.60 s. There is no
correction and no warning, because there is no alignment step to correct
with.

**And the count itself flips.** The intake counts a hand-off when channel B
is loud at the instant A's silence *ends*. Shifting B by 300 ms moves its
utterance off that instant, and the hand-off count for that scene goes
**from 1 to 0** — a scene that contains a hand-off is reported as
containing none.

(The intake's own 100 ms measurement grid cannot even resolve a 40 ms
offset: it reads 0.00 s of movement. The 5 ms column is the underlying
signal; the grid column is what the intake's resolution would let a
downstream consumer see.)

### The fix, measured

The clap the README already asks for recovers the offset exactly. The
1 ms energy envelope of the first 2 s of each file, normalised and
cross-correlated over ±600 ms, peaks at the true offset:

| true offset | clap offset detected | peak r | residual gap error after correcting | hand-offs after correcting |
|---:|---:|---:|---:|---:|
| 40 ms | −40 ms | 0.7587 | **0.00 s** | 1 (was 1) |
| 120 ms | −120 ms | 0.7587 | **0.00 s** | 1 (was 1) |
| 300 ms | −300 ms | 0.7587 | **0.00 s** | **1 (was 0)** |
| 600 ms | −600 ms | 0.7587 | **0.00 s** | **1 (was 0)** |

The correction costs nothing at the recording stage — the clap is already
in the protocol, in both channels, at the start of every scene.

### The limit of that fix: drift

A single clap offset corrects the offset *at the clap*. Two independent
recorders also drift. Measured with a 1.25% time-base error (250 ms over
20 s — far larger than a real crystal's, used to make the consequence
measurable rather than to claim a realistic magnitude):

| | value |
|---|---:|
| drift at the gap | 187.5 ms |
| true gap | 0.40 s |
| measured gap with no drift | 0.40 s |
| measured gap with the drift | **0.21 s** |
| **gap error caused by the drift alone** | **−0.19 s** |
| clap offset detected (at the start, where there is no drift yet) | 4 ms |
| drift reported by the intake | **nothing — no drift field exists** |

So the clap fixes the *start* of a take and leaves the rest to drift, and
the intake cannot see drift at all.

## 6. The hand-off semantics — the second big one

The intake's rule, from its own code: **a CLOSED pause on channel A whose
END coincides with channel B being loud**. A silence on A that never closes
— A stays quiet to the end of the file — is an *open* pause and is skipped.

Measured on isolated pairs:

| shape | what two people actually did | intake hand-offs |
|---|---|---:|
| **canonical hand-off** | A stops, B takes the floor, A stays quiet | **0** |
| canonical, A resumes much later while B is quiet | A stops, B answers, A comes back later | **0** |
| barge-back | A stops, B takes the floor, A comes back *while B is still talking* | **2** |

And at corpus level, with the protocol corpus built so that every turn is
canonical:

| | value |
|---|---:|
| canonical hand-offs present in the corpus | **8** |
| hand-offs the intake counts | **0** |
| status | **NOT YET USABLE** |
| reason | *"zero hand-offs: no silence is followed by the other speaker, so the positive class is still absent"* |

The tool rejects a perfectly protocol-shaped recording for lacking the
thing the recording contains.

Two further details in the same counter:

- **there is no minimum duration.** In the barge-back probe the intake's
  own pause list is `[(0.025, 0.285), (0.385, 0.985), (2.045, 2.985),
  (3.005, 3.025)]` — and the last of those, a **20 ms** dip, is counted as a
  hand-off, because there is no duration filter between "a pause" and "a
  hand-off". `energy.MIN_SILENCE_S` is 0.150 s, but the watcher re-derives
  its segmentation on every push and the merge does not protect the count.
  In the eight-scene corpus, three such sub-150 ms pauses survive in scene 2.
- **the count is a subset of the overlaps.** Every counted hand-off is a
  moment where both channels are loud, so a corpus with *no* overlap at all
  necessarily has zero hand-offs too. Removing all overlap therefore moves
  two gates at once, which is why the overlap perturbation in §4 keeps one
  barge-back: without it, the corpus would be rejected by the hand-off gate
  and the overlap gate would remain untested.

## 7. The level floor

The intake's absolute floor is `energy.ABS_FLOOR` = 1e-4 RMS (≈ −80 dBFS).
A quiet lived-in room is **23.5 dB above** that. The same corpus, replayed
with a realistic room tone (1.5e-3 RMS):

| | quiet room (1e-5 RMS) | realistic room (1.5e-3 RMS) |
|---|---:|---:|
| `signal_frame_frac` | 0.00 – 0.90 | **1.0 in every scene** |
| `both_loud_frac` | 0.0000 – 0.1255 | **1.0 in every scene** |
| cross-talk `corr` | None – 0.983 | **0.968 – 0.997** (0.413 in the room-tone scene) |
| hand-offs counted | 1 | **38** |

**In a realistic room the intake reports 100% signal and 100% overlap in
every scene, and a cross-talk correlation of 0.97–0.997, on a corpus that
contains no cross-talk at all.** Scene 8 — the room-tone scene, nobody
speaking — is reported as `signal_frame_frac: 1.0`, `both_loud_frac: 1.0`:
"a full minute of two people talking over each other" out of an empty room.
And the hand-off count rises from 1 to 38, every one of them an artefact of
the floor rather than a turn.

The watcher survives this because it uses an *adaptive* threshold
(`energy.adaptive_threshold`, a local moving quantile): it finds **the same
38 silences in both rooms**. The intake's own measures use the *absolute*
floor, and every one of them collapses — `signal_frame_frac` → 1.0,
`both_loud_frac` → 1.0, the hand-off count → 38 (one per silence, because
channel B is "loud" everywhere once the room tone is above the floor), and
the cross-talk number → 0.97+. The two halves of the repo disagree about
what "loud" means, and the intake's half is the wrong one for a real room.
`src/vad.py` already has a gate calibrated on exactly this kind of room tone,
and the intake does not use it.

## 8. The latent-defect checklist — the corpus capture's three defects

| # | the corpus capture's defect | the intake | verdict |
|---|---|---|---|
| 1 | the A/V correspondence was never verified | the intake never verifies the A/B correspondence either — there is no alignment and no drift measure | **FAIL** (§5) |
| 2 | the frame rate was assumed, never measured | the **duration** is read from the file and survives resampling. The **rate** is not: `raw_rate_hz` is always 16000 | **PARTIAL** |
| 3 | the achievement check never ran; the import resolved to the wrong path and the exception was swallowed | `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` — built from `__file__`, not from the cwd. Runs identically from `/` and from the repo root, both exit 0 with parseable JSON. The one `try/except` returns an error dict that is used (`problem: "unreadable"`); there is no other. Copied somewhere with no `src` beside it, it **fails loudly** (non-zero exit, no report). | **PASS** |
| 4 | (the exit-code half of defect 3) | `main()` returns `None` and never calls `sys.exit(1)`. The tool exits **0** while reporting `NOT YET USABLE`, and exits **0** while reporting `NOTHING RECORDED YET`. A pipeline that runs it cannot tell a rejected corpus from an accepted one. | **FAIL** |

Two more latent defects found while looking:

- **`ok` is dead code.** `main()` computes the matched-pairs verdict and
  never uses it (§4, row 1).
- **`src/wavio.py` has no writer.** It reads 8/16/32-bit PCM at any rate;
  nothing in the repo writes the WAVs the protocol asks for. The selftest
  supplies one (stdlib `wave`, the same module `src.wavio` reads with) and
  verifies the round-trip through `src.wavio.read_wav` on every file.

## 9. What must be fixed before Casey records

> **CORRECTION (2026-10-19, the fixes lane).** The paragraph below is no
> longer true. **All seven of the P0–P3 items in this section have been
> fixed**, in `experiments/dialogue_intake.py` and `src/wavio.py`, and the
> measurements that motivated each one were re-taken against the fixed tool:
> see **§12** and `experiments/dialogue_intake_fixes_results.json`. The
> section is kept as it was written because it is the reason §12 exists — a
> fix with no recorded audit behind it is a claim.
>
> One item is only PARTIALLY fixed, and the audit was right to call it out
> separately: **the drift**. The clap gives one alignment point, and with
> one alignment point the drift between two recorders **cannot be
> measured** — the two channels speak different words, so their envelopes
> are uncorrelated everywhere except at the clap. What the tool now does is
> report a **bound** derived from an assumed maximum clock error, and say
> that it is a bound. A bound is not a measurement and the report does not
> pretend otherwise.

Prioritised. **No change to `experiments/dialogue_intake.py` was made by
this lane** — it is the tool Casey will rely on and it was not this lane's
to change. Each item below is a precise fix, with the measurement that
motivates it.

### P0 — the hand-off rule measures the wrong thing

In `main()`:

```python
k = min(len(loud_b) - 1, int(e.t1 / 0.1))
if loud_b[k]:
    hand_offs += 1
```

should test whether B is loud **during** A's silence, not at its end:

```python
k0 = max(0, int(e.t0 / 0.1))
k1 = min(len(loud_b) - 1, int(e.t1 / 0.1))
if any(loud_b[k] for k in range(k0, k1 + 1)):
    hand_offs += 1
```

That counts "B takes the floor while A is quiet" — the canonical hand-off.

**Measured consequence** (the proposed rule is implemented in
`experiments/dialogue_intake_selftest.py` and run over the protocol-strict
corpus; the intake itself is untouched):

| | protocol-strict corpus |
|---|---:|
| canonical hand-offs actually in the corpus | 8 |
| hand-offs the intake counts today | **0** |
| hand-offs under the proposed rule, clap region skipped | **7** |
| hand-offs under the proposed rule, clap region included | 14 |

Three honest qualifications, all measured:

1. **the rule needs the clap excluded.** The clap is loud in *both* channels
   by construction, so it sits inside the lead-in silence and is counted as a
   hand-off: exactly one per scene, which is why the count doubles from 7 to
   14 when the clap region is included.
2. **the rule needs a minimum duration.** The proposed rule counts any
   pause, and the same 20 ms envelope dips that the current counter counts
   (see §6) still get counted. `energy.MIN_SILENCE_S` (0.150 s) should
   filter them.
3. **one canonical hand-off is still missed.** In scene 1 the last turn
   leaves A quiet to the end of the scene, so A's silence never *closes* and
   the closed-pause rule cannot see it. 7 of 8 is the number this change
   alone buys; the open-pause case needs handling too.

With those three, the "zero hand-offs" gate stops rejecting correct
recordings and starts doing its job.

### P0 — align the two channels with the clap, before any cross-channel number

Add a step that, for each pair, finds the clap in both files and shifts
channel B by the measured difference before the crosstalk, overlap and
hand-off measurements. The measurement above: the offset is recovered
exactly at 40/120/300/600 ms (peak r = 0.7587), the residual gap error after
correcting is **0.00 s** in every case, and the hand-off count returns to
its correct value at every offset.

**Also measure the drift**, or reject the pair: the clap fixes the offset at
the clap only, and a 1.25% time-base error leaves a 187.5 ms gap error by
the middle of a 20 s scene, with nothing in the report to say so.

### P1 — make the exit code carry the verdict

`main()` should end with a non-zero exit when `gates.status != "USABLE"`,
and the "nothing recorded" case should exit non-zero too. Today both exit
0.

### P1 — gate the things that are only reported

Unmatched pair, unreadable channel, silent channel B, low-level channel:
all four are visible in the report and invisible in the verdict. The dead
`ok` variable is the "matched pairs" gate, unwired.

Note the constraint on the silent-B gate: scene 7 is *specified* to have a
silent channel B, so "B is silent" cannot be an error on its own. It has to
be scoped to the scenes where B is supposed to speak.

### P1 — replace the cross-talk statistic

*"Of the time A is loud, how much of it is B loud too?"* separates
0.000–0.3125 (clean) from 1.0000 (B hears A) — a margin of **0.6875** on
the same signals where the intake's statistic's margin is **0.0067**. Then
gate on it.

### P2 — fix the level floor

Either raise the floor used by `signal_frame_frac` and `both_talking_frac`
to something a real room does not trip, or use the repo's existing
`src/vad.py` gate, which is calibrated on room tone. At 1e-4 RMS an empty
room is reported as 100% signal and 100% overlap.

### P2 — align the length gate with the protocol

The gate is 600 s; the README's eight scenes are ≈ 1230 s (20.5 min). A
corpus that satisfies the protocol trips the gate.

### P3 — add a WAV writer to `src/wavio.py`

The protocol specifies "WAV, PCM 16-bit" and the repo has no writer for it.

## 10. Is the path ready for the eight-scene recording?

> **CORRECTION (2026-10-19, the fixes lane).** The four numbered defects
> below **have been fixed**, and the tool now reads a protocol-shaped
> recording the way it is supposed to. The recording instruction did not
> change: clap **once per scene, loudly, before either person speaks** —
> which was already the protocol, and is what the alignment step consumes.
>
> What is still true, and is not a defect in the tool: **the drift between
> two recorders cannot be measured from one clap**, so the tool reports a
> bound rather than a number (§12.7).

**The recording: yes. The intake's verdict on it: no, not yet.**

Casey's 25 minutes of human time is the bottleneck and the protocol does
not need to change — it already asks for two separate microphones, a clap
before each scene, and eight specific scenes, which is exactly the right
design. The clap, in particular, is what the alignment fix needs, so it
costs nothing extra to record.

What is not ready is the tool that will read what Casey records. In its
current state the intake would:

1. **reject** a correct, protocol-shaped recording for having "zero
   hand-offs" (§6);
2. report **100% overlap** and **0.97 cross-talk** on a clean recording made
   in a real room (§7);
3. report a hand-off count that **flips to zero** from a 300 ms difference
   in when two recorders were started, with no warning that any offset
   exists (§5);
4. accept a corpus that is **missing a channel entirely**, or whose channel
   is **dead**, or whose **rate is wrong**, and **exit 0** while doing it
   (§4, §8).

None of those is a reason to delay the recording. All four are reasons not
to trust a green light from this tool until P0 and P1 are done.

**If the recording goes ahead before the fixes land, one instruction makes
the corpus worth more than it would otherwise be:** clap **once per scene,
loudly, before either person speaks** — which is already the protocol. Do
not clap between scenes within a take, and do not add a second clap later
in a scene, because the alignment step proposed above assumes exactly one
alignment point per pair, at the start.

## 11. What could not be measured here

- **Real audio.** Real speech, real room tone, real two-microphone
  crosstalk, real recorder start skew and real clock drift. The 1.25%
  drift used here is roughly a thousand times a real crystal's drift; it
  was used to make the *consequence* measurable, not to claim a realistic
  magnitude.
- **The intake has still never seen a real file.** Everything above is
  synthetic with known truth, exercising the gates. It says nothing about
  what a real recording will look like.
- **The cross-talk margin is a lower bound.** Synthetic speech built from a
  deterministic harmonic stack is more mutually similar than real speech, so
  the clean corpus's cross-talk numbers are pessimistic and the measured
  margin is a floor.
- **Gap resolution.** The sample-resolution gap measurement uses 5 ms
  frames, so every gap number carries ±5 ms of quantisation. The intake's
  own grid is 100 ms and carries ±50 ms.
- **Not exercised:** 24-bit and stereo WAVs, non-PCM encodings, truncated
  files, a `raw` directory containing non-WAV files with matching names, and
  the `--raw` argument missing its value (which raises `IndexError`).
- **The matrix corpus is short by construction**, so the protocol-length
  gate fires on it; only the 615 s run clears it.

---

## 12. What changed — the fixes, and what they were worth

**2026-10-19.** Every P0–P3 item in §9 was fixed in
`experiments/dialogue_intake.py`, with `src/wavio.py` changed for P3 and for
the sample-rate field. The measurements below were re-taken against the
fixed tool by `experiments/dialogue_intake_fixes.py`, which writes
`experiments/dialogue_intake_fixes_results.json`. The tests are
`tests/test_dialogue_intake.py` (behaviour) and
`tests/test_dialogue_intake_fixes.py` (the margins).

The audit's own numbers are in §4–§8 and are unchanged; what follows is the
same quantities measured on the fixed tool, so the two can be compared
directly.

### 12.1 The hand-off rule

| | old rule | new rule |
|---|---:|---:|
| canonical hand-off (A stops, B takes the floor, A stays quiet) | **0** | **1** |
| canonical, A resumes much later | **0** | **1** |
| barge-back (A comes back while B is still talking) | 1 | **1** |
| protocol-strict corpus, hand-offs actually in it | — | **8 canonical + 3 barge-in = 11** |
| protocol-strict corpus, counted | **1** | **11** |

The old rule scored the common shape zero and the rare shape one. The new
rule counts "channel B loud **during** channel A's silence", which is what
"the other person took the floor" means. Three conditions, each measured:

- the silence must be at least `energy.MIN_SILENCE_S` (0.150 s) — the old
  rule counted a **20 ms** envelope dip as a hand-off;
- the clap must be excluded — it is loud in **both** channels by
  construction and sits inside the lead-in silence, which is exactly one
  false hand-off per scene;
- a leading sliver of silence is not a rest. `src/energy.py:290` keeps its
  first segment whatever it is, so a file that opens with a sliver of
  silence reports a sliver of silence. That defect is **measured but not
  patched** in `src/energy.py` (`SILENCE-LEADING-ARTEFACT.md`); the
  intake's own segmentation applies the rule to the first segment as well,
  because a hand-off count that saw a leading sliver would be wrong.

### 12.2 The clap

The protocol's alignment point is now read, and the correction is applied
**before** any cross-channel number is computed.

| true offset | detected | peak r | old rule's hand-offs | new rule's hand-offs |
|---:|---:|---:|---:|---:|
| 0 ms | 0 ms | 0.7586 | 1 | 1 |
| 40 ms | −40 ms | 0.7586 | 1 | 1 |
| 120 ms | −120 ms | 0.7586 | 1 | 1 |
| 300 ms | −300 ms | 0.7586 | **0** | **1** |
| 600 ms | −600 ms | 0.7586 | **0** | **1** |

The old rule flipped at 300 ms and 600 ms. The new rule never flips.

One thing the fix exposed, and it is a measurement worth keeping: **the clap
is shorter than `energy.MIN_SPEECH_S` (0.150 s)** — it is about 40 ms — so
the new segmentation does not count it as speech at all. That is why the
overlap budget no longer contains the clap, and it is why the clap has to
be excluded from the hand-off rule explicitly rather than by falling out of
the segmentation.

A clap is also a **transient**, not merely the loudest thing in the window.
In the room-tone scene there is no clap, and the loudest thing in the first
two seconds is a noise peak; reading it as a clap produces a confident
alignment between two files that were never aligned. The tool therefore
requires the peak to stand at least 4× above the 90th percentile of the
same window (measured: a real clap scores ≈ 15, a Gaussian noise peak
≈ 3.1) and to be 3× above the measured noise floor.

### 12.3 The exit code

| situation | exit code |
|---|---:|
| the corpus is usable | **0** |
| the corpus is rejected (any unmet requirement) | **1** |
| nothing has been recorded | **2** |

Before the fix, all three exited **0**.

### 12.4 The reported-only failures

All four are gates now: an unmatched pair, an unreadable channel, a silent
channel, and a low-level channel. The dead `ok` variable — the matched-pairs
verdict that was computed and never used — is gone; the verdict it computed
is what the gate uses.

Two of them needed scoping, and the audit was right to insist on it:

- **"channel B is silent"** cannot be an error on its own, because scene 7
  is *specified* to have a silent B. The gate fires only in the scenes
  where the protocol has B speaking, and the tool says which scenes those
  are (`B_expected_to_speak` in the report). Scene 7 with a silent B is
  accepted; scene 1 with a silent B is rejected.
- **"the channel is low"** is now a statement about the **balance between
  the two channels of a pair**, not about one channel in isolation: a
  microphone on the table instead of at the mouth is tens of dB down. The
  limit is 20 dB.

The perturbation battery, all ten cases from the audit, now behaves
correctly: the eight defects are caught, and the two legitimate variations
(a corpus with no barge-in, and two channels at different sample rates) are
**not** rejected. The battery distinguishes "caught by the relevant gate"
from "caught by the length gate", which every one of these short synthetic
corpora trips.

### 12.5 The cross-talk statistic

| statistic | clean | contaminated | separation |
|---|---|---|---:|
| old (correlation of window means) | 0.974 – 0.983 | 0.990 – 0.998 | **0.0067** |
| new (of the time A is loud, how much is B loud too) | 0.000 – 0.306 | **1.000** | **0.6939** |

A margin of 0.0067 cannot be gated on. A margin of 0.6939 can, and is:
the gate fires above 0.60, which sits between the two measured ranges.

### 12.6 The level floor

The old floor was `energy.ABS_FLOOR` — 1e-4 RMS, about −80 dBFS — which is
**23.5 dB below an ordinary lived-in room**. The new floor is **measured**,
from the place the protocol says to measure it from: *"Also record 30 s of
nobody talking … This is the noise floor every other number is measured
against."* The room-tone scene is read first and every other number is
measured against it.

| the empty room-tone scene | quiet room (1e-5 RMS) | ordinary room (1.5e-3 RMS) |
|---|---:|---:|
| old floor, `both_loud_frac` | 0.0 | **1.0** |
| new floor, `both_loud_frac` | **0.0** | **0.0** |
| new floor, `signal_frame_frac` (A and B) | **0.0** | **0.0** |

The new floor is **room-independent**: the same corpus in a quiet room and
in an ordinary one produces the same numbers, and an empty room reads as
empty in both.

Three things were measured on the way, because the obvious rule fails:

- a threshold at a fixed fraction of the file's **peak** works on continuous
  speech (scene 4, 90% detected) and fails on room tone (100% detected in
  an empty room);
- a threshold at a fixed multiple of the channel's **quietest frames**
  works on room tone and fails on continuous speech (0.2% detected);
- the **watcher's adaptive threshold** has the same failure on continuous
  speech (14% detected in scene 4), because a local noise-floor estimator
  in a room with no silence estimates the floor from the voice.

The measure that survives both is the one the protocol already specifies: a
**reference scene** with a known noise level, and a bar 3× above it.

### 12.7 The drift — a bound, and labelled as one

The audit was right that drift has to be dealt with, and wrong about how it
can be dealt with. The clap gives **one** alignment point; with one
alignment point the drift **cannot be measured**, because the two channels
speak different words and their envelopes are uncorrelated everywhere except
at the clap.

What the tool does:

- reports the offset it measured at the clap;
- reports a **bound**: the error an assumed clock error of 100 ppm would
  produce by the end of the scene (**2.0 ms** over a 20 s scene), and the
  same at the audit's deliberately extreme 1.25% (**250 ms**);
- says in the report that the drift is **not measurable** from one
  alignment point, and says why.

Measured with the audit's 1.25% time-base error: the tool detects the clap
offset as **4 ms** (there is no drift yet, at the clap) and the bound at
the scene's end is **250 ms**. The gap error that drift alone would cause
(−0.19 s, §5) is still a real consequence — it is simply not something the
tool can see from one clap, and it does not claim to.

### 12.8 The length gate, and the writer

- The gate wants the protocol's own number: **1230 s** (300 + 180 + 180 +
  120 + 180 + 120 + 120 + 30). The old gate wanted 600 s, so a corpus that
  satisfied the protocol tripped it.
- `src/wavio.py` has a **writer** now (`write_wav`), so the repo can
  produce the files the protocol asks for instead of every test carrying
  its own copy. It clips rather than wraps — a wrapped sample is a sign
  flip — and it round-trips through the repo's own reader to within the
  16-bit quantisation (1 LSB = 3.05e-5 full scale).
- `src/wavio.py` now keeps the file's **real** sample rate on the `Audio`
  object (`source_sample_rate`). `read_wav` resamples to the analysis rate,
  so before this the original rate was unrecoverable and every file
  reported 16000 — the intake's `raw_rate_hz` field was 16000 whatever the
  file said.

### 12.9 What the fixes exposed, and what is still open

Two new findings came out of fixing the old ones, and both are recorded
here rather than quietly fixed:

1. **The scene names are zero-padded** (`scene01_free`, not `scene1_free`),
   and the first version of the scene-number pattern found **nothing** in a
   padded name — a single digit that is not followed by another digit does
   not exist in `scene01`. Every scene then fell through to the keyword
   rules, which know only three of the eight, and the "is B supposed to
   speak here" gate silently did not fire for five of them. Fixed, and
   pinned by `test_the_scene_expectations_match_the_protocol`.
2. **A noise reference taken from the wrong scene is worse than no
   reference.** The first fallback picked the *quietest scene*, which in a
   corpus without a room-tone scene is scene 7 — one speaker alone — whose
   channel A is the quietest channel in the corpus for the wrong reason: it
   is the median of a scene that is mostly **speech**, and a bar derived
   from it is three times a voice. The fallback is now the 10th percentile
   of the frame levels pooled over every scene, and the reference is the
   **larger** of the room-tone estimate and that pooled estimate, so a
   room-tone scene recorded at a lower gain cannot lower the bar for the
   rest of the corpus.

Still open, and not fixable by this tool:

- **The drift.** A bound, not a measurement, and it stays a bound until the
  protocol grows a second alignment point.
- **The intake has still never seen a real file.** Everything in this
  section is synthetic with known truth, exercising the gates. It says
  nothing about what a real recording will look like.

---

## Reproducing this

```bash
/home/proart/projects/glasses/.venv/bin/python \
    experiments/dialogue_intake_selftest.py --matrix-seconds 12 --long
/home/proart/projects/glasses/.venv/bin/python \
    experiments/dialogue_intake_fixes.py
/home/proart/projects/glasses/.venv/bin/python \
    -m pytest tests/test_dialogue_intake.py tests/test_dialogue_intake_fixes.py -q
```

The self-test takes ≈ 4 minutes and writes
`experiments/dialogue_intake_selftest_results.json`. The fixes battery takes
≈ 4 minutes and writes `experiments/dialogue_intake_fixes_results.json`.
The two test files take ≈ 30 s. All of them use only numpy and the repo's
own `src/`.
