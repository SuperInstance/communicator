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

## Reproducing this

```bash
/home/proart/projects/glasses/.venv/bin/python \
    experiments/dialogue_intake_selftest.py --matrix-seconds 12 --long
/home/proart/projects/glasses/.venv/bin/python -m pytest tests/test_dialogue_intake.py -q
```

The self-test takes ≈ 4 minutes and writes
`experiments/dialogue_intake_selftest_results.json`. The test suite takes
≈ 59 s. Both use only numpy and the repo's own `src/`.
