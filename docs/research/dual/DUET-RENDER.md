# THE DUET, MEASURED — what two cameras of the same 236 seconds actually give

Casey's brief was to render the dual capture positionally, more than one way, and
keep all. The pair has landed: `~/dual-capture/dual-3.json`, the phone's room-lens
(513 frames, audio present), against the ProArt's own corpus sessions. This is what
the three renders say, measured rather than eyeballed, with the alignment labelled
for what it is.

**The short of it: the duet does not yet show the synergy, and the reason is
measured, not a bug.** The two views' change maps are not the same quantity, the
phone's channel is saturated, and the alignment is the wall clock rather than the
claps. Each of those three is quantified below, and each has a specific remedy. The
one instrument that does work — the corpus's own *crossing* detector, asked whether
the two views find the same moments — reaches **z = +1.97, p = 0.087**, which is
suggestive and not significant, and survives a misalignment of five seconds.

## 1. The pair, and how the clock was found

The phone's bundle is zcode's room-lens session, and the clock inside it is a trap.
The bundle's `meta.startedAt` reads 07:52:12.428Z, but the room-lens stamps every
frame `t = Date.now() - walkStart`, and `walkStart` is when the walkabout began.
Those are different clocks: the session was created 435.840 s before the walkabout
started, and the bundle proves it — three of its seven room measures carry no `t`
at all, because they were taken before the phone was walking.

So the anchor is derived rather than believed. Five audio parts and four walk-time
room measures each carry a wall stamp and an offset from the walkabout's start, and
all nine estimators agree **to 0.0 ms**. The walkabout began at 07:59:28.268Z; the
phone's camera ran 07:59:28.350Z to 08:03:24.283Z, 235.93 s of picture.

The ProArt side is eight corpus sessions, and they are merged into one stream only
after the merge is verified. The capture page stamps every frame with
`performance.now()` minus the page's load base, so the sessions' offsets share a
clock, and the page base is recoverable as `wall_clock_start - frame_offsets_ms[0]`.
All eight agree on that base **to 1.2 ms across 12.8 minutes**. The tool refuses a
merge whose bases disagree by more than 250 ms, because merging two clocks invents
a timeline. The merged stream is 1866 frames over 767.7 s, with 14.4 s of real gaps
between the sessions, and the gaps are listed rather than papered over.

That leaves the alignment, and it is the wall clock: **offset +500538.0 ms**, drift
assumed zero. The two devices' clocks have never been compared, so this carries the
device-to-device skew — the one constant the corpus cannot measure — and the sidecar
says so in every output it appears in. The 236 s of the duet cannot separate a clock
drift from a movement.

| what | value |
|---|---|
| the walkabout's start | 07:59:28.268Z (9 estimators, spread 0.0 ms) |
| the phone's camera | 07:59:28.350Z → 08:03:24.283Z, 235.93 s |
| the ProArt, merged | 1866 frames, 767.7 s, page base spread 1.2 ms |
| the shared minutes | 235.93 s |
| the alignment | +500538.0 ms, WALL CLOCK, not the claps |
| the alignment's resolution | **not measurable from this sidecar** — it carries no clap times, and the wall clock's own relation to the two devices has never been compared |
| the phone's change maps | 488 of 512 pairs (24 lost to a geometry change) |

## 2. The two views are not the same quantity, and the gap is measurable

The ProArt's change map is the corpus's own: the mean absolute luma difference over
a 4×4 cell of the camera's picture, in luma, at 2.0 fps. Its noise floor is the
corpus's measured 3.0 luma, and its duty cycle — the share of frames showing a
change above its own floor — is 58.6 %.

The phone's change map is a different kind of object entirely. The room-lens renders
CHARACTERS, and its real ramp is `" .:-=+*#%@@@@@@@@"` — seventeen glyphs whose top
**eight are all `@`**, so half the brightness range carries no information in the
picture. The renderer also applies a 2.2× contrast stretch before bucketing, and
clips at both ends. This lane's first draft assumed a ten-glyph ramp and was wrong
by a factor of two across the whole top half; the ramp is now read out of the
room-lens's own source and each glyph is read as the LOWER edge of its band, so
every value is a lower bound on the true difference.

The consequence is measurable and it is stark. The phone's per-pair maximum cell has
a median of 142.38 against a ceiling of 143.44 — and 143.44 is exactly one full
brightness band, the largest difference the phone's own scale can express. Half the
phone's frame pairs contain a cell that flipped a whole band. Its duty cycle is
**99.8 %**: the phone's channel is on essentially always.

| | the ProArt | the phone |
|---|---|---|
| the change map is in | the camera's luma | the room-lens's own brightness (2.2× stretch, clipped) |
| the floor | 3.0 luma (the corpus's, measured) | 15.9375, one band — the sensor's *resolution* |
| the frame interval | 499.7 ms | 406.5 ms |
| the duty cycle | 58.6 % | **99.8 %** |

The phone's floor is its resolution, not a grain estimate, and the graded fixture
proves why: the room-lens is a *deterministic renderer* of a noisy sensor, so 12.8 %
of the fixture's frame pairs are at **exactly zero** — below one band, a change
cannot be expressed at all. The corpus's 3.0 is a grain floor; the phone's 15.9375
is a resolution floor. They are different kinds of thing, and borrowing one for the
other would have been a threshold with no meaning.

## 3. The three renders, in three scale modes, all kept

The dual-ASCII, the fused score and the differenced view are in
`DUET-RENDER-FLOOR_RELATIVE.txt`, `DUET-RENDER-SHARED_AUTO.txt` and
`DUET-RENDER-PER_VIEW_AUTO.txt`, with the machine-readable numbers beside each.
Three scale modes, because "more than one way, keep all" is the brief and because
the three answer different questions.

`shared-auto` gives both panels one scale. It is the tool's default because a
side-by-side of two scales says nothing about which view saw more — and on this
pair it renders **the ProArt's panel black**, because the phone's numbers are
~300× the ProArt's in a different unit. That black panel is the honest picture of
the units mismatch, so it is filed rather than hidden.

`per-view-auto` scales each view against its own maximum. Both panels are legible,
and the output says in its header that they are not comparable in magnitude.

`floor-relative` divides each view by that view's own floor and then uses one scale.
A cell at the top of the ramp then means the same thing on both sides: this many
times this sensor's own grain. It is the only comparison the duet makes possible,
and it is the one the filed headline render uses.

## 4. The witness count is degenerate, not merely weak

The fused score's marks are the 2×2 contingency the lane built for this: both views
moved, only the fixed, only the roving, neither. On the duet it reads:

| | slices | share |
|---|---|---|
| both moved | 151 | 64.0 % |
| only the fixed moved | 0 | 0.0 % |
| only the roving moved | 85 | 36.0 % |
| neither moved | 0 | 0.0 % |

The roving view moved in **236 of 236 slices**. One row of the table is constant, so
the excess over chance is exactly **+0.0 points** and no chi-square can be formed at
all. This is not a weak result; it is a degenerate one, and the cause is the duty
cycle: a channel that is on 99.8 % of the time cannot time a movement, so the
contingency table has nothing left to measure. The alignment's sensitivity table
shows the count is not moving when the alignment is deliberately shifted by ±2 s,
which confirms the null is saturation rather than a clock artefact.

## 5. The graded reading, and why the slide cannot find the alignment

The binary count throws away everything but the sign. The graded question — do the
two channels' change energies rise and fall *together* — uses the whole signal. Its
null is the slide: the roving channel is slid against the fixed one and the
correlation recomputed at every offset, because the null here is the
**autocorrelation of two slow series**, not the correlation of two noises. A person
walking about a room produces bursts minutes long.

| reading | r at the alignment | null mean | null sd | z | slides that match it |
|---|---|---|---|---|---|
| raw | +0.2553 | −0.0221 | 0.1598 | +1.74 | 18 of 237 |
| log | +0.3599 | −0.0335 | 0.1921 | +2.05 | 4 of 237 |

The raw reading is the one that was asked for first and it is not significant: it
sits at the 92nd percentile of the slide null. The log reading sits at the 98th. The
log was added *after* the raw was seen, so it is a second reading and not a
replacement, and it is reported beside the raw one rather than instead of it.

The sweep cannot localise the alignment, and this is the finding that matters most.
The raw sweep's own maximum is at **−44 s**; the log sweep's is at **−2 s**; and in
both the region within one null standard deviation of the maximum spans over fifty
seconds. Two slow series correlate somewhere by luck. A sweep that cannot say
*where* is saying nothing about the skew, and the two readings disagreeing about
where the peak is says it twice. **The duet cannot measure its own alignment. The
claps are not optional; they are the only way.**

## 6. The crossing witness, and the instrument bug it exposed

The corpus already has the right instrument and this lane had not used it. A
**crossing** is a connected blob of change that holds its shape while it
*translates* — the person is one object, it keeps its size, and it moves. The
sensor's grain is below the floor, a highlight is one frame, a fidget does not
travel, and neither does a shake. So each view is asked the question it can
actually answer: did a *thing* move across the picture?

**AND THE INSTRUMENT HAD A HIDDEN ASSUMPTION.** The corpus's detector
takes a gap in **seconds** — 0.35 s — which is 21 frames at the 60 fps
the corpus was captured at, and **0.70 frames** at the 2 fps the corpus
is captured at now. At 0.70 frames no blob can EVER chain, every blob
is a singleton, and the walk-by study finds **nothing at all**. The gap
has to be counted in *frames*, and it is now the corpus's own rule:
`max(0.35 s, 1.5 × the session's own frame interval)`.

Measured over the whole corpus, that fix is exactly what it should be:

| the corpus's sessions | the flat 0.35 s gap | the gap in frames |
|---|---|---|
| the 92 fast ones (60 fps) | 739 crossings | **739** — the same number, so no filed result moves |
| the 11 slow ones (2 fps) | 2 crossings | **25** — five sessions the corpus could not see at all |

That is a defect every one of the corpus's 2 fps sessions has been
carrying, and it is why the duet needed a second look at the corpus's
own tools. With the gap fixed, the duet gives its best number:

| criterion | observed | null mean | null sd | z | slides that match it | p |
|---|---|---|---|---|---|---|
| the intervals overlap at all | 11 of 18 | 4.40 | 3.17 | **+2.08** | 12 of 241 | **0.050** |
| they share half of the shorter one | 9 of 18 | 3.70 | 2.93 | +1.81 | 29 of 241 | 0.120 |

The looser criterion is the one that was asked for first and the tighter one is
reported beside it, not instead of it. The honest reading is **suggestive and not
significant**: the two views find the same crossings about twice as often as a
random alignment would, and the result is of the same order under either criterion
(z between +1.8 and +2.1) and under either gap rule. It is also stable: the
alignment can be wrong by five seconds before it goes (z +2.08 at 0 s, +2.08 at
1 s, +2.40 at 2 s, +2.08 at 5 s, and +1.14 at 10 s). A result that survives a
five-second misalignment is **not** a measurement of the alignment — it is a
statement that the two views find the same crossings at the same moments, which is
the thing two sovereign logs of the same minutes give that neither gives alone.

## 7. The phone's camera luma, and why the roving eye is always on

The duet produced a surprise worth filing on its own: **the phone's channel is
saturated because the roving eye is IN the scene, not only because of the
posterisation.** The ProArt sits on a stand, so when the room goes still its picture
goes to **exactly zero** change — the quietest tenth of its pairs is 0.000 luma.
The phone's minimum per-pair change over 2.1 s is **40.438 luma** and it never
reaches zero. A hand-held camera is always moving, so its picture always changes.

That is structural, and it means the witness question cannot be "did anything
change" — for a roving view the answer is always yes.

**THE KEYFRAME CHANNEL.** The room-lens already computed the camera's own luma
before it posterised it, and the bundle banks 103 **keyframes** of the camera's
actual picture (320×569 JPEG). Those are the camera's *luma*, which is the very
quantity the ProArt's change map is in. The ingest now builds a second bundle from
them — 160×96 luma in the corpus's own format, cropped to the corpus's aspect
rather than stretched — so `experiments/lens.py` computes its change map with **no
changes at all**, and the floor is the corpus's own 3.0 because the construction
*is* the corpus's. The two views are commensurable in that bundle for the first
time.

It is also a lesson about intervals. The keyframes are one per 2.105 s, 4.2× the
ProArt's cadence, and the corpus's 3.0 luma floor — measured on 0.5 s frames — is
saturated at that interval (the keyframe channel's duty cycle is 100 % at 3.0 and
89.2 % at its own measured floor of 94.938). **A noise floor is a property of a
sensor AND an interval**, and the corpus's floor is a property of the corpus's
interval. At its own floor the keyframe channel finds 2 crossings, and 2 crossings
say nothing at all.

## 8. What the duet does and does not show

It does not show the synergy the brief is after, in the form the brief hoped for.
The two channels' energies correlate at +0.26 (raw), which is suggestive and not
significant against the honest null; the binary witness test cannot be formed at
all; and the crossing coincidence reaches z = +1.97, p = 0.087. Three things are
known, measured, and are not alignment artefacts: the phone's change map is a mask
of band flips that is on 99.8 % of the time, the two change maps are in different
units so no single threshold means the same thing on both sides, and a roving view
in the room is always changing.

It does show the machinery, working on the real pair, and it shows three traps that
would have produced false results. The clock trap — reading `meta.startedAt` as the
frames' origin — would have moved the whole duet 435.840 s and paired the phone
with the wrong ProArt session. The units trap — one scale for two sensors — would
have rendered a black panel and a confident-looking number underneath it. And the
instrument trap — the corpus's 60 fps gap on 2 fps sessions — would have produced
an empty crossing count that reads exactly like "the two views never agree".

## 9. What would make it work, and who owns it

**zcode's clap alignment** replaces the wall clock and settles the skew the sweep
could not. The tool already consumes a clap sidecar, and the sidecar's own note is
carried into every output, so the re-run is one command and the comparison against
this render is immediate.

**BUT NOT ON THIS TAKE: the clap channel was dead, and that is now measured.**
zcode's dual-3 pair report (`docs/research/DUAL-3-PAIR-REPORT.md`, 2026-10-22) ran
**four independent detectors** at this duet's sync and every one of them sits at its
noise floor — envelope correlation (r ∈ [0.765, 0.809], but sweeps 3–6 all "arrive"
at the same witness instant, a template self-match), the TF-path matched filter
(pathSNR = 1.00 on all 7), the spectrogram correlation (r ∈ [0.074, 0.082] over a
±90 s search), and the 8–16 kHz band energy at the expected windows (mean 0.01
against a global p99 of 0.04). The phone's speaker at gain 0.4 does not reach the
Realtek witness across the room, and the phone's ambient audio — the only place its
claps would have lived — is **4.4 s of 3.5 minutes**, a parts race that has since
been fixed (room-lens `474d6c0`).

**SO THE WALL CLOCK IS PERMANENT FOR DUAL-3, AND THESE RENDERS ARE AS GOOD AS
THIS TAKE CAN BE MADE.** The clap sidecar belongs to the next take, and the
capture lane's staged `pair_analysis.py` estimates onsets on a **5 ms hop** — which
is sub-frame against the ProArt's 16.667 ms and very much sub-frame against this
duet's own 499.75 ms — so the sync the route asks for is the sync the next take
will carry.

**AND THE POSE CARRIES NO SCALE.** The phone's bundle holds the device's alpha,
beta and gamma and nothing else: no position, no translation, and **no heading in
any of the 513 frames**. The phone's view cannot be registered to the ProArt's in
space from its own bundle, so the metric scale has to come from the ranging/TDOA
side, and until it does these renders carry no metric claim.

**The phone's camera luma, banked every frame rather than every fifth.** The
room-lens computes a luma field for its own ASCII and then posterises it; the
posterisation is the lossy step, and keyframes every fifth frame are 4.2× too
coarse to carry the corpus's own noise floor. Banking the luma — or a change map
computed from it — at the ASCII's own cadence would put the phone on the corpus's
footing, in the camera's luma, where the two views' change maps are the same
quantity and the witness test becomes a test.

**A witness that asks about objects, not about change.** The crossing detector is
the right instrument and it is already in the corpus. Its gap is now counted in
frames, which leaves the corpus's 739 fast-session crossings exactly where they
were and lets its 2 fps sessions be seen at all.

**A slice length above the frame interval.** At 2.0 and 2.5 fps a 0.25 s slice
holds at most one frame from each view, and the fused score's slices are better at
1 s, which is what the filed render uses.

**And the sync at SUB-FRAME resolution, because the route asks for it.** The
literature route for two uncalibrated views is audio-sync first and at sub-frame
precision, and the resolution of a frame-quantised sync is set by the **coarser**
of the two streams. On this pair that is the fixed side: the merged ProArt stream's
frame interval is **499.75 ms** against the phone's **406.5 ms** and the keyframe
channel's **2105.0 ms**. A sidecar written from frame timestamps would therefore
carry a half-second quantisation, and the renderer now says so in every output —
it measures each clap time's distance from its own stream's frame grid and reports
whether the sidecar carried **FRAME TIMES** or **SUB-FRAME times**. The duet's own
alignment carries no clap times at all, so its output says that the resolution
cannot be measured from it, and the wall-clock skew stands.

**AND NO SCALE IS BAKED IN.** The relative pose that comes after the sync is
scale-free, and the metric scale comes from the ranging/TDOA side, which is
zcode's lane. Until it lands, these renders carry no metric claim: a cell in the
left panel and a cell in the right panel are not the same part of the room, and
that has been true of every output this lane has ever filed.

## 10. The commands, so the whole thing re-runs

```
experiments/dual_ingest.py --room-lens ~/dual-capture/dual-3.json --out /tmp/duet-3 \
  --label "the phone room-lens (the roving eye)" --fixed-label "the ProArt, merged" \
  --fixed-session 1792655768 --fixed-session 1792656069 --fixed-session 1792656077 \
  --fixed-session 1792656100 --fixed-session 1792656118 --fixed-session 1792656155 \
  --fixed-session 1792656183 --fixed-session 1792656235

experiments/dual_render.py --fixed /tmp/duet-3/fixed --roving /tmp/duet-3 \
  --sync /tmp/duet-3/wallclock-fixed.json --out /tmp/duet-render-floor-relative \
  --scale floor-relative --moments 4 --slice 1.0 --window 0.5
```

The selftests that grade this: `dual_ingest` 29/29 and `dual_render` 68/68. The
ingest's room-lens path is graded against a fixture with a known truth — a person
walking a known path, rendered through the room-lens's own rules — and the change
map has to land on the truth at five points along the walk.

Committed for Muse to review; not merged.
