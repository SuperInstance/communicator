# BROWSER-CAPTURE ACCEPTANCE — the referee's verdict

**Referee:** the acceptance worktree (`~/projects/communicator-acceptance`,
branch `acceptance/browser-capture`)
**Subject under test:** `browser_capture/` in the main checkout
(`~/projects/communicator`), built by ZCode
**Date:** 2026-10-20, re-measured 2026-10-21 against `6c8dc1b` (§1.5)
**Expectations:** `docs/research/ACCEPTANCE-EXPECTED.md` (written blind, before
reading any of ZCode's code)
**Checker:** `experiments/browser_capture_acceptance.py`
**Measured report:** `experiments/browser_capture_acceptance_results.json`

---

## 0. THE VERDICT, IN ONE PARAGRAPH

**The binding requirement of the base brief is MET, and I measured it
independently.** The browser capture produces takes the existing corpus
pipeline accepts unchanged: the four pipeline files per take, the reference's
own schemas, the reference's own array set, and the change maps and drift
re-derive **exactly** from the stored luma when recomputed with the
reference implementation's own functions. The repo's gate
(`experiments/corpus_integrity.py`) exits 0 on the session, on the directory
as it stands.

**The addendum's rich artifacts are PARTIALLY met, and the part that matters
most for "maximum nuance" — the full-resolution WebM — is NOT met by
anything that was actually produced.** Every `_full.webm` in the verified
session is a 37-byte header-only stub with **no video track and no audio
track at all**, and one take has no WebM whatsoever. The addendum asks for the
mic audio to be muxed in; that requirement is therefore **unverified**, and
the only artifact against it fails.

**One substantive open risk, and the sharpest finding of this exercise: the
speech path is not exercised by the verification at all.** The verified
session's `speech_frac` values are **hardcoded constants** (0.72 / 0.01 /
0.02), not measured from the audio that was written — and the audio is the
same synthetic tone in every take. I re-measured the WAVs with the
reference's Silero gate and got **0.0064 against a recorded 0.72**, and I ran
the **page's own** speech gate (extracted verbatim from `index.html`, under
node) over the same files and got **0.6084 on all of them**. Had the page's
real gate produced those numbers, **C1 and D1 — both labelled `speak=False`
— would have been flagged `NOT_SILENT`**, and the manifest carries no flags
at all. The production path computes the number for real
(`index.html:806`); the *verification* substitutes constants for it and so
proves nothing about it.

**My checker's verdict on the session: FAIL** — 3 of 12 checks failed
(`LAYOUT`, `SPEECH`, `WEBM`), all three of them on the addendum's rich
artifacts or on the speech gate, and none of them on the pipeline-exact
artifacts. **This verdict was re-measured twice, against the build as it
stands after ZCode's five further commits (§1.4) and again after the sixth
(§1.5); the result is identical both times.** **That is not a verdict on the
production build's correctness**:
every one of the three is a property of the *synthetic verify session*, and
each is disclosed in ZCode's own `DONE.md`. The production path for two of
them is unexercised offline.

---

## 1. WHAT WAS MEASURED, AND HOW

### 1.1 The session under test

`browser_capture/DONE.md` appeared in the main checkout during this
acceptance, and it reports a synthetic session produced by
`browser_capture/verify.py`. **`verify.py` deletes its output directory when
it finishes**, so the session had to be reproduced to be inspected.

I reproduced it with ZCode's own code, running the real server on a scratch
port, generating the takes with the **shipped JavaScript executed under
node**, and POSTing them through the real HTTP path. The only change was
**where the output is kept**:

```
--- browser_capture/verify.py
+++ /tmp/accept/communicator/browser_capture/verify.py
@@ -324 +324,2 @@
-    out_dir = tmp / "corpus"
+    import os as _os  # ACCEPTANCE EDIT: keep the output
+    out_dir = Path(_os.environ.get("BC_OUT", str(tmp / "corpus")))
```

Two lines, one of them a comment. The generator, the shipped JS, the POST
path, `server.py`, `index.html` and the gate are ZCode's own bytes: the
scratch tree symlinks `src/`, `experiments/`, `models/` and `examples/` to
the real repo and `index.html` / `server.py` to the real `browser_capture/`.
**Nothing in `browser_capture/` was edited, and nothing in the real corpus
was touched.**

### 1.2 ZCode's own claims, re-run

| ZCode's claim in `DONE.md` | My measurement |
|---|---|
| "145 checks passed, 0 failed" (the summary header) | **STALE** — the patch log under it says "154 passed, 0 failed" for the same build, and I measure **154 passed, 0 failed, 0 skipped**. The header was not updated with the patch log |
| "91 passed under bare system python3 (numpy suite skipped there)" | **CONFIRMED, at the old build** — 91 at `e42143b`; at `6c8dc1b` the same suite under bare system python3 3.14.4 (no numpy) gives **100 passed, 0 failed, 1 skipped**, because the nine new checks do not need numpy. The header is stale for the same reason |
| "every take produces the script's four files" | **CONFIRMED** — 4 files per take, correct names |
| "compact-timing / indented-meta JSON styles" | **CONFIRMED** — timing has no newline; meta starts `{\n  "` |
| "the npz contains exactly maps, scale, maps_raw, scale_raw, luma" | **CONFIRMED** |
| "maps_raw is byte-identical to a numpy re-implementation" | **CONFIRMED, and stronger** — see §3.1 |
| "the row-removed maps differ on 0.000% of cells" | **CONFIRMED** — I measured 0 differing cells |
| "drift_raw is exactly equal" | **CONFIRMED** — my re-derivation matched |
| "src/nv12.py detects the stored luma as luma_only, 15,360-byte stride, right frame count, margin ≫ 5" | **CONFIRMED** — `luma_only`, stride 15,360, 150 frames, **margin 25.244** |
| "src/wavio.py reads every WAV at its stated 16 kHz with the full sample count" | **CONFIRMED** — 79,950 samples at 16,000 Hz, `source_sample_rate == 16000` |
| "THE GATE passes" | **CONFIRMED** — gate exits 0, 0 of 5 takes failed |
| "duplicate POSTs get distinct stamps" | **CONFIRMED** — two A1 takes, stamps `…735` and `…739` |
| "out root holds only `<name>_<stamp>.json` and `_timing.json` files" | **IMPRECISE** — see §4.4 |
| "the rich artifacts live in `out/rich/`, not at the out root" | **CONFIRMED** — and it deviates from the addendum; see §5.2 |
| "speech_frac comes from an adaptive-RMS gate, not Silero" | **CONFIRMED** — and the verification does not measure it at all; see §4.1 |
| "the `_full.webm` … was only tested as a header-only synthetic blob" | **CONFIRMED** — 37 bytes, no tracks; see §4.2 |
| "F1 (no webm posted) has no `_full.webm`" | **CONFIRMED** — and the page does record one for every take; see §4.3 |

### 1.3 The checker, and why its verdict can be trusted

A referee that only ever says FAIL is worthless, and a checker shaped around
one producer's layout is not a referee. So the checker was validated two
ways before its verdict was read.

**(a) It passes a session built from the derivation alone.**
`experiments/browser_capture_acceptance_fixture.py` builds a 16-take session
from `ACCEPTANCE-EXPECTED.md` — the reference's own functions for the maps
and the drift, the reference's own schemas, real speech from the repo's own
corpus for the speaking takes, a **real** WebM container with a video track
and an Opus audio track, and the manifest **at the out root, as the addendum
asks**. The checker passes it: **12 of 12 checks PASS, exit 0.** A checker
that failed a correct session would have been discarded.

**(b) It catches corruption, with the right named check.** Ten mutations of
that known-good session, each detected:

| Mutation | Check that caught it |
|---|---|
| drop the `VOICE_MISSING` flag from a take that needs it | `FLAGS` |
| add a `VOICE_MISSING` flag to a take that does not need it | `FLAGS` |
| lie about the WAV's sample rate | `WAV`, `SPEECH` |
| truncate the luma inside the npz | `ARRAYS`, `NV12` |
| replace the WebM with a header-only stub | `WEBM` |
| remove the session manifest | `MANIFEST` (+`FLAGS`, which live in it) |
| lie about `n_frames` | `JSON`, `ARRAYS`, `NV12`, `WAV`, `GATE` |
| make the achieved frame rate wrong by 20 % | `JSON`, `GATE` |
| break the manifest's file list | `MANIFEST` |
| remove the rich sidecar | `LAYOUT`, `RICH` |

**10 of 10 detected.** The flag check is bidirectional: it fails a missing
flag *and* a spurious one.

The checker also found three bugs **in itself** during this validation (a
tuple type test that rejected `true`, an `nv12.luma_view` call that passed
bytes where the reader wants an array, and a crash on a missing rich file).
They were fixed; they are recorded here because a referee's instruments are
part of the evidence.

---

## 1.4 THE BUILD MOVED; THIS VERDICT WAS RE-MEASURED AGAINST IT

While the camera lane was running, ZCode shipped five more commits on top
of the build this verdict originally measured:

```
e42143b  the performer never guesses again: a phase banner, STOP & SAVE, bounded ends
49f4053  every camera parameter on screen: a controls panel from the real capabilities
5194314  stop driving the camera like the terminal script did: near-default constraints
3c0b2fe  the dark camera made diagnosable: a picker that avoids the IR impostor
a941354  the page learns --stimulus-file
```

`index.html` grew from 55,807 to 79,702 bytes. So the verdict below is **not**
a measurement of the build I first looked at: I re-derived the session from
the **current** `verify.py` (the same one-line output-location patch, nothing
else) and re-ran the checker against it.

**The result is unchanged.** The same three checks fail (`LAYOUT`, `SPEECH`,
`WEBM`) and the same nine pass; the gate still exits 0 on the directory and on
the takes alone; ZCode's own suite still reports **145 passed, 0 failed**.
That is the expected outcome, because all five patches are client-side page
changes and none of them touches the take format — which is exactly what
"the pipeline accepts the takes unchanged" requires of them.

One of the new commits did land a finding I had made independently: `3c0b2fe`
adds a camera picker that "avoids the IR impostor", and my device inventory
found the same thing from the Windows side — this machine has **two**
physical cameras, the ASUS FHD webcam and the ASUS IR camera, and the IR one
is the classic dark-picture impostor. See
`docs/research/CAMERA-RND-FINDINGS.md` §3.

## 1.5 THE BUILD MOVED AGAIN; THE VERDICT WAS RE-MEASURED A SECOND TIME

ZCode has since shipped one more commit on top of the build §1.4 measured:

```
6c8dc1b  FREE RECORD is the front door: one RECORD, one STOP, four states, the whole raw session saved for offline analysis -- plan mode intact behind it
```

`browser_capture/` grew by 825 lines: a new `free_record.html` (481), a new
`POST /free` endpoint and the session writer in `server.py` (224), nine new
checks in `verify.py` (82), and a `DONE.md` section. `index.html` changed by
three lines — a link to the new front door.

**The result is unchanged, and the reason is now visible.** The same three
checks fail (`LAYOUT`, `SPEECH`, `WEBM`) and the same nine pass; the gate
still exits 0 on the directory and on the takes alone.

ZCode's own suite, re-run on this build under all three interpreters:

| interpreter | at `e42143b` | at `6c8dc1b` |
|---|---|---|
| the audio venv (numpy + the gate) | 145 passed, 0 failed | **154 passed, 0 failed, 0 skipped** |
| the bare system python3 3.14.4 (no numpy) | 91 passed, 0 failed, 1 skipped | **100 passed, 0 failed, 1 skipped** |
| the bare Windows python 3.14.3 (numpy 2.5.1) | — | **154 passed, 0 failed, 0 skipped** |

The nine new checks are the free-session path, and they are the ones that
raise the count. The "91" was never a second opinion about the same
coverage: it was the same suite with the numpy-dependent checks **skipped**,
because bare system python3 has no numpy. The Windows python has numpy, so
it runs everything and agrees with the venv exactly.

What did not change is what matters: **the new commit adds a free-session
mode and its server endpoint, and it does not touch the take format.** So
the checks that are about the take format still pass, and the three that
fail still fail for exactly the reasons they failed before — and all three
are properties of the *synthetic verify session*, not of the page.

Two harness details worth recording, because both look like findings and
neither is one:

- **The mirror's symlinked `server.py` wrote into the real tree.** The
  server derives its diagnostics path from `Path(__file__).resolve().parent`,
  so a symlinked `server.py` resolves to the real checkout and its
  `diag.log` went there. The file is gitignored (`.gitignore:6`, `*.log`),
  nothing tracked changed, and the main checkout is clean — but the mirror
  now holds a **real copy** of `server.py`, `index.html` and
  `free_record.html` (byte-identical) and a patched `verify.py`
  (byte-identical apart from the one-line output-location patch), so the
  run is self-contained. The mirror is the harness, and the harness should
  not be able to write into the thing under test.
- **`verify.py` reads `free_record.html` from its own directory.** A mirror
  built before the free-record mode has no such file, and the verifier
  raises `FileNotFoundError` before it checks anything at all. That is a
  property of the mirror, not of the build: the file is in the build.

**One consequence for the free-session path, which is new and worth
recording.** The new `POST /free` writes a session folder of five files —
`meta.json`, `audio.wav`, `luma.raw`, `timestamps.json`, `video.webm` —
and the verifier checks that it is honest (the luma length is
`n_frames × 15,360`, the timestamps have one entry per frame, the WAV
header matches the claimed rate, a mismatched body is rejected with a 400)
and that it adds **no json to the out root**, so the gate still sees only
takes. Those checks pass. The session folder is not a take and never
pretends to be one, which is the right design: a free session is a
recording, and the labels are derived from it afterwards — offline, by
`experiments/free_session_analyze.py`
(`docs/research/FREE-SESSION-ANALYZER.md`).

## 2. PASS / FAIL PER CHECK

Measured on the pristine session produced by the **current** `verify.py`
(`/tmp/zcode_6c8dc1b_b/corpus`, 5 takes: A1 ×2, C1, D1, F1).

| Check | Result | What it means here |
|---|---|---|
| `LAYOUT` | **FAIL** | F1 has no `_full.webm` (the verifier posted none) |
| `JSON` | **PASS** | both schemas, all types, all cross-field arithmetic |
| `ARRAYS` | **PASS** | the five arrays, right shapes and dtypes; maps re-derive exactly |
| `NV12` | **PASS** | 150 × 15,360 = 2,304,000 bytes, byte-identical round trip |
| `WAV` | **PASS** | mono 16-bit at 16,000 Hz, 79,950 samples, duration agrees |
| `DRIFT` | **PASS** | 3.992 after row removal, under the script's 5.0; matches the re-derivation |
| `FLAGS` | **PASS** | no flag is demanded by the recorded `speech_frac`, and none is present |
| `SPEECH` | **FAIL** | recorded 0.72 against an independent Silero measurement of 0.0064 and the page's own gate at 0.6084 — the recorded value is a constant, not a measurement |
| `RICH` | **PASS** | all eight addendum items present, matched semantically |
| `WEBM` | **FAIL** | every WebM is a 37-byte stub with no video track and no audio track |
| `MANIFEST` | **PASS** | every take listed, with files, labels, flags and notes |
| *(reference figures)* | **FAIL** | the page's on-screen "empty room 50" is the torn-frame number; the correct value is 30 — see §4.6 |
| `PLAN` | **PASS** (reported) | 4 of 16 plan conditions present — a synthetic session, as expected |
| `GATE` | **PASS** | the gate exits 0 on the directory as it stands, and on the takes alone |

---

## 3. WHAT I MEASURED THAT CONFIRMS THE BUILD

### 3.1 The browser's JavaScript reproduces the reference's mathematics EXACTLY

I recomputed the change maps from the **stored luma** using the reference
implementation's own `row_band_remove` / `change_maps` / `quantise`
functions (`experiments/corpus_capture_win.py`, byte-identical to
`/mnt/c/proart_capture/corpus_capture_win.py`) and compared them with what
the browser wrote:

```
maps      re-derives from the stored luma exactly (max difference 0 counts)
maps_raw  re-derives from the stored luma exactly (max difference 0 counts)
```

Not "within tolerance" — **zero differing cells on both arrays, on every
take**. The shipped JS is not a re-implementation of the script's maths; it
*is* the script's maths. The two scales also agree with the reference's
99.5-percentile to float64 precision.

### 3.2 The frame stream round-trips through the repo's own reader

`src/nv12.py` `detect_layout` on the stored luma:

```
layout luma_only   stride 15,360   150 frames   luma 15,360 bytes
mean consecutive |diff| 3.391907   runner-up y_uv_halfres 85.623604
margin over the runner-up: 25.244
```

The margin is 5× the reader's own `min_margin` of 5.0. The reader that once
mis-strided every capture in this repo reads the browser's frames
unambiguously.

### 3.3 The WAV is honest

`src/wavio.py` reads every WAV at `source_sample_rate == 16000`, the header
rate matches `timing.audio_sample_rate`, and 79,950 samples at 16,000 Hz is
4.997 s against a video span of 5.000 s — a 0.06 % disagreement, far inside
the gate's 2 % tolerance. **The defect this repo spent a night fixing — a
WAV header that lies about its rate — is not present.**

### 3.4 The schemas are the reference's schemas

Compared key-by-key against the reference's own dict literals, parsed with
`ast` rather than by eye:

```
timing:  reference 11 keys   browser 11 keys   missing 0   extra 0
meta:    reference 24 keys   browser 26 keys   missing 0
         extra: capture_tool, speech_gate
```

The two extra keys are provenance, they are documented in `DONE.md`, and the
gate ignores keys it does not read.

### 3.5 The gate passes

`experiments/corpus_integrity.py` exits **0** on the session directory as it
stands — 5 takes, 0 failed — and also on the takes in isolation. ZCode's
claim holds, and I confirmed the mechanism that makes it hold (§5.2).

---

## 4. THE DISCREPANCIES

### 4.1 The speech path is NOT exercised by the verification, and the verified data would flag differently — **the substantive one**

Three measurements, in order.

**(a) The pipeline's gate does not reproduce the recorded number.** I
re-measured every WAV in the verified session with the reference's own Silero
gate (`src.vad.SileroGate`, threshold 0.5, **a fresh gate per take**, exactly
as `corpus_capture_win.py:502` does):

| take | recorded `speech_frac` | my independent Silero measurement |
|---|---|---|
| A1 | 0.7200 | **0.0064** |
| C1 | 0.0100 | 0.0064 |
| D1 | 0.0200 | 0.0064 |
| F1 | (unmeasured) | 0.0064 |

**(b) The recorded numbers are hardcoded, not measured.** `verify.py`'s
`build_take` takes `speech_frac` as a *parameter*, and the specs list passes
the constants 0.72 / 0.01 / 0.02 / None. Nothing in the verify path measures
speech from the audio it writes. And every take's WAV is the same synthetic
bursty tone — `synth_wav(n_samples, rate)` does not depend on the condition
— so the four takes carry identical audio under different labels.

**(c) The page's OWN gate disagrees with the recorded numbers too.** I
extracted `speechStats` verbatim from the shipped `index.html` and ran it
under node on the WAVs the server actually wrote (the same technique ZCode
used for the change maps, applied to the speech gate instead):

```
A1_...735.wav   page_gate_frac = 0.6084   (recorded 0.72)
C1_...736.wav   page_gate_frac = 0.6084   (recorded 0.01)
D1_...737.wav   page_gate_frac = 0.6084   (recorded 0.02)
F1_...738.wav   page_gate_frac = 0.6084   (recorded unmeasured)
```

**What this means.** The verified session's speech fractions are not
reproducible by the pipeline's gate, and they are not reproducible by the
browser's own gate either. They are constants.

**And the consequence is concrete.** Had the page's real gate produced those
numbers, **C1 and D1 — both labelled `speak=False` — would have been flagged
`NOT_SILENT`** (0.6084 > 0.05), and the manifest would carry that flag. The
manifest in the verified session carries `"flags": []` for every take,
because the server's `take_flags` reads the *recorded* fraction.

So: **the achievement check — the `VOICE_MISSING` / `NOT_SILENT` mechanism
the whole corpus plan depends on — is not exercised by the verification at
all, and the verification's own synthetic data would produce different flags
if it were.**

**What is NOT wrong.** The production path computes the number for real:
`index.html:806` does `const speech = speechStats(rec.samples, rec.nSamples,
realRate)` and `:840` puts `speech.frac` into the metadata. The gate exists,
runs on the actual samples, and records which gate produced the number
(`speech_gate: "browser_rms_adaptive"`). The defect is in the
*verification*, which substitutes constants for the measurement and so proves
nothing about it.

**And what remains unproven.** The reference computes `speech_frac` with
Silero; the browser computes it with an adaptive-RMS gate. The thresholds the
corpus gate enforces (0.15 / 0.05) are defined on Silero's output. To
separate the gates from the signal, I built a fixture with **real speech**
from the repo's own corpus
(`examples/realaudio/corpus/poetry_weeds_by_the_wall_1702_librivox.wav`):
Silero measures **0.806** on it. Whether the adaptive-RMS gate gives a
comparable number on real speech is **unproven** — ZCode says so in
`DONE.md`, and so do I. This is the one finding that could change the
corpus's *meaning* rather than its format, and it is measurable in ten
seconds on Casey's machine: record one take, then run both gates over it.

### 4.2 The WebM artifacts carry no tracks at all

My container parser (stdlib EBML: magic → DocType → Segment → Tracks →
TrackEntry → TrackType / CodecID / PixelWidth / PixelHeight /
SamplingFrequency) reports, for every `_full.webm` in the session:

```
no video track
no audio track
```

The files are 37 bytes: an EBML header with `DocType "webm"` and nothing
else. The addendum requires "a full native-resolution video file of the
take (browser MediaRecorder, WebM, camera native res/fps, **WITH the mic
audio track muxed**)". A 37-byte stub satisfies none of that.

**This is disclosed**: `DONE.md` says the WebM "was only tested as a
header-only synthetic blob here; whether the file plays everywhere is a
real-camera question". I am not reporting a hidden defect. I am reporting
that **the addendum's richest requirement is entirely unverified**, and that
the artifact standing in for it fails the check.

For the record, the production path looks right: `index.html` creates
`new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 4000000 })`
on the **raw getUserMedia stream** — so native resolution and native fps by
construction — and `pickMime()` prefers `video/webm;codecs=vp9,opus`, then
`vp8,opus`, then bare `video/webm`. The first two name the Opus audio codec
explicitly, so the muxed audio track is requested. The fallback to bare
`video/webm` is the one path where the audio track is browser-dependent.

### 4.3 One take has no WebM at all

F1's `_full.webm` is absent. That is the **verifier's choice** — `verify.py`
posts F1 with an empty WebM section to exercise the server's "no webm"
branch — and not a production path: `index.html` starts a MediaRecorder for
every take, free sessions included. But the consequence for acceptance is
real: **the verified session does not demonstrate the addendum's WebM
requirement for even one take.**

### 4.4 `DONE.md` overstates one thing about the out root

`DONE.md` says "the out root holds only `<name>_<stamp>.json` and
`_timing.json` files". The root also holds every `.npz`, every `.wav` and
every `_full.webm` — those are, after all, where the takes live. What the
claim is *for* is gate safety, and that is true: **the only JSON at the out
root is take metadata and timing**, so the gate cannot misread anything
there. The sentence is imprecise; the property it defends is real.

### 4.5 Plan coverage in the verified session is 4 of 16

The session contains A1 (twice), C1, D1 and F1. That is expected of a
synthetic verification session, and `DONE.md` does not claim otherwise. It
is reported, not failed. The page's own plan is correct: `REPEATS = 3` over
A, B, C, D, F, then one long FREE — **16 takes**, matching the reference
`PLAN` and `CAPTURE-NEXT.md`.

---

### 4.6 The deliverable's on-screen reference numbers are the torn-frame numbers

This one came out of the camera lane, and it is a defect in the deliverable.

`index.html:212` tells Casey, on the page:

> The existing captures on this camera and in this room sat around 110;
> **the empty room read 50.**

The occupied figure is right. **The empty-room figure is the torn-frame
number.** Recomputed from the files at the correct 19,200-byte stride:

| capture | the page says | correct (19,200) | torn (23,040) |
|---|---|---|---|
| A.nv12 — occupied, facing | ~110 | **105.94** | 110.73 |
| B.nv12 — occupied, turned | ~111 (DONE.md) | **110.17** | 114.16 |
| empty room | 50 | **30.07** | 49.69 |

The page's number matches the torn read exactly. `browser_capture/DONE.md`
repeats it ("111–114", "empty room: 50").

It is a small thing, and it is a real thing: the number Casey is told to
compare the live readout against is the wrong one, by 20 luma counts, and it
was carried over from a bug this repo already corrected everywhere else
(`docs/research/NV12-STRIDE.md`; `experiments/room_map.py` keeps the old
value in its "supersedes" block and recomputes it correctly). The page's
"amber below 80" threshold is not affected — but the *reference* it sits
beside is.

**It also matters for the camera work**, because the corrected numbers change
the shape of the problem: occupied 105.9 against an empty room of 30.1 is a
3.5× dynamic range — a camera working properly. So "the camera is starved"
was never supported by the evidence pack; the question was always "why is
the browser's stream darker than the camera can be", with the unconstrained
reference capture at 105.9 as the yardstick.

### 4.7 `DONE.md`'s summary header still quotes the pre-free-record test counts

A small documentation discrepancy, found while re-measuring against
`6c8dc1b` (§1.5), and worth one line because the numbers are the sort of
thing a reader quotes.

`browser_capture/DONE.md` opens with a summary of what was verified:

> - **145 checks passed, 0 failed** (audio-venv run, numpy + gate included);
>   91 passed under bare system python3 (numpy suite skipped there).

**Both numbers are stale.** The same file's patch log, two hundred lines
below, says:

> Verify suite after this patch: 154 passed, 0 failed (nine new checks:
> front-door routing, plan intact at /plan, the /free write path, the
> gate-safe subfolder, the dishonest-body 400, and /log).

The header was not updated with it — `git diff e42143b..6c8dc1b --
browser_capture/DONE.md` touches the patch-log line and not the header. I
measured **154 passed, 0 failed, 0 skipped** in the audio venv and
**100 passed, 0 failed, 1 skipped** under bare system python3 3.14.4 (the
91 was the same suite at the previous build, with the nine new checks
absent; they need no numpy, so the numpy-less count rises by nine).

The claims are true of the build they were written for, and stale for the
build that is in the tree. Nothing here is a defect in the product; it is
the reason this verdict quotes measurements rather than `DONE.md`.

## 5. THE TWO WRITTEN-CONFLICT FINDINGS

Both were derived blind in `ACCEPTANCE-EXPECTED.md` §12 and both are now
**demonstrated**.

### 5.1 The base brief says "19 takes"; the plan has 16

`BRIEF-BROWSER-CAPTURE.md` says "Plan driver for all 19 takes". The
reference's `PLAN` is `REPEATS = 3` over `[A, B, C, D, F]` plus one `FREE`
= **16**, and `CAPTURE-NEXT.md`'s table lists the same 16. ZCode found this
too, resolved it to 16, and documented the reasoning in `DONE.md` format
decision 1. **The brief's arithmetic is wrong; ZCode's reading is right.**

### 5.2 The addendum's manifest at the out root is individuated as a take by the gate

The addendum says "Session manifest **at the out root**: `session.json`".
`corpus_integrity.py` discovers takes with

```python
corpus.glob("*.json")  # minus *_timing.json
```

so a root-level `session.json` **is** picked up as a take, and it fails: no
`condition` field, no `session.npz`. The briefs cannot both be satisfied by
a flat directory, and neither brief may be edited.

I demonstrated this rather than arguing it. My fixture puts `session.json`
**at the out root, exactly as the addendum asks**, and the gate says:

```
on the directory as it stands: FAIL (17 takes, 3 failed)
   session.json: label: metadata has no condition field
   session.json: file: session.npz is missing
on the takes alone:             FAIL (16 takes, 2 failed)
```

**17 takes for a 16-take session.** The conflict is real, and it is a defect
in the *briefs*, not in either implementation.

**ZCode's resolution is correct and is the better reading of the addendum's
intent**: the rich artifacts live in `out/rich/`, so the out root holds only
take metadata and timing, and the gate passes on the directory as it stands.
The cost is that the manifest is not literally "at the out root", which
`DONE.md` states plainly as format decision 5.

**My checker's treatment of this**: the manifest is *found* wherever it is,
its contents are checked wherever it is found, and its **location** is
reported as a discrepancy rather than silently normalised. The gate is run
**twice** — on the directory as it stands, and on the takes alone — and the
verdict says which one failed and why. A checker that picked one silently
would be hiding the very thing the verdict is for.

---

## 6. WHAT STILL NEEDS CASEY'S REAL-CAMERA TEST

Nothing in this verdict certifies the production path. These need the real
machine, and ZCode's own list agrees with mine:

1. **`getUserMedia` end to end** — permissions, device selection, and a
   camera that delivers a usable 160×96 stream.

   **The camera lane has since narrowed this considerably**
   (`docs/research/CAMERA-RND-FINDINGS.md`): the device is an **ASUS FHD
   webcam**, it is **not starved** (the corrected references are 105.9
   occupied against 30.1 for the empty room — a working auto-exposure), and
   the camera is **single-consumer**: a second consumer gets
   `NotReadableError: Device in use`. There are **two** physical cameras on
   the machine, and the **ASUS IR camera** is the dark-picture impostor that
   `3c0b2fe` now avoids with a picker. So this item is no longer "will the
   camera work" — it is "which camera did the page pick, and was anything
   else holding it". Both are now answerable in one look at the page's
   picker and luma readout.
2. **The achieved frame rate.** The whole point of `timing.json` is that the
   time base is measured, not assumed. A browser tab throttled to 15 fps
   produces takes whose time base is wrong by 2× — the same class of defect
   as the old 1.2× capture, and only a live run exposes it. The page records
   the achieved rate honestly and the server flags `FPS_OFF`, so this is
   checkable on the first real take.
3. **The mic's actual rate.** The old captures claimed 16,000 Hz when the
   data was not. The page requests `AudioContext({sampleRate: 16000})` and
   records `realRate`, flagging `MIC_RATE` when the browser refuses. If
   Chrome on Casey's machine does not honour it, **the gate will fail every
   take** until the mic runs at 16 kHz — a real risk, not a hypothetical one.
4. **The shared clock.** One `performance.now()` reading per tick, with the
   audio paced by the audio clock. Whether the browser's video and audio
   clocks really agree is only measurable live; the spans must agree to
   within 2 %.
5. **The MediaRecorder output.** Whether `_full.webm` plays, carries the
   muxed mic audio, and holds the camera's native resolution — including
   whether the bare `video/webm` fallback keeps the audio track.
6. **The speech gate on real speech, and the flags it produces.** Whether the
   adaptive-RMS gate agrees with Silero on Casey's actual voice (§4.1), and
   whether the `VOICE_MISSING` / `NOT_SILENT` flags land correctly on real
   takes. This is the one finding that could change the corpus's *meaning*
   rather than its format, and it is measurable in ten seconds: record one
   take, then run both gates over it and compare the flags. The verification
   does not exercise this path at all (§4.1), so nothing short of a real
   recording will.
7. **Which constraint set, measured.** The camera lane could not answer this
   from outside the browser: the device is single-consumer, the browser
   holds it, and the one fallback stack available from the Windows side
   (DirectShow) negotiates badly with this UVC device and returns digital
   black — a measurement error I made and then corrected, recorded in
   `CAMERA-RND-FINDINGS.md` §1. The **probe** is the instrument, it is
   running on `127.0.0.1:8766`, and `?sweep=1` measures P0, P1, P3 and P4
   back to back and writes the whole decision table. ZCode's near-default
   constraint patch (`5194314`) is the right prior — the reference capture
   set *no* constraints and produced 105.9 — but it is unverified until
   those numbers land.

8. **The luma kernel.** The browser's canvas resampling versus the script's
   `cv2.INTER_AREA`. Same geometry, different kernel. If browser takes and
   script takes will ever be pooled pixel-wise, that difference should be
   measured before they are pooled.
9. **Casey performing the conditions.** A real 90° turn, the poem recited
   from memory, steady lighting. `CAPTURE-NEXT.md` is blunt that a posture
   shift is indistinguishable from a head turn in a 160×96 frame: "it is the
   one confound that cannot be measured away."

---

## 7. HOW TO REPRODUCE THIS VERDICT

```bash
# the checker, against any session directory
python3 experiments/browser_capture_acceptance.py <session_dir> --json report.json

# the checker's own validation: a session built from the derivation alone
python3 experiments/browser_capture_acceptance_fixture.py /tmp/fixture
python3 experiments/browser_capture_acceptance.py /tmp/fixture --require-full-plan
#   -> 12/12 PASS, exit 0

# ZCode's own verification, reproduced
cd /tmp/accept/communicator && BC_OUT=/tmp/zcode_venv/corpus \
    /path/to/audio-venv/bin/python browser_capture/verify.py
#   -> 154 passed, 0 failed, 0 skipped   (at 6c8dc1b; 145 at e42143b)

# the same suite under a python with NO numpy -- the numpy checks are skipped
cd /tmp/accept/communicator && BC_OUT=/tmp/zcode_bare/corpus \
    /usr/bin/python3 browser_capture/verify.py
#   -> 100 passed, 0 failed, 1 skipped   (91 at e42143b)

# the page's OWN speech gate, over the WAV the server wrote
python3 experiments/browser_capture_speech_probe.py \
    /tmp/zcode_venv/corpus/A1_*.wav
#   -> the page's gate 0.6084, the pipeline's gate 0.0064, recorded 0.72
```

The mirror at `/tmp/accept/communicator` holds **real copies** of
`browser_capture/{server.py,index.html,free_record.html}` and of
`verify.py` — byte-identical to the main checkout apart from the one-line
output-location patch that makes the corpus land in `BC_OUT`:

```
out_dir = tmp / "corpus"
  ->  out_dir = Path(_os.environ.get("BC_OUT", str(tmp / "corpus")))
```

Real copies, not symlinks, because the server derives its diagnostics path
from `Path(__file__).resolve().parent` (§1.5): a symlinked `server.py`
resolves into the real checkout and writes there.

The checker's provenance block records the script's SHA-256, the numpy
version, the Silero model it used, and the fact that the reference
implementation and the gate were both loaded from this checkout.

---

## 8. WHAT I DID NOT DO

- I did not edit anything under `browser_capture/`, and I did not write to
  the real corpus (`/mnt/c/proart_capture/corpus`). The verified session was
  reproduced in a scratch directory and read read-only.
- I did not merge. This branch is for Muse to review.
- I did not certify the real-camera path, because nothing in this exercise
  can.
- I did not soften a failure into a note. The checker's verdict on the
  verified session is FAIL, and it is reported as FAIL, with the reason for
  each failure separated into "defect" and "not verifiable offline" in §0
  and §6 rather than being buried.
- **One file in the main checkout was appended to, and it is a log.**
  Running ZCode's own verifier through a mirror whose `server.py` was a
  symlink made the server resolve `Path(__file__).resolve().parent` into
  the real tree and append one line to `browser_capture/diag.log`. The file
  is gitignored (`.gitignore:6`, `*.log`), nothing tracked changed, and
  `git status` in the main checkout is clean. The mirror now holds real
  copies so it cannot recur. I am recording it rather than leaving it to
  be found later, because the constraint was not to write into that tree
  and the one write that happened was mine.
