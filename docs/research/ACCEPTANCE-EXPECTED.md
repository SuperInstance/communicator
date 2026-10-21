# ACCEPTANCE-EXPECTED — what a browser-capture take must contain, derived blind

**Written by:** the acceptance referee (this worktree, `acceptance/browser-capture`)
**Date:** 2026-10-20
**Subject:** `browser_capture/` in the main checkout (`~/projects/communicator`)

---

## 0. WHAT THIS DOCUMENT IS, AND HOW IT WAS DERIVED

This is the referee’s **independent** statement of what a correct browser-capture
take looks like, written **before reading any of ZCode’s code**. ZCode writes its
own `verify.py`; a judge cannot certify itself. So the expectations below were
derived only from sources that predate the browser build and are named as
authoritative by the brief itself.

**Sources read, in this order:**

1. `/mnt/c/proart_capture/corpus_capture_win.py` — the reference implementation.
   Byte-identical to this repo’s `experiments/corpus_capture_win.py` (verified by
   `diff`: identical). This defines what `--plan` writes per take.
2. `CAPTURE-NEXT.md` (repo root) — the plan, the conditions, the warnings, the
   thresholds’ meaning.
3. `docs/research/BRIEF-BROWSER-CAPTURE.md` (main checkout) — the base brief.
4. `docs/research/BRIEF-BROWSER-CAPTURE-ADDENDUM.md` (main checkout) — the
   addendum.
5. `experiments/corpus_integrity.py` — **the gate the output must satisfy**.
6. `src/nv12.py`, `src/wavio.py` — the readers the files must round-trip through.
7. `experiments/seed_corpus_from_captures.py` + the real corpus on disk
   (`/mnt/c/proart_capture/corpus_v2/*.json`) — the canonical shape of a
   pipeline-format take, for cross-checking the derivation.

**Deliberately NOT read at the time of writing:** anything under
`browser_capture/` in the main checkout (`index.html`, `server.py`, `verify.py`).
Those were read only **after** this file was written, and only to *check* the
deliverable against what is written here.

**The rule applied throughout:** where a format is fixed by the reference
implementation, this document states it exactly. Where the briefs leave a
detail open (notably the addendum’s JSON key names), this document says so
explicitly and defines the **semantic** requirement instead of inventing a
format the brief never asked for.

---

## 1. THE PLAN: 16 TAKES

The reference `PLAN` is built as: `REPEATS = 3` × conditions `[A, B, C, D, F]`,
then one trailing long free take. That is **3 × 5 + 1 = 16 takes**.

`CAPTURE-NEXT.md`’s table lists exactly those 16: A1–A3, B1–B3, C1–C3, D1–D3,
F1–F3, FREE.

> **DISCREPANCY IN THE BASE BRIEF.** `BRIEF-BROWSER-CAPTURE.md` says “Plan
> driver for all **19** takes”. 3+3+3+3+3+1 = **16**, not 19; both the
> reference `PLAN` and `CAPTURE-NEXT.md` agree on 16. The “19” is an
> arithmetic slip in the brief. The expectation here is **16** — a browser
> with 19 takes has invented five conditions that the corpus plan does not
> contain, and a browser with fewer has dropped takes the plan requires.

| Condition | Kind | `pose` | `speak` | Duration | Count |
|---|---|---|---|---|---|
| `A1`–`A3` | `condition` | `facing` | `true` | 50 s | 3 |
| `B1`–`B3` | `condition` | `turned` | `true` | 50 s | 3 |
| `C1`–`C3` | `condition` | `facing` | `false` | 30 s | 3 |
| `D1`–`D3` | `condition` | `turned` | `false` | 30 s | 3 |
| `F1`–`F3` | `free` | `any` | `null` | 60 s | 3 |
| `FREE` | `free` | `any` | `null` | 300 s (5 min) | 1 |

The distinction that the gate enforces and that must not be blurred:
`kind == "condition"` ⇒ `speak` is a **boolean**; `kind == "free"` ⇒ `speak` is
**null**. `F1`–`F3` are *free* takes in the reference (`"speak": None`), not
speaking conditions, so they carry no speech ground truth.

---

## 2. THE PER-TAKE ARTIFACT SET

Let `<STEM>` be `<COND>_<STAMP>`, where `<COND>` is the condition label
(`A1`, `FREE`, …) and `<STAMP>` is `int(time.time())` — epoch seconds, as an
integer, as the reference writes it (`stamp = int(time.time())`).

### 2.1 REQUIRED — the pipeline-exact set (base brief + reference `--plan`)

| File | Kind | Written by reference as |
|---|---|---|
| `<STEM>.npz` | numpy archive | `np.savez_compressed` |
| `<STEM>.wav` | mono 16-bit PCM WAV | `wave.open(..., "wb")` |
| `<STEM>_timing.json` | timing metadata | `json.dumps(timing)` |
| `<STEM>.json` | take metadata | `json.dumps(meta, indent=2)` |

> Note the **timing file’s name**: `<STEM>_timing.json` — the suffix is
> `_timing.json`, appended *after* the stamp, not `_timing.json` replacing
> `.json`. The gate looks it up as
> `meta_path.with_name(f"{stem}_timing.json")`
> (`experiments/corpus_integrity.py:215`), and excludes `*_timing.json` from
> its take list. A file named `<STEM>_timing` or `<STEM>.timing.json` would be
> silently invisible to the gate.

### 2.2 REQUIRED — the addendum set (browser only, in addition)

| File | Kind | Addendum item |
|---|---|---|
| `<STEM>_full.webm` (or `<COND>_full.webm`) | WebM, native res/fps, **video + muxed mic audio** | 1 |
| `<STEM>_rich.json` | per-take rich sidecar | 3 |
| `session.json` at the **out root** | session manifest | 4 |

The PCM WAV stays its own file (addendum item 2) — the browser must not fold
the audio into the WebM only.

**On `<label>` in the addendum’s file names.** The addendum writes
`<label>_full.webm` and `<label>_rich.json` but never defines `label`. The
reference’s per-take file prefix is `<COND>_<STAMP>`, and the base brief
requires re-recording individual takes out of order — so a bare condition label
(`A1_full.webm`) would be overwritten by a second take of the same condition
and could not be attributed to a take. The acceptance expectation is therefore:
**one WebM and one rich JSON per take**, named with the take stem (preferred)
or with the condition label; a per-condition file shared between two takes of
the same condition is reported as a discrepancy, because it cannot be
attributed to a take.

### 2.3 WHAT IS *NOT* REQUIRED

- No `.nv12` file. The reference `--plan` writes **no** raw frame file; the raw
  frames live inside the `.npz` as the `luma` array. A browser that writes an
  extra raw frame file is not wrong — but if it does, that file is checked too
  (§7, `NV12`).
- No MediaRecorder/webm **substitute** for the raw frames. The base brief is
  explicit: “the raw frames are the data product”.

---

## 3. FRAME GEOMETRY AND STRIDE ARITHMETIC

Constants, from the reference (`corpus_capture_win.py:221-224`):

```
W, H      = 160, 96          # capture dimensions
MAP_W, MAP_H = 80, 48        # stored change-map resolution
RATE      = 16000            # requested audio sample rate
FPS       = 30               # assumed frame rate
```

### 3.1 The luma stream

The reference converts each camera frame with `cv2.COLOR_BGR2YUV_I420` and
stores **only the Y plane**: `yuv[:96, :160]`, i.e. **160 × 96 = 15,360 bytes
per frame**.

`src/nv12.py` knows three layouts for a 160×96 frame:

| Layout | Stride | What it is |
|---|---|---|
| `standard_nv12` | `W*H*3//2` = **23,040** | luma + interleaved UV — *the wrong assumption that destroyed the repo’s video evidence* (`docs/research/NV12-STRIDE.md`) |
| `y_uv_halfres` | `W*H*5//4` = **19,200** | luma + interleaved UV at 80×24 — what `nv12small.py` wrote |
| `luma_only` | `W*H` = **15,360** | luma only — **the layout of the pipeline’s `luma` array** |

> The capture tool stores luma-only frames, so the browser’s raw frame stream
> has stride **15,360**, *not* 19,200 and *not* 23,040. A browser that
> interleaves chroma into the pipeline artifact would not be the layout the
> pipeline expects.

**The round-trip requirement** (base brief: “file size == frames × stride via
`src/nv12.py`”) therefore reads: the take’s raw luma stream is
`n_frames × 15,360` bytes, and `src/nv12.py`’s `luma_view(data, "luma_only",
160, 96)` must return exactly `(n_frames, 96, 160)` with byte-identical
content. `n_frames` here is the **frame count**, and the change-map count is
`n_frames − 1` (see §3.2) — a take that reports `n_frames` but stores
`n_frames` change maps has an off-by-one that the gate also catches.

### 3.2 The change maps

```
d    = |frame(t+1) − frame(t)|            shape (n−1, 96, 160)
map  = d.reshape(n−1, 48, 2, 80, 2).mean(axis=(2, 4))    shape (n−1, 48, 80)
```

- 2×2 box average → **48 × 80** per map (this is `MAP_H × MAP_W`).
- `n_frames − 1` maps, because there are `n−1` consecutive-frame differences.
- Row `r` of a map covers luma rows `2r, 2r+1`; column `c` covers luma
  columns `2c, 2c+1`.

Two variants are stored, with the **same** count and shape:

| Array | Definition |
|---|---|
| `maps` | change maps of the **row-removed** luma |
| `maps_raw` | change maps of the **raw** luma (unfiltered) |

The row-removal (the reference’s own `row_band_remove`): per-row **median**
across the columns, minus the row-profile mean, subtracted from the frame.

Quantisation (the reference’s own `quantise`): `hi = percentile(maps, 99.5)`,
floored at `1e-6`; `q = clip(maps / hi * 255, 0, 255).astype(uint8)`; the
`hi` is stored **alongside** as `scale` / `scale_raw` so nothing is silently
clipped.

### 3.3 The `.npz` array set

`np.savez_compressed` with exactly these five arrays:

| Key | dtype | Shape | Meaning |
|---|---|---|---|
| `maps` | `uint8` | `(n_frames−1, 48, 80)` | row-removed change maps |
| `scale` | `float32` (scalar) | `()` | the 99.5-percentile scale of `maps` |
| `maps_raw` | `uint8` | `(n_frames−1, 48, 80)` | unfiltered change maps |
| `scale_raw` | `float32` (scalar) | `()` | the 99.5-percentile scale of `maps_raw` |
| `luma` | `uint8` | `(n_frames, 96, 160)` | the raw luma frames |

The gate requires `maps` and `scale` and checks `maps.shape[0] == n_frames − 1`
(`corpus_integrity.py:344-364`). The reference’s own `meta["arrays"]` field
records `["maps", "maps_raw", "luma"]` and `meta["filters"]` records
`{"maps": "row", "maps_raw": "none", "luma": "none"}` — the representation is a
**post-hoc decision**, recorded per array, not one property of the take.

---

## 4. WAV REQUIREMENTS

| Property | Requirement | Source |
|---|---|---|
| Channels | **1** (mono) | reference `f.setnchannels(1)` |
| Sample width | **2 bytes** (16-bit PCM) | reference `f.setsampwidth(2)` |
| Frame rate in header | the **achieved** mic rate, honest | reference `f.setframerate(real_rate)`; `CAPTURE-NEXT.md` “the WAV is written with the rate it was ACTUALLY captured at” |
| Sample count | ≈ `n_frames × int(rate/30)` | reference reads `blk = int(real_rate / FPS)` samples per frame |
| Duration | `n_samples / rate` | = `audio_span_s` |

With the plan’s rate of 16,000 Hz: `blk = int(16000/30) = 533` samples per
frame. For a 50 s take: `n_frames = 1500`, `n_samples = 799,500`,
`audio_span_s = 49.97`, `video_span_s = 1500/30 = 50.00`.

**The gate’s two rate/span checks** (`corpus_integrity.py:269-329`):

- `|achieved_fps − assumed_fps| / assumed_fps ≤ 0.05` — the camera-rate
  consistency check (this is the one that catches the old 1.2× time-base
  defect).
- `|audio_span_s − video_span_s| / video_span_s ≤ 0.02` — the A/V span
  agreement check.
- `audio_sample_rate == requested_sample_rate` — the audio rate must be what
  was requested (16,000).

**How the file must be read.** `src/wavio.read_wav` returns an `Audio` whose
`sample_rate` is **always** the analysis rate 16,000 (it resamples), and whose
`source_sample_rate` is the file’s **actual** rate. So “the WAV reads at its
stated rate” means: `read_wav(...).source_sample_rate` equals the WAV header
rate, equals `timing.json`’s `audio_sample_rate`, and equals
`requested_sample_rate`. The duration check is
`len(audio.samples) / 16000 ≈ audio_span_s` (exact when the file rate is
16,000, because the resample is then the identity).

---

## 5. `timing.json` SCHEMA

File: `<STEM>_timing.json`. Keys and types, exactly as the reference writes
them (`corpus_capture_win.py:481-494`):

| Key | Type | Value / constraint |
|---|---|---|
| `condition` | str | the take label, e.g. `"A1"` |
| `n_frames` | int | frame count; `len(frame_offsets_s)` must equal it |
| `frame_offsets_s` | list[float] | one timestamp per frame, relative to the first, 6 dp |
| `assumed_fps` | int | 30 |
| `achieved_fps` | float | **measured**, `(n_frames−1) / (last−first offset)` |
| `video_span_s` | float | `n_frames / assumed_fps` |
| `audio_span_s` | float | `n_samples / achieved_rate` |
| `audio_sample_rate` | int | the rate the audio was **actually** captured at |
| `requested_sample_rate` | int | 16000 |
| `mic` | str | the input device name |
| `capture_wallclock_s` | float | seconds spent capturing |

Cross-field arithmetic the acceptance checker re-derives:

```
achieved_fps  == (n_frames − 1) / (frame_offsets_s[-1] − frame_offsets_s[0])
video_span_s  == n_frames / assumed_fps
audio_span_s  == n_samples / audio_sample_rate
```

The **achieved** rate is the truth; the frame index is not. This is the whole
point of the file: `CAPTURE-NEXT.md` — “The timestamps in timing.json are the
truth; use them, not the frame index.”

---

## 6. `meta.json` SCHEMA

File: `<STEM>.json`. Keys and types, exactly as the reference writes them
(`corpus_capture_win.py:519-554`):

| Key | Type | Value / constraint |
|---|---|---|
| `condition` | str | must match the file’s condition prefix |
| `kind` | str | `"condition"` or `"free"` |
| `note` | str | the condition instruction |
| `pose` | str | `"facing"` / `"turned"` / `"any"` |
| `speak` | bool \| null | bool iff `kind == "condition"`; null iff `kind == "free"` |
| `stimulus` | str | the stimulus name — **must be identical across the corpus** |
| `stimulus_title` | str | human title of the stimulus |
| `stimulus_estimated_s` | float | estimated stimulus duration |
| `speech_frac` | float \| null | the gate’s speech fraction (null if the gate was unavailable) |
| `n_frames` | int | frame count |
| `seconds` | float | video span |
| `achieved_fps` | float | achieved camera rate |
| `audio_sample_rate` | int | achieved mic rate |
| `audio_seconds` | float | audio span |
| `map_shape` | list[int] | `[48, 80]` |
| `map_scale` | float | the `scale` stored in the `.npz` |
| `map_scale_raw` | float | the `scale_raw` stored in the `.npz` |
| `drift_raw` | float | mean \|diff\| of the raw luma |
| `drift_after_row_removal` | float | mean \|diff\| after row removal — **must be < 5.0** |
| `timestamp` | int | epoch seconds — the **session identity** |
| `rep` | int | repetition number (1..3) |
| `filter` | str | `"row"` (the label the reference corrected to) |
| `filters` | dict | `{"maps": "row", "maps_raw": "none", "luma": "none"}` |
| `arrays` | list[str] | `["maps", "maps_raw", "luma"]` |

**The `timestamp` field is load-bearing.** The gate counts *distinct
sessions* by distinct `timestamp` — “one session recorded twice is one
session”. Two takes with the same `timestamp` are one session for coverage
purposes.

**Drift.** `drift_after_row_removal < 5.0` is the capture script’s own
usability threshold (`corpus_capture_win.py:446`) and the gate’s `drift`
check. Note that the threshold was re-derived against the corrected frames:
occupied baseline 0.545, empty-room baseline 0.005.

---

## 7. THE ADDENDUM ARTIFACTS IN DETAIL

### 7.1 `<label>_full.webm`

Required by addendum item 1: a **full native-resolution** video file of the
take, **WebM**, camera native res/fps, **with the mic audio track muxed**,
saved next to the take.

Acceptance checks: the file exists and is non-empty; it is a real WebM
(EBML magic `1A 45 DF A3`, DocType `webm`); it has **both** a video track and
an audio track (`TrackType` 1 and 2). A WebM with no audio track is a
half-done artifact, not a rich copy.

### 7.2 `<label>_rich.json`

Required by addendum item 3, with these **contents**:

1. condition label
2. the instruction text shown on screen
3. the stimulus name
4. wall-clock **start and end**
5. achieved fps **and** sample rate
6. a **reference** to the per-frame timestamp array
7. browser/camera device label, if available
8. a free-text **notes** field, default empty — “what I actually did”

> **The addendum specifies contents, not key names.** Unlike the pipeline
> artifacts, whose schemas are fixed byte-for-byte by the reference
> implementation, the rich sidecar’s key names are not specified anywhere in
> the brief, the addendum, or the reference. The acceptance checker therefore
> matches each required item **semantically**, by normalised key alias
> (case- and separator-insensitive), and reports which key it matched. Inventing
> a required key name would be the referee improvising a format — precisely
> what the base brief forbids (“do not improvise formats”). The one item the
> addendum pins down is the **notes** field’s *default*: it must exist and be
> empty by default.

Item 6 is a **reference**, not a copy: the timestamps live in
`<STEM>_timing.json` (`frame_offsets_s`). A reference may be a path, a
filename, or an in-file pointer.

### 7.3 `session.json` at the out root

Required by addendum item 4: “listing every take, its files, labels and
flags, so an external tool can ingest the whole session without knowing our
repo layout.”

Acceptance checks: the manifest exists at the out root; it lists **every** take
found in the session; each listed take carries its files, its label, and its
flags; every file it names exists.

> **DISCREPANCY BETWEEN THE ADDENDUM AND THE GATE — and it is real.**
> `corpus_integrity.py` discovers takes with
> `corpus.glob("*.json")` minus `*_timing.json`
> (`corpus_integrity.py:392-395`). A `session.json` sitting in the same
> directory is therefore **individuated as a take**, and it fails: it has no
> `condition` field (→ `label` failure) and no `session.npz` (→ `file`
> failure). The gate would exit non-zero on a directory that satisfies the
> addendum perfectly.
>
> The briefs cannot both be satisfied by a flat directory, and neither brief
> may be edited. The resolution is structural: the manifest must not sit in
> the directory the gate scans (e.g. takes in a per-session subdirectory,
> manifest at the out root), or the gate must be pointed at the takes
> directory. The acceptance checker therefore reports the gate’s verdict
> **twice** — on the directory as it stands (what a user actually runs), and
> on the takes alone (what the pipeline would see if the manifest were out of
> the way) — and says which one failed and why. This is reported as a
> **discrepancy**, not silently normalised away, because it is a genuine
> conflict between two written requirements.

> **On locating the rich artifacts.** The addendum fixes `session.json` at
> the out root and the base brief puts the rich sidecar "next to the take",
> but neither statement is a *format*, and a producer that resolves the
> conflict above structurally (by keeping the manifest out of the gate's
> scan) is not thereby wrong. The checker therefore **finds** the rich
> sidecar and the manifest wherever they are -- next to the take, in a
> `rich/` subdirectory, or elsewhere in the session -- checks their contents
> wherever they are found, and **reports the location** as a discrepancy
> against the addendum's wording. A missing artifact is a failure; a
> relocated one is a reported deviation.

---

## 8. THE FLAGS: `VOICE_MISSING` / `NOT_SILENT`

The capture script’s own achievement thresholds (`corpus_capture_win.py:505-511`),
which the gate reuses verbatim (`corpus_integrity.py:99-100`):

```
speak == True   and speech_frac < 0.15   ->  VOICE_MISSING
speak == False  and speech_frac > 0.05   ->  NOT_SILENT
```

The reference **prints** these warnings; it does not persist them. The browser
brief requires them “shown in the UI, take kept but flagged exactly as the
script does”, and the addendum requires the session manifest to list “labels
and **flags**”. So in the browser they must be **persisted** somewhere findable.

**Acceptance expectation, bidirectional.** For each take:

- `speak == true` ⇒ `VOICE_MISSING` must be present **iff** `speech_frac < 0.15`.
- `speak == false` ⇒ `NOT_SILENT` must be present **iff** `speech_frac > 0.05`.
- `speak == null` (free takes) ⇒ neither flag is required; either one is
  reported as a discrepancy (a free take has no ground truth to fail).

“Present” means the token appears in the take’s own JSON sidecars (`<STEM>.json`,
`<STEM>_timing.json`, `<STEM>_rich.json`) or in that take’s entry in
`session.json`. The check is bidirectional on purpose: a take that claims
speech and carries `VOICE_MISSING` is as wrong as one that should carry it and
does not.

**Independent re-measurement.** The referee re-runs the **same** Silero gate
the reference uses (`src/vad.SileroGate` over `src/wavio.read_wav`, threshold
0.5, one probability per 32 ms frame) on the take’s WAV and compares the
result with the recorded `speech_frac`. This is what makes the flag check more
than a tautology: a producer could write `speech_frac: 0.9` into its metadata
to avoid a flag, and only a re-measurement catches that. If the Silero weights
are unavailable, the re-measurement is reported as **unverified**, not as a
pass.

---

## 9. WHAT THE CHECKER ASSERTS, AND WHAT IT ONLY REPORTS

`experiments/browser_capture_acceptance.py` (this worktree) takes a session out
directory and checks every take. Named checks, non-zero exit on any failure:

| Check | Asserts |
|---|---|
| `LAYOUT` | the required artifact set exists per take, with the reference’s names |
| `NV12` | the raw luma stream round-trips through `src/nv12.py`: size == frames × stride, content byte-identical |
| `WAV` | `src/wavio.py` reads it; header rate == `timing.audio_sample_rate` == requested; duration == `audio_span_s` |
| `JSON` | `timing.json` and `meta.json` parse and match the schemas in §5–§6, and their cross-field arithmetic holds |
| `ARRAYS` | the `.npz` holds the five arrays with the right shapes/dtypes; the maps re-derive from the stored luma via the reference’s own functions |
| `DRIFT` | `drift_after_row_removal` is < 5.0 and matches the value re-derived from the stored luma |
| `FLAGS` | `VOICE_MISSING` / `NOT_SILENT` present exactly where the data demands them; the recorded `speech_frac` is corroborated by an independent Silero re-measurement |
| `RICH` | `_rich.json` carries all eight required items |
| `WEBM` | `_full.webm` exists, is a WebM, and carries both a video and an audio track |
| `MANIFEST` | `session.json` lists every take with its files, labels and flags |
| `GATE` | `experiments/corpus_integrity.py` passes on the takes |
| `PLAN` | which of the 16 plan conditions are present (reported; a hard failure only under `--require-full-plan`) |

**Reported but not failed:** plan coverage (the synthetic session is expected
to be partial), the gate/manifest conflict of §7.3, and any `_full.webm` /
`_rich.json` naming ambiguity.

---

## 10. THRESHOLDS, COLLECTED

| Constant | Value | Where it comes from |
|---|---|---|
| `SPEECH_MIN` | 0.15 | `corpus_integrity.py:99` / `corpus_capture_win.py:506` |
| `SPEECH_MAX` | 0.05 | `corpus_integrity.py:100` / `corpus_capture_win.py:509` |
| `FPS_REL_TOL` | 0.05 | `corpus_integrity.py:104` / `corpus_capture_win.py:417` |
| `SPAN_REL_TOL` | 0.02 | `corpus_integrity.py:109` |
| `DRIFT_MAX` | 5.0 | `corpus_integrity.py:113` / `corpus_capture_win.py:446` |
| `TARGET_SESSIONS_PER_CLASS` | 63 | `corpus_integrity.py:116` |
| capture dims | 160 × 96 | `corpus_capture_win.py:221` |
| map dims | 48 × 80 | `corpus_capture_win.py:222` |
| audio rate | 16,000 Hz | `corpus_capture_win.py:223` |
| fps | 30 | `corpus_capture_win.py:224` |
| `REPEATS` | 3 | `corpus_capture_win.py:187` |

---

## 11. WHAT NO AMOUNT OF OFFLINE CHECKING CAN SETTLE

These are for Casey’s real-camera test, and the verdict must say so plainly:

1. **`getUserMedia` actually works on Casey’s machine** — permissions, device
   selection, and a camera that delivers a usable 160×96 stream.
2. **The browser’s achieved frame rate** lands within 5 % of the assumed 30 fps.
   A browser tab throttled to 15 fps produces a take whose time base is wrong
   by 2× — the same class of defect as the old 1.2× capture, and only the
   live run exposes it.
3. **The audio rate the browser actually gets** from the mic, and whether the
   `AudioWorklet` clock and the video clock share a real time base. The
   reference took one clock reading per iteration; the browser must not take
   two clocks and hope.
4. **The mic’s real sample rate.** The old captures claimed 16,000 Hz when
   the data was not; only a live take reveals what the browser’s
   `AudioContext` rate is.
5. **The MediaRecorder WebM actually carries the muxed audio** in the
   browser Casey uses, at native resolution.
6. **Whether the speech gate inside the browser** (the page’s own
   SPEAKING/SILENT meter) agrees with the referee’s Silero re-measurement.
   A browser-side energy meter is not the Silero gate, and the two can
   disagree exactly where it matters.
7. **Everything about Casey actually performing the conditions** — the
   head turn really being 90°, the poem really being recited from memory, the
   lighting really being steady. `CAPTURE-NEXT.md` is blunt that a posture
   shift is indistinguishable from a head turn in a 160×96 frame: “it is the
   one confound that cannot be measured away.”

---

## 12. SUMMARY OF THE TWO WRITTEN-CONFLICT FINDINGS

1. **`BRIEF-BROWSER-CAPTURE.md` says “19 takes”; the plan has 16.** The
   reference `PLAN` and `CAPTURE-NEXT.md` both give 16 (A1–A3, B1–B3, C1–C3,
   D1–D3, F1–F3, FREE). The acceptance expectation is 16.
2. **The addendum’s `session.json` at the out root is individuated as a take
   by `corpus_integrity.py`**, which globs `*.json` and would fail it for
   having no `condition` and no `.npz`. The manifest and the gate cannot both
   sit in the same scanned directory. The checker reports the gate’s verdict
   both ways rather than picking a winner silently.
