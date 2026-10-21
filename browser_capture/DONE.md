# DONE — browser_capture

Built 2026-10-20, per `docs/research/BRIEF-BROWSER-CAPTURE.md` and its
`BRIEF-BROWSER-CAPTURE-ADDENDUM.md`.

## Start command and URL

The server is a **systemd USER service** (`browser-capture.service`):
it starts with WSL (linger is on), restarts itself on failure, and logs
to `browser_capture/server.log`. The bench command is a VIEW, never the
server:

```
python3 browser_capture/bench.py    # status + /info + live log tail
```

The one status command: **`systemctl --user status browser-capture`**.

Open **http://localhost:8765** in the Windows browser — the FREE RECORD
AUTO front door. Stdlib-only server, binds 127.0.0.1:8765, default out
dir `/mnt/c/proart_capture/corpus` (`--out` / `--port` to change).
README.md has the plain-language walkthrough.

(For a scratch run outside the service, `python3
browser_capture/server.py --out /tmp/x --port 8799` still works — the
service and a scratch server can share the machine on different ports,
but only one thing may own 8765.)

## What was verified, and how

`browser_capture/verify.py` (run it with system python3 for the stdlib
suite, with the audio-venv python for everything). It starts the real
server on a scratch port and a temp out dir, generates synthetic takes,
and pushes them through the real HTTP POST path. The change maps, scales
and drift values in the POSTs are computed by the **actual shipped
JavaScript** — the MATH block is extracted from the `index.html` the
server serves and run under node — so the test exercises the same code the
browser will run, not a Python translation of it.

- **145 checks passed, 0 failed** (audio-venv run, numpy + gate included);
  91 passed under bare system python3 (numpy suite skipped there).
- Files and names: every take produces the script's four files with the
  script's compact-timing / indented-meta JSON styles; the npz contains
  exactly `maps, scale, maps_raw, scale_raw, luma`.
- **The shipped JS equals the script's math**: `maps_raw` is **byte-identical**
  to a numpy re-implementation of `corpus_capture_win.py`'s row-median →
  change-map → quantise pipeline; `drift_raw` is exactly equal; the row-removed
  `maps` differ on **0.000% of cells**; both scales agree with numpy's
  percentile to float64 precision (and round-trip through the npz's float32
  exactly, the script's own storage choice).
- `src/nv12.py` detects the stored luma stream as `luma_only`, 15,360-byte
  stride, the right frame count, margin ≫ 5; `src/wavio.py` reads every WAV
  at its stated 16 kHz with the full sample count.
- **THE GATE passes**: `experiments/corpus_integrity.py` exits 0 on a corpus
  produced entirely through the POST path — with the addendum's rich
  artifacts present (this is also asserted directly: the out root holds
  only `<name>_<stamp>.json` and `_timing.json` files).
- Server robustness: inconsistent takes → 400 with a reason; path-traversal
  condition names rejected; duplicate POSTs get distinct stamps (the gate
  identifies sessions by timestamp); `/notes` round-trips into the sidecar
  and `session.json`; oversized notes rejected.

## What still needs the real-camera test (Casey's 10-second take)

- The actual `getUserMedia` path: permission flow, a real camera's frame
  delivery under the audio-paced loop, and whether Chrome on his machine
  honors `AudioContext({sampleRate: 16000})` (if it doesn't, takes are
  saved with the honest rate and flagged `MIC_RATE`, same as the script
  flags a misbehaving mic — but the flag means the gate will fail that
  take until the mic runs at 16 kHz).
- `speech_frac` here comes from an **adaptive-RMS gate** (10× the 20th-
  percentile noise floor), not Silero. Thresholds mirror the script's
  0.15 / 0.05 checks, but equivalence on real speech is unproven — the
  provenance key `speech_gate` in every take's metadata says exactly which
  gate produced the number.
- The `_full.webm` is a real `MediaRecorder` output in the browser but was
  only tested as a header-only synthetic blob here; whether the file plays
  everywhere is a real-camera question.
- The canvas downscale uses the browser's resampling; the script uses
  `cv2.INTER_AREA`. Same target geometry (160×96), different kernel — if
  takes from both tools will ever be pooled pixel-wise, that difference
  should be measured first.
- End-to-end timing under a real 5-minute FREE take (memory and processing
  pauses) — the math runs in one blocking pass (~a few seconds at 9,000
  frames) after recording stops, with the UI showing "processing".

## Format decisions, and why

1. **The brief said "all 19 takes"; the plan has 16.** `CAPTURE-NEXT.md`
   and the script's `PLAN` define 3×(A,B,C,D,F) + FREE = 16 (~16 min). The
   brief names those same takes and miscounts them; the brief also says
   the script and CAPTURE-NEXT.md are the sources of truth, so 16 it is.
2. **Maps are computed in the browser, not the server.** stdlib Python
   cannot do 34M-cell median/percentile work in reasonable time; typed-
   array JS does it in seconds. The server stays stdlib-only and writes
   the npz with a hand-rolled NPY-in-deflate-zip writer that matches
   `np.savez_compressed` — proven by the `np.load` round-trip above.
3. **The capture loop copies the script's loop.** One video frame + one
   `floor(rate/30)`-sample audio block + one `performance.now()` reading
   per tick, paced by the audio clock — the same construction that makes
   the script's spans agree and its timestamps trustworthy.
4. **`capture_wallclock_s`** measures the browser's processing step
   (math + WAV + speech build), not the upload. The script's field covers
   its write step; this is the analogous cost in the browser flow.
5. **The rich JSON artifacts live in `out/rich/`, not at the out root.**
   The gate reads *every* `*.json` in the out root as take metadata, and
   the gate is not ours to modify — a root-level `session.json` or
   `*_rich.json` would fail every future gate run. The addendum's WebM
   does sit at the root next to its take (the gate only reads JSON). This
   deviates from the addendum's letter to preserve both its intent and
   the base brief's "the existing pipeline accepts the takes unchanged".
6. **Take stamps are globally unique**, even for takes recorded within the
   same second. The script gets this for free from 30-second takes; the
   browser UI allows fast re-records, and the gate counts sessions by
   timestamp, so a collision would silently merge sessions.
7. **Extra metadata keys** `capture_tool: "browser_capture"` and
   `speech_gate` ride along in each take's JSON. The gate ignores keys it
   doesn't read; an honest record of what produced the take is worth more
   than a bit-exact key set. Everything else is the script's schema.
8. **The take format stores luma only** (160×96), exactly as the script's
   npz does — the take format has no chroma plane. "NV12" in the brief is
   the layout family `src/nv12.py` knows; the browser's luma stream
   round-trips through it as `luma_only` at the 15,360-byte stride.
9. **Audio processing is off** (echo cancellation, noise suppression, auto
   gain) so the mic data is as raw as sounddevice's — a silent condition
   must not be cleaned into digital silence by the browser.
10. **Mirror is preview-only** — a CSS transform on the preview element.
    The data canvas is never mirrored.

## Not done, on purpose

- No pip installs, no CDN, no analytics, nothing leaves localhost — per
  the brief. `index.html` is one file; the only "module" it loads is an
  in-page blob for the AudioWorklet.
- `corpus_capture_win.py`, `CAPTURE-NEXT.md`, the gate and `src/` are
  untouched.

## Patch 2026-10-20 (live session): the operator's stimulus — the page's `--stimulus-file`

Casey does not know Invictus by heart, and the page gave him no way to use
a piece he does. The poem card shown for the recite conditions (A and B)
now holds a **stimulus name** field and an **editable stimulus text** area
(default: Invictus, verbatim from `src/stimulus.py`). Whatever is in them
when a take finishes is stamped into that take's metadata —
`stimulus`, `stimulus_title`, `stimulus_estimated_s`, exactly where
`corpus_capture_win.py` records them — and into the rich sidecar's
`stimulus` / `stimulus_title`; `session.json` follows from the metadata.
Nothing else about the take changed.

- The untouched default records the corpus identity (`invictus`, 40 s), so
  those takes stay pooled with the script's.
- Anything typed over it records under its own slug (lowercase,
  non-alphanumerics collapsed to `-`, so "Sea Fever" → `sea-fever`, the
  `ALTERNATIVES` key convention in `src/stimulus.py`), estimated the way
  `load_stimulus` estimates a supplied file: 0.4 s per word. `invictus` is
  reserved for the default poem — a different text under that name records
  as `custom` rather than silently pooling with real Invictus takes.
- The gate fails a corpus whose takes mix stimuli, by design. A different
  piece means a separate out dir, or a gate run that says so.

**Picking it up: a browser reload is enough — no server restart.** The
server reads `index.html` from disk on every GET (served `no-store`), and
this patch changes nothing server-side. Verify suite after the patch:
145 passed, 0 failed.

## Patch 2026-10-20 (same session): the dark camera, made diagnosable

The page's picture was REALLY dark while the same camera and room produce
mean luma 111–114 in the existing captures (empty room: 50) — the hardware
can do it, so something in the page's device choice or rendering was off.
Three additions, all client-side; the recorded data path is untouched:

1. **Camera picker.** The camera card now has a dropdown of the system's
   videoinputs (labels appear once permission is granted). The choice is
   remembered in `localStorage`; when several cameras exist, the page
   defaults to the first whose label does NOT read IR / Infrared / Virtual
   (the Windows face-auth sensor is the classic dark-picture impostor) and
   switches to it right after the first frames if the browser's own pick
   was one of those. Switching opens the new camera first, then swaps the
   video track on the live stream — the mic track and its worklet are
   never re-opened, so the audio path and its rate cannot change under a
   take. Pause the plan before switching (the picker is disabled mid-take
   anyway); a remembered camera that has been unplugged is forgotten
   automatically.
2. **Live mean-luma readout.** Next to the picker: the mean luma of the
   current 160×96 data frame, computed with the same BT.601 luma the takes
   record, updated twice a second, shown with the reference numbers —
   *old captures ~110, empty room 50* — and amber below ~80. During a take
   it reads the last captured frame, so it is the data, not a styled
   preview.
3. **Boost preview toggle.** A header button applying
   `filter: brightness(1.6)` to the preview element ONLY. Preview-only by
   construction: CSS filters style the element's rendered box and are
   invisible to `drawImage` (the data canvas) and to `MediaRecorder` (the
   WebM). Both stay raw; the small data view next to the preview keeps
   showing the honest darkness. Remembered in `localStorage`, like mirror.

Also picked up by a browser reload — no server restart, nothing
server-side. Verify suite after this patch too: 145 passed, 0 failed.

## Patch 2026-10-20 (same session, third): the exposure-starved stream

Casey at the page confirmed it IS the front camera and it responds to
light (a lamp brought into frame registers) — but the stream is
exposure-starved in the browser, while the old Windows-stack captures in
the same room averaged luma ~111. The camera is fine; OUR `getUserMedia`
constraints were the difference: the page asked for 640x480 at 30 fps,
driving the camera the way the terminal script did, and the browser
answered with a fast, dark exposure. Changes, all client-side:

1. **Near-default video.** No pinned `frameRate` ideal, no 640x480 ideal
   — the only constraint left is the `deviceId` when the picker chose a
   camera, so the camera's own auto-exposure runs the show, as it does in
   every normal app. A floating camera rate is safe by construction: the
   take loop paces on the audio clock (one frame drawn per audio block;
   slower camera frames are simply drawn twice) and `timing.json` keeps
   recording the achieved rate, exactly as before.
2. **`exposureMode: "continuous"`** is applied to the track where the
   track supports it (capability-checked via `getCapabilities()`, errors
   swallowed) — a nudge for stacks where it is not already the default, a
   silent no-op everywhere else.
3. **Visible track state.** Next to the luma readout: the active track's
   `getSettings()`, updated twice a second — the frameRate the camera
   actually delivers, and exposureTime in ms when the driver exposes it
   (`n/a` when it does not; the spec unit is 100 µs, converted here).
   State visible, not guessed.

**Server vs reload, confirmed:** the server reads `index.html` from disk
on **every request** and serves it with `Cache-Control: no-store`
(`server.py`, `do_GET` → `self.index_path.read_bytes()`). Casey reloads
the page; nobody restarts the server for any of these page patches. A
restart would only ever be needed for a `server.py` change, and none of
these patches touched it.

**Race note:** MiniMax is running a parallel empirical lane on this same
problem — a probe page measuring luma per constraint preset on the real
camera. This patch is the product fix shipped ahead of those numbers;
its expectation (near-default constraints → auto-exposure → luma back
toward ~110) is UNVERIFIED on real hardware until that lane reports.
Evidence wins: when the measured numbers land, the brighter verified
approach is what the page keeps.

Verify suite after this patch: 145 passed, 0 failed.

## Patch 2026-10-20 (same session, fourth): every camera parameter on screen

The fallback for "automatic does not fix the exposure": a CAMERA CONTROLS
panel under the camera picker (collapsed `<details>`, so it costs nothing
when unused), built from the **live track's `getCapabilities()`**:

- For every supported key among `exposureTime, exposureCompensation, gain,
  iso, brightness, contrast, saturation, frameRate, colorTemperature`: a
  slider ranged to the track's **real** min/max/step, current value shown
  (exposureTime in ms, frameRate in fps, colorTemperature in K; "auto"
  when the driver reports no current value), applied live via
  `applyConstraints({advanced: {...}})` on input — bursts coalesced into
  one driver round-trip per 150 ms. Unsupported keys are simply absent.
- An **exposureMode AUTO/MANUAL toggle** where the driver offers the
  choice — MANUAL is what makes the exposureTime slider actually take
  (while exposure is auto the driver owns that value, so it is not sent).
- Choices **persist** in `localStorage` (`bcCamCtl`) and re-apply on
  reload and camera switch; a driver rejection is shown in the panel and
  applies nothing.

These sliders drive the **REAL stream — the recorded data included**;
that is their purpose (a manual exposure that brightens the data is a
property of the takes, and the metadata records the world the camera
saw). Boost preview remains the separate cosmetic control it already
was. Still a browser reload, never a server restart. Verify suite after
this patch: 145 passed, 0 failed.

## Patch 2026-10-20 (same session, fifth): the take that looked hung

Casey clicked a take from the table, performed 50+ seconds, and from his
chair it was indistinguishable from hung: one ambiguous "running", no
stop control, and — confirmed server-side — no POST ever arrived. The
record loop itself was already bounded (done at `captured >= n`, noaudio
at 2.5 s, a target+15 s cap); the hang was AFTER the loop. Three fixes:

1. **Explicit take states.** A large phase banner in the condition card
   carries the whole arc: STARTING → **RECORDING 0:37 / 1:40** (blinking
   red dot; elapsed/target counts up with every captured frame) →
   PROCESSING THE TAKE → UPLOADING → **SAVED — TAKE A1** (green, three
   seconds). The banner is driven by the phase transitions around
   finalize/upload themselves, so it cannot lie about where the take is;
   the takes-table chip remains as the per-take view.
2. **STOP & SAVE.** A green button beside ABORT TAKE, enabled only while
   recording. It finalizes the take with the frames that exist and
   stamps `short_take: true` into the metadata (and the rich sidecar) —
   an honest take that did not run to the condition's full length, kept
   rather than discarded. Under the script's 10-frame minimum nothing
   can be saved and the state line says so; SKIP/ABORT remain the
   discard paths.
3. **The stall audit — every end is bounded on the wall clock.**
   - The genuinely unbounded wait — `await recorder.onstop` after
     `recorder.stop()`, which if the driver never fires it leaves a
     finished take hung forever with no POST (the observed failure) — is
     now raced against a 3-second timer; whatever was buffered is used.
   - A new no-progress watchdog ends the take when no frame has arrived
     for 5 s (audio clock or worklet dying mid-take): ≥10 frames are
     SAVED flagged `short_take` with the reason on the state line,
     fewer are discarded with the reason.
   - The original target+15 s cap stays as the backstop, with the same
     save-if-possible semantics.
   - The upload fetch aborts after 120 s and says so, instead of
     sitting in UPLOADING forever; the take stays in memory with the
     Retry path either way.

Also a browser reload, nothing server-side. Verify suite after this
patch: 145 passed, 0 failed.

## Patch 2026-10-20 (same session, sixth): THE PIVOT — FREE RECORD is the front door

Casey is right that the plan page is over-built for what he needs at the
chair; this supersedes the queued instrumentation patch. The heartbeat
survives only where it rides for free — and it does, on the new page.

**`/` is now FREE RECORD** (`free_record.html`): one RECORD button, one
STOP button, a live preview, an elapsed clock and an audio level meter —
nothing else. FOUR states — idle, recording, saving, saved — through a
single `setState`; anything more is wrong. While recording it captures
the native-res WebM (video + mic via MediaRecorder), the raw 160×96 luma
canvas stream with a per-frame timestamp (one frame per
requestAnimationFrame), and full-rate 16-bit mono WAV (the AudioContext
opens at the browser's default rate; no resampling, no processing). At
STOP it all POSTs once and the server writes `<out>/free_sessions/
<stamp>/` holding five files: `video.webm`, `luma.raw`,
`timestamps.json`, `audio.wav`, `meta.json` (camera track settings,
achieved video fps, the honest audio rate, the luma layout). No
conditions, no plan, no in-browser math pass — the browser records,
analysis happens offline. Offline loading is one line:
`np.fromfile("luma.raw", np.uint8).reshape(-1, 96, 160)`.

- **The heartbeat rides for free**: while recording or saving, the page
  POSTs `/log` once a second (state, frame/sample counts, recorder and
  AudioContext state, tab visibility), plus every click and phase change
  the instant they happen; the server appends JSONL to
  `browser_capture/diag.log`, stamped on receipt. Watch live with
  `tail -f browser_capture/diag.log`. A failed save keeps the session in
  the tab behind a RETRY SAVE button; the upload is bounded at 300 s.
- **Plan mode is intact at `/plan`** — the old page, unchanged behavior
  (it gained one link back to the front door). The plan page's own
  instrumentation was superseded and was not added.
- **Gate safety preserved**: `free_sessions/` is a subdirectory, so its
  JSONs are invisible to the gate's non-recursive out-root glob — the
  same reasoning as `rich/`, now asserted in verify.py.
- **`/free` validates honesty, not a schema**: luma length ==
  n_frames × 15,360, one timestamp per frame, the WAV parses as mono
  16-bit at exactly the rate meta claims, the WebM starts with the EBML
  magic; bodies cap at 2 GB (~35-minute sessions).

**RESTART THE SERVER for this one, then reload the page.** Routing
changed (`/` serves a different page) and `/free` + `/log` are new
endpoints — the first patch in this series that needs the restart.
After it, http://localhost:8765 lands on FREE RECORD.

Verify suite after this patch: 154 passed, 0 failed (nine new checks:
front-door routing, plan intact at /plan, the /free write path, the
gate-safe subfolder, the dishonest-body 400, and /log).

## Patch 2026-10-20 (same session, seventh): AUTO — the page arms itself

Even one button per sitting is 20th century. FREE RECORD is now AUTO by
default: **after permissions, the page ARMS ITSELF** — a segment opens
when speech or presence is detected, closes after ~8 s of silence and
stillness, and every closed segment uploads **in the background as it
finishes**. Chunked, no end-of-session crunch, no STOP needed for the
data to be safe. Open the page once, put it aside, forget it; the corpus
accrues. A reload of the page is enough (server unchanged this time).

- **The indicator is big and honest at all times**: ARMED (listening) /
  RECORDING (blinking red, segment elapsed) / PAUSED — visible from
  across the room, plus a live line of speech/presence levels against
  their thresholds.
- **Exactly two controls survive**: PAUSE EVERYTHING (the kill switch —
  closes and saves whatever is open, then holds until RESUME) and
  RECORD/STOP, the manual fallback (a manual segment has no auto-close).
- **Detection, honestly named**: speech is the corpus's own rule — RMS
  at 10× the 20th-percentile noise floor of a ~5.5 s rolling ring, the
  floor FROZEN while a segment is open so continuous speech cannot
  recalibrate itself into silence. "Presence" is an adaptive MOTION
  proxy over the 160×96 luma frames (mean |Δluma| at 5× its
  20th-percentile floor, frozen likewise) — not a face detector; it
  answers "is a person present and doing something". Every level,
  floor, threshold and decision is in the 1 s heartbeat, so the
  thresholds can be tuned from `diag.log` on the real machine before
  anyone changes a line of code.
- **~1 s of pre-roll** (audio + luma + timestamps) seeds each segment,
  so the syllable that triggered it is inside the artifact. The WebM
  starts at segment open (MediaRecorder cannot reach back); meta.json
  says so. The pre-roll rings are cleared on close, so a new segment
  never seeds with stale data.
- **Artifacts unchanged**: each segment is one `POST /free` → one
  `<out>/free_sessions/<stamp>/` folder with the same five files;
  meta.json gains `mode` (auto/manual), `trigger`, `close_reason`, and
  the full detection context (floors and thresholds at open — the
  offline record of WHY the segment exists). The server needed no
  changes.
- **Background-tab honesty**: speech detection runs on the audio
  thread and keeps working with the tab aside, but browsers stop
  rAF-driven video frames in background tabs — a 500 ms interval keeps
  capturing degraded luma and keeps the auto-close clock ticking. The
  page says so, meta.json says so, and the heartbeat's visibility field
  shows when it is happening.
- Failed background saves are held in the tab and retried every 30 s
  (upload bounded at 300 s); the heartbeat carries the retry count.

Verify suite after this patch: 154 passed, 0 failed (the free-path
checks cover the segment artifact unchanged; the AUTO logic itself is
real-camera territory — tune it from diag.log).

## Self-verify of AUTO, end-to-end (2026-10-21, step 1 of the polish queue)

Not simulated — the real page in a real (headless) Chrome against the
real server, with **file-backed fake media** chosen so the decisions
must be made honestly: a static-gray Y4M camera (motion stays honestly
absent) and a fake-mic WAV that is 4 s of room tone, 10 s of
speech-level noise, then silence (speech must genuinely trigger, and
quiet must genuinely close). Windows Chrome headless with
`--use-fake-device-for-media-stream --use-file-for-fake-audio-capture=
...wav --use-file-for-fake-video-capture=...y4m
--use-fake-ui-for-media-stream --autoplay-policy=no-user-gesture-required`
(unique `--user-data-dir`, PID-scoped stop), server on a scratch port
and scratch out dir, 95 s run. Chrome loops the fake audio, so the run
covers TWO full auto cycles.

**The first run FOUND A REAL BUG**: the detection thresholds bootstrap
at 1e-4 and no noise floor exists before the first 500 ms recompute, so
room tone above −60 dB RMS opened a segment 100 ms after boot and — with
the floor then frozen — never closed it. On a real machine AUTO would
have recorded one eternal segment from page load. Fixed: a
`FLOOR_WARMUP_MS = 2000` gate — no auto open/close decision until the
floors are measured from real data (`FLOORS_READY` event in diag.log);
until then speech/motion read honestly as unknown, and the manual
fallback still works.

**The second run passed, both cycles** (all numbers from the artifacts
and diag.log, none from the DOM):

- `BOOT armed` → `FLOORS_READY` at +2.0 s → `SEGMENT_OPEN
  trigger=speech` at +3.6 s with ~1.2 s pre-roll → `SEGMENT_CLOSE
  reason=quiet` at +21.6 s (the file's silence began at 14 s + the 8 s
  grace) → `SEGMENT_SAVED` in the background, `tries: 0` → armed again
  → a second identical cycle at +70 s (the re-arm, for free).
- Heartbeats carried the AUTO states throughout, including the frozen
  threshold while open (speech level 0.105 vs thr 0.030; room tone
  0.0033 below it) and `saved` ticking 0 → 1 → 2.
- Both session folders held all five files, intact: `luma.raw` ==
  n_frames × 15,360 exactly (1,148 and 1,146 frames), one timestamp per
  frame spanning ~19.2 s + 1.2 s pre-roll, an honest mono 16-bit 48 kHz
  WAV with the full 925,696 samples, a 300 KB `video.webm`, and meta
  with `mode: auto`, `trigger: speech`, `close_reason: quiet`,
  achieved fps ≈ 59.8 (headless rAF runs at 60; the 30 fps fake camera
  means duplicate frames — identifiable offline from the timestamps,
  which is what they are for).

What this proves and what it does not: the decision loop, the pre-roll,
the quiet close, the background upload, the re-arm, the artifacts and
the heartbeat are all verified against real browser behavior. It does
not verify real-microphone speech/noise statistics or real-camera
motion — those are what the heartbeat numbers are for on Casey's
machine.

## Step 2 of the polish queue: the single-consumer guard, in plain language

Per MiniMax's camera findings: when `getUserMedia` fails
`NotReadableError` (the classic single-consumer collision — the camera
is held by another app or tab), the page must SAY so, never fail dark
and silent.

- **FREE RECORD** now names the failure on the big indicator itself —
  `CAMERA BUSY` (amber), with the plain-language body: the camera is
  being used by ANOTHER app or tab right now, close it (another tab
  left open on this page, Teams/Zoom/OBS, the terminal capture script),
  then press **RETRY DEVICES** — a button that appears ONLY on failure
  and re-runs the whole device boot (old streams/contexts torn down,
  floors re-measured, no page reload needed). `NO CAMERA` and
  `PERMISSION REFUSED` get the same treatment, and every failure is a
  `DEVICE_FAILURE` line in diag.log with the error name.
- **The plan page** gets the same distinction in its error banner (it
  previously said "permission was refused" for every failure, and told
  you to click a RETRY DEVICES button that did not exist — starting a
  take is the actual retry).

Honest limit, stated: fake media devices cannot contend, so the
`NotReadableError` branch itself could not be reproduced in the
headless self-verify — the branch is code-reviewed, the failure UI is
exercised live, and the error name is logged for the first real
occurrence.

## Step 3 of the polish queue: receipts

Every segment that lands gets a receipt on the page: a RECEIPTS panel
(one line per saved segment, newest first, this sitting) reading
`✓ 03:41:22  free_sessions/<stamp>  —  19.3 s, 1,148 frames, 19.8 MB
(luma 17.6 MB / wav 1.9 MB / webm 0.3 MB)` — the saved duration, frame
count, and the bytes that actually landed, per artifact. The same
numbers ride in the `SEGMENT_SAVED` heartbeat line as a `receipt`
object, so the page's claims are checkable against diag.log AND against
the disk.

**Proof, by re-running the headless end-to-end after the change**: two
full auto cycles again; for both segments the receipt's `luma_bytes`,
`wav_bytes`, `webm_bytes` and `total_bytes` equal the on-disk file
sizes exactly, and `duration_s` equals the WAV-derived duration to the
millisecond (19.267 s vs 19.267 s; 19.256 s vs 19.256 s). The verify
suite stays at 154 passed, 0 failed. Still just a page reload — the
server has not changed since the FREE RECORD pivot.

## Patch 2026-10-21: the server outlives the terminal (systemd user service)

The live incident: the bench terminal hosting `python3
browser_capture/server.py` got Ctrl-C'd by accident and localhost:8765
died silently mid-session. Fix — the server is now **`browser-capture.service`,
a systemd USER service** (linger is on, so it starts with WSL):

- `Restart=on-failure`, `RestartSec=2`, logs appended to
  `browser_capture/server.log` (`*.log` is gitignored).
- The unit's source of truth lives in the repo
  (`browser_capture/browser-capture.service`); install = copy to
  `~/.config/systemd/user/`, `daemon-reload`, `enable --now`.
- **The bench start command became a view**: `python3
  browser_capture/bench.py` shows the service status, the live `/info`
  answer, and follows the log — Ctrl-C there exits the VIEW, nothing else.
- **The one status command: `systemctl --user status browser-capture`.**
- Cutover from Muse's plain server was a single swap (stop PID, start
  service): exactly one listener on 8765 at every moment, verified.
- Crash-tested live: `kill -9` on the service's python → back up under a
  new PID in ~2 s, one listener, `/info` answering. The incident is
  structurally impossible now — there is no controlling terminal to
  Ctrl-C.

Deploying future server.py changes means `systemctl --user restart
browser-capture` (a ~2 s gap; the pages' saves retry by design).

## Patch 2026-10-21 (AUTO polish 1): segments rotate at max length, mid-activity

Muse's live telemetry during Casey's first auto session: a segment only
closed on silence+stillness, so a long active stretch pooled EVERYTHING
in the tab — exactly the risk AUTO exists to kill. Auto segments now
**rotate at `MAX_SEG_MS` (default 5 minutes) even mid-activity**: close,
upload in the background, and — the activity still going — reopen on the
next tick (~1 s seam while the pre-roll refills; the timestamps are
honest about it). `close_reason: "maxlen"` lands in meta.json, and
`?maxseg=<seconds>` tunes the length (and made the path provable).
Manual segments do NOT rotate — the operator is present, and STOP is
theirs.

**Proven by the headless end-to-end at `?maxseg=8`**: two `maxlen`
closes mid-speech at 8.0/8.1 s, an 18 ms reopen seam while speech
continued, and a correct no-reopen when the second rotation landed after
speech had ended (armed, waiting — silence does not reopen). Both
session folders intact, `close_reason: maxlen`, `max_segment_s: 8`
recorded in meta, 480 frames/segment at the headless 60 fps rAF.

**The same run caught and fixed a real server bug**: any request with a
query string (`/?maxseg=8`) 404'd — the handlers matched `self.path`
exactly, query included, so the routing fix strips `?...` before
matching (GET and POST). Found because the rotation test page refused
to load at all; the fix rides in this commit and deploys with the
service restart that ships the /live monitor (next section) — the
running pages (already loaded, POSTing bare paths) are unaffected
either way.

Test-harness lesson, recorded: the earlier headless runs' Stop-Process
killed only Chrome's launcher process, leaving browser processes alive
— ghosts that kept posting heartbeats and confused this run's forensics
until they were killed by command-line match
(`Get-CimInstance Win32_Process` filtered on the unique
`--user-data-dir`). All headless verification now kills that way, and
each run gets a fresh profile directory.

## Patch 2026-10-21 (AUTO polish 2): the LIVE MONITOR — the stream's window

Casey asked for a live stream; the stream exists (the 1 s heartbeat) —
**http://localhost:8765/live** is its window. Read-only: the monitor
never touches the capture page or a session; it polls `/live-state`
once a second and shows:

- **Page state from the heartbeat**: ARMED / RECORDING / PAUSED (with a
  STALE marker when the heartbeat is >5 s old and NO PAGE when >15 s),
  segment age, frames, audio samples, saved count, uploads retrying,
  floors status, tab visibility, recorder state.
- **The detectors, live**: speech level and motion as bars against
  their thresholds — the white tick IS the trigger; a bar crossing it
  is a segment opening before your eyes.
- **What the pipeline sees**: the latest 160×96 luma thumbnail, POSTed
  once a second by the capture page (`POST /thumb`, in-memory only,
  latest wins — a window, not a record), dimmed when stale.
- **The event feed**: SEGMENT_OPEN / CLOSE / SAVED, rotations, device
  failures, from the diag tail.
- **Saved segments as they land**, with close reason, duration, frames
  and MB — and **harvest verdicts**: the contract is that a harvester
  (MiniMax's lane) writes `free_sessions/<stamp>/verdict.json` — any
  JSON object — and the monitor surfaces it the next second. Nothing
  else is agreed; the harvester owns the file, the server only reads.

The capture page gained a "live monitor" link; its thumbnail rides in
the same 1 s cadence as the heartbeat. Verify suite: **159 passed,
0 failed** (five new checks: /live serves the page, /live-state returns
events + segments, /thumb round-trips into /live-state, and the saved
session appears in the segment list).

**Deployed by `systemctl --user restart browser-capture`** — the same
restart ships the query-string routing fix from the previous section.
The capture pages already open in a browser are unaffected (they
POST bare paths); the restart's ~2 s gap is absorbed by the pages'
30 s save-retry by design.
