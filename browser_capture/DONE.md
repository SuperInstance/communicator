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

## Patch 2026-10-21 (AUTO polish 3): a deaf page is obvious at a glance

Live telemetry caught a sitting where the speech level sat at exactly 0
for minutes while motion kept firing and segments kept saving — the
page was holding a mic that delivered digital silence, and nothing on
the AUTO face said so. Now:

- **The active devices are named, always**: a device line under the
  meters — `mic: <label> @ <rate> Hz · cam: <label>` — is part of the
  face, not a debug event.
- **The meter carries a number**: the det line shows the live mic dB.
- **The deafness watchdog**: samples arriving but flat-silent for >20 s
  turns the device line RED — "AUDIO SILENT for Ns — mic '<label>' is
  delivering silence; a healthy room is never this quiet; check the
  Windows input device / mic mute, or the browser took the wrong mic".
  No samples at all is distinguished: "AUDIO DEAD — the audio tap
  failed; reload". The heartbeat now carries `mic`, `cam`,
  `audio_blocks`, and `audio_silent_s` so telemetry sees deafness the
  same second the page does.

Page-only — a reload picks it up; the server is unchanged.

## SICK LANE 2026-10-21: the silent WAVs — evidence, cause, fix, and measured proof

**The report**: every free-session WAV saved tonight is digital silence
— Muse measured 0 nonzero samples out of 12.4 million.

**The evidence, all measured from the artifacts and diag.log:**
- All 34 real sessions (two sittings, mic = "Microphone Array
  (Realtek(R) Audio)"): **82,383,616 WAV samples, every one zero.**
- The WebM audio tracks (opus, recorded by MediaRecorder straight off
  the track, bypassing our worklet entirely) are **also exactly zero** —
  so the silence was upstream of every consumer the page has.
- The worklet agreed: `speech_floor_rms ≈ 7.16e-10` in every meta, and
  **every trigger was `motion`** — speech never fired once in either
  real sitting (max speech_level = exactly 0 across 1,268 real-sitting
  heartbeats). The "meter fires on speech" reading came from the
  fake-device test runs sharing diag.log, not from Casey's mic.
- **Conclusion: meter and samples never diverged in code — both
  faithfully recorded an input device that delivered literal zeros.**
  The code path stands cleared by the same evidence: the identical
  pipeline fed by the fake mic produces healthy files (below). The
  failing component was the device Chrome held — muted, dead, or the
  wrong input — invisible to the operator all night while MOTION
  accrued sessions.

**The fix (page-side; a reload picks it up):**
1. **Mic and camera pickers on the AUTO face** — the held devices are
   NAMED and SWITCHABLE in one click. A mic switch re-taps the worklet
   and re-measures the floors (a different device is a different
   instrument); choices persist; a remembered-but-vanished device falls
   back to defaults automatically.
2. **The 60-second deafness GATE**: with samples flowing but flat
   silent for 60 s, the indicator goes **DEAF — MIC SILENT**, any open
   auto segment is closed (`close_reason: "deaf"`), and **no new
   segments open — motion included — until the mic hears something**.
   Tonight's 34 worthless folders become one visible stop. Manual
   RECORD stays available (explicit intent outranks the gate).
3. **`audio_silent: true` + `peak_rms` stamped into every session's
   meta.json** — a silent-audio segment is distinguishable forever, on
   disk, for the analysis and the live monitor.

**Measured proof, per the standard "the files measure speech, not the
code reads right" — headless Chrome on this machine's real Chrome build,
fake mic speaking speech-shaped bursts (room tone + three phrases),
scratch server and out dir:**
- 3 speech-triggered segments saved (`trigger: speech`,
  `close_reason: quiet`):
  - **WAV**: 590,592 samples each, **589,411–589,443 nonzero (99.8%)**,
    **RMS 0.0443–0.0457**, peak 0.39–0.40.
  - **WebM audio, decoded with pyav**: 1,054,080 samples,
    **100% nonzero**, **RMS 0.0468–0.0482**, peak 0.42–0.44.
- Deaf-gate run (fake mic = exact zeros, fake camera = constant motion):
  one motion segment during the 60 s grace, closed at 60 s with
  `close_reason: "deaf"`, `audio_silent: true`, `peak_rms: 0` (its WAV:
  0 nonzero of 2,876,416 — honestly stamped), then **zero further
  SEGMENT_OPENs despite constant motion**, `DEAF_ON` in diag.log, deaf
  flags in the heartbeats.

The one step no automation can do: Casey picks a live mic at the page
(the picker names them; the deaf line goes quiet the moment the room is
heard) and speaks once — the receipts and the live monitor then show
measured speech, or the page says exactly why not.

## Patch 2026-10-21: the ROOM LIAISON v1 (docs/research/BRIEF-ROOM-LIAISON.md)

The local host graduates from a web page into the AGENT IN THE ROOM:
a resident JSON API for agents, an outbox shipped toward Muse, and
self-reported health. Deployed with `systemctl --user restart
browser-capture` (server-side change). The pages are untouched — the
API is additive.

**The agent API (localhost, JSON):**
- `GET /api/status` — uptime, the page's live state from the last
  heartbeat (state, deaf, segment age, speech level/threshold, devices,
  heartbeat age), sessions saved, and the rendezvous ship health.
- `GET /api/sessions` — every free session with its meta PLUS **measured
  audio health: peak abs sample, nonzero count, RMS — computed from the
  WAV files with `array`/`math.sumprod`, memoized on (size, mtime),
  never asserted.** The meta's page-asserted `audio_silent` rides along
  for contrast with the measured truth.
- `POST /api/inbox {from, text}` — appends `liaison_inbox.jsonl`;
  `GET /api/inbox` reads it back. Local agents message the room through
  this; every message also lands in the outbox.

**The outbox and self-checks** (`liaison_outbox.jsonl`, gitignored):
- Event classes: `session_saved` (with the measured audio stats),
  `page_segment_closed`, `page_deaf_on/off`, `page_device_failure`,
  `page_boot`, `inbox_message`, `liaison_started`.
- **SILENT_AUDIO self-alert**: the moment a saved session's WAV measures
  peak 0, the liaison files an `alert` about itself — tonight's bug
  class is now a self-report, measured at save time. Proven in verify by
  FORCING a silent WAV through the real POST path and finding the alert
  in the outbox.
- Each event is appended durably, then shipped (background thread) to
  `wsl-tunnel@147.224.38.131:~/rendezvous/room-liaison/events.jsonl`
  (the Oracle the reverse tunnel terminates on); failures retry every
  5 minutes and are reported in `/api/status.rendezvous`, never silent.

**Rendezvous reality, stated plainly (v1)**: the `wsl-tunnel` account on
the Oracle is nologin for shells AND sftp, so every ship attempt refuses
until a writable path is granted — the liaison says so itself rather
than pretending. The read path that works TODAY: Muse's rounds reach
this host through the existing reverse tunnel (Oracle loopback `:2222`
→ this host's sshd) and can `curl http://127.0.0.1:8765/api/status`,
`/api/sessions`, `/api/inbox` from there.

**Muse-side contract (documented, not built, per the brief)**: Muse is
this liaison's MAINTENANCE AGENT (Casey owns it). On its rounds (the
tunnel path above, or the rendezvous once writable), Muse reads status
+ alerts, and replies via `POST /api/inbox {from: "muse", text: ...}`;
the operator sees inbox messages at the bench. Latency is minutes-scale,
maintenance-grade — this is a resident agent, not a chat.

Verify suite: **166 passed, 0 failed** (seven new liaison checks: the
status shape, the measured-audio contract — 800 samples of value 1
measure as peak 1 / nonzero 800 — the inbox round-trip, the forced
silent WAV raising its own SILENT_AUDIO alert, and the ship-refusal
honesty).

## Design-law compliance note (docs/research/BRIEF-TIERS-AND-DECOMPOSITION.md)

The tiers/decomposition law applies to all liaison work; v1 complies by
construction, and the compliance is load-bearing, not decorative:

- **Local-first, fleet as upgrade**: the outbox is durable locally
  BEFORE any ship; the Oracle ship is a best-effort upgrade whose
  failure changes nothing functional and is self-reported. On the boat
  tier (offline Thor) the liaison runs whole: API, sessions, measured
  health, self-alerts — only the rendezvous ship waits for a link.
- **Visible seams, no monolith**: every stage is a small inspectable
  piece — energy VAD, luma frames, the timestamps spine, per-file
  artifacts — connected by files, replaceable one at a time. No stage
  hears sentiment end-to-end.
- **Capture is destiny-safe TODAY**: the session bundles save the full
  raw fidelity (WAV, luma, timestamps spine), and the score is DERIVED
  from raw — transcript from the WAV, melody from pitch tracking,
  face/motion from the luma — so a conversation captured tonight can
  become a score later even before the scorer (MiniMax, build-order
  item 3) lands. The conversation page (item 4) therefore waits on the
  scorer per the brief's own order, without losing anything.

## North-star anchor (docs/research/BRIEF-SYNERGY-LEARNING-LOOP.md)

The liaison is steps 1 and 2 of the synergy loop — CAPTURE (automatic,
everything on one clock: the session bundles already carry audio + luma
+ the timestamps spine; the score slot is what build-order item 3 fills)
and LOCAL SCORING (the measured-not-asserted health is the first local
computation on the box; the decomposition stages plug into the same
bundles as files) — plus the FAILURE-DETECTOR HOME: "every failure
becomes (a) a detector in the liaison, (b) a test in the suite, (c) a
line in the digest." Audit of this build's failure classes against that
law:

- **SILENT_AUDIO** (the brief's named prototype): detector at save time
  ✓, suite test (forced silent WAV through the real POST path) ✓, outbox
  line ✓. The full triple, live — it fired in production within minutes
  of deployment.
- **Query-string 404**: fix ✓, suite test added this change ✓.
- **Deaf input / AUDIO DEAD / DEVICE_FAILURE / cold-start floors**:
  detectors ✓ (page gate + heartbeat fields), outbox lines ✓
  (`page_deaf_on/off`, `page_device_failure`), and E2E-proven — but the
  browser-dependent tests are not in the stdlib suite; recorded as the
  known gap, honestly, until the E2E is suite-enclosed.

The teacher-dependence ratio is the loop's headline metric; nothing in
this lane computes it yet — it belongs to the digest tier, and the
liaison's job is only to make its inputs honest and measured.

## Patch 2026-10-21: the LIVE FORWARDER for the Muse+Jetson pair

While AUTO runs, the Jetson seat's analyzer gets its feed: every ~2 s
the capture page POSTs the newest ~90 luma frames (with per-frame
offsets) to `POST /luma` (in-memory, latest-wins — same pattern as the
thumbnail), and a SEPARATE sidecar — `browser_capture/forwarder.py`,
installed as **`browser-capture-forwarder.service`** — pushes every ~3 s
to `jetson-lan:~/work/pair/live/`:
`latest_luma.raw`, `latest_luma.json` (n, offsets_ms, frame shape,
format), `heartbeat.json` (the newest page heartbeat). Writes land via
temp+mv so the analyzer never reads a torn file. It ships only while a
page heartbeat is fresher than 10 s; a dead Jetson or a failed cycle is
logged and retried — **additive only: the sidecar is a separate process
with zero coupling to the capture path, and the page's POST is
fire-and-forget.**

Deployed and proven live: a 3-frame batch POSTed to the running server
was on the Jetson within one cycle (46,080 bytes = 3 × 15,360, offsets
[0, 33, 67], format documented), alongside the real live heartbeat.
The Jetson seat's replies ride the existing `verdict.json` contract.

Verify suite: **169 passed, 0 failed** (two new checks: the /luma POST
and the /luma-latest round-trip — same offsets, same bytes).

## SICK LANE CLOSED ON EVIDENCE (Muse verification, 2026-10-21)

Muse verified from the artifacts, independently of this seat: the
newest saved session (1792560597) **measures speech — 422,585 nonzero
samples of 1,312,128, peak 3006, RMS 0.00394** — and the live heartbeat
reads `deaf: false` with the silence counter at 0. This seat re-measured
the same file and got the same numbers to the sample. The session was a
MANUAL take (trigger manual, close manual_stop) on the same Realtek
array that delivered zeros the night before — the device hears, the
pipeline carries it, the file holds it. Sick lane stood down.

**The mechanism, for the next seat, in one breath**: nothing diverged
inside the code — the meter and the WAV (and the WebM's audio track)
all consumed the SAME stream, and on the bad night that stream was the
input device delivering literal zeros while MOTION kept opening
segments; 82.4M samples of silence accrued across 34 folders with no
signal anywhere in the UI that the mic was dead. What changed:

1. the held mic and camera are NAMED on the face and switchable in one
   click (a mic switch re-taps the worklet and re-measures floors);
2. a 60-second deafness GATE stops motion-only accrual and says so on
   the indicator (DEAF — MIC SILENT), resuming the moment audio lives;
3. every session's meta carries measured-not-asserted `peak_rms` /
   `audio_silent`, `/api/sessions` recomputes them from the WAV itself,
   and a save-time peak of 0 files a SILENT_AUDIO alert in the liaison
   outbox — the failure class became a self-report.

**Where the truth lives now, for whoever inherits this**: audio health
is read from `/api/sessions` (measured) or the outbox (alerted), never
from the page's own claims; the heartbeat's `deaf` / `audio_silent_s` /
`speech_level` fields are the live instruments; and the reproducible
proof method is the headless file-backed fake-mic run documented above
(speech-shaped WAV in → measured nonzero WAV and WebM out; exact-zero
WAV in → gate, no accrual, honest stamp).

## THE EVOLVE LAB IS LOADABLE — /evolve (Casey: open it once)

Per `BRIEF-EVOLVE-PAGE.md`: **open http://localhost:8765/evolve once and
leave it** — that tab is the lab. It is the full AUTO capture page
(devices, segments, rotation, deaf gate, pickers) plus the probe engine:
every 20 s it re-fetches `browser_capture/probes/manifest.json` from the
server's disk, and any probe whose module version changed is re-imported
and RUNS THE NEW CODE WITHOUT A RELOAD — proven live tonight: the open
page ran sonar-band v1, the module was rewritten on disk, and the same
open tab produced v2's corrected verdict minutes later.

- **The rails live in the shell, not the probes**: audible-flagged probes
  are refused outright 22:00–07:00 local; EVERY stimulus is gain-clamped
  (≤0.3) and duty-capped (≤5 s per 60 s per probe) regardless of what a
  probe requests; PAUSE EVERYTHING is the one kill switch for capture
  AND probes; every run, refusal, and failure is posted to /log and
  shown in the lab rows — no silent experiments.
- **Camera-busy coexistence** (learned on the first real run: Casey's
  own tab holds the camera, as it should): the shell degrades to
  AUDIO-ONLY — probes, speech detection and the deaf gate live; motion
  and video frames honestly stop; the face says why. The lab and the
  capture page coexist overnight.
- **Agents iterate** by editing `browser_capture/probes/<name>.js`,
  bumping its `meta.version`, and (optionally) the manifest's
  schedule — the open page picks it up next cycle. Results land as
  `probe_result` lines in diag.log; artifacts as `mode: "probe"` session
  folders. Tenant 1: **sonar-band** (room sonar step 1 — verdict filed
  in `docs/research/ROOM-SONAR-REPORT.md`: **usable band NONE on this
  hardware**, return 12–29 dB under the room floor in every excited
  band, lane closed with a number unless Casey reopens it with an
  external speaker or a raised ceiling). Tenant 2: **audio-health**
  (patrols the measured WAV health of recent sessions; tonight it
  correctly flagged the pre-fix silent era and passed the new ones).

Verify suite: **173 passed, 0 failed** (evolve shell served, manifest +
modules served from disk, path traversal refused).

## Patch 2026-10-21: the OVER probe (BRIEF-WAKE-AND-TURN mechanism 3)

The trained transmit word, as evolve-lab probe `over-detector` (v4):
OVER fires ONLY as the conjunction -- the word in terminal position
(template match, multi-duration fixed-grid mel matcher against
OPERATOR-CALIBRATED templates in localStorage), terminal falling
prosody (48k-geometry YIN on the run's tail, slope-or-below-mean rule),
and >= 1.5 s trailing silence (energy envelope; 1.2-1.5 s near-misses
reported separately). Page-tier only -- 20-year-old tools, decomposed,
no model. Calibration is self-instructing (say OVER alone five times;
first run without templates does it); every ambient run reports the
brief's headline numbers: fires + component counts as false-fires per
hour, with the full per-event table saved in the probe artifact.

**Proven by a three-phase browser E2E** (headless Chrome, synthetic
fake-mic speech through the REAL worklet/localStorage/probe chain):
calibration (5 templates), clean ambient (7 runs, 0 word-matched,
**0 FIREs**), over-like ambient (5 runs, **1 word-matched, 1 FIRE** --
falling prosody + 13.6 s trailing silence; every non-over burst
measured dist >= 0.42 vs the over burst's 0.104, which set the honest
0.25 threshold floor). Missed-yield measurement needs meant-OVER ground
truth -- the drill-deck class the brief plans; the pieces are exposed.

Bugs found and fixed on the way (all by the E2E, kept as the record):
trailing-slash URLs 404'd (`/evolve/` -- routing now normalizes, server
deployed); "over" is TWO syllables and the energy detector split every
spoken OVER in half (runs now merge across <80 ms intra-word gaps); the
YIN port's 512-frame geometry was 16k-native and returned empty tracks
at 48k (now 1024); a shadowed variable in the distance function crashed
every calibration (the `b` loop index vs the `b` parameter); identical
synthetic calibrations produced a degenerate 0.02 threshold (floor 0.25
from the measured separation). audio-health v2 now skips probe
artifacts (sonar returns are often honestly silent).

## Patch 2026-10-21: the retention ladder, plumbed and DRY (BRIEF-RETENTION-GC.md + addendum)

The GC is a curation system; deletion is the side effect; collection is
the one sanctioned deletion in a never-delete house -- so it runs DRY
FIRST and **gc.py contains no deletion code, by construction**. Live
collection stays blocked until the quintessential miss rate is measured
near zero and Casey rules on it (the report says so in every run).

- **Per-session tier state**: `retention.json` inside each session
  folder (gate-invisible) carries pins and graduated annotations; tier
  verdicts are COMPUTED from files, never asserted -- age from the
  session's wall clock, derivation completeness by file presence
  (`score.mid`/`score.json`, `transcript.json` -- the analysis tier's
  contract), quintessence by `quintessential.json`.
- **The ladder (v1 constants, stated because they are per-person)**:
  audio HOT < 48 h; audio dry-collectible only when derivation is
  complete, unpinned, non-quintessential, and older than 72 h; MIDI
  collectible after 14 days AND ONLY when its tone essentials have
  graduated into transcript annotations (the addendum's rule -- the
  sarcasm and the question-rise become marks on the words before the
  score is let go); transcript kept forever.
- **Error-driven PINS**: `POST /api/pin {stamp, reason, source}` makes
  a session exempt at EVERY layer; `POST /api/unpin` is refused without
  the lesson -- the debt clears when the lesson exists, not when time
  passes. Both ride the outbox to the digest.
- **The nightly dry run**: `browser-capture-gc.timer` (04:30,
  persistent) runs gc.py; the report lands at
  `gc-report-latest.json`, is served at `GET /api/gc-report`, and a
  `GC_DRY_RUN` event rides the outbox with the headline counts. First
  real-corpus run: **97 sessions, would collect 0 audio / 0 MIDI** --
  every session honestly blocked by derivation-incomplete (the scorer
  does not exist yet; the ladder is telling the truth about the
  pipeline). The audit section reserves the miss-rate field and states
  it is unmeasurable until MiniMax's selection study exists.
- Verify suite: **179 passed, 0 failed** (six new: an old
  derivation-complete session IS dry-collectible; a pinned session is
  exempt at every layer; unpin without a lesson refused; with a lesson
  clears; the report serves).

## Scope notes: the four conversation-surface addenda (read, folded into the lane's queue)

- **ADDENDUM-MODES** (modal agnosticism + mode-aware digestion): the
  digestion-policy hook in the conversation surface takes `mode` from
  the shading layer (ramble -> key points; deep thought -> point by
  point); gestures/posture/phone as first-class channels arrive as
  more session-clock tracks -- the bundle already carries arbitrary
  per-clock files, so the hook is a policy table keyed on mode, not
  new plumbing. Blocked on the scorer like the rest of that surface.
- **ADDENDUM-AWARENESS-CALCULATORS** (double-hearing delay as
  geometry, output routing by located context, calculators not
  answers): SCREEN AS A MODE (co-reading) is mine -- the liaison
  surfaces own a screen-mode hook, scoped as: the live monitor's
  family gains a co-reading view whose content is chosen by the same
  mode/digestion policy; build when the conversation surface builds.
- **ADDENDUM-ASCII-PHONE** (phone video as an ASCII score stream,
  motion-indexed adaptive fidelity -- still = high-res deltas-only,
  moving = low-res fast; tilt/GPS/video pose for walked maps): the
  evolve page hosts the first ASCII phone probe; the fidelity rule is
  a per-frame function of the measured motion index (the corpus's own
  change-maps define it), so the probe composes PROVEN pieces (luma,
  motion, braille conversion from the seed proofs). Scoped next after
  the scorer lands; needs a phone-side capture page (BRIEF-PHONE-NODE
  lineage).
- **ADDENDUM-PERCEPTION-ADAPTATION** (normals are verified by
  self-checks forever): the liaison owns normals, and most of the
  verification skeleton already runs -- the deafness watchdog (channel
  death), audio-health (archive truth), sonar (a deliberate railed
  stimulus + measured return: touching the shirt on purpose). The
  named gap: a scheduled `normals-verify` probe that plays a known
  quiet speech-band stimulus and measures the return against the baked
  per-context normal, reporting DRIFT, so adaptation cannot bake in a
  limp unnoticed. Queued as the next self-check build.

## Patch 2026-10-21: the normals registry -- LENS FILES (ADDENDUM-PERCEPTION-ADAPTATION closing section)

Perception reads the world through each sensor's living self-model,
never raw -- so the self-models are first-class: **one lens file per
sensor under `browser_capture/lenses/`** (`mic.json`, `camera.json`,
`speakers.json`, seeded from what this lane has MEASURED, not
believed): the speech rule and its frozen floor, the healthy-room luma
normal, the speaker rails, and the FAILURE SIGNATURES with their
detectors -- SILENT_AUDIO (floor ~7e-10, zero-peak WAVs everywhere),
EXPOSURE_STARVED (dark era: pinned constraints vs the ~110 healthy
normal), SINGLE_CONSUMER_BUSY (NotReadableError), and the speakers'
measured ULTRASONIC_DEADBAND (return 12-29 dB under the room floor,
ROOM-SONAR-REPORT). Every lens carries a `verified` stamp with
provenance -- unverified normalisation is the silent-audio failure in
perception's clothes, and the stamp is what the queued normals-verify
probe closes when it touches the shirt on purpose.

Registry API: `GET /api/lenses` reads all; `POST /api/lens
{sensor, updates, verified_by?}` merges (provenance fields are the
registry's to stamp), stamps updated_at, optionally stamps the
verification, and rides a `lens_updated` event to the outbox/digest.
Verify suite: **182 passed, 0 failed** (three new: seeded lenses
served with the SILENT_AUDIO signature; merge + verification stamp;
persistence).

## Patch 2026-10-21: the pre-wake buffer + wake assembly (BRIEF-WAKE-AND-TURN mechanism 1)

The conversation surface's my-lane half that the scorer does NOT gate
(capture is destiny): while ARMED, the evolve page keeps a rolling
~5-minute mic ring (the PRE-WAKE BUFFER); a WAKE is the first speech
onset after >= 15 s of quiet, observed in the audio path itself. At
wake, the buffer window is assembled and saved as a `wake_prompt`
bundle -- description only, no judgments banked, with a `resolver:
null` slot whose note says pronoun resolution (recency x salience,
Jev-style) lands with the scorer. The prompt itself is the ordinary
AUTO/manual segment that opens on the same onset; the two join on the
timestamps clock, and the meta says so. `WAKE_FIRED`/`WAKE_SAVE_FAILED`
ride the heartbeat feed and the liaison outbox; the heartbeat carries
`wake_buf_s` and `wakes_fired`.

**Proven by a headless E2E** (fake mic looping speech-2s + quiet-28s):
wakes fired exactly on the quiet cadence (quiet 26.5-29.2 s measured),
buffers growing to the window cap, and each saved bundle carrying the
full wake record with its WAV matching the buffer length. Three real
bugs the E2E caught and killed: a wake check gated on `!openSeg` can
NEVER observe an onset (AUTO opens its segment on the same onset --
the wake now rides the onset in the audio path); `fireWake` could die
in a silent unhandled rejection (the wake stub seg lacked the floors
`buildPayload` reads -- call-site catch added so a failure speaks);
and mutating `payload.meta` after `buildPayload` touches nothing (the
blob is serialised inside -- the wake record now rides the seg). Also
learned, kept as design truth: the buffer ends quiet BY DESIGN -- the
first prompt syllable belongs to the segment, not the buffer.

## DUAL-CAPTURE liveness check (2026-10-22T02:15Z, per BRIEF-DUAL-CAPTURE.md)

The corpus had banked nothing since 07:26 local. Found, fixed, proven:

- **Ghosts**: my own E2E headless Chromes (fake devices from the
  OVER/wake runs) were squatting the camera/mic hours after their runs;
  killed by profile-path sweep (the lesson is now standing practice:
  kill by `user-data-dir` under Temp, never by image name).
- **A lying health number**: `/api/status` reported `heartbeat_age_s: 0`
  for a page 7 hours dead -- `time.mktime` read the UTC `received_at`
  as local (AKDT), making every age negative and clamped to zero.
  Fixed with `calendar.timegm`; the honest age after the fix was
  26,541 s. A wrong liveness number is a silent-audio-class failure for
  the API; this one is now measured correctly.
- **Witness revived**: a real headful Chrome at `/evolve` on the
  Windows desktop, REAL devices, AUTO armed. Status: ARMED ->
  RECORDING, deaf:false, mic "Microphone Array (Realtek(R) Audio)",
  cam "ASUS FHD webcam".
- **Proven by a fresh banked session** (`free_sessions/1792635819`,
  wall 02:18:37 -> 02:23:37Z): 300.1 s rotation segment, 14,404,096
  samples @ 48 kHz, 96.7% nonzero, audio_silent=false. The chain --
  page -> POST /free -> disk -> measured health -- is whole.

The pair analysis (clap sync, offset+drift, TDOA track) is staged and
waits on the phone upload.

## The /lens route: room-lens served live from the bench (2026-10-22)

`GET /lens` serves `../room-lens/index.html` from disk per request (the
evolve pattern: every push to the room-lens checkout is live on reload).
This is the deploy path for the audio fix Casey's acceptance watches:
the phone opens `http://<bench-LAN-IP>:8765/lens`. The audio-defect
story -- cause from source, fix, self-test numbers (audioParts 2 with a
rirMeasure mid-walk) -- is `room-lens/docs/AUDIO-FIX-REPORT.md`.

## Witness outage 2026-10-22 ~04:37Z-07:51Z: what was wrong

The witness Chrome window was simply GONE (closed or crashed -- no
chrome process held the page; heartbeat went stale at ~04:37Z, ~3.2 h
before the 07:50Z check). Not a server fault: the liaison was up, the
last-heartbeat page state read stale-but-armed. Cost: one phone-only
take (23:47 local) the ProArt never heard. Relaunched the same page
(/evolve, same profile so device picks persist) at 07:51Z: AUTO armed,
Realtek mic + ASUS cam live, deaf:false, and the first segment BANKED
(free_sessions/1792655768) within ~5 min. Liveness reported from page
state + the bank, per house rule. FOLLOW-UP for the incident class:
the liaison already alerts on silent AUDIO; it should also alert when
the PAGE disappears (heartbeat stale > 60s while out-of-session hours)
-- queued.

## TAKE-2 lanes (BRIEF-DUAL-CAPTURE-TAKE2.md), all four landed

1. **The 20.000 s transient is CLOSED — my own detector's chunk-seam
   artifact**: the transient scan processed energy in 4000-frame chunks
   at a 5 ms hop = exactly 20.000 s; the seam frames spiked the
   frame-diff, uniformly, forever. (Round numbers are artifacts until
   proven innocent — proven guilty.) The raw-waveform click detector
   finds the REAL transients: 4 irregular events (07:52:00, 07:54:19,
   07:57:34, 08:01:18Z) at 8-12x floor — room sounds, no periodicity.
2. **ESS recompute with per-band SNR + r2: ALL BANDS GRADE DISCARD.**
   Band SNR is ~0 dB (-1.6 to +0.1) with r2 0.81-0.85: after
   band-limiting, the deconvolved peak does not rise above the phone's
   own noise. The previously reported T20s (2.6/3.8 s) are
   noise-dominated and WITHDRAWN as measurements — the room's decay
   remains unmeasured at the phone. Slopes only, distrust graded, as
   briefed.
3. **The heartbeat watchdog is live and already blooded**: PAGE_GONE
   fired at 16:56:58Z when the (second) witness outage was found — the
   3-hour-unalerted class is closed. Alert rides the outbox; PAGE_BACK
   on recovery.
4. **The TAKE-2 build**: phone beacons PHONE_CHIRP on every sweep
   (room-lens); the witness shell polls /cue at 500 ms and answers with
   a railed 1-4 kHz chirp (level from the speakers lens file); the
   chirp-cal bench probe played the level ladder ON THE LIVE PAGE and
   measured: **g0.05=5.4 dB, g0.12=4.9 dB, g0.25=6.3 dB — flat across a
   5x level range, i.e. the detector saw ambient, not the chirp** (the
   pass ran during an active segment: speech_now, motion 0.49). No
   chirp_gain was set — the level-from-data rule refuses a bad
   measurement; the probe re-runs hourly and sets it in a quiet room.
   The flat ladder also says the bench speakers at current rails are too
   quiet for two-way chirp ranging across a room — validating the
   brief's ordering: loud claps are the primary sync, two-way behind it.

## FULL STOP (2026-10-22T18:23Z, Casey's order)

All lanes halted on this laptop: the witness capture page is CLOSED
(heartbeat dead at 18:22Z; the watchdog's PAGE_GONE fired at 18:24Z and
will keep reporting the room as quiet), the forwarder service and the
GC timer are STOPPED, the conversation V0 build halted mid-wire (the
/converse endpoint landed moments before the stop; it speaks only when
a page asks it to, and no page is running). The liaison server alone
stays up so Casey can reach status and history when he wants them.
NOTHING restarts without his explicit word -- no page relaunch, no
service start, no probes. Quiet confirmed by measurement: no page
heartbeat, no capture processes, both side services inactive.
