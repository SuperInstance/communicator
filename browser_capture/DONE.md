# DONE — browser_capture

Built 2026-10-20, per `docs/research/BRIEF-BROWSER-CAPTURE.md` and its
`BRIEF-BROWSER-CAPTURE-ADDENDUM.md`.

## Start command and URL

```
python3 browser_capture/server.py        # from the repo root, inside WSL
```

Open **http://localhost:8765** in the Windows browser. Stdlib-only server,
binds 127.0.0.1:8765, default out dir `/mnt/c/proart_capture/corpus`
(`--out` / `--port` to change). Click **START PLAN**; the page drives the
rest. README.md has the plain-language walkthrough.

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
