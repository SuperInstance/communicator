# BRIEF — Browser-native corpus capture (Casey, via Muse, 2026-10-20)

## Why
Casey: the Windows-terminal capture flow is clunky. He wants the corpus
recording (CAPTURE-NEXT.md) as a local browser page: opens in his browser,
shows him his own camera feed LIVE while recording, big on-screen prompts,
so he can see that each take is doing what it asks. Same data, humane surface.

## Read first (sources of truth — do not improvise formats)
- CAPTURE-NEXT.md (repo root): the plan, the conditions, the checks, the warnings.
- /mnt/c/proart_capture/corpus_capture_win.py: what --plan mode writes per
  take — file names, layouts, metadata, timing.json schema, the arrays it
  stores (maps / maps_raw / luma), frame dimensions and stride, achieved-rate
  handling. The browser version must produce takes the existing pipeline
  accepts unchanged.
- experiments/corpus_integrity.py: the gate the output must satisfy.
- src/nv12.py and src/wavio.py: the readers your files must round-trip through.

## Build (in this repo, new folder browser_capture/)
1. server.py — Python STDLIB ONLY http server, binds 127.0.0.1:8765. Serves
   index.html and accepts one POST per finished take (raw frame data, WAV,
   timing/meta JSON), writing takes to the out dir (default
   /mnt/c/proart_capture/corpus, --out flag) in EXACTLY the layout
   corpus_capture_win.py --plan writes. No pip installs, no external calls.
2. index.html — single file, vanilla JS, zero CDN/network dependencies.
   - getUserMedia camera + mic. LARGE live self-preview (mirror toggle).
   - Video: draw frames to canvas at the capture dimensions the Python
     script uses; keep per-frame timestamps from ONE clock shared with the
     audio; store raw luma/NV12 exactly as the pipeline expects. Do NOT
     substitute MediaRecorder/webm — the raw frames are the data product.
   - Audio: capture PCM (AudioWorklet), write 16-bit WAV at the achieved
     rate, header honest about the rate.
   - Plan driver for all 19 takes from CAPTURE-NEXT.md: A1-A3 facing recite
     50s; B1-B3 turned-90 recite 50s; C1-C3 facing silent 30s; D1-D3 turned
     silent 30s; F1-F3 free 60s; FREE 5 min. Per take: huge condition text
     (e.g. TURN 90 DEGREES - STAY SILENT), 5s countdown, live timer, poem
     on screen for recite takes, progress bar across the whole plan.
   - Live audio level meter + SPEAKING / SILENT indicator while recording.
   - Per-take end checks mirroring the script: speech %, drift, achieved
     rates; VOICE_MISSING / NOT_SILENT flags shown in the UI, take kept but
     flagged exactly as the script does.
   - Controls: start/pause the plan, redo THIS take, skip take, click any
     take to (re)record it out of order. Saved takes listed with status.
   - A take uploads to the server as it finishes (no giant browser
     downloads, no data held only in the tab).
3. README.md in browser_capture/: the one start command and the URL
   (http://localhost:8765), plus what Casey clicks, in plain language.

## Verify before you call it done
- Unit-level: push synthetic frames + a synthetic WAV through the POST path;
  prove the written files round-trip: file size == frames x stride via
  src/nv12.py, WAV readable via src/wavio.py at the stated rate,
  timing/meta JSON parses and matches the Python script schema.
- State plainly what you could NOT verify without a real camera (the actual
  getUserMedia take on Caseys machine is his 10-second test, not yours).
- Write browser_capture/DONE.md: start command, URL, what was verified and
  how, what still needs the real-camera test, any format decision you had to
  make and why.

## Discipline
- Do not modify corpus_capture_win.py, CAPTURE-NEXT.md, or the gate.
- Stage explicit paths, commit, push. No git add -A.
- Localhost only. Nothing leaves the machine. No analytics, no CDN.
