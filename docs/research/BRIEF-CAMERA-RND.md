# BRIEF — Camera exposure R&D: the empirical lane (Casey via Muse, 2026-10-20)

## The race
Casey has ordered a head-to-head. The browser capture page shows the front
camera REALLY dark in Chrome on this machine. ZCode (other window) is
shipping a fix in the product page from reasoning. YOUR lane is the
empirical one: measure what this actual camera does under different
constraint sets and let numbers pick the winner. Casey would rather burn
both subscriptions than wait. Your acceptance-referee brief is PAUSED, not
cancelled - resume it after this.

## Evidence pack
- Same camera + room through the Windows stack (corpus_capture_win.py):
  A.nv12 mean luma 110.7, B.nv12 114.2, empty room 49.7 (0-255 scale).
- The page asked: video ideal 640x480, frameRate ideal 30, no deviceId.
  Casey confirms: correct front camera, responds to light, just starved.
- Suspects: pinned frame rate capping exposure; driver behaving differently
  under explicit constraints than under defaults.

## Build (this worktree only, branch acceptance/browser-capture)
experiments/camera_probe/: a single probe.html + probe_server.py (stdlib,
port 8766, binds 127.0.0.1). The page offers PRESET BUTTONS; each applies a
constraint set to the SAME camera and shows: live video, live mean luma
(computed on a 160x96 canvas exactly like the capture page), and the
track getSettings() + getCapabilities() dump. Presets:
  P0  shipped: {width ideal 640, height ideal 480, frameRate ideal 30}
  P1  bare:    {video: true} semantics - no constraints beyond device
  P2  defaults + exposureMode continuous via applyConstraints (catch if unsupported)
  P3  {width ideal 1280, height ideal 720} no frameRate pin
  P4  your own best idea from the capabilities dump (e.g. exposureTime manual
      near max, if the track exposes it)
A LOG PRESET button samples luma ~3s and POSTs {preset, settings,
capabilities summary, luma mean/min/max} to the server, which appends to
experiments/camera_probe/results.jsonl in this worktree.
Start the server yourself (nohup, port 8766) and leave it running.

## Deliverable
docs/research/CAMERA-RND-FINDINGS.md: this cameras reported capabilities
(ranges that matter: exposureTime, frameRate, width/height), the measured
luma per preset once Casey has clicked through (until then, the tool ready
+ whatever you can verify without a lens), and a concrete recommendation:
which constraint set the product page should ship, with the numbers.
Commit explicit paths, push the branch. Do not touch browser_capture/.
