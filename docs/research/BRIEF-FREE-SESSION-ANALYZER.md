# BRIEF - Free-session analyzer (Casey via Muse, 2026-10-20) - supersedes the probe as your active lane

## The pivot
Casey, verbatim in spirit: "make it so I can click record and just talk
freely and turn and things, and use OpenCV and other things to see what I
was doing - you just need me to do a bunch of stuff in front of the camera."
He is right. The choreographed 16-take plan manufactured labels live and
broke in the browser. New division of labor: the page only RECORDS (ZCode
is simplifying it to one record/stop free-session mode); YOUR job is the
offline analysis that derives the labels afterwards.

## Build (this worktree, branch acceptance/browser-capture)
experiments/free_session_analyze.py (+ a short design note in
docs/research/FREE-SESSION-ANALYZER.md):
Input: a free-session folder as the simplified page will write it -
session dir under /mnt/c/proart_capture/corpus/free_sessions/<stamp>/ with
the full-res WebM, the raw luma stream + per-frame timestamps, the WAV,
and meta.json (negotiate the exact layout from ZCodes page code in the main
checkout; read it, do not guess).
Output: 
1. A segment table (JSON + printed): contiguous segments with start/end,
   face state (facing / turned / absent), speech state (speaking / silent)
   and a confidence per label. Face/head direction from OpenCV (Haar or
   DNN face + landmarks if available in the environment; MediaPipe if it is
   already installed in a project venv - check, do not install the world).
   Speech from the repo own VAD/energy tooling on the WAV.
2. Slices: for each segment of usable length, emit a take in EXACTLY the
   corpus take format the integrity gate eats (same npz/json layout as
   corpus_capture_win.py --plan writes), labelled by derived condition
   (facing+speaking ~ A, turned+speaking ~ B, facing+silent ~ C,
   turned+silent ~ D) with provenance "derived: free_session_analyze"
   and the confidence in the metadata.
Validate on data that exists TODAY: you have no real free session yet, so
(a) unit-test the slicer on synthetic arrays, and (b) run the face/speech
machinery over the existing seed takes + A/B/C nv12 captures in
/mnt/c/proart_capture and report whether it recovers their known labels.
That recovery report is your acceptance evidence.
Fold whatever the camera probe measured (results.jsonl) into a short
section of the design note, then the probe lane is closed.
Commit explicit paths on this branch, push. Do not touch browser_capture/.
