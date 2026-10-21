# BRIEF - Motion as text (Casey via Muse, 2026-10-20) - the communicator/syzygy bridge

## The idea (his, paraphrased to be buildable)
Video is heavy; edges are cheap. When Casey moves, you do not need to save
a wall of video: save a few REFERENCE FRAMES and, between them, a TEXT
STREAM of where the edges moved and how fast. Because the reference frames
anchor the picture, the motion events reconstruct everything that matters
about the movement - and a text stream can run at far higher frame rates
than anything that ships pixels could afford. Motion becomes a document.

## Build (this worktree): experiments/motion_as_text.py
Use tonights free session luma stream
(/mnt/c/proart_capture/corpus/free_sessions/1792553040/luma.raw, 160x96,
15,514 frames @60fps; the change-map machinery already exists in this
repo - reuse it, do not reinvent):
1. REFERENCE FRAMES: emit one reference frame per second (or on scene
   change above a threshold - your call, report it).
2. EVENT STREAM: between references, encode per-frame motion as text -
   regions where edges moved, with magnitudes. Design the encoding
   yourself (grid cells, sparse edge coords, whatever is honest); it must
   be plain text an LLM can read.
3. RECONSTRUCT + MEASURE: rebuild an approximation from references +
   text stream. Report: (a) bytes/sec of the text representation vs the
   raw stream; (b) reconstruction error per frame (mean abs diff, and your
   judgment of where it breaks); (c) an ASCII rendering of one real
   gesture from the session (Casey, arms wide, ~180s in) - the syzygy
   payoff: text is the storage AND the rendering.
4. CLOCK CONTRACT: every reference and event carries the session clock
   (frame indices + wall time from timestamps.json). This stream is meant
   to sit in the SAME alignment table as transcript words and MIDI notes.
Deliverable: docs/research/MOTION-AS-TEXT-REPORT.md with the numbers, the
encoding spec, and the ASCII gesture, committed and pushed. Known
limitation to name, not hide: what the encoding loses (texture, identity,
small deformations) - say it plainly.
