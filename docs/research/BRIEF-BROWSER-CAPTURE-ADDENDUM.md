# ADDENDUM to BRIEF-BROWSER-CAPTURE (Casey, via Muse, 2026-10-20)

Casey, while watching you work: the takes must not only feed our pipeline.
He will later send the same recordings to SEVERAL other understanding
systems (cloud multimodal models, future pose/affect tools) to extract
maximum nuance about what he was doing in each take. So, in ADDITION to the
pipeline-exact files the base brief requires, every take must ALSO save a
portable rich copy:

1. A full native-resolution video file of the take (browser MediaRecorder,
   WebM, camera native res/fps, WITH the mic audio track muxed) saved next
   to the take as <label>_full.webm. This runs in PARALLEL with the raw
   canvas capture - it is an extra artifact, never a substitute for the raw
   frames the pipeline needs.
2. Keep the full-rate PCM WAV as its own file (already in the base brief).
3. A per-take sidecar <label>_rich.json: condition label, instruction text
   shown, stimulus name, wall-clock start/end, achieved fps + sample rate,
   per-frame timestamp array reference, browser/camera device label if
   available, and a free-text notes field the page lets Casey type into
   after the take ("what I actually did") - default empty.
4. Session manifest at the out root: session.json listing every take, its
   files, labels and flags, so an external tool can ingest the whole
   session without knowing our repo layout.

Principle, in Caseys framing: capture once at the highest fidelity we can,
in open formats; every analysis is a later, re-runnable decision. Nothing
about todays features may throw away tomorrows nuance. (The overnight
correction to CAPTURE-NEXT.md made the same call for maps vs maps_raw vs
luma - this extends it to the full-resolution original.)
