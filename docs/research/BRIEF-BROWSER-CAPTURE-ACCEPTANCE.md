# BRIEF — Independent acceptance of the browser-capture build (Casey via Muse, 2026-10-20)

## Your role: referee, not co-builder
ZCode is building browser_capture/ in the main checkout (~/projects/communicator)
from BRIEF-BROWSER-CAPTURE.md + its ADDENDUM. A judge cannot certify itself:
ZCode writes its own verify.py, so YOU independently check the deliverable.
You work ONLY in this worktree (~/projects/communicator-acceptance, branch
acceptance/browser-capture). NEVER edit browser_capture/ or ZCodes files.

## Do this
1. BLIND DERIVATION FIRST, before reading any ZCode code: from
   /mnt/c/proart_capture/corpus_capture_win.py (what --plan writes per take),
   CAPTURE-NEXT.md, and the two briefs (docs/research/BRIEF-BROWSER-CAPTURE*.md
   in the main checkout), write down the exact expected artifact set per take
   and per session: file names, layouts, stride arithmetic, WAV requirements,
   timing/meta schema, the addendum artifacts (_full.webm, _rich.json fields,
   session.json manifest). Save as docs/research/ACCEPTANCE-EXPECTED.md here.
2. Then write experiments/browser_capture_acceptance.py: given a session out
   dir, it checks every take against your derived expectations and the repo
   readers (src/nv12.py round-trip: file size == frames x stride; src/wavio.py
   reads the WAV at its stated rate; JSON schemas parse and match; flags
   VOICE_MISSING / NOT_SILENT present where the data demands them). Exit
   non-zero with named failures. Stdlib + the repo venv numpy only.
3. When browser_capture/DONE.md appears in the MAIN checkout and ZCode has
   produced a synthetic/test session via its verify path, run your checker
   against that session dir (read-only) and write the verdict to
   docs/research/BROWSER-CAPTURE-ACCEPTANCE.md here: PASS/FAIL per check,
   discrepancies between ZCodes claims and your measurements, and anything
   that still needs Caseys real-camera test. If DONE.md has not appeared
   within a reasonable wait, validate your checker against a synthetic
   session you construct from your own derivation, and say so in the verdict.
4. Commit on this branch with explicit paths and push the branch. No
   git add -A. Do not merge; Muse reviews and merges.
