# Lane for the other agent — do not touch anything else in this repo

Two agents are now working this tree. This file is the boundary, so that
neither of us has to rediscover the collision cost by having it again.

## DO NOT TOUCH

    src/score.py                src/local_tev.py
    src/local_judge.py          src/judged.py
    src/question_health.py      experiments/small_model_sweep.py
    experiments/gate_prompt_diag.py
    experiments/local_question_health.py
    PROGRESS.md                 docs/

## YOUR LANE: real-speech fixtures

Every test in this repo runs on synthetic tones. That was a deliberate choice
while the pitch tracker was being calibrated, and it is now the largest hole:
**octave errors are the classic failure of a YIN-style F0 estimator on human
voice, and nothing here exercises one.**

Produce, under `tests/fixtures/` and `tests/test_real_speech.py`:

1. **A small fixture corpus of real recorded speech**, at least 6 clips,
   covering: a male and a female fundamental; a low voice near FMIN (70 Hz)
   and a high one near FMAX (400 Hz); at least one clip with a known octave
   error risk (breathy onset, or a falling glide that briefly doubles).
   Record them yourself or synthesise with an explicitly human-like
   glottal source. **Do not use a pure sine tone** — that is what the existing
   tests already do, and it cannot produce an octave error.
2. **Ground truth** per clip: the F0 you expect, in Hz, with the tolerance you
   believe is defensible. State where the ground truth came from. If you
   derived it by listening, say so — that is legitimate and it is different
   from deriving it from the tracker under test, which would be circular.
3. **Tests that run the existing `src/pitch.py` over them** and report the
   worst per-frame error, including any octave errors, rather than asserting
   only on clips that pass.
4. **A written statement of what the corpus does NOT cover.**

## The rules this repo works by

* **A test that cannot fail is not evidence.** For every claim you make, break
  the thing that makes it true and watch the suite go red. Record the result.
  If a `sed` pattern does not match, the sabotage did not apply — a control
  that silently changed nothing looks exactly like a test that passed.
* **Written is not run.** Two modules in this repo were complete, documented,
  committed and published, and had never been executed. Both failed on their
  first run. Run your code before you commit it.
* **Never push a key.** `.env` and `.scratch/` are gitignored. If you need a
  credential, ask.
* **Do not weaken an assertion to make a run green.** If real speech fails the
  pitch tracker, that is the finding — write it down with the number, do not
  move the tolerance.

## If you touch something outside your lane

Stop, commit only your lane, and say so in your summary. Do not run
`git add -A`: sweeping the tree has already pulled another agent's in-flight
files into an unrelated commit twice.