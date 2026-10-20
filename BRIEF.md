New lane in a fresh session. Read this fully before starting.

REPO AND OWNERSHIP

/home/proart/projects/communicator, branch master. Two other agents are working
right now. They own src/watcher.py, src/vad.py, tests/, and these docs:
docs/research/IN-THE-WILD-AUDIO.md, MIC-HARDWARE.md, DEVICE-GENERALISATION.md,
GATE-PEAK-COVERAGE.md, TWO-SPEAKER-TURNS.md, F0-SPEAKER-CHANGE.md,
OTHER-REALM-IDEAS.md, CROSS-MODAL-FUSION.md, ADOPTION-PLAN.md.
Do not touch any of those. Do not modify src/ or tests/.

THE MEASURED PROBLEM

We need to know whether someone is handing you the floor. Two routes are now
MEASURED DEAD ENDS and must not be re-proposed:
  - silence duration: AUC 0.529 over 1396 pauses from two four-party AMI
    meetings; every duration threshold loses to the trivial "always predict
    change" baseline of 0.6225
  - pitch/F0: separates two voice CLASSES but not speakers within one (0.5 and
    0.3 sigma apart)
So the question is what signal architecture could carry this information inside
an always-on budget.

Read experiments/turn_eval.py and docs/research/CROSS-MODAL-FUSION.md before
starting. The second one matters to you directly -- see the corrections below.

YOUR OBJECTIVE

Test whether a vision model can classify a hand-off from a spectrogram IMAGE,
on the real two-speaker AMI corpus in examples/realtwospeaker/.

Two corrections, both established AFTER this brief was first drafted. Take them
as constraints on your conclusions, not as trivia:

1. Do NOT report this as a fair test of "image-native fusion". The OmniVLA work
   trains per-sensor projectors and spatialises cues; neither OmniVLA paper
   licenses "just render it as an image". If your result is negative, it is
   evidence about OUR SPECTROGRAM ROUTE, not about the pattern.

2. Phase is UNTESTED, not disproven. A spectrogram discards phase and phase is
   what distinguishes two overlapping voices -- so a negative result here may be
   an artifact of the representation rather than of the problem. If you fail,
   say explicitly whether your failure is consistent with "phase is the missing
   ingredient", because that is a live hypothesis and your data can bear on it.

METHOD REQUIREMENTS

1. Read experiments/spectrogram_vision.py first. Report what it actually does,
   what model it uses, and what it measured. Do not assume it works.
2. Establish what is actually available BEFORE designing around it: torch,
   transformers, timm, open_clip, ollama, or a local VLM. Do not add a heavy
   dependency to the project. If nothing usable is present, that IS the finding.
   Note: numpy IS present in /home/proart/projects/glasses/.venv/bin/python;
   torch/transformers were not confirmed.
3. Build image examples that make the classes separable IN PRINCIPLE, and state
   the construction and its weaknesses. A turn spectrally unlike every other
   turn is a trivial win and proves nothing.
4. Use the same discipline as experiments/turn_eval.py. ALWAYS report the
   trivial baseline, currently 0.6225 pooled. Any method that cannot beat 0.6225
   has learned nothing, because the corpus supplied the answer.
5. Run the no-information control: a vision model shown a spectrogram with NO
   turn-end information must score near chance. If it does not, the model is
   reading an artefact and your result is void. This is the check that catches a
   rigged experiment.

HARD CONSTRAINTS

- Write lane, ONLY these three paths:
    experiments/spectrogram_turn_eval.py        (new)
    experiments/spectrogram_turn_eval_results.json (new)
    docs/research/SPECTROGRAM-TURN.md            (new)
- Do NOT run any git write command. No git add, commit, stash, checkout, clean.
  Read-only git is fine. Commit nothing.
- Use /home/proart/projects/glasses/.venv/bin/python (system python lacks
  typesafe_sdk).
- There is NO audio hardware here. /dev/snd has only a timer; PulseAudio exposes
  only an RDP bridge. Do not attempt to capture a microphone. Work from files.

ACCEPTANCE CRITERIA

- A measured number, or an honest statement that it could not be run and why.
- The 0.6225 baseline stated alongside any accuracy.
- The no-information control run, and its result.
- Every number reproducible from a command you state.
- An explicit statement of whether a negative result is consistent with "phase
  was the missing ingredient".

HOUSE RULE

This project has a documented history of plans written from other people's
numbers failing on contact with real audio, and a documented history of agents
citing artifacts that turn out not to exist. If the answer is that this route
does not work, that is a complete and valuable result. Do not soften it. If you
find your own experiment is rigged, say so loudly and early.

Report back under 400 words.
