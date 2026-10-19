# Two-or-more-speaker conversational corpus

**This material is CONVERSATIONAL.** It is spontaneous multi-party meeting
audio in which several people genuinely take turns inside a single file,
including handing the floor over to somebody else. It is **not** read speech,
**not** narration, and **not** dramatic reading.

It is not a substitute for `examples/realaudio/corpus/`, and it must never be
quietly merged with it: that corpus is one person talking continuously, which
is exactly why it could not answer the turn-taking question.

## Why this directory exists

The blocking finding (`docs/research/ADOPTION-PLAN.md`, commit `17ae46a`) was
that the whole audio corpus was single-speaker. Measured consequence: after any
silence ≥ 0.6 s, the **same** speaker resumes 98.1–100 % of the time. The
positive class — *a speaker yields the floor and someone else takes it* — was
unrepresentable, so turn-taking accuracy could not be measured at all.

Measured on the full AMI annotations:

| scope | gaps ≥ 0.6 s | followed by a **different** speaker |
|---|---|---|
| `examples/realaudio/corpus/` (pre-existing) | — | **1.9–0 %** (same speaker 98.1–100 %) |
| AMI EN2002a, whole meeting | 260 | **191 (73.5 %)** |
| AMI EN2002b, whole meeting | 191 | **129 (67.5 %)** |
| AMI EN2002c (partial) | 69 | 43 (62.3 %) |

## What is here

20 minutes of audio, 16 kHz mono 16-bit PCM WAV, 4 speakers, 2 meetings.
Full per-file provenance in `MANIFEST.json`.

| file | what it is | length |
|---|---|---|
| `ami_sdm_EN2002a_turns.wav` | EN2002a on the **single far-field microphone** | 420 s |
| `ami_sdm_EN2002b_turns.wav` | EN2002b, same | 420 s |
| `ihm_EN2002a_channels_90s/spk_*.wav` | four **per-speaker headset channels**, same 90 s window | 4 × 90 s |

Each `.wav` has a same-named `.segments.json` giving, for every annotated
utterance, its `speaker_id`, `begin_time`, `end_time` and transcript text.
That file is the ground truth — **no diarizer was run and no label guessed.**

## Source and licence

- **AMI Meeting Corpus**, HuggingFace mirror `edinburghcstr/ami`, split `test`.
  <https://huggingface.co/datasets/edinburghcstr/ami> ·
  upstream <https://groups.inf.ed.ac.uk/ami/corpus/>
- **Licence: CC-BY-4.0**, declared in the dataset card frontmatter. The card's
  prose licensing sections are empty, so attribute the upstream Edinburgh
  corpus (Carletta et al.), not the mirror.
- Fetched from the public HuggingFace `datasets-server` rows endpoint: **no
  token, no account, no registration, no licence click-through, no paywall.**
  Only stdlib `urllib`; no pip dependency was added to this project.
- `_downloads/fetch_ami.py` regenerates every byte here.

## What is represented — and what is not

**Is:** real turn-taking between real people, including hand-offs to another
speaker, real overlap (24–30 % of adjacent utterance pairs), real far-field
room conditions from a single microphone, and per-utterance speaker ground
truth.

**Is not:**

- **Continuous capture.** AMI publishes per-utterance slices. Where no segment
  exists, this corpus contains **digital silence**, and we do not claim the room
  was silent. A gap measured here is a *lower bound* on the real gap.
- **Four identifiable voices.** Median F0 resolves **two** voice groups, not
  four people: FEO070 and FEO072 sit ~10 Hz apart (0.5 MAD σ), MEE071 and
  MEE073 ~7 Hz (0.3 MAD σ). Pitch therefore proves *two distinct voices exist*,
  but **a hand-off within a group produces no reliable pitch jump.** A
  turn-taking stage must not assume "speaker changed" ⇒ "F0 jumped".
- **Clean far-field F0.** Pooling overlapped segments inflates the
  within-speaker spread from 22.3 Hz to 52.8 Hz and drops the best speaker
  separation from 4.7 to 1.8 MAD σ, because one microphone carries two voices
  at once. Exclude overlapped segments, or use the headset channels.
- **Every annotated utterance, audibly.** Overlapping segments are slices of the
  *same* channel; the fetcher paints rather than sums (summing would
  comb-filter), so 111 of 314 segments in EN2002a contribute no samples. Labels
  are still in the sidecar.
- **A representative sample.** Each 420 s window was deliberately chosen for
  the *highest* density of "silence then different speaker" events, so the
  slice-level rates (0.803 / 0.714) **overstate** the meetings. Quote the
  meeting-wide rates (0.735 / 0.675) for anything about hand-off frequency.
- **Demographic or room variety.** Both meetings are the same four people.

## Verifying it

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/twospeaker_verify.py
```

Writes `experiments/twospeaker_verify_results.json`, exits 0 only if two
distinct speakers are confirmed. It measures per-speaker median F0 with
voiced-frame counts, uses the **within-speaker MAD σ as the noise floor**, and
reports every pair's separation in units of that floor. The vectorised YIN is
cross-checked frame-by-frame against `src/pitch.py` on every run (agreement is
written into the results, not assumed).

Current verdict: `TWO_OR_MORE_DISTINCT_SPEAKERS_CONFIRMED` — best separation
**4.7 MAD σ** (103.5 Hz), channel-separated reference **5.4 MAD σ**, label-free
F0 bimodality **5.17 pooled σ**.