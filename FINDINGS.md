# FINDINGS — the ProArt capture corpus, extracted before purge (2026-10-22)

A bounded extraction pass over `/mnt/c/proart_capture` on the ProArt WSL,
run the day the capture programme was stopped. Everything measurable was
measured, the labelled core was copied off-box, and this file is the plain
record of what the corpus contained and what it actually taught. The full
archive (census, transcripts, stats, room profile, preserved sessions,
manifest with sha256) lives with the coordinator.

## What the corpus contained

- **241 free sessions**, 2026-10-20 19:19 AKDT to 2026-10-22, totalling
  **5.12 hours of audio** (1.77 GB of WAV). The corpus directory holds
  8.48 GB by file length (6.8 GiB allocated on the Windows mount); the
  bulk is video side-channels (luma.raw + video.webm), not audio.
- **56 sessions (23%) are digital silence** — every sample exactly zero.
  All of them fall in the deaf-microphone era of 2026-10-20 evening, when a
  selected input device delivered literal zeros while the page reported
  recording.
- **36 sessions (15%) carry speech** by energy VAD. Fresh transcription
  (faster-whisper small, RTX 4050, 42 items in 52 s wall) found **2,786
  words** across them — Casey talking with his AI setup, testing the rig,
  and, in the duet-window witness sessions, narrating the dual capture
  itself.
- Only **one true conversation** exists in the whole corpus: the rig-test
  session 1792560758 (2026-10-20 21:30 AKDT, 229 words). The planned
  two-microphone dialogue was never recorded; its `raw/` stayed empty.

## Correction of record: the "Invictus fixture" is not Invictus

The loose takes A.wav and B.wav were carried in fleet memory as Casey
reciting *Invictus*. Transcription says otherwise — both are the same
prose reading passage, near-identical takes:

> "The library opened at nine, and the light came through the east window
> in the long bar across the floor. A man in a gray coat sat near the
> window with a cup of coffee he did not drink…"

68 words each. The fixture still works — a known text, read twice — but
the known text is the library passage, and it is now on record in the
archive transcripts. C.wav, the third take, is near-silent (peak
−53 dBFS): no speech is recoverable from it even at +30 dB gain.
mic.wav is four seconds of pure zeros — the original deaf-mic evidence.
probe.wav is a very low-level probe, no speech.

## What it taught

**Fixtures earn their keep.** The corpus's durable value was as test
material with known answers. The deaf-mic signature is exact and now
measured: a healthy session's noise floor (20 ms frame RMS, 20th
percentile) medians **−68.3 dBFS** and never falls below 0.000136, while a
deaf session is 0.000000 on every sample. `zero_fraction == 1.0` is a
perfect single-feature detector — no model required. The page's own
`deaf:true` self-flag (fix 8cfebe0) was validated against this data.

**His pauses break naive endpointing.** From the deliberate takes
(A, B, the rig-test conversation, and the same-evening speech test;
energy VAD, −60 dBFS floor): 65 inter-phrase pauses, **median 0.72 s**,
mean 1.41 s; **33.8% of pauses are ≥ 1 s and 6.2% are ≥ 2 s**. Any
endpointing rule that closes the floor at ~1 s of silence would cut in on
roughly a third of his phrase boundaries mid-thought. Speech segments
median ~1.3 s; speech rate ≈ 200–225 words per minute of VAD-active
speech (reading and animated conversation — brisk). Ambient room speech
(610 pauses, not attributed to Casey alone) medians 0.94 s with a long
tail (p90 7.1 s).

**The room has a measurable character.** The empty-room take (empty1.wav,
150 s) sits at RMS −73.6 dBFS. The ambient noise spectrum (median across
eight quiet sessions) is LF-dominated: 100–300 Hz is the loudest band
(−101.9 dB/Hz), speech bands fall away above it (1–3 kHz −112.9), and
everything above ~6 kHz drops off a shelf (−146 dB/Hz) — the webcam
microphone's rolloff, not the room's. Decay-slope (RT60) claims from the
phone sweeps were already withdrawn by the seats for 0 dB band SNR; this
pass confirms the honest position: trust the spectrum and the slopes you
can measure, distrust the levels.

**The duet's witness ear worked.** The laptop witness sessions from the
2026-10-21 duet window hold Casey's side of the experiment in full —
session 1792635819 alone carries 524 words — even though the phone's
chirps never reached the witness microphone and the pair alignment stayed
unmeasurable. One ear was enough to keep the record; it was never enough
to locate anything.

## What it could not teach

- **No depth, no spatial model, no room map.** One fixed viewpoint, no
  labelled geometry, audio that is mostly a single room's tone. Nothing
  in this data trains or validates spatial understanding; the walkabouts'
  value was always in their labelled takes, not their volume.
- **No general speech model.** One speaker, one room, 5 hours, of which
  only minutes are dense speech. It fits statistics about Casey in this
  room; it generalises nowhere.
- **No conversation corpus.** One rig-test conversation is a fixture,
  not a dataset.

## Preservation and release

Preserved off-box in the extraction archive (2.95 GB, sha256-manifested):
the loose labelled WAVs and seed files; the seven longest Oct-20 AKDT
session dirs in full (the only ones long enough to hold the pair
protocol) plus audio+metadata for the remaining 46 sessions of that day;
the seven duet-window witness dirs in full; and all analysis outputs
(census, transcripts, speech stats, room profile).

The purge therefore releases the whole corpus directory — **8.48 GB by
file length** — on the laptop. Of that, ~5.5 GB — bulk luma/video of
ambient and silent sessions — exists nowhere else and is destroyed by
the purge. That is the honest price, stated before it is paid: the
lessons above are extracted, the fixtures are kept, and the rest was a
room being recorded, not a room being understood.
