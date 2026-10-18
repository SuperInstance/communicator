# Real-audio experiments — depth beyond the transcript

*Design doc for the podcast/YouTube line. Thesis first sharpened, then six
experiments with protocols and pass/fail bars, then Day-0 data (the first
real speech ever run through this pipeline, 2026-10-18) and the
infrastructure gaps the Day-0 run exposed.*

## 0. The thesis, sharpened

The framing on the table:

> The waveform is a scalar representation; the MIDI is a vector that snaps
> to the inferred intended meaning and can change tempo by the phrase. The
> MIDI captured a sound ON THE WAY TO MAKE MUSIC, not the music itself.
> This makes meaning a first-class citizen in time, the way a band changes
> tempo and time signature inside a complicated song.

Three parts, each with a precise version:

1. **"Scalar → vector."** Operationally true: the waveform carries one
   untyped degree of freedom per sample; the nine-track score carries nine
   TYPED dimensions per event. The honest sharpening: the waveform contains
   ALL the information — the score is a **lossy projection whose loss is
   chosen** to keep gesture, structure, and character while dropping noise.
   Like a pianist's reduction of an orchestral score: fewer notes, all of
   the music. So the claim to test is not "more information than audio"
   (false) but **"the retained axes are the ones meaning lives on, at the
   granularity meaning acts"** (testable).

2. **"More depth than STT."** Trivially true for prosody, rhythm, voice
   character, and speaker identity — a transcript discards F0, energy,
   timing, and voice quality BY CONSTRUCTION. The scientifically honest
   baselines are therefore not STT but: (a) generic paralinguistic feature
   sets — eGeMAPS is the standard [Eyben et al. 2016, TAC — cited-not-
   judged, verification pending], and (b) mel-spectrogram front-ends. The
   score's differentiator against THOSE is the typed/semantic layer
   (judged tracks), the musical time structure (per-phrase tempo map,
   rests, event grid), and 100× smaller representations.

3. **"Meaning first-class in time."** The judged tracks are time-stamped
   semantic events riding the same grid as the acoustic ones — urgency,
   attention and speech-act become notes with positions, not metadata
   about a whole clip. The band analogy is exact: tempo maps change at
   section boundaries; here, sections are PHRASES (Day-0: median phrase
   1.25 s, 46 phrases in 90 s — there is material for a real tempo map).

## 1. The experiments

Ordered by information gained per unit of work. Each names what it does
NOT prove. Shared corpus needs are in §2.

### E1 — Depth test: score vs transcript on paralinguistic questions

**Question.** Can a consumer answer prosody-dependent questions from the
score that they cannot from the transcript?
**Protocol.** N=60 clips (30–90 s). Produce (a) transcript, (b) transcript
+ 9-track score rendered as a readable event table, (c) audio itself
(human ceiling). Consumers answer fixed questions: "did urgency rise in
the second half?", "which phrase was the interruption?", "who dominated
the exchange?", "was the final statement a question or rhetorical?" Use
human raters for a subset (n=15, 3 raters) and a strong LLM as a cheap
full-set proxy — with the known weakness that LLMs read prosody poorly,
which cuts AGAINST condition (b), making a win conservative.
**Metric.** Accuracy per condition; McNemar (b) vs (a).
**Pass.** (b) > (a) by ≥ 10 points on ≥ half the question types.
**Does not prove.** That the score beats raw audio (it won't; the claim is
beats-the-transcript and approaches-audio-cheaper).

### E2 — Snap-to-meaning validity: multi-track coherence at annotated moments

**Question.** When a speaker demonstrably shifts (excitement, whisper,
rapid-fire list, solemnity), do multiple tracks move coherently?
**Protocol.** 20 clips with deliberate shifts (stand-up cadence breaks,
podcast gasps-to-whispers, narrators changing register). 3 annotators mark
shift timestamps blind; keep moments with ≥2 agreement. Measure track
deltas in a ±2 s window: ΔTEMPO, ΔVELOCITY, ΔINTENTION, ΔPITCH spread.
**Metric.** Fraction of annotated moments with ≥2 tracks moving >1 within-
clip SD; correlation of annotated "energy" curve with a composite score
curve.
**Pass.** ≥ 60% of moments coherent.
**Does not prove.** Causality of meaning — this is construct validity of
the vector, not a theory of intent.

### E3 — Multi-voice: diarization by score clustering

**Question.** Does the score separate speakers without supervision —
"multi-voice as first-class citizen"?
**Protocol.** Two-host podcasts and dramatic readings (LibriVox drama is
public domain and multi-voice). Cluster voiced frames on (F0, local
velocity, local rate) with k=2. Known turns from a light human pass.
**Metric.** Cluster purity vs turns; F0 bimodality (Hartigan dip or
two-Gaussian BIC) per clip.
**Day-0 hint.** The drama excerpt's F0 p10–p90 spans 187–331 Hz AND piles
mass at both tracker bounds (70/399) — speaker mix is visible in F0 alone
before any clustering.
**Pass.** Mean purity ≥ 0.8 on clean two-host audio.
**Does not prove.** More than 2–3 speakers; overlapping speech needs the
channel-per-speaker extension (MIDI channels exist in the spec; our writer
currently emits no channel nibble — see gaps).
**Build first.** Frame-level speaker features + per-speaker track
emission (one MIDI track per voice is the natural 9-track-compatible
encoding).

### E4 — Tempo by the phrase

**Question.** Is a per-phrase tempo map materially better than one BPM?
**Protocol.** Same corpus. Compute local rate per speech segment (De Jong
& Wempe syllable-nuclei method — see PAPERS.md §3); report the
within-clip variance of local tempo; correlate phrase tempo with
annotated excitement (E2 moments).
**Pass.** Within-clip tempo SD ≥ 15% of mean AND tempo tracks excitement
(r ≥ 0.4) on ≥ half of clips.
**Day-0 evidence.** Phrases exist at the right granularity (1.25 s
median); the CURRENT global mapping is inadequate on real speech (285
"bpm" on 90 s of drama) — the crude voiced-duration→BPM heuristic is
retired for real audio by this result, per-phrase maps replace it.

### E5 — Round trip: score → synth → human recognition (the thesis, made literal)

**Question.** If we RENDER the score through a synthesizer (timbre→
instrument register, velocity→dynamics, tempo map→tempo, rests→rests),
can listeners recover the structure?
**Protocol.** Render 20 clips' scores to audio (simple synth; no speech
synthesis — this is the reduction, not a reconstruction). Listeners
(n≥5) answer: how many speakers? which phrase was the question? where
did energy peak? did the ending resolve or interrupt?
**Pass.** Speaker count ≥ 80% on two-voice clips; question-phrase
identification ≥ 70%.
**Does not prove.** Fidelity of any single track — it probes whether the
REPRESENTATION carries recoverable structure, which is the entire "sound
on the way to make music" claim. Failure here is the most informative
outcome of the set: it would locate the loss exactly.

### E6 — Representation economics: probe accuracy per kilobyte

**Question.** How much paralinguistic signal per byte, versus baselines?
**Protocol.** Fixed label set (E2 moments, E3 speakers, speech acts from
E1). Three feature front-ends at matched pipeline simplicity: eGeMAPS-
style acoustic features, mel-spectrogram means, our 9-track score
features (all per phrase). Boring classifiers only (logistic regression —
fleet law: labeled traffic → boring classifiers win); matched tiny
parameter budgets.
**Metric.** Accuracy vs feature-set size (bytes/event and bytes/clip).
**Pass.** Score features ≥ acoustic baseline accuracy at ≤ 10% of the
bytes.
**Does not prove.** Anything about deep models — deliberately. The claim
is about the REPRESENTATION, so the learner is held boring on purpose.

## 2. Corpus and sources

- **LibriVox dramatic readings** (public domain, multi-voice, prosody-
  rich) — working Day-0 source; item `aliceinwonderland_drama_1310_librivox`.
- **Two-host podcasts** via RSS enclosures (curl) where license permits
  local research use; keep files out of the repo.
- **YouTube** needs yt-dlp + ffmpeg — NOT on this box today (see gaps);
  install before this lane widens.
- Deliberate variety for the matrix: monologue, banter, interview with
  interruptions, audiobook narration, news-anchor formal cadence,
  stand-up (comic timing), poetry reading, rap (the literal music-boundary
  case). 8 cells × 5–8 clips.
- Legal posture: research use, local storage, no redistribution of
  copyrighted audio; public domain where possible.

## 3. Day-0 data — first real speech through the pipeline (2026-10-18)

Source: LibriVox dramatic Alice ch.1, 20s–110s, 90 s mono @16 kHz after
decode (`examples/realaudio/`, mp3 kept out of git).

    frames 8996 · voiced 4478 (50%) · poisoned 0
    F0 median 229 Hz · p10 187 · p90 331 · min 70 (FMIN) · max 399 (FMAX)
    adjacent-voiced octave jumps 134/4477 = 3.0%  (downward-halving 62)
    segments 91 (46 speech / 45 rest) · median phrase 1.25 s
    bpm 285 (crude global heuristic — retire for real audio, see E4)
    inflection FLAT 216 · RISING 151 · FALLING 145

Findings this one run already produced:

1. **The octave-error rate on real speech is 3.0% of adjacent voiced
   pairs**, split near-evenly up/down. The suite's most-feared,
   never-exercised failure now has a number. pYIN-style smoothing is the
   canonical fix (PAPERS.md §1) and now has a baseline to beat.
2. **Both tracker bounds saturate** (mass at 70 and 399 Hz): the speaking-
   range assumption (70–400) clips real dramatic voices, and the pile at
   the bounds is itself speaker-mix signal (E3's starting evidence).
3. **50% voiced** — plausible for dramatic reading with real rests; the
   energy floor's real-speech rejection rate is now measurable (and
   worth an honest check against quiet-but-real voiced frames).
4. **Phrase granularity is right** for per-phrase tempo maps (E4).
5. The pipeline digested 90 s / 9k frames in one pass without incident —
   windowing for hour-long input is an engineering concern, not a science
   blocker.

## 4. Infrastructure gaps the Day-0 run exposed

| gap | status | unblocks |
|---|---|---|
| MP3 decode | worked around: `miniaudio` in scratch venv `~/.local/audio-venv` (no ffmpeg on box) | corpus build |
| yt-dlp + ffmpeg | ABSENT — YouTube lane blocked until installed | corpus breadth |
| tempo on real speech | crude heuristic falsified on Day-0 (285 bpm) | E4 — implement syllable nuclei |
| octave smoothing | 3% baseline measured; fix not built | E1/E2/E5 quality |
| diarization / per-speaker tracks | not built; channel nibble not emitted by smf writer | E3 |
| long-file windowing | single-pass OK at 90 s; needs phrase-windowed scoring loop | everything at scale |
| ASR leg | absent — judged tracks need text; local whisper or Windows-side | E1/E2 (judged tracks on real audio) |

## 5. What I would run first

E4's per-phrase tempo map (small, purely acoustic, Day-0 already demands
it), then E3's F0-bimodality on the full 38-minute drama chapter (zero
new labels needed), then the ASR leg — because E1, the headline test of
the thesis, is blocked on transcripts for the judged tracks. E5 last and
deliberately: it is the most direct test of the "sound on the way to
make music" claim, and its failure would be the most informative event
in the whole program.
