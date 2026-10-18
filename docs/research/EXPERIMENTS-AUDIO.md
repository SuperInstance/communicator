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

**Day-0 correction (Day-1):** the 285 "BPM" was called "falsified" above.
Wrong: `src/tempo.py` documents one-syllable=one-beat deliberately
(conversational 4–5 syll/s = 240–300 BPM, intentionally not rescaled).
285 BPM = 4.75 syll/s — in-band for read speech. What Day-0 actually
showed is that a GLOBAL rate erases the structure; E4's per-phrase map is
the fix, and it passed (below).

## 3a. Day-1 (2026-10-18) — first experiment results

**E4 PASSED.** `phrase_tempo_map()` (src/tempo.py): 42 rated phrases in
the 90 s excerpt, tempo 146–396 BPM (2.4–6.6 syll/s), **SD = 19% of mean**
against a ≥15% bar. The speaker changes tempo by the phrase, and the map
keeps it.

**E3 first evidence.** F0 medians by 90 s window across the 38-minute
chapter: 229/236/225 Hz early vs **174/194 Hz** at 1800 s/2200 s — the
cast mix changes scene by scene and windowed F0 tracks it (~6 semitones).
Pooled n=23,911: lower-half median 169 vs upper-half 262 Hz. Speaker
signal is plainly present; formal dip-test diarization is now E3's
remaining work, not a question of whether the signal exists.

**ASR leg unblocked.** faster-whisper small, CPU int8, raw-PCM input
(bypassing the broken PyAV path): **8 s for the 90 s clip (11× realtime)**,
language en @1.00, clean multi-speaker transcript saved
(`alice01_transcript.txt`; GPU blocked only by missing libcublas on this
box).

**Judged tracks on real dialogue (jev-1.14.0, gated).** Transcript lines
scored directly:

| line | GESTURE | INTENTION | ATTENTION | TIMBRE |
|---|---|---|---|---|
| "What did you do then?" | question 122 | 69 | listener 117 | 9 |
| "This way, you see, Alice?" | question 67 | 51 | listener 117 | 8 |
| "It's like a battle." | statement 64 | 6 | task 23 | 2 |
| "With soldiers?" | question 112 | 29 | listener 55 | 2 |

Questions vs statements separate by 40+ CC; addressee shifts with the
scene; TIMBRE's presence gate holds it near zero on text-only input —
exactly the designed honesty. The chain real audio → acoustics + ASR →
judged tracks now works end to end.

**Cross-model battery** (same 5 real + 4 garbage probes, urgency + gesture
questions):

| model | urgency split | garbage | gesture split | stability on re-ask |
|---|---|---|---|---|
| jev-1.14.0 | correct | parks (pos 0.06) | correct | 7%-class flips (jev-lattice) |
| jev-1.13.0 | correct | parks (0.04) | correct | — |
| tev1 (local, ollama --ask) | correct (73/95%) | **declines** no@77-80% | correct (96/89%) | **bit-stable 94%×3** |
| laya:en (Ollaya) | correct but weak on urgent (0.80) | declines | **INVERTED** (no@0.69 / yes@0.97) | stable 0.67×3 |
| gemma2:2b (chat control) | 1 of 2 | declines (no confidence) | 1 of 2 | — |

Findings: (1) tev1 is a credible free/private judged-track fallback —
deterministic, declines garbage; (2) laya's question/command inversion on
typed criteria is a real cross-model divergence worth a question-rewrite
probe; (3) the chat-model control loses calibration and half the answers,
as expected.

**Cross-build calibration (jev-1.13.0 vs 1.14.0, same corpus):** 1.14.0
dominates — ATTENTION 0.917/0.097 vs 0.833/0.162 acc/ECE, INTENTION
Spearman 0.978 vs 0.956; GESTURE perfect on both; ATTENTION more
ill-posed on garbage in 1.13.0 (0.90 vs 0.79). The default build choice
is validated. TypeSafe's public model list exposes only the two aliases
(jev-latest→1.13.0, jev-preview→1.14.0); explicit probes of
jev-1.15.0/jev-2.0.0/jev-mini all return 400 Unknown model — no newer
public builds as of today.

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

## 6. Later additions

**E7 — the pedal swing observable** (pre-event rises before annotated
moments) and **E8 — prime-then-payload landing quality** (operator
response with vs without variance-primed delivery) live in
`docs/research/TIMING.md` §4 with the timing theory that motivates them
(capture point, T-minus scheduling, priming as receiver calibration).


## 3b. Day-2 (2026-10-18) — GPU stretch: full scene, formal E3, tev1 table

**E3 formally PASSED (existence level).** 2-Gaussian EM on pooled log-F0
(5 windows, n=23,911): BIC difference **1933** vs one Gaussian (decisive
>10). Components: **124 Hz (20% share) and 235 Hz (80%)** — two voices
from pitch alone, no training, no labels. Remaining for full E3 pass:
cluster purity against known turn boundaries (`bimodality_probe.py`).

**tev1 full corpus** (39 texts × 4 binary questions, 68 s on GPU):
GESTURE question-vs-other **15/15 = 1.000** (matches Jev); INTENTION
yes-rate monotone across designed levels 0.00 / 0.00 / 0.67 / 1.00;
ATTENTION listener-vs-other 0.667 (weak vs Jev's 0.917); warm yes-rate
6/39 (conservative). **Flip stress 5×3 texts: byte-identical every time**
— and the borderline line "It's like a battle." answers *yes @53%*,
identically on every ask: tev1 states near-coin uncertainty
DETERMINISTICALLY. A stable-uncertainty model is exactly what a
controller wants from a fallback. Interface: `ollama run tev1 --ask
"<question>"` with state on stdin (0.4 s/ask).

**Whole-scene run** (28 sentence-lines sampled across the full 38-minute
transcript, Jev on text + measured tracks on the aligned audio windows,
`scene_analysis.json`):

    intent vs line RMS median   r = +0.070   (nothing)
    intent vs syllable rate     r = −0.181   (nothing)
    E7 proxy: RMS rise in the 2 s BEFORE a top-quartile-intent line:
    6/7 = 86%  (against ~50% chance, one-sided p≈0.06, n=7 — suggestive,
    not conclusive)

The dissociation IS the finding: **within-line delivery intensity does
not carry word-intent — but intensity RISES INTO high-intent lines.**
The signal is in the approach, not the utterance: the pedal-swing law
(TIMING.md §2) observed in real dramatic audio, while the naive
"loud = urgent" hypothesis is refuted on the same data. Honest caveats:
n=7 high-intent lines; narration/dialogue confound unseparated; 2 s RMS
windows are crude; a staged replication on more scenes is required
before this is a result, and it is designed above (E7 proper needs
annotated punchlines, not quartile cutoffs).

Also: full transcript committed (`alice01_full_transcript.txt`, 467
timestamped segments, 247 s CPU); gesture distribution across the real
scene — statements 15, questions 6, commands 4, reactions 3.

## 3c. Day-3 — E3 completed on the full chapter; corpus built

**E3 full pass** (120,762 voiced frames, whole 38 min): EM components
**125 Hz (18%) vs 239 Hz (82%)**, BIC difference **10,097** (decisive;
the 5-window probe's 1933 was conservative). 30 s windows: 54% pure-HIGH,
**0% pure-LOW**, 46% mixed — the low voice appears only in short stints
inside predominantly-high windows, with one long HIGH run over the first
~22 min and H/L alternation concentrated in the later, song-heavy scenes
(handoff rate 18.7%). The 7 LOW-dominated sampled lines are CHARACTER
voices ("With extras?" — Gryphon; "Soup of the evening, beautiful soup"
— Mock Turtle's song) while HIGH carries Alice and narration;
P(question∪reaction | HIGH) = 0.31 vs 0.14 for LOW (n_LOW=7 — directional
only). **The confound stands and is now sharper**: one dramatic reader
dropping to a creaky 125 Hz register for characters produces the same
signal as multiple readers. For the SCORE this is moot — the
register-to-character mapping is the per-voice encoding either way — but
"two speakers" is NOT established by pitch alone (consistent with
PAPERS.md §14: no canonical F0-only diarization result exists).

**Corpus built** (§2 matrix filled, public domain): poetry / essay /
short-story / Shakespeare full-cast / children's / JFK oration, 75 s
excerpts, pipeline sanity green — the oration cell (64% voiced, 5 long
segments) vs children's (156 segments) brackets the cadence range the
matrix was designed to span. MANIFEST in `examples/realaudio/corpus/`.

## 3d. Day-3 — the laya "inversion" resolved; winnow is clean

Five phrasing variants × 10-text battery on Ollaya (local), plus a
winnow:e4b baseline:

| model × phrasing | accuracy | garbage-yes |
|---|---|---|
| laya:en V0 "is this a question..." (bare criteria) | 3/7 | 1/2 |
| laya:en V1 grammar-detailed criteria | 1/7 | 1/2 |
| **laya:en V2 "is this a command..." (polarity swapped)** | **7/7** | 1/2 |
| laya:en V3 intent-framed | 4/7 | 0/2 |
| laya:en V4 examples-inline | 4/7 | 0/2 |
| **winnow:e4b V0 baseline** | **7/7** | 0/2 |

Day-1's "laya inverts the gesture question" is downgraded and resolved:
laya has a **label-orientation bias** — the command-framed polarity
scores perfectly while the question-framed one sits near chance (a
blocked agent's probe caught it at exact chance, 0.502/0.498), and RICH
criteria make it worse (V1 < V0), echoing the fleet's small-model finding
that elaborate instructions can hurt sub-1B encoders. Practical rules:
on Ollaya, use **winnow:e4b for speech-act questions** (clean at
baseline, declines garbage); if laya must be used, swap the polarity and
map the answer accordingly. The QUESTION-REWRITE method worked exactly
as designed — before/after kept, claim corrected.
