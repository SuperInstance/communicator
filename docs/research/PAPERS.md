# The research corpus — verified, annotated

Every citation below was verified against the published record on
2026-10-18 (publisher pages, ISCA/IEEE/AMS archives, arXiv) — not written
from memory. Where memory drifted from the record, the correction is kept
in the ledger at the bottom: the drift itself is data about how citations
rot. Entries are grouped by the layer of the system they underpin; read
`docs/SCIENCE.md` for the mechanics. Fleet repos cite this file as the
shared bibliography.

Format: citation · **why it is here** — the specific machinery in this repo
it underpins.

---

## 1. Pitch tracking (PITCH)

**de Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency
estimator for speech and music. *Journal of the Acoustical Society of
America*, 111(4), 1917–1930.** · The algorithm `src/pitch.py` implements:
cumulative mean normalized difference, absolute threshold, first-dip rule,
parabolic interpolation. Everything in SCIENCE.md §1.1–1.3.

**Mauch, M., & Dixon, S. (2014). pYIN: A fundamental frequency estimator
using probabilistic threshold distributions. *Proc. IEEE ICASSP 2014*,
659–663.** · The probabilistic successor (threshold distribution + HMM
smoothing). The obvious upgrade path if the stdlib tracker's octave errors
on real speech prove frequent — the known trade is a dependency.

**Camacho, A., & Harris, J. G. (2008). A sawtooth waveform inspired pitch
estimator for speech and music. *JASA*, 124(3), 1638–1652.** · SWIPE':
spectral-template matching instead of the difference function; the standard
baseline reported to beat YIN and RAPT. The alternative lineage, kept for
the day the comparison is worth running.

**Kim, J. W., Salamon, J., Li, P., & Bello, J. P. (2018). CREPE: A
convolutional representation for pitch estimation. *Proc. IEEE ICASSP
2018*, 161–165 (arXiv:1802.06182).** · The deep-learning tracker
(raw-waveform CNN); canonical on noise robustness. Benchmarked against the
7-bit CC scale's actual needs, almost certainly overkill here — noted so
the next person checks before importing torch.

**Rabiner, L. R., Cheng, M. J., Rosenberg, A. E., & McGonegal, C. A.
(1976). A comparative performance study of several pitch detection
algorithms. *IEEE Trans. Acoustics, Speech, and Signal Processing*, 24(5),
399–418.** · The classic head-to-head that established the gross-error
(including octave doubling/halving) vs fine-error taxonomy — the exact
taxonomy this repo's "what this does not prove" sections use. Our octave
gap (synthetic fixtures only) is the open item this taxonomy names.

**Gerhard, D. (2003). *Pitch extraction and fundamental frequency
estimation* (survey). McGill University.** · The most compact treatment of
octave errors specifically; read before building real-speech fixtures.

## 2. Voice activity, endpoints, hysteresis (SILENCE, VELOCITY)

**Rabiner, L. R., & Sambur, M. R. (1975). An algorithm for determining the
endpoints of isolated utterances. *Bell System Technical Journal*, 54(2),
297–315.** · The founding energy + zero-crossing endpointer that all later
VAD cites; `src/energy.py`'s segmentation is its modern descendant with
hysteresis added.

**Sohn, J., Kim, N. S., & Sung, W. (1999). A statistical model-based voice
activity detection. *IEEE Signal Processing Letters*, 6(1), 1–3.** · The
likelihood-ratio statistical VAD lineage; the alternative to energy
thresholds once real-room noise makes RMS floors lie.

**Canny, J. (1986). A computational approach to edge detection. *IEEE
Trans. PAMI*, PAMI-8(6), 679–698.** · Canonical for **hysteresis
thresholding** itself (high threshold seeds, low threshold extends) — there
is no canonical audio-specific paper for dual-threshold segmentation, so
the concept cites Canny and the speech lineage runs informally through
Rabiner & Sambur. `src/energy.py`'s on/off deadband is this idea.

## 3. Speech rate (TEMPO)

**De Jong, N. H., & Wempe, T. (2009). Praat script to detect syllable
nuclei and measure speech rate automatically. *Behavior Research Methods*,
41(2), 385–390.** · The standard transcription-free speech-rate method
(intensity-peak syllable nuclei). `src/tempo.py` currently maps voiced
speech to BPM and asserts shape only; this is the reference for doing it
properly on real speech.

## 4. Prosody & intonation (INFLECTION)

**Pierrehumbert, J. B. (1980). *The Phonology and Phonetics of English
Intonation*. Ph.D. dissertation, MIT.** · Founding document of the
autosegmental-metrical model; the theory that pitch contours decompose into
tones — the scientifically grounded version of what INFLECTION's three-way
classification approximates.

**Silverman, K. E. A., Beckman, M. E., Pitrelli, J. F., Ostendorf, M.,
Wightman, C. W., Price, P., Pierrehumbert, J. B., & Hirschberg, J. (1992).
ToBI: A standard for labeling English prosody. *Proc. ICSLP 1992*, Banff,
vol. 2, 867–870.** · The annotation standard with inter-transcriber
reliability data. If INFLECTION ever grows beyond rising/falling/flat,
grow it toward ToBI's inventory, not an invented one.

**Cutler, A., Dahan, D., & van Donselaar, W. (1997). Prosody in the
comprehension of spoken language: A literature review. *Language and
Speech*, 40(2), 141–201.** · The canonical review of prosody signaling
segmentation, syntax, and speaker intention — the empirical basis for
claiming intonation carries INTENTION at all.

**Banse, R., & Scherer, K. R. (1996). Acoustic profiles in vocal emotion
expression. *Journal of Personality and Social Psychology*, 70(3),
614–636.** · The reference acoustic-profile study for emotional prosody
(14 portrayed emotions → pitch/intensity/rate/voice-quality profiles). The
bridge between our measured tracks and the judged INTENTION track.

## 5. Voice production & quality (TIMBRE's future)

**Fant, G. (1960). *Acoustic Theory of Speech Production*. The Hague:
Mouton.** · THE source-filter citation — glottal source, vocal-tract
filter, formants. TIMBRE-from-text is measured dead here (delivery
property); when the acoustic pipeline arrives, this is the physics it
measures.

**Stevens, K. N. (1998). *Acoustic Phonetics*. MIT Press.** · The
comprehensive modern reference (quantal theory, features ↔ acoustics);
the companion to Fant.

**Hirano, M. (1981). *Clinical Examination of Voice*. Springer.** · Origin
of the GRBAS perceptual voice-quality scale (Grade, Roughness, Breathiness,
Asthenia, Strain, each 0–3) — a validated human-rater timbre taxonomy.
TIMBRE's warm/cold/nasal/breathy/bright scale should be reconciled against
GRBAS rather than coexisting unknowing beside it.

## 6. Affect dimensions (INTENTION's construct validity)

**Russell, J. A. (1980). A circumplex model of affect. *Journal of
Personality and Social Psychology*, 39(6), 1161–1178.** · The valence ×
arousal circumplex (~27k citations) — the standard two-dimensional affect
space. INTENTION's idle→urgent line is an arousal-like axis; the wording
should be checked against this space, not assume urgency is one-dimensional.

**Mehrabian, A., & Russell, J. A. (1974). *An Approach to Environmental
Psychology*. MIT Press.** · The PAD (pleasure–arousal–dominance) model; the
three-dimensional alternative if dominance/potency turns out to be a
confound in urgency judgements.

## 7. Speech acts (GESTURE)

**Austin, J. L. (1962). *How to Do Things with Words*. Oxford: Clarendon
Press.** · Founding text of speech act theory (performatives, illocutionary
force) — the philosophical grounding for GESTURE's question/command/
statement/reaction/greeting taxonomy.

**Searle, J. R. (1969). *Speech Acts: An Essay in the Philosophy of
Language*. Cambridge University Press.** · The systematized taxonomy
(assertives, directives, commissives, expressives, declarations). Our five
classes map onto Searle's; the mapping should be written down when GESTURE
grows, so the classes come from theory rather than vibes.

## 8. Expression & the controller surface (the MIDI layer)

**Palmer, C. (1997). Music performance. *Annual Review of Psychology*, 48,
115–138.** · The standard review of expressive performance — timing,
intensity, articulation as continuous controlled parameters. The academic
version of "a MIDI CC is a continuous controller value."

**International MIDI Association (1983). *MIDI 1.0 Specification*;
maintained as MMA/JMSC, *MIDI 1.0 Detailed Specification* (midi.org).** ·
The standard itself. Note for implementers: a note-on with velocity 0 is a
note-off (SCIENCE.md §1.6's asserted quirk), and Control Change values are
7-bit — both from here.

## 9. Probability & calibration science (the measurement layer)

**Brier, G. W. (1950). Verification of forecasts expressed in terms of
probability. *Monthly Weather Review*, 78(1), 1–3.** · The original proper
scoring rule; `calibration.brier` is this formula.

**Murphy, A. H. (1973). A new vector partition of the probability score.
*Journal of Applied Meteorology*, 12(4), 595–600.** · The Brier
decomposition into reliability (calibration) + resolution + uncertainty —
the theory that makes "calibration is one component of skill" precise.

**Platt, J. C. (1999). Probabilistic outputs for support vector machines
and comparisons to regularized likelihood methods. In *Advances in Large
Margin Classifiers*, MIT Press, 61–74.** · Platt scaling — the sigmoid
post-hoc calibrator. Fleet rule grounded here and measured in
decision-ledger: repair for stretched models (laya 0.374→0.077), harm for
already-calibrated ones (winnow 0.033→0.053).

**Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities
with supervised learning. *ICML 2005*, 625–632.** · The canonical empirical
study of which models are natively calibrated and when Platt vs isotonic
helps — the citation behind "measure first; scale only on evidence."

**Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into
accurate multiclass probability estimates. *KDD '02*, 694–699.** ·
Multiclass calibration (one-vs-rest + normalization) — relevant to
Choice-track calibration beyond the winning option.

**Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well
calibrated probabilities using Bayesian binning. *AAAI-15*, 2901–2907.** ·
The paper that introduced **ECE itself**. `calibration.ece` and every bin-
property lesson in SCIENCE.md §3.2 traces here.

**Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration
of modern neural networks. *ICML 2017*, PMLR 70, 1321–1330.** · Modern DNNs
are badly overconfident; temperature scaling is the simplest effective fix.
The reason "the model said 0.95" is never trusted without a measured ECE —
including Jev's.

## 10. Rank association (INTENTION ordering)

**Spearman, C. (1904). The proof and measurement of association between two
things. *American Journal of Psychology*, 15(1), 72–101.** · ρ itself;
`calibration.spearman` implements it with tie-averaged ranks.

**Kendall, M. G. (1938). A new measure of rank correlation. *Biometrika*,
30(1–2), 81–93.** · τ, the concordant/discordant alternative; and Kendall
(1945), *Biometrika* 33, 239–251 for the ties treatment. τ should accompany
ρ on small-n corpora (it is less sensitive to one wild pair).

## 11. Output variance & prompt sensitivity (why the cache and fingerprint)

**Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S.,
Chowdhery, A., & Zhou, D. (2023). Self-consistency improves chain of
thought reasoning in language models. *ICLR 2023*.** · The canonical
sampling-aggregation result; the alternative to caching (average many
samples) at N× the cost — rejected here on cost, not on merit.

**Sclar, M., Choi, Y., Tsvetkov, Y., & Suhr, A. (2024). Quantifying
language models' sensitivity to spurious features in prompt design. *ICLR
2024*.** · FormatSpread: up to 76-point accuracy spreads across semantically
equivalent prompt formats — the published version of why question wording
is fingerprinted and re-measured rather than trusted to be neutral.

**Ouyang, S., et al. (2025). LLM is like a box of chocolates: An empirical
study of the non-determinism of ChatGPT in code generation. *ACM TOSEM*.** ·
The closest thing to a canonical citation for output divergence at
temperature 0 under identical prompts. **No canonical peer-reviewed source
exists for non-determinism of decision models specifically** — jev-lattice's
7%-flip measurement and our verdict cache are primary evidence in that
vacancy, which is worth stating when citing.

## 12. System-1 decision models (the framing)

**Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and
Giroux.** · The dual-process framing the entire "System-1/decision-model"
category borrowed its name from. A framing reference, not evidence.

**TypeSafe AI (2026-09-15). Introducing System One Models & Jev. Vendor
blog, typesafe.ai.** · The primary source for Jev — docs/announcement
only, not peer-reviewed. Cite as such.

**Deußer, O., et al. (2026). Evaluating and benchmarking the System One
model Jev. arXiv:2609.37647.** · The independent third-party benchmark that
exists so far; preprint, treat accordingly. Our own live numbers
(CALIBRATION.md, gate-replay's JEV-RESULTS.md) are the load-bearing
evidence for this repo's claims.

---

## Corrections ledger (memory vs record, 2026-10-18)

Recorded because citation drift is a defect class this fleet keeps meeting:

1. ToBI: remembered as Chicago Linguistic Society — it is **ICSLP 1992,
   Banff**.
2. SWIPE: remembered co-author "Smith" and an IEEE venue — it is **Harris,
   JASA 2008** (Camacho's 2007 UF dissertation introduced the variants).
3. CREPE: remembered Interspeech — it is **ICASSP 2018**; also the commonly
   quoted arXiv id 1802.06168 belongs to an astronomy paper; CREPE is
   **1802.06182**.
4. Rabiner & Sambur: remembered IEEE TASSP and "endpoint detection" — it is
   **Bell System Technical Journal**, "An algorithm for **determining** the
   endpoints of isolated utterances."
5. Sohn et al.: remembered IEEE Trans. Speech & Audio — it is **IEEE
   Signal Processing Letters**.
6. Naeini et al.: remembered AISTATS — it is **AAAI-15**.
7. Murphy 1973: remembered Journal of the Atmospheric Sciences — it is
   *Journal of Applied Meteorology*.
8. Sclar et al.: remembered EMNLP 2023 — it is **ICLR 2024**.
9. Hysteresis: no canonical audio paper exists; the concept cites **Canny
   1986**.
10. Jev: no peer-reviewed vendor paper exists; primary source is the
    **vendor blog** plus one independent arXiv preprint.

Six of thirty-plus citations drifted in venue or author; two would have
been wrong papers. Verification is not optional.

---

## 13. Day-1 additions (2026-10-18, verified by research lane)

**Audio→symbolic intermediates.** Wu, Y.-H. et al. (2022). MIDI-DDSP:
Hierarchical musical modeling. *ICLR 2022*, arXiv:2112.09312 · the
canonical argument FOR a symbolic/expressive mid-layer between waveform
and score — this repo's architecture in published form. Gardner, I. et
al. (2022). MT3: Multi-task multitrack music transcription. *ISMIR*,
arXiv:2111.03017 (note: Gardner, not "Garcia") · MIDI-like tokens as
universal audio→symbolic interchange. Bittner, R. et al. (2022). Basic
Pitch. arXiv:2203.09893 · lightweight instrument-agnostic transcription
— the small end of the field. Zeng, M. et al. (2021). MusicBERT.
*Findings of ACL*, arXiv:2106.05630 · MIDI tokens made LM-friendly.

**Prosody/emotion representation.** Yang, S.-W. et al. (2021). SUPERB.
*Interspeech*, arXiv:2105.01051. **CORRECTION (Day-3, verified against
the paper itself): SUPERB contains NO question/statement prosody task —
"PR" is Phoneme Recognition, and a search-engine claim that it holds an
intonation task was hallucinated and propagated here by an earlier
research lane. The verified prosody-adjacent line is Lin et al. below
and the audio-only speech-act results in §14.** Lin, J. et al. (2022). SUPERB-Prosody. *IEEE SLT* (Best
Paper) — middle wav2vec/HuBERT layers carry prosody. Ma, Z. et al.
(2024). emotion2vec. *ACL Findings*, arXiv:2312.15185 + EmoBox
(arXiv:2406.01655): IEMOCAP SOTA ~73–77%, MSP-Podcast ~0.38 macro-F1 —
the ceilings that justify presence gates over raw probabilities. Wang,
Y. et al. (2018). Style Tokens. *ICML*, arXiv:1803.09017; Skerry-Ryan,
R. et al. (2018). Prosody transfer. *ICML* — delivery mapped to a small
controllable bank: the CC-bank move, in TTS.

**Pre-event intent (the T-minus precedents).** Ekstedt, E., & Skantze, A.
(2022). Voice Activity Projection. *Interspeech*, arXiv:2205.09812 —
self-supervised forecast of FUTURE voice activity: the published analog
of an intention track. Inoue, K. et al. (2024). Real-time VAP turn-taking,
arXiv:2401.04868. Chang, S. et al. (2022). Turn-taking prediction.
*Interspeech*, arXiv:2208.13321 (>97% shift recall). **Udupa et al.
(2026). Endpoint anticipation for low-latency dialogue. *Interspeech*,
arXiv:2606.13450 — forecasts end-of-turn up to 2.56 s ahead and cuts
505 ms via speculative execution: published T-minus scheduling for
speech.** Raux & Eskenazi (2008). Endpointing thresholds. *SIGdial*.

**Anticipatory timing.** Repp, B., & Su, Y.-H. (2013). Sensorimotor
synchronization review. *Psychon. Bull. Rev.* 20(3) — negative mean
asynchrony is the NORMAL regime of skilled performance. Fujii, S. et al.
(2011). Drum kit synchronization error. *Music Perception* 28(5) — pro
drummers strike measurably AHEAD of the click: the kick-drum law with
numbers.

**System-1 ecosystem 2026.** Deußer et al. (2026). arXiv:2609.37647 (Jev
benchmark); JEVal arXiv:2610.03935 (11,257 instances/36 datasets);
DecisionRobust arXiv:2610.19118 (injection robustness — validates our
presence-gate posture); multi-family arXiv:2610.00346. Microsoft
Decision-1 announced 2026-10-09 (vendor-only, unflagged). Ollaya:
github.com/ollaya-dev/ollaya (Apache-2.0).


## 14. The open questions, answered from the record (Day-3 lane)

**Per-utterance reliability.** Jiang et al. (2018). To trust or not to
trust a classifier. *NeurIPS*, arXiv:1805.11783 — the per-instance trust
score. Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a
Random World* — conformal prediction; Campos et al., TACL 2024 survey
(arXiv:2405.01976); Quach et al., Conformal Language Modeling, ICLR 2024.
**The ceiling: Gupta, Podkopaev & Ramdas, NeurIPS 2020 (arXiv:2006.10564,
Thm. 3) — distribution-free instance-conditional calibration is
IMPOSSIBLE except trivially.** So: per-utterance guarantees exist only as
prediction-SET coverage; the controller's per-item trust signal is the
set size. That is the implementable answer to PROGRESS's "per-utterance
reliability unmeasured."

**Speech acts from acoustics, no ASR.** Chen et al. (2022). WavLM.
arXiv:2110.13900 — 98.6% intent (SLURP), 70.2% emotion (IEMOCAP) audio-
only. Abdulrahman et al. (2025). SQEBSP dataset. *Data in Brief* — ~88%
three-way question/statement/exclamation from pitch. Dang et al.
(2020). arXiv:2004.11419 — speech-to-dialog-act end-to-end. So yes,
published, and our measured tracks are the hand-rolled version.

**Per-item stability.** Kuhn, Gal & Farquhar (2023). Semantic
uncertainty. *ICLR*, arXiv:2302.09664; Farquhar et al. (2024). *Nature*
630 — per-question entropy over meaning clusters predicts confabulation.
Recipe for our flip measurements: resample the same utterance N times,
cluster, stability = 1 − normalized cluster entropy. tev1's stable-coin
behavior is the degenerate (zero-entropy) case.

**F0-only diarization.** No canonical clean comparison exists (open).
Anguera et al. (2012). Diarization review. *IEEE TASLP* 20(2) — pitch
unreliable as a feature; Honorof & Whalen (2005). *JASA* — F0 is
relative to the speaker's own range. Our Alice bimodality works because
the cast is mixed-voice; same-gender overlap is the known killer.

**Prosody→intent causality.** Hellier et al. (2002). Perceived urgency of
speech warnings: semantics versus acoustics. *Human Factors* 44(1) —
orthogonal manipulation, causal acoustic effect (the published form of
our Day-2 dissociation). Kobayashi, Hamada & Akagi (2022). *Speech
Communication* 139 — F0 mean + dynamics drive evacuation-announcement
urgency. Bänziger & Scherer (2005). *Speech Communication* 46 —
intonation resynthesis changes emotion recognition.
