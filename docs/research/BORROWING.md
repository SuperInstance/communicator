# Mini-repos worth borrowing — annotated, verified via GitHub API 2026-10-18

Constraints: pure stdlib (no numpy/scipy in the runtime), 6 GB GPU for
optional local models. Best-first; license noted; AGPL/no-license = ideas
only, never code.

| repo | loc ~ | license | borrow |
|---|---|---|---|
| rhasspy/energy-vad | 300 | MIT | the only zero-dep VAD; auto-threshold from leading silence — complements our hysteresis |
| patriceguyot/YIN | 150 | MIT | cleanest plain-python YIN to diff ours against |
| kenziyuliu/py3pYIN | — | none | **ideas only**: the pYIN blueprint (threshold distribution + HMM) for the 3.0% octave-error fix |
| wq2012/SpectralCluster | 650 | Apache-2.0 | diarization clustering; needs a power-iteration eigensolver to go stdlib |
| tango4j/Auto-Tuning-Spectral-Clustering | med | unverified | eigengap speaker-count estimation (unknown-k diarization) |
| drfeinberg/PraatScripts (syllable_nuclei.py) | 230 | MIT | De Jong & Wempe dB-drop parameters, to upgrade our peak heuristic toward the reference method |
| timmahrt/promo | k-LOC | MIT | F0-contour interpolation/stylization utilities for contour gaps |
| bixb922/umidiparser | 800 | MIT | stdlib SMF reader w/ running status (beyond our parser's needs today) |
| MarkCWirt/MIDIUtil | 650 | MIT-ish | battle-tested writer, if ours diverges from spec |
| absadiki/pywhispercpp | — | MIT | whisper.cpp bindings, quantized tiny/base under 6 GB — the GPU ASR path (libcublas currently blocks ctranslate2 GPU here) |
| SYSTRAN/faster-whisper | big | MIT | the ASR leg in use today (CPU int8, 11× realtime) |

Immediate uses: energy-vad's calibration idea for our floor; py3pYIN's
blueprint for E-fix; SpectralCluster for E3's unknown-k phase.
