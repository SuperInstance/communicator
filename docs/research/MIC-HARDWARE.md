# Microphone hardware and mounting as a physical noise-rejection layer

Lane: hardware / prior art. Written for the 150-200 Hz gate weakness in
`tests/test_vad.py::TestDeclaredLimits` (pure tones at 150/175/200 Hz score
0.66-0.82; a 200 Hz harmonic stack scores 0.97), which is exactly where
50/60 Hz mains and their harmonics live: 50, 100, 150, 200, 250 Hz.

## How to read the labels

Three kinds of claim are mixed in this document and they are not equivalent:

- **[read]** a number read out of a datasheet, standard or paper fetched during
  this session, with a URL.
- **[measured]** a number produced by `experiments/mic_filter_sim.py` on this
  repo's own audio, in this environment, reproducible by running it.
- **[derived]** arithmetic I did from a [read] number, shown so it can be
  checked or thrown away.
- **[unverified]** a claim I believe but did not confirm from a primary source.
  These are labelled because this project has a documented history of other
  people's numbers failing on contact with real audio
  (`src/vad.py`, `tests/test_vad.py`, `docs/research/prior-art-audit.md`).

## THE HEADLINE, MEASURED FIRST

Before any hardware argument, the cheapest possible fix was measured on the
repo's own speech, because "a high-pass is standard practice" is a claim about
somebody else's tracker. `experiments/mic_filter_sim.py` runs this repo's
`src/pitch.PitchTracker` over `examples/realaudio/alice01_excerpt.wav`
(90.0 s, 16 kHz, 2999 analysed positions, 1484 voiced in the unfiltered
reference, median F0 228.915 Hz) through RBJ filters, and reports attenuation
at the mains harmonics plus the resulting pitch error against the *unfiltered*
tracker on the *same* frames. Results: `experiments/mic_filter_sim_results.json`.

Filter: RBJ biquad, high-pass = 2 cascaded sections (4th order, Q=0.707);
notch = band-stop, Q=10 (~20 Hz wide at 16 kHz). Attenuation is measured by
pushing unit sines through the same code path, not asserted from the design.

| condition | dB @50 | dB @100 | dB @150 | dB @200 | median err (cents) | p90 (cents) | frames >20c | octave errors | voiced frames lost |
|---|---|---|---|---|---|---|---|---|---|
| unfiltered (reference) | - | - | - | - | 0.00 | 0.00 | 0.000 | 0.000 | 0 |
| HPF 60 Hz | -9.79 | -1.08 | -0.23 | **-0.08** | 0.14 | 1.43 | 0.005 | 0.001 | 2 |
| HPF 100 Hz | -24.49 | -6.04 | -1.58 | **-0.53** | 0.24 | 2.57 | 0.013 | 0.002 | 3 |
| HPF 150 Hz | -37.56 | -15.65 | -6.04 | **-2.40** | 0.47 | 4.82 | 0.026 | 0.004 | 17 |
| HPF 200 Hz | -45.98 | -24.56 | -12.40 | **-6.03** | 0.73 | 8.73 | 0.051 | 0.005 | 42 |
| notch 200 Hz (Q=10) | -0.00 | -0.02 | -0.13 | **-20.99** | 0.18 | 4.54 | 0.040 | 0.003 | 18 |
| notches 50/100/150/200 Hz | -15.01 | -18.17 | -20.12 | **-21.33** | 0.24 | 5.00 | 0.040 | 0.003 | 19 |

Three things follow, and the first two are negative results that matter more
than the third.

1. **A high-pass does not fix this, and the reason is structural, not a
   tuning failure.** A high-pass attenuates *below* its corner. Mains' third
   and fourth harmonics sit at 150-200 Hz, which is *above* the corner of any
   filter whose corner is low enough to keep the voice. Measured: a 4th-order
   60 Hz high-pass removes **0.08 dB** at 200 Hz. Even a 200 Hz corner -- which
   starts eating the tracker's own range -- only removes 6.03 dB. The folklore
   "80 Hz rumble filter" removes 0.08-0.23 dB at 150-200 Hz. **Any design that
   proposes a high-pass to defeat 150-200 Hz hum is proposing a filter that
   cannot touch it.**
2. **A narrow notch can, and the cost on our tracker is small.** -21 dB at
   200 Hz for 0.24 cents median F0 error and 19 of 1484 voiced frames lost.
   Note the asymmetry: the four-notch bank attacks 3.5x harder at 200 Hz than
   the 200 Hz high-pass while costing *less* pitch, because it does not delete
   the bottom of the range.
3. **The high-pass's real cost is not pitch error, it is lost frames, and the
   mechanism is the energy floor.** `src/pitch.py:58` gates on
   `ENERGY_FLOOR = 1e-3` RMS *before* YIN runs. Removing low-frequency energy
   lowers RMS and pushes borderline frames under the floor. At a 200 Hz corner
   that is 42 frames (2.8% of voiced); the median cents error is still only
   0.73. A filter that "does not damage pitch" can still quietly delete speech.

### The tension, stated precisely

The measured median F0 of this fixture is 228.9 Hz, and a 200 Hz corner sits
just below it, which is the friendliest possible case. Read the attenuation row
for the harmonics *below* the corner: at fc=200 Hz the filter removes 24.56 dB at
100 Hz and 45.98 dB at 50 Hz. **[derived]** For a male speaker whose F0 sits at
100 Hz, that filter attenuates his fundamental by 24.6 dB while attenuating the
200 Hz hum by only 6.0 dB -- it removes **four times more voice than hum**. The
small median-cents number above is a property of this one narrator, not a
general licence. `src/pitch.py` declares 70-400 Hz; the gate weakness is at
150-200 Hz. **Those two ranges overlap, and no linear filter can separate them.**
That is the whole reason this problem belongs to hardware and mounting rather
than to filtering.

## Ranked interventions, by how well each attacks 150-200 Hz specifically

Ordered for a Pi-class wearable on glasses. "Fixes 150-200 Hz?" means the
component that fools the gate.

| # | intervention | mechanism | fixes 150-200 Hz? | side effects | confidence |
|---|---|---|---|---|---|
| 1 | **Conducted-hum hygiene**: differential/balanced signalling, star ground, isolation mounts, mic physically remote from DC-DC converters and the charger | hum is not the loop's *presence* but the loop current flowing through an audio ground; a balanced line rejects it | **Yes, and it is the only lever that is not frequency-selective** -- it removes the interferer without touching 200 Hz voice | cost is board/wiring effort, no power penalty, no signal penalty | mechanism [read]; magnitude [unverified] |
| 2 | **Notch at mains harmonics** (50/100/150/200 Hz, Q~10), in firmware | removes a steady narrowband component at known frequencies, leaving the rest of the band | **Yes: -21 dB at 200 Hz [measured]** | needs re-tuning when the grid frequency or the offending device changes; a moving or switching source (motor whine, ballasts) smears across the notch; 4 biquads = trivial CPU and zero power | high [measured] |
| 3 | **Contact / structure-borne mic** on the frame, temple or larynx | senses vibration of a solid, not air pressure, so *airborne* mains is rejected by mechanism rather than by frequency | **Yes, for airborne hum, at every frequency** | the wearer's voice arrives via bone conduction and is much weaker; you lose the room entirely (this is a feature if privacy is the goal and a bug if turn-taking needs ambient speech); ManiWAV uses two on a gripper, not for speech | mechanism [read]; speech usefulness [unverified] |
| 4 | **Bone-conduction coupling** at the temple/jaw | same class as (3); a transducer/mic is a mass on a spring | **Yes** -- and LF response is where BC is weakest, which is *good* for hum and *bad* for our F0 feature | this is the sharpest trade in the table: it rejects exactly 50-200 Hz because it barely hears 50-200 Hz, and `src/pitch.py` needs 70-400 Hz | [unverified] -- mechanism direction is certain, the numbers are not |
| 5 | **Capsule acoustic high-pass** via vent-hole design | the membrane's ventilation holes are the acoustic resistance that sets the corner frequency | **No.** By construction it acts *below* its corner; pushing the corner to ~200 Hz still leaves 150-200 Hz nearly untouched, and gives only a few dB | free if the vendor part already has it; requires a custom part to change | [read], high |
| 6 | **Airborne MEMS self-noise floor** (SNR 61-66 dB(A)) | lowers the noise floor | **NO -- this does not actually help.** A noise floor is not a selectivity. A quiet capsule hears the hum *more* cleanly, not less. It bounds the SNR available for quiet speech and nothing else | cost/power: none beyond the part | [read] + [derived], high |
| 7 | **Directivity: cardioid / differential / hypercardioid capsule** | rejects sound arriving off-axis by interference across the capsule aperture | **NO -- this does not actually help at these frequencies.** Directivity needs an aperture comparable to a wavelength | the requirement is physically out of reach for a 1.4-2.5 mm capsule: see the derivation below | [derived], high |
| 8 | **Naive hardware high-pass at 60-80 Hz** | standard rumble filter | **NO -- provably.** -0.08 dB at 200 Hz measured | cheap; genuinely useful for knocks, wind and handling noise, which is what vendors actually use it for | [measured], high |

### The two "this does not help" results in full, because they are the point

**Self-noise (row 6) does not attack hum at all.** A 65 dBA-floor capsule is a
statement about the *lowest sound it can hear*, not about which sounds it
rejects. A tonal interferer sits far above that floor and passes through
untouched. Choosing a quieter MEMS part makes the hum *cleaner*, which for a
gate that is fooled by periodicity is arguably the wrong direction. The real
consequence of the 61-66 dB(A) class is a dynamic-range budget question for
quiet speech, and it is a constraint to respect, not a fix to apply.

**Directivity (row 7) is physically unavailable at 50-200 Hz.**
**[derived]** Directivity requires the aperture to be a non-negligible fraction
of a wavelength. At 200 Hz, c=343 m/s gives lambda=1.715 m; a cardioid's first
null sits near lambda/2, about 0.86 m. A 1.4 mm chip needs ka ~ 1 to begin to
form a null at all. At 200 Hz the small Infineon capsule in [read] below is
roughly **600x too small**, so it is acoustically omnidirectional exactly where
the hum is. Buying a cardioid means buying a multi-centimetre capsule -- which
is not a MEMS part, and is not a glasses-temple part. The honest version of
this lever is: *two* MEMS capsules plus a real time-delay or spatial filter
cannot fix it either, because the aperture is the array size, not the element
size, so the array would be ~0.8 m across. Row 7 stays "no".

## Supporting sources

### Airborne MEMS capsules: what the numbers actually say

**[read]** Deh, Würzer, Flödner, Krumbin, *The Infineon Silicon MEMS
Microphone*, AMA Conferences 2013 SENSOR/OPTO/IRS, DOI
[10.5162/sensor2013/A4.3](https://doi.org/10.5162/sensor2013/A4.3), PDF at
<https://www.ama-science.org/proceedings/getFile/ZGD4Zt==>:

- SNR **61 / 64 / 66 dB(A)** for chip areas 1.0 / 1.2 / 1.4 mm, membranes 0.7 /
  0.9 / 1.1 mm, all at -38 dBV/Pa. The paper notes mobile expectations moved
  from "telephone quality with SNR of 58dB(A)" to a state-of-the-art 66dB(A).
- **[derived]** Self-noise = 94 - SNR, so this class of part sits at
  **28-33 dBA (A-weighted)**, matching the "65 dBA SNR" figure in the brief.
- The membrane's **ventilation holes form an acoustic high-pass**; the measured
  corner frequency is **80 Hz** ("the 3dB high pass at 80Hz"), with a port
  resonance near 22 kHz.
- The corner frequency is a **design variable**: hole counts/diameters of
  4x5 um, 4x10 um, 18x10 um and 36x10 um give measured corners of
  **2 Hz, 24 Hz, 106 Hz and 211 Hz** respectively. So a capsule *can* be built
  with its corner at 211 Hz.
- But the paper's own conclusion is the reason this is row 5 and not row 1:
  **"Environmental noise suppression of 5dB by acoustical high pass design is
  demonstrated"** -- against wind noise, i.e. broadband low-frequency, not
  against a narrowband hum 4x higher than the corner.
- Microphone noise is specified **"A-weighted and non weighted in the bandwidth
  of 10Hz to 10kHz"**.

**[read]** A-weighting is defined in IEC 61672:2003 and is used because "the ear
is less sensitive to low audio frequencies"
(<https://en.wikipedia.org/wiki/A-weighting>). **[derived]** A consequence for
this project: a spec written in dBA is close to blind in the 50-200 Hz band,
so vendor SNR numbers and published transformer/charger noise limits understate
low-frequency problems systematically. **[unverified]** the exact per-frequency
A-weighting values -- I did not read the IEC table in this session, so no
numbers are quoted.

### Contact and structure-borne microphones

**[read]** <https://en.wikipedia.org/wiki/Contact_microphone>: a piezoelectric
disc glued to a metal disc is the common form; contact mics are "almost
completely insensitive to air vibrations but transduce only structure-borne
sound", are passive and high-impedance, and "sound tinny without a matching
preamp".

**[derived]** the mechanism and its consequence for us: rejection of mains hum
is not a frequency-selective effect at all -- it is an *air-versus-structure*
discrimination, so it applies equally at 50 Hz and at 20 kHz, and it costs
nothing in speech bandwidth because the hum never entered the signal in the
first place. This is why it outranks every filter in the table for this
specific failure.

**[read]** ManiWAV (`real-stanford/maniwav`, README fetched from
<https://raw.githubusercontent.com/real-stanford/maniwav/main/README.md>)
mounts **two** contact microphones in the gripper CAD model
(`CE3S1_soft_finger_contact_mic.gcode`), held by **grip tape**, cabled to a
**GoPro media-mode external microphone port**. The microphone is listed only as
the Amazon ASIN **B08RBSQ7QD**.
**[unverified] what that ASIN actually is.** Amazon was not fetchable from this
environment and the part identity could not be confirmed; I am not going to
guess a part number, because an unverified mic part is exactly the kind of
number that later turns out to be the reason something does not work. What
*is* verified is the arrangement, and it is worth noting what it implies: two
contact sensors on a rigid gripper, summed or differenced, fed to a camera
mic input. That is a **structure-borne, mechanically-coupled, heavily
mechanically-isolated-from-the-room** signal chain -- it is not an
always-on-listen-to-the-room design, and its filtering is whatever the GoPro
does.

### Hum: conducted versus airborne is the whole question

**[read]** RaneNote 110, *Cause and prevention of ground loops*, written 1985,
revised 11/2015 (<https://www.ranecommercial.com/legacy/note110.html>):
"It is not simply the presence of this current that creates the hum -- it is
when this current flows through a unit's audio signal ground that creates the
hum", and balanced interconnect "was developed to be immune to these noise
currents, which can never be entirely eliminated". The note cites AES48-2005
for connector/shield grounding practice.

**[derived]** This is the single most decision-relevant fact in the document.
If the hum is conducted, then the mitigation list is row 1 (wiring, grounding,
isolation) and *nothing on the frequency axis helps at all* -- no capsule, no
filter, no notch. If the hum is airborne, then rows 2-4 are the whole game.
The project's current threat model (`experiments/hum_sweep.py`) uses synthetic
harmonic stacks at the ADC input and therefore **cannot distinguish the two**:
the same spectrum is produced by a radiating transformer and by an
unbalanced cable, and only one of them is a hardware problem.

### Bone conduction

**[read]** <https://en.wikipedia.org/wiki/Bone_conduction>: "Bones are generally
more effective at transmitting lower-frequency sounds compared to
higher-frequency sounds."
**[unverified]** the numbers that actually matter here -- the frequency
response of a commercial bone-conduction *transducer* or bone-conducted
*microphone*, and the crossover below which it stops responding. I did not
obtain a datasheet. The honest statement is directional: the skin/skull
coupling path and the actuator resonance both make BC weak at the bottom of the
band, which means it is **good at rejecting 50-200 Hz and bad at providing the
70-400 Hz that `src/pitch.py` tracks**. The brief asked whether the poor
low-frequency response makes bone conduction useless for hum rejection or
ideal for it: on the physics it is **ideal for rejection and simultaneously
unusable for the pitch feature this project is built around**. That trade, not
the rejection, is the objection.

## Recommended order of work for a Pi-class wearable

1. **Establish whether the hum is airborne or conducted before choosing a
   part.** Simultaneous capture of a contact mic and an airborne mic on the same
   suspected source, plus a null-test/differential capture, separates the two in
   one session. Every row in the table is conditional on the answer, and the
   rows that win under one answer lose under the other.
2. **Conducted-hum hygiene** (row 1): balanced/differential mic input, one
   ground point, the capsule physically away from DC-DC converters, inductors
   and the charging path. No frequency-selective compromise, so it costs
   nothing in speech and nothing in F0.
3. **Notch at the mains harmonics** (row 2) in the signal path: -21 dB at
   200 Hz for 0.24 cents median F0 error, measured on this repo's own tracker.
4. **Only then** consider spending money on a capsule (rows 3-5, 7) or on a
   lower self-noise part (row 6). None of them is a fix for a conducted hum,
   and the quietest available part is not a fix for anything here.

## What I could not verify

- **The ManiWAV contact mic part.** ASIN B08RBSQ7QD could not be resolved;
  Amazon is not fetchable from this environment. No part guessed.
- **Bone-conduction transducer/microphone frequency response.** No datasheet
  obtained; only the qualitative statement above is sourced.
- **Contact mic resonance frequencies.** The mechanism is sourced; the
  1-10 kHz resonance typical of piezo contact capsules is **[unverified]**.
- **Specific wearable MEMS part datasheets** (e.g. the common
  SPH0645HM4H-class parts). ST and TDK both refuse requests from this
  environment (timeouts and HTTP 403), so the only capsule numbers quoted here
  are from the Infineon paper. The 65 dBA figure in the brief is consistent
  with [derived] 28-33 dBA self-noise for the 61-66 dB(A) SNR class, but no
  part-level datasheet was read.
- **Psychophysical speech-intelligibility versus high-pass cut-off.** The
  ANSI S3.5 / ITU-T P.835 band-importance data is the right source and neither
  was retrievable in this session (ITU returned an error page). Per the
  measurement above this matters less than expected: **any cut-off low enough
  to preserve speech is far too low to touch 150-200 Hz**, so the
  speech-intelligibility question does not arise for this failure. It would
  arise immediately for the *notch* option, where the 150/200 Hz components of
  a genuine voice F0 are removed, and that cost is not yet characterised
  across speakers -- only across the frames of one 229 Hz narrator.
- **Whether the gate's scores change** when fed filtered versus unfiltered
  audio. This file loads no model; it reports only what the filter does to the
  signal. If the notch is adopted, the end-to-end gate behaviour still has to
  be measured by whoever owns the gate lane.
