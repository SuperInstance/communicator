# ZzFX — a complete synth in 125 lines, and what makes it work

- **Repo**: https://github.com/KilledByAPixel/ZzFX
- **Branch audited**: `master` @ `aab7e2b6b9086746b6e55fab75c75ce03e716c49`
- **License**: **MIT** (`LICENSE`, "Copyright (c) 2019 Frank Force")
- **Stars / forks**: 803 / 50 · **disk**: 1.5 MB
- **Last push**: 2026-10-01

## Inventory — the entire repository

| | |
|---|---|
| blobs | **15** |
| JS source | **4 files**: `ZzFX.js` (10.1 KB), `ZzFXMicro.js` (5.6 KB), `ZzFXMicro.min.js` (1.2 KB), `wav.js` (2.3 KB) |
| markdown | 1 |
| tests | **0** |
| CI | **none** (`.github/` contains only `FUNDING.yml`) |
| vendored deps | **none** |

Of that 10.1 KB, ~44 lines are the MIT licence header. The entire synthesiser is the
125-line `buildSamples` function. Read it as a piece of writing, not as code.

## The architecture: render it all, then hand it over

There is no audio graph. `buildSamples` synthesises the **entire sound** into a single
pre-allocated `Float32Array`, which is then handed to WebAudio as a finished buffer
(`ZzFX.js:77-95`). No nodes, no per-sample callbacks, no graph to traverse. The WebAudio
part is pure delivery.

```js
// allocate the full sample buffer up front, much faster than growing an array
length = attack + decay + sustain + release + delay | 0;   // :168
b = new Float32Array(length > 0 ? length : 0);             // :169
```

**Why this matters for us:** it makes ZzFX trivially testable with **no audio hardware at
all**. `buildSamples` is a pure function `params -> Float32Array`. You can call it in
Node, assert on the samples, and you have a regression test suite for a synthesiser in
about ten lines. Our WSL environment has no audio device; this is the shape of DSP that
is testable anyway.

## The primitives — and how each one is compressed

### 1. Six wave shapes in a nested ternary, one of which is *noise*

`ZzFX.js:176-182`:

```js
s = shape? shape>1? shape>2? shape>3? shape>4? // wave shape
    (t/PI2%1 < shapeCurve/2? 1 : -1) :         // 5 square duty
    Math.sin(t**3) :                           // 4 noise
    Math.max(Math.min(Math.tan(t),1),-1):      // 3 tan
    1-(2*t/PI2%2+2)%2:                         // 2 saw
    1-4*abs(Math.round(t/PI2)-t/PI2):          // 1 triangle
    Math.sin(t);                               // 0 sin
```

- **tan is clamped, not approximated**: `max(min(tan(t),1),-1)` — the exactly-clipped
  ramp. One call, correct behaviour, no series expansion.
- **square is a phase comparison**: `t/PI2%1 < shapeCurve/2 ? 1 : -1`. `shapeCurve`
  becomes the **duty cycle**. PWM from a modulo and a compare.
- **triangle is a folded distance to the nearest integer**: `1-4*|round(t/PI2) - t/PI2|`.
  Distance-to-nearest-integer folded and scaled — correct triangle with no branches.
- **noise is `sin(t³)`.** Cubing the oscillator phase before the sine is a chaotic
  function: the phase advance is already irregular because the frequency is being
  modulated, so cubing it decorrelates successive samples. It produces something
  noise-*like* from pure phase, deterministically, with no RNG and no state. Two
  multiplications.

### 2. Noise as a function of the sample index — no RNG in the inner loop

`ZzFX.js:208`:

```js
t += f + f*noise*(i*i*PI2%2-1);         // noise
```

`i*i*PI2 % 2` cycles through `[0,2)` and `-1` maps it to a deterministic triangular
waveform in `[-1, 1]`. The `noise` parameter is the depth of frequency perturbation. So
"noise" is a reproducible, allocation-free, branch-free modulation signal derived from
the sample counter.

**This is the single most immediately stealable idea here for our purposes.** If you need
a deterministic test signal — dither, a noise burst, a stress input for a pitch tracker —
that is reproducible from the sample index and needs no RNG state. It also means two
runs of the same parameter set produce bit-identical output, which makes a regression
test meaningful.

### 3. Bitcrush as a duty cycle on the whole update

`ZzFX.js:174`:

```js
if (!(++crush%(bitCrush*100|0)))                   // bit crush
{
    ... everything below, including oscillator advance ...
}
```

Bitcrushing is normally "round to N bits". Here it is implemented as: **only recompute the
sample every Nth sample; in between, hold.** That single modulo gate gives you
sample-and-hold, reduces the effective update rate (its own kind of coarsening), and
skips the biquad and the envelope evaluation entirely on held samples. Six DSP features
fused into one integer test, and it makes the inner loop *cheaper* at high crush.

Note also the consequence: the oscillator phase `t` does **not** advance while held
(`t += ...` is inside the gate). That is why high bitcrush settings lower the perceived
pitch. Probably unintended, definitely worth knowing if you lift this.

### 4. LP and HP from one coefficient set via a sign flip

`ZzFX.js:146-152`:

```js
// biquad LP/HP filter
quality = 2, w = PI2 * abs(filter) * 2 / sampleRate,
cos = Math.cos(w), alpha = Math.sin(w) / 2 / quality,
a0 = 1 + alpha, a1 = -2*cos / a0, a2 = (1 - alpha) / a0,
b0 = (1 + sign(filter) * cos) / 2 / a0,
b1 = -(sign(filter) + cos) / a0, b2 = b0,
x2 = 0, x1 = 0, y2 = 0, y1 = 0;
```

with `sign = v => v<0?-1:1` at `:131`. The sign of the filter parameter selects
high-pass vs low-pass; `abs(filter)` is the cutoff. One parameter, both filters.
`quality = 2` (i.e. Q = 0.707) is hardcoded — the library does not expose Q at all, which
is a real limitation, not a choice.

And the whole Direct-Form-II transposed section is one line (`:202-203`):

```js
if (filter)
    s = y1 = b2*x2 + b1*(x2=x1) + b0*(x1=s) - a2*y2 - a1*(y2=y1);
```

The assignments are sequenced through the state variables so `x1`/`x2`/`y1`/`y2` are
read and written in the right order without temporaries.

### 5. Reverb as a 3-token feedback delay reading its own output buffer

`ZzFX.js:198-200`:

```js
s = delay ? s/2 + (delay > i ? 0 :
    (i<length-delay? 1 : (length-i)/delay) *
    b[i-delay|0]/2/volume) : s;              // sample delay
```

A feedback delay line read out of `b` — the array already written earlier in the same
loop — with a fade-in and a fade-out tail. No extra buffer.

**The subtlety that makes it correct:** `b[i]` was written as `s * volume` (`:172`), so
the feedback term must be divided by `volume` to undo that gain. Otherwise the loop gain
would scale with the master volume and the reverb would change character when you turn the
volume down. One `/volume` token, and it is the difference between a reverb and a
bug. Note the fixed `/2` on both terms — the loop gain is hardcoded at 0.5 total, so
this cannot self-oscillate. There is **no feedback parameter**; you cannot build a
reverb longer than the buffer.

### 6. The NaN guard that is not a guard

`ZzFX.js:155-156`:

```js
const minAttack = 9; // prevent pop if attack is 0
attack = attack * sampleRate || minAttack;
```

`attack = 0` would make `i/attack` (`:188`) evaluate `0/0 = NaN` and poison the whole
buffer. `|| minAttack` catches it — for a cost of 9 samples of attack at 44.1 kHz, which
is ~0.2 ms, inaudible, and prevents the click that zero attack would otherwise cause.
Written as a defaulting expression rather than an `if`. This is the kind of thing that
takes an afternoon to find the hard way.

### 7. The full ADSR, in one chained conditional

`ZzFX.js:188-196`:

```js
(i < attack ? i/attack :                 // attack
i < attack + decay ?                     // decay
1-((i-attack)/decay)*(1-sustainVolume) : // decay falloff
i < attack  + decay + sustain ?          // sustain
sustainVolume :                          // sustain volume
i < length - delay ?                     // release
(length - i - delay)/release *           // release falloff
sustainVolume :                          // release volume
0);                                      // post release
```

A five-stage envelope as a single nested ternary, each branch annotated with which stage
it is. Also supports a decaying "sustain" segment (`sustainVolume` below 1), so it is a
five-stage envelope in the classic sense, not the three-stage ADSR.

### 8. Everything scaled by sample rate, once, up front

`ZzFX.js:154-165`:

```js
attack *= sampleRate; decay *= sampleRate; sustain *= sampleRate;
release *= sampleRate; delay *= sampleRate;
deltaSlide *= 500 * PI2 / sampleRate**3;
modulation *= PI2 / sampleRate;
pitchJump *= PI2 / sampleRate;
pitchJumpTime *= sampleRate;
repeatTime = repeatTime * sampleRate | 0;
```

All user-facing parameters are in seconds or Hz. They are converted to per-sample
increments **once**, before the loop, so the inner loop contains no division. The
`sampleRate**3` on `deltaSlide` is because slide is a second derivative of frequency —
slide of slide. Getting that wrong by one power of sample rate is the classic
portability bug; here it is correct.

## ZzFXMicro — proof the DSP is separable from the delivery

`ZzFXMicro.js:88-140` is the **same loop, character for character** as `ZzFX.js:172-223`.
The only differences are the I/O: `b = []` (a plain JS array, `:55`) instead of a
`Float32Array`, and a `createBufferSource()` created inline at `:62` instead of a
`playSamples` call. No panner, no gain node.

This is a useful structural fact: **the DSP core and the audio delivery are fully
separable in ~60 lines.** When you lift a ZzFX-style kernel into Python for WAV-file
testing, you are lifting precisely that separable part.

It is also, bluntly, a code smell — the same 60 lines pasted twice with no shared module
and no test to catch divergence.

## Parameter list, for the record

`ZzFX.js:104-124`, in call order (this order **is** the API — ZzFX sounds are positional
arrays, so appending is safe and reordering is not):

`volume, randomness, frequency, attack, sustain, release, shape, shapeCurve, slide,
deltaSlide, pitchJump, pitchJumpTime, repeatTime, noise, modulation, bitCrush, delay,
sustainVolume, decay, tremolo, filter`

`getNote(semitoneOffset, rootNoteFrequency=440)` at `:229-232` is `440 * 2**(n/12)` —
note-naming kept separate from synthesis.

## What to take, what to skip

**Take (MIT, copy freely):**
- `i*i*PI2%2 - 1` as a deterministic, RNG-free, reproducible noise source from the sample
  index (`:208`)
- `sin(t**3)` as a noise-shaped oscillator from pure phase (`:178`)
- one-parameter LP/HP via `sign(filter)` (`:147-152`)
- `attack * sampleRate || minAttack` as a NaN guard written as a default (`:156`)
- the up-front buffer allocation + "render the whole thing then hand it over" architecture
  (`:167-169`), which makes the synth testable with no audio device
- `ZZFXSound` (`:237-259`): pre-render with `randomness = 0`, then re-apply randomness
  as a **playbackRate** offset. Variation without re-synthesis.

**Skip:**
- The positional-parameter API. With 21 positional parameters you will ship a bug you
  cannot see. Use keyword arguments.
- `Math.random()` in `buildSamples` for the frequency jitter (`:134`) — it makes output
  irreproducible, which defeats the point of a pure function.
- Bitcrush-as-sample-hold, unless you *want* the pitch drop that comes with it.

## Verification notes

- No tests, no CI, no issues infrastructure beyond a funding file. This is a
  demonstrator, not a maintained library.
- I did not run `buildSamples` — no Node/jsdom check was in scope and there is no test
  harness. Every claim above is from reading `master` at the commit above. The DSP claims
  (formulas, control flow) are verifiable by reading; the *audible* quality claims are not.
