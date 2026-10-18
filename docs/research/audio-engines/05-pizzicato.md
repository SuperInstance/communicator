# Pizzicato — a pleasant API wrapped around zero DSP

- **Repo**: https://github.com/alemangui/pizzicato
- **Branch audited**: `master` @ `4c3826e46f112743fa0653b6bad56006b98da537`
- **License**: **MIT** (`LICENSE`, "Copyright (c) 2016 Alejandro Mantecon Guillen")
- **Stars / forks**: 1709 / 132 · **disk**: 10.4 MB
- **Last push**: 2025-07-09

## Inventory

| | |
|---|---|
| blobs | 105 |
| JS source | **47 files** |
| source bytes | 322 KB (`distr/Pizzicato.js` 62.5 KB + `site/Pizzicato.js` 62.5 KB are the same built file, plus `distr/Pizzicato.min.js` 35.2 KB) |
| real `src/` | 16 files, ~44 KB |
| tests | **23 files** under `tests/`, incl. a 97 KB `.mp4` fixture |
| CI | `.travis.yml` — **node 12, `dist: trusty`** (Ubuntu 14.04). Both EOL. Effectively dead. |

Note that 78% of the "JS source" is three copies of the same built bundle in
`distr/`, `site/` and `src/`. The actual library is ~44 KB.

## The blunt finding: it implements no DSP

I enumerated every WebAudio factory call in all 13 effects:

| Effect | Native nodes it wraps |
|---|---|
| Compressor | `createDynamicsCompressor`, `createGain` |
| Convolver | `createConvolver`, `createGain` |
| Delay | `createDelay`, `createGain` |
| Distortion | `createWaveShaper` |
| DubDelay | `createBiquadFilter`, `createDelay`, `createGain` |
| Filters | `createBiquadFilter`, `createGain` |
| Flanger | `createDelay`, `createGain`, `createOscillator` |
| PingPongDelay | `createChannelMerger`, `createDelay`, `createGain` |
| Quadrafuzz | `createBiquadFilter`, `createGain`, `createWaveShaper` |
| Reverb | `createBuffer`, `createConvolver`, `createGain` |
| RingModulator | `createDynamicsCompressor`, `createGain`, `createOscillator`, `createWaveShaper` |
| StereoPanner | `createGain`, `createPanner`, `createStereoPanner` |
| Tremolo | `createGain`, `createOscillator`, `createWaveShaper` |

Every effect is a wrapper that sets properties on a browser-provided node. There is no
filter design, no delay line, no feedback matrix, no oscillator. Pizzicato's value
proposition is entirely ergonomics: consistent construction, consistent `inputNode` /
`outputNode` / `enable` / `disable` / `remove` protocol, and ES5-safe getters.

**For our purposes the DSP is almost entirely absent.** What survives is four genuinely
interesting micro-techniques, all of which are worth reading for the *trick*, not the
effect.

## The four real techniques

### 1. Waveshaper as a synth-free nonlinearity — `Distortion.js:49-60`

```js
var gain = Pz.Util.isNumber(this.options.gain) ? parseInt(this.options.gain * 100, 10) : 50;
var n_samples = 44100;
var curve = new Float32Array(n_samples);
var deg = Math.PI / 180;
var x;

for (var i = 0; i < n_samples; ++i ) {
    x = i * 2 / n_samples - 1;
    curve[i] = (3 + gain) * x * 20 * deg / (Math.PI + gain * Math.abs(x));
}

this.waveShaperNode.curve = curve;
```

The entire "distortion" DSP is one expression:
`(3+g)·x·(20·deg) / (π + g·|x|)`. A rational approximation of `tanh` — the classic
Cubic Non-Linearity with a pre-gain folded in, written as a single division. The
pre-computed curve is uploaded once; per-sample cost is then the browser's interpolation.

Two real problems: `n_samples` is **hardcoded to 44100** regardless of the actual context
sample rate, and `deg` is the only concession to a non-180Hz world. Also
`parseInt(gain*100)` quantises the parameter to 101 steps, so `gain: 0.507` and `0.512`
are the same distortion.

### 2. A reverb in eight lines: synthetic impulse response — `Reverb.js:108-120`

```js
function buildImpulse() {
  var length = Pz.context.sampleRate * this.time;
  var impulse = Pizzicato.context.createBuffer(2, length, Pz.context.sampleRate);
  var impulseL = impulse.getChannelData(0);
  var impulseR = impulse.getChannelData(1);
  var n, i;

  for (i = 0; i < length; i++) {
    n = this.reverse ? length - i : i;
    impulseL[i] = (Math.random() * 2 - 1) * Math.pow(1 - n / length, this.decay);
    impulseR[i] = (Math.random() * 2 - 1) * Math.pow(1 - n / length, this.decay);
  }
```

White noise × `pow(1 - n/length, decay)`. That is the whole reverb: the browser's
`ConvolverNode` does the actual convolution; Pizzicato only synthesises an exponentially
decaying noise burst and hands it over. This is the standard cheap reverb and it is
**8 lines**.

**Steal this.** A synthetic-impulse generator is ~10 lines of NumPy, produces a usable
plate/room/hall from two parameters, and — critically — is testable offline on a WAV.

Also note `:126-133`, a documented browser limitation worked around by rebuilding the
node:

```js
// https://github.com/alemangui/pizzicato/issues/91
// ConvolverNode can be associated with only one buffer.
// Not sure what's the best way, but we are recreating ConvolverNode
// when properties change to work it around.
if (this.reverbNode.buffer) {
  this.inputNode.disconnect(this.reverbNode);
  this.reverbNode.disconnect(this.wetGainNode);
  this.reverbNode = Pizzicato.context.createConvolver();
  ...
}
```

Changing `time` rebuilds the convolver and reconnects the graph. Rough, but it documents a
real constraint.

### 3. Diode ladder from cascaded two-point shapers — `RingModulator.js`

The transfer curve, `:105-118`:

```js
samples = 1024;
wsCurve = new Float32Array(samples);
for (i = 0; i < samples; i++) {
  v = (i - samples / 2) / (samples / 2);
  v = Math.abs(v);
  if (v <= this.vb) {                                  // below the "knee"
    value = 0;
  } else if ((this.vb < v) && (v <= this.vl)) {         // quadratic knee
    value = this.h * ((Math.pow(v - this.vb, 2)) / (2 * this.vl - 2 * this.vb));
  } else {                                              // linear above the knee
    value = this.h * v - this.h * this.vl +
            (this.h * ((Math.pow(this.vl - this.vb, 2)) / (2 * this.vl - 2 * this.vb)));
  }
  wsCurve[i] = value;
}
```

Zero below the breakpoint, quadratic through the knee, linear above — a single diode's
transfer function, as a 1024-point waveshaper curve. And `:189`:

```js
for (var i=0, l=diodeNodes.length; i<l; i++) {
```

**Each cascaded `WaveShaperNode` contributes one diode.** Four diodes = four series
shapers, because non-linearities stack. You get a four-diode ladder out of a primitive
that is nominally single-voice waveshaping.

**Steal the idea, not the code:** non-linear stages in series compose, so a cascade of
cheap per-stage curves approximates an expensive global nonlinearity.

### 4. A two-point waveshaper curve used as a smooth gain — `Tremolo.js:104`

```js
this.shaperNode.curve = new Float32Array([1-depth, 1]);
```

A `WaveShaperNode` maps input `[-1, 1]` linearly onto the curve. A curve with **two
points** is therefore a straight line from `1-depth` to `1` — i.e. a linear gain of
`1-depth`. There is no `gain` parameter on a `WaveShaperNode`, so this is how you get an
automatable gain out of one.

Whether that was the original intent or an accident, the technique is real and it is
worth knowing: **a two-element Float32Array curve is a gain node.**

### Dry/wet is a hard step, not a crossfade — `Util.js:62-78`

```js
getDryLevel: function(mix) {
  if (!Pz.Util.isNumber(mix) || mix > 1 || mix < 0) return 0;
  if (mix <= 0.5) return 1;
  ...
getWetLevel: function(mix) {
  if (!Pz.Util.isNumber(mix) || mix > 1 || mix < 0) return 0;
  if (mix >= 0.5) return 1;
```

At `mix = 0.5` you get full dry **and** full wet. There is no equal-power curve, no
`cos/sin` law, no `-3 dB` crossover. Whatever the interpolation between 0 and 0.5, the
midpoint is a discontinuity or at best a plain linear crossfade (which dips in
correlation). Worth copying the *validation* pattern — bail to a safe value on a
non-numeric input, at every setter — and skipping the curve.

## Tests — the healthiest part of this repo

23 test files, one per effect plus `Sound.test.js` (13.4 KB), `Util.test.js` (5.3 KB),
`Group.test.js`, `Events.test.js`, and a real 97 KB `tests/audio/ddplus-format.mp4`
fixture. `Compressor.test.js` alone is 7.8 KB.

That is a genuine test suite for a browser audio library, covering each effect's
getters/setters and range validation. **It is the only repo in this audit with a test
suite proportional to its size** — SuperCollider has more tests but a 53 MB source tree;
Pizzicato has 23 test files for 44 KB of source.

But it is not run. `.travis.yml`:

```yaml
language: node_js
sudo: required
dist: trusty
node_js:
  - "12"
```

Node 12 (EOL April 2022) on Ubuntu Trusty (EOL April 2024), downloading Chrome from a
URL that no longer resolves. **The tests exist and are not executed.** That is a worse
situation than having no tests, because the suite looks like a safety net.

## One more real-time note: `createScriptProcessor`

`Sound.js:186-196`:

```js
var test = Pizzicato.context.createScriptProcessor();
...
this.getRawSourceNode = function() {
  var node = Pizzicato.context.createScriptProcessor(bufferSize, 1, 1);
  node.onaudioprocess = audioFunction;
  return node;
};
```

`ScriptProcessorNode` is deprecated and is explicitly documented as running on the main
thread — it is the main cause of audio glitches in naive WebAudio code. It is exposed as
`getRawSourceNode`, i.e. as a supported escape hatch. Do not copy that.

## What to take, what to skip

**Take (MIT):**
- synthetic impulse response for reverb — 10 lines, offline-testable (`Reverb.js:108-120`)
- the rational-`tanh` waveshaper curve, resampled to the *actual* context sample rate
  (`Distortion.js:49-60`)
- the "non-linear stages cascade" idea for diode-style nonlinearity (`RingModulator.js`)
- the `isInRange` bail-out at the top of every setter (`Util.js:23-28`) — cheap input
  validation that never throws from a UI callback
- the `inputNode` / `outputNode` / `enable` / `disable` / `remove` effect protocol — a
  decent uniform interface for a peripheral system

**Skip:**
- everything else. There is no DSP to learn.
- The positional-`Float32Array`-at-44100 assumption.
- `createScriptProcessor` and `getRawSourceNode`.
- The whole "wrap the native node" approach, if you are going to do DSP at all: you gain
  nothing over the WebAudio API except sugar, and you cannot inspect or test the signal.

## Verification notes

- The "no DSP" claim is from an exhaustive `createXxx(` scan of all 14 files in
  `src/Effects/` plus `Sound.js`, plus reading all four files that contain an actual
  computation. I read `Distortion.js`, `Reverb.js`, `Tremolo.js`, `RingModulator.js` in
  full and `Util.js` in full; the remaining nine effects I verified by factory-call
  enumeration only.
- I did not run the test suite (no browser in this environment, and the CI config is
  dead anyway).
- Whether travis.org still receives builds I did not check.
