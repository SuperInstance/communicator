# Audio engines — the learning journey

*2026-10-18. Six repos audited by reading source: SuperCollider, Sonic Pi, ZzFX,
Glicol, Pizzicato, PTS. This file is the diary — what I expected, what surprised me,
what I got wrong. The evidence is in `0N-*.md`; the cross-cutting result is in
`SYNTHESIS.md`.*

The brief said not to waste time hunting for fake claims, and it was right: all six are
real and mature. The instruction that earned its keep was **"focus on mechanism"**,
because the interesting part turned out to be almost entirely *how*, not *what*.

---

## What I expected, and the scorecard

| Expected | Verdict |
|---|---|
| A ring buffer in the big audio engine | **yes** — and it is 45 lines, which I did not expect |
| A scheduling model worth copying | **yes** — a fixed heap with a tiebreak counter I would never have invented |
| ZzFX to be a clever trick or two | **no** — it is a complete synth in 125 lines and every trick is load-bearing |
| Glicol to get the Rust real-time story right | **partly** — the Rust half is good, the JS host allocates in its own audio callback |
| Pizzicato to have interesting effects | **no** — it implements no DSP at all |
| Sonic Pi to teach something about perceived structure | **yes, but not where I looked** |
| At least one project I could vendor | **no** — the two best architectures are GPL and AGPL |

---

## What surprised me

### 1. The most valuable 45 lines in the entire audit are a ring buffer

`supercollider/common/MsgFifo.h` is 124 lines and contains two ring buffer variants. The
second, `MsgFifoNoFree`, is 33 lines.

I expected to find ring buffers, because that is what the brief asked about. I did not
expect them to be *this* small, and I did not expect the crucial design decision to be a
single line:

```cpp
if (next == mReadHead.load(memory_order_relaxed))
    return false;                       // fifo is full
```

**It returns false. It does not block.** That one line is why a real-time audio thread
can never be stalled by its producer. SuperCollider's stated backpressure policy is "drop
the newest event" — not stated anywhere as prose, just written as the natural thing to
do in the function. I read a lot of real-time systems code before this one and none of
them said it this plainly.

I also did not know the three-head trick until I saw `mFreeHead` (`MsgFifo.h:34-80`):
letting the *consumer* free the messages means deallocation happens on the thread that
can afford it, and the producer never blocks waiting for memory.

### 2. Sonic Pi's C++ audio server does not exist any more

The brief described Sonic Pi as "built on SuperCollider" and asked how it achieves
low-latency operation. I went looking for `app/server/sonic-pi-server/` — the C++
`sequencer.cpp` / `player.cpp` that every article about Sonic Pi discusses.

`app/server/native/` contains one file: `.gitkeep`. It is gone from **both** `dev` and
`stable`. The tree now has `.gitmodules` pointing at
`https://github.com/samaaron/supersonic`, and `LICENSE.md:48-52` says SuperSonic and
Clockwork are AGPL.

I checked twice, because this is exactly the kind of thing I would have mis-reported
from a single lookup. It is real, and it is out of my assigned repo list, so I did not
audit it. It is the largest gap in this work and it is flagged in `SYNTHESIS.md`.

### 3. The new scheduler is AGPL — so the best architecture is the one we cannot use

Even after the C++ server left, the replacement is fascinating. `app/web/runtime/lib/
sonic_pi/scheduler.rb` opens with:

> Sonic Pi's threads without threads: every thread is a fiber with its own logical clock,
> and one scheduler runs whichever is due next. Nothing waits on a wall clock; a sleep is
> a jump in logical time, so **a whole program runs in an instant and its trace is the
> same every time.**

And `LICENSE.md:35-41` says `app/web/` is AGPL-3.0-or-later. The MIT parts are the
*older* Ruby runtime. So the interesting new architecture is exactly the unusable one.

GraphQL reported `licenseInfo.spdxId = NOASSERTION` for Sonic Pi. That is technically
correct and practically useless — it did not tell me the repo has three different licenses
across three directories. **Do not trust the API's license field for a composite repo.**

### 4. ZzFX is not "a trick or two". It is a real synth, and `sin(t³)` is noise

I expected `ZzFX.js` to be a curiosity — a few clever hacks in a tiny file. It is
10.1 KB including a 44-line licence header, and it contains 21 parameters covering
oscillators, 6 wave shapes, ADSR with a decaying sustain stage, slide, delta-slide, pitch
jump, repeat, vibrato via tremolo, noise, ring modulation, bitcrush, a feedback delay
with a tail, and an LP/HP biquad. All of it in one `for` loop.

The specific surprises:

- **`Math.sin(t**3)` is the noise oscillator** (`:178`). Cubing the phase before the sine.
  Chaotic, deterministic, no RNG, no state, two multiplications. I would not have thought
  of it and I could not have invented it.
- **`i*i*PI2%2-1` is a noise source** (`:208`). A reproducible triangular wave in [-1,1]
  derived from the sample counter. No RNG in the inner loop *at all*.
- **The feedback delay divides by `volume`** (`:200`) to undo the gain applied on write
  (`:172`). Miss that and the reverb changes character when you change the master volume.
  One token, and it is the entire difference between correct and subtly broken.
- **`attack = attack * sampleRate || minAttack`** (`:156`). A NaN guard written as a
  defaulting expression. `attack = 0` would make `i/attack` evaluate `0/0` and poison the
  whole buffer.

I also found that `ZzFXMicro.js` is the *same loop, character for character* — only the
I/O differs. That told me the DSP core is cleanly separable from delivery, which is
exactly the shape we can test on WAV files with no audio device.

### 5. Glicol allocates in its own audio callback

The Rust half is genuinely well done: a `StableGraph` so node indices survive live-coding
edits, `max_nodes: 1024` declared in one place, and — the bit I liked most — a
**persistent `DfsPostOrder`** so the graph traversal does not allocate
(`rs/synth/src/graph.rs:41,64-67`). Most graph engines allocate a traversal object per
pass. Glicol does not.

Then I read the JS host. `js/npm/glicol-engine.js:120-175`:

```js
this._inPtr = this._wasm.exports.alloc(128)        // :143
this._inBuf = new Float32Array(...)                // :144
...
outputs[0][0].set(this._outBuf.slice(0, 128))      // :172
outputs[0][1].set(this._outBuf.slice(128, 256))    // :173
```

Six allocations and four `slice()` copies **every 128 frames**, in an `AudioWorkletProcessor`,
in the audio thread. And `alloc(128)` at `:143` is never freed — roughly 124 million
unreachable floats of wasm memory per hour of playback.

This is the finding I am most confident about and it is also the one I should be most
careful with: **I did not measure it.** It is a direct reading of the source. But it
contradicts the surrounding architecture's entire purpose, and it contradicts the Rust
half's own discipline. A stale repo (no commits since April 2025) whose CI runs clippy and
rustfmt but never `cargo test` is exactly the profile of code that rots here.

### 6. Glicol's sequencer is O(events × samples) and silently mis-times

```rust
for out in &mut *output[0] {
    for event in &self.events {
        if (self.step % (bar_length as usize)) == ((event.0 * bar_length) as usize) {
```

`rs/synth/src/node/sequencer/seq.rs:45-65`. For every sample, every event, two integer
modulos. And `(event.0 * bar_length) as usize` **truncates**, so an event that does not
land on a whole sample after scaling either never fires or fires at the wrong moment.

There is no event queue and no absolute-time comparison anywhere in glicol. It achieves
"sample accuracy" by brute force, and the brute force does not scale and does not round
correctly. **This is the clearest mature-project-still-wrong result in the audit**, and
it is the direct argument for mechanism #2 in `SYNTHESIS.md`.

### 7. Pizzicato implements no DSP whatsoever

I went in expecting to find clever effects. I ran an exhaustive scan of every WebAudio
factory call across all 13 effects:

```
Compressor    → createDynamicsCompressor, createGain
Distortion    → createWaveShaper
Reverb        → createBuffer, createConvolver, createGain
RingModulator → createDynamicsCompressor, createGain, createOscillator, createWaveShaper
... (all 13 in 05-pizzicato.md)
```

Every effect wraps a browser-provided node. The entire DSP content is four micro-techniques
— a rational-`tanh` waveshaper curve, an 8-line synthetic impulse response, a cascaded
diode ladder built from N stacked `WaveShaperNode`s, and a **two-element `Float32Array`
curve used as a gain** because `WaveShaperNode` has no `AudioParam`.

That last one is delightful and I would not have found it without reading. But the honest
conclusion is that Pizzicato's value is ergonomics, and it is worth roughly 44 KB of
MIT-licensed sugar.

### 8. Its test suite is real, well-proportioned, and never runs

23 test files for 44 KB of source — including a 97 KB audio fixture and a 7.8 KB
`Compressor.test.js`. That is a better test-to-source ratio than anything else in this
audit.

And `.travis.yml` is `dist: trusty`, `node_js: "12"` — Ubuntu 14.04 and Node 12, both
long EOL — downloading Chrome from a URL that no longer resolves. **A suite that looks
like a safety net and is not is worse than no suite**, because it changes how you read
the README.

### 9. PTS's value was not the graphics. It was the seed string.

The brief said PTS was "only relevant for its generative/noise patterns and its approach
to compact self-contained systems". That undersold it. The thing I actually want from PTS
has nothing to do with drawing:

```typescript
static seed(seed: string): void { this.generator = generator(seed); }
static random(): number { return this.generator ? this.generator.random() : Math.random(); }
```

`pts/src/Num.ts:197,207`. **A seeded, reproducible PRNG, keyed by a string.** Every
mechanism else in this audit is about real-time plumbing; this one is about being able to
debug a stochastic system. We *want* randomised parameter response — that is what makes
the peripheral feel alive. Without one seed that replays the whole stochastic behaviour we
cannot answer "why did it do that?", regression-test it, diff two runs, or reproduce a
bug report.

The generator is **public domain** (Gibson Research UHEPRNG, Marsaglia MWC over 48 words,
seeded through Baagoe's `Mash`). The only unconditionally reusable code in the entire
audit.

And the bit PTS added is small and good (`src/uheprng.ts:110-119`): it normalises the
seed before hashing, so `"run\n"` and `"run"` collide. Seeding from a filename with a
trailing newline and getting a different run every time is a real class of
non-reproducible-run bug, and they closed it and documented it.

### 10. The three-clock-rate model is the answer to a question we had not asked well

`supercollider/server/scsynth/SC_Rate.cpp:33`:

```cpp
inRate->mBufRate = inSampleRate / inBufLength;
```

SuperCollider declares every UGen at sample rate, control rate (~750 Hz), or audio rate.
It has *always* had the answer to "how do I measure cheaply and continuously" — **you do
not run the analysis at the sample rate.** YIN pitch tracking at 48 kHz is not
"continuous measurement", it is 48,000 measurements per second of which ~47,900 are
redundant. Run it at control rate and interpolate the contour.

I did not go looking for this. It fell out of reading the scheduling code.

### 11. `step()` returning `nexts.min` is TIMING.md in code form

```ruby
nexts = @queue.map { |(wake, _, p)| p.start + wake } +
        @fx_frees.map { |(at, _)| at - FREE_LEAD } +
        @node_frees.map { |(at, _)| at - FREE_LEAD }
nexts.min
```

`sonic-pi/app/web/runtime/lib/sonic_pi/scheduler.rb:417`. The scheduler takes the min over
three deadline sources and returns it, so the host sleeps exactly that long. Its own
comment says: "The scheduler never sleeps itself; the host owns the one real wait."

This is our own **T-minus law** (`docs/research/TIMING.md` §2) expressed as code. We wrote
it as a principle about compensating for downstream latency. Sonic Pi implements it as a
function that returns its own next deadline. Ours almost certainly polls on a fixed timer
and burns CPU discovering it has nothing to do.

The genuinely surprising framing is that Sonic Pi's `EventHistory` (`:190-203`) is the
same idea again in a different place: "get the last value at or before t" and "get the
next value after t". That is *our score*, queryable. MIT, so liftable.

---

## What I got wrong

**I misread the CI situation for three repos and nearly published it.**
My first pass searched for `.github/workflows/` with a pattern that did not match the
nested paths, and concluded SuperCollider, PTS and Glicol had **no CI at all**. That was
wrong. All three have workflows; SuperCollider has
`test_linux.yml`/`test_macos.yml`/`test_windows.yml` plus five build workflows, and PTS's
is the best in the audit. I only caught it because the numbers looked wrong — SuperCollider
with 245 test files and no CI is not a credible combination for a 20-year-old project.
Lesson: when a finding contradicts a well-established project's reputation, suspect the
search, not the project.

**I nearly concluded Sonic Pi's C++ server was merely relocated rather than removed.**
The first `main`/`master` branch probe I ran returned 200 for branches that do not exist
in the repo (`gh branches` lists only `dev`, `stable`, `translations`). I trusted raw
fetch results over the authoritative branch list. Listing branches properly took one call
and settled it.

**I was wrong about what Pizzicato's `createScriptProcessor` was for** until I read
`Sound.js:186-196`. It is not an effect — it is `getRawSourceNode`, an escape hatch that
exposes the deprecated main-thread node as a supported feature. Worse than if it had been
an oversight.

**I under-read the Sonic Pi license.** I saw `NOASSERTION` from GraphQL and moved on. It
is a three-part composite license (MIT for `app/`, AGPL for `app/web/`, AGPL for
SuperSonic/Clockwork, MIT-with-GPL-exceptions for synthdefs). The single most important
fact about reuse in this whole audit was sitting in a file I had opened and skimmed.

**I nearly over-claimed the glicol traversal order.** `Reversed(&graph)` + DFS post-order
from the destination means consumers are emitted before producers, which implies one block
of latency on acyclic chains. I found no test that pins it and I did not run it. It is
in the write-up explicitly marked as inference, and it should stay that way.

**I expected PTS to have a noise implementation and it does not.** There is no noise
function in the numeric core. The `uheprng.ts` header even has a careful port note
explaining that the nondeterministic startup path was *removed*. PTS does not do noise;
it does reproducible randomness. The brief's framing was close but not quite right.

---

## The through-line

Six projects, four licences, ~60 MB of source, one conclusion.

**Every mechanism worth taking is 30 to 60 lines, and not one of them is a framework.**

The ring buffer is 33. The event heap is 40. `step()` returning a deadline is one line.
The bounded history store is 40. The seeded PRNG is 30. What surrounds them — the
1266-line `SC_World.cpp`, the 1687-line `SC_SequencedCommand.cpp`, the 2113-line
`runtime.rb` coexisting with a 1708-line `scheduler.rb`, the 92 KB flat `Op.ts`, the
1024-node graph engine — is the accumulated weight of supporting every case a
general-purpose public tool must support. None of it transfers.

And the licensing is hostile: the two best architectures are GPL-3.0 and AGPL-3.0, so
**we were never going to vendor anything from the top of this list anyway.** That
settles the question. We write the 60 lines ourselves, in our own idiom, against our own
needs — and the reason we can defend every line of them is that they are small enough
and standard enough that we can.

The one thing I would genuinely want to read is the thing I could not:
`samaaron/supersonic`. The real-time scheduling engine left this repository, went AGPL,
and is now a submodule. If anyone picks this work up again, that is where the next
30 minutes should go.
