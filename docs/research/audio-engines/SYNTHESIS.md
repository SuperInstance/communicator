# Audio engines — cross-cutting synthesis

*2026-10-18. Six repos: SuperCollider, Sonic Pi, ZzFX, Glicol, Pizzicato, PTS.
Read the per-repo files for full evidence; this is the layer that survives all six.*

Method note: every line reference below is from the default branch at the commit
recorded in each per-repo file. Nothing here was executed — the WSL box has no audio
device, so claims about *audible* or *measured* behaviour are explicitly marked as
unverified. Claims about *code structure* are directly readable.

## Reuse posture (this is the first filter, not the last)

`BORROWING.md` already established: **AGPL = ideas only, never code.** That rule does
most of the work here.

| repo | license | verdict for us |
|---|---|---|
| supercollider | **GPL-3.0** | ideas only. Read + reimplement. |
| sonic-pi `app/` | MIT | **copyable** (`runtime.rb`, `event_history.rb`, `clockwork_arena.rb`) |
| sonic-pi `app/web/` | **AGPL-3.0** | ideas only. **The best scheduler in the audit is here.** |
| sonic-pi supersonic/clockwork | **AGPL-3.0** | out-of-tree submodule; not audited |
| ZzFX | **MIT** | **copyable** |
| glicol | MIT | **copyable** (except vendored `ringbuf.js`, **MPL-2.0**) |
| pizzicato | **MIT** | copyable, but there is almost nothing to copy |
| pts | **Apache-2.0** | **copyable** |
| pts → uheprng | **PUBLIC DOMAIN** | **unconditionally copyable** |

**The uncomfortable shape of this table:** the two projects whose *architecture* we most
want to learn from — SuperCollider's real-time machinery and Sonic Pi's new scheduler —
are the two we cannot copy a line from. Pizzicato, the one we can copy freely, contains
no DSP at all.

That is the actual finding: **we have to reimplement the real-time primitives from
first principles regardless, so we should write them small, in our own idiom, against
our own needs — and stop looking for a library to vendor.**

---

## The five most stealable mechanisms

### 1. A 2-head SPSC ring buffer as the only thread boundary

**Where:** `supercollider/common/MsgFifo.h:84-117` (GPL-3.0 — reimplement)
**Also:** `glicol/js/npm/glicol-engine.js:17-21` (MIT — same shape over `SharedArrayBuffer`)
**Instances:** `server/scsynth/SC_HiddenWorld.h:91-94`, four fixed-size FIFOs

```cpp
bool Write(MsgType& data) {
    unsigned int next = NextPos(mWriteHead.load(memory_order_relaxed));
    if (next == mReadHead.load(memory_order_relaxed))
        return false;                       // fifo is full
    mItems[next] = data;
    mWriteHead.store(next, memory_order_release);
    return true;
}
int NextPos(int inPos) { return (inPos + 1) & (N - 1); }   // :113
```

Why it is the mechanism and not a detail: it is the **only** thing standing between the
voice-analysis thread and the audio thread, and it has three properties that together make
it un-blockable:

1. **Power-of-two capacity → `(p+1) & (N-1)`**. No modulo, no division, one cycle.
2. **Two atomics, no CAS.** Single-word heads mean SPSC release/acquire is *sufficient*;
   there is no CAS loop and therefore no possibility of livelock.
3. **`Write` returns `false` when full.** It does not block, allocate, or retry. A full
   queue drops the newest event. That is the entire backpressure policy, in one line, and
   it is the reason the audio thread can never be stalled by its producer.

The 3-head variant (`MsgFifo.h:34-80`) adds `mFreeHead` so the producer never reuses a
slot the consumer hasn't drained — and frees on the *consumer's* schedule, which is what
`SC_Bela.cpp:152` (`mFromEngine.Free()`) does at the top of the callback.

**For us:** the speech analyser writes `{feature values, timestamp}` into a 1024-entry
ring; the MIDI/synthesis thread drains it once per block. Same shape, our fields. **Plan
the drop policy deliberately** — SuperCollider drops the newest; for measurement, dropping
the *oldest* (overwrite) is usually correct, because stale audio is worse than a gap.

### 2. A fixed-capacity time-ordered heap with a stability counter

**Where:** `supercollider/server/scsynth/PriorityQueue.h:31-121` (GPL-3.0 — reimplement)

```cpp
bool Add(Event& inEvent) {
    if (mSize >= N) return false;                        // :36-37
    inEvent.mStabilityCount = mStabilityCounter++;       // :39
    ... // percolate up: (mom - 1) >> 1
}
void Perform(int64 inTime) {                            // :57-62
    while (NextTime() <= inTime) { Event event = Remove(); event.Perform(); }
}
int64 NextTime() { return mEvents[0].mTime; }
```

Three properties worth having:

- **Capacity is a template parameter.** The event array is inline. Zero allocation on the
  audio path, enforced by the type system.
- **`mStabilityCounter` gives FIFO order within one timestamp** (`SC_SequencedCommand.h`
  event classes, and independently Sonic Pi's `@seq` at
  `app/web/runtime/lib/sonic_pi/scheduler.rb:407`). Two independent mature projects both
  needed this. Without it, two events at the same timestamp fire in heap order, which is
  effectively random. **Two of our own MIDI tracks would jitter relative to each other.**
- **Empty is `kMaxInt64`**, so `NextTime()` is always safe and `Perform` terminates.

**For us:** this is the clock our 9-track score plays against. `Add(cc_event)` at an
absolute sample time; `Perform(now)` at the top of every audio block.

### 3. Publish your own next deadline instead of polling

**Where:** `sonic-pi/app/web/runtime/lib/sonic_pi/scheduler.rb:417` (AGPL — the *idea* is
a one-liner; the code is ~10 lines and not worth the licence)

```ruby
nexts = @queue.map { |(wake, _, p)| p.start + wake } +
        @fx_frees.map { |(at, _)| at - FREE_LEAD } +
        @node_frees.map { |(at, _)| at - FREE_LEAD }
nexts.min
```

Three independent deadline sources, `min`-ed, returned to the host. The host sleeps
exactly that long. **The scheduler never sleeps itself** (`:317-318`: "The scheduler never
sleeps itself; the host owns the one real wait").

This is `docs/research/TIMING.md`'s **T-minus law** in code form. TIMING.md says
"Everything upstream schedules against T-minus, not T-zero." Sonic Pi says the same thing
differently: the event is handled at `T` but stamped `T + sched_ahead`
(`scheduler.rb:1147`), and the *next* wakeup is `min` over everything pending.

Our peripheral nervous system almost certainly polls on a fixed timer. It should not. It
should compute `min(next_segment_boundary, next_measurement_refresh, next_midi_event,
next_score_event)` and sleep exactly that — with the measurement refresh as an explicit
input to the `min`, not a separate thread with its own timer.

The same idea appears in SuperCollider as a fixed control block, which is the degenerate
version: always sleep 64 samples, always check. Correct but 750 wakeups/second doing
nothing most of the time.

### 4. A bounded, time-indexed "what was the value at t?" store

**Where:** `sonic-pi/app/server/ruby/lib/sonicpi/event_history.rb:178-209, 401-417`
(**MIT — copyable**)

```ruby
def initialize
  @trim_history = true
  @min_history_size = 20
  @history_depth = 32
  @state = EventHistoryNode.new
...
# Get the last seen version (at or before the current time)
def get(t, ...)
# Get next version (after current time)
def get_next(t, ...)
...
def __insert_event!(e, idx=0, sn=@state)
  if idx == e.path_size
    sn.events.unshift(e)
    bubble_up_sort!(sn.events)
    if @trim_history
      cutoff_t = (Time.now - @history_depth).to_i
      while (sn.events.size > @min_history_size) && (sn.events.last.time.to_i < cutoff_t)
        sn.events.pop
      end
    end
```

A trie keyed on address path; each leaf holds a newest-first sorted event list; two
queries (`at-or-before t`, `next-after t`); auto-trimming with a floor of 20 and a depth
of 32.

This is the exact structure for rendering a score "as of time T", for scrubbing, and for
asking "when does CC 74 next change?". MIT, so it can be lifted rather than reinvented.

**Take the design; do not take line 412.** It evicts on `Time.now` (wall clock) while the
events carry logical time. Under `time_warp`, in tests, or in trace mode, that evicts the
wrong events. Correct rule: evict on `e.time - depth`, with the caller passing the cutoff.

### 5. Reproducible randomness with a normalised string seed

**Where:** `pts/src/Num.ts:190-209`, `pts/src/uheprng.ts` (**public domain** algorithm,
Apache-2.0 port)

```typescript
static seed(seed: string): void { this.generator = generator(seed); }
static random(): number { return this.generator ? this.generator.random() : Math.random(); }
```

and the part PTS actually added (`uheprng.ts:110-119`):

```typescript
// so users always obtain the same EFFECTIVE uheprng seeding key: seeds that
// differ only by surrounding whitespace or embedded control characters collide.
function cleanString(inStr: string): string {
  inStr = inStr.replace(/(^\s*)|(\s*$)/gi, "");
  inStr = inStr.replace(/[\x00-\x1F]/gi, "");
  return inStr;
}
```

Marsaglia MWC over 48 words, seeded through Baagoe's `Mash` avalanche hash. Explicitly
**public domain** — the only unconditionally reusable code in the audit.

Why it is top-5 for *this* project: every other mechanism here is about real-time
plumbing. This one is about **being able to debug a stochastic system.** We already want
randomised parameter response (it is what makes the peripheral feel alive). Without a
single seed that replays the entire stochastic behaviour, we cannot answer "why did it
do that?", regression-test it, diff two runs, or reproduce a user's bug report. Seed
from a session id; store it with the session.

Normalise the key before hashing, or seeding from a filename with a trailing newline
will produce a different run every time.

**Honourable mentions** (real, useful, but not top-5 for us):

- **Deterministic noise from the sample index** — `ZzFX.js:208`,
  `t += f + f*noise*(i*i*PI2%2-1)`. No RNG, no state, reproducible, branch-free. The
  cheapest possible deterministic test signal. MIT.
- **Power-of-two delay buffers** — `supercollider/server/scsynth/SC_World.cpp:1112`,
  `BUFMASK(x) = (1 << (31 - CLZ(x))) - 1`. `CLZ` is one instruction; every delay line and
  oscillator phase then wraps with a mask instead of a modulo. GPL-3.0, but this is a
  one-line idiom, not copyrightable expression in any meaningful sense.
- **Sample/subsample offset for sub-block timing** —
  `supercollider/server/scsynth/SC_Bela.cpp:206-214`. The scheduler ticks per 64-sample
  block, but every event fired inside a block gets an integer *and* fractional offset, so
  effective resolution is sub-sample. This is how to get sample-accurate MIDI scheduling
  from a coarse scheduler.
- **Three clock rates** — `supercollider/server/scsynth/SC_Rate.cpp:26-41`. Sample rate,
  control rate (`mBufRate = mSampleRate / mBufLength`, `:33`), audio rate. **This is the
  direct answer to "measure cheaply and continuously": YIN does not need to run at 48 kHz.**
  Run it at control rate (~750 Hz) and interpolate the f0 contour.
- **`StableGraph` + a persistent DFS object** — `glicol/rs/synth/src/context.rs:89`,
  `rs/synth/src/graph.rs:41,64-67`. Node identity survives hot-reload; the traversal does
  not allocate. Correct pattern, stated once, in a real codebase.
- **Synthetic impulse response** — `pizzicato/src/Effects/Reverb.js:108-120`. Decaying
  white noise, 8 lines. Offline-testable.

---

## The four recurring architecture patterns

Across six unrelated codebases, the same four shapes appear. That convergence is the
strongest evidence in this audit.

### Pattern A — the audio thread never allocates, never blocks, never waits

SuperCollider: `MsgFifo.Write` returns false (`MsgFifo.h:94`); `PriorityQueue.Add`
returns false (`PriorityQueue.h:36`); the event array is a template-sized member.
Glicol (Rust half): `with_capacity` on the DFS stack (`graph.rs:64-67`), fixed
`max_nodes` (`context.rs:28`).

Three independent mechanisms enforce it: (a) fixed-capacity containers, (b) non-blocking
insert that reports failure, (c) preallocated traversal state.

**Corollary that both projects state explicitly:** the *consumer* does the freeing
(`SC_Bela.cpp:152`, `mFromEngine.Free()`). Deallocation is scheduled on the side that
can afford it.

### Pattern B — a continuous sensor loop inside the audio callback

`SC_Bela.cpp:138-146` re-measures the host sample rate *every callback*, forever, and
smooths it with a one-pole. It does not assume the clock is correct; it checks.

**This is a pattern, not a bug workaround.** Our peripheral nervous system is a sensor,
not a controller. It should be continuously validating its own timing assumptions the same
way — and SuperCollider's version includes the escape hatch: if the measured rate deviates
by more than 10 Hz, treat it as a fault (`if (fabs(smoothSampleRate - mSampleRate) > 10.)
smoothSampleRate = mSampleRate;`) rather than believing it.

### Pattern C — sub-block time resolution from a coarse scheduler

SuperCollider schedules at control rate and adds `mSampleOffset` + `mSubsampleOffset`
per event (`SC_Bela.cpp:206-214`). Sonic Pi's `step()` returns a sub-second horizon.
Both separate *when the scheduler wakes* from *when the event is considered to happen*.

Glicol gets this by brute force — a per-sample modulo scan over every event
(`rs/synth/src/node/sequencer/seq.rs:45-65`) — which is O(events × samples) and truncates
to whole samples. **Do not copy glicol's sequencer.** This is the clearest
"mature project, still wrong" finding in the audit.

### Pattern D — bounded everything

`MsgFifo` at 1024/512 compile-time (`SC_HiddenWorld.h:91-94`). `PriorityQueue` at
compile-time `N`. `glicol` at `max_nodes: 1024`. `EventHistory` at depth 32 / floor 20.
`AudioContextBuilder` at fixed sample rate 44100 and 2 channels (`context.rs:24-26`).

**Every one of these is a hard bound with a stated policy when it is hit.** That is the
discipline: unbounded growth in a real-time system is an outage, and these projects each
chose a number and wrote down what happens at the number.

---

## What we should NOT build

Stated plainly, because the temptation is real and the evidence is against it.

**1. Do not build an audio synthesis engine.**
SuperCollider is 53 MB of source. Glicol's node zoo is 60+ files to make *sounds*. We
need *measurement*, and our synthesis need is "MIDI CC values driving real-time
parameters" — i.e. we need the **control signal**, per `TIMING.md`'s capture-point
diagram. Synthesis is the last mile and it belongs to someone else's synth.
ZzFX (125 lines, MIT) and a `midi-synth` binary cover the whole requirement.

**2. Do not build a node-graph DSP engine.**
`glicol` needed `StableGraph`, a persistent `DfsPostOrder`, a `NodeData<T, N>` wrapper
with `Buffer<N>` arrays, `BoxedNode`, `const N: usize` threaded through everything, and a
1024-node capacity story — and its sequencer is still O(n·m). We would re-derive all of
that to produce a `float` we can already compute directly. If we ever need a graph, it
should be a list of `(filter_fn, state)` pairs, not a framework.

**3. Do not copy Sonic Pi's or SuperCollider's scale.**
Sonic Pi's `runtime.rb` is 2113 lines and `scheduler.rb` is 1708; they coexist because
two architectures were never reconciled. SuperCollider's `SC_World.cpp` is 1266 lines of
one translation unit doing allocation, graph management, buffer management and message
dispatch. `SC_SequencedCommand.cpp` is 1687 lines of the same four stages repeated for
every buffer operation.

Every mechanism worth taking from them is **30 to 60 lines**. The rest is the accumulated
weight of supporting every case a public language server must support. Write the 60 lines.

**4. Do not adopt Glicol or Pizzicato.**
Glicol: 18 months stale, 14 KB of docs, CI runs clippy and rustfmt but never `cargo test`
(`.github/workflows/ci.yml`), and the JS host allocates six objects and does four
`slice()` copies **inside its own audio callback** (`js/npm/glicol-engine.js:143-173`)
plus leaks a wasm block per callback at `:143`. Pizzicato: no DSP, and a CI config pinned
to Node 12 on Ubuntu Trusty that cannot run.

**5. Do not trust anything that needs an audio device to verify.**
The most valuable thing in this audit is code that could be verified with no hardware:
`PriorityQueue.h` (pure data structure), `MsgFifo.h` (pure), `ZzFX.buildSamples`
(pure `params → Float32Array`, testable in Node), `EventHistory` (pure, and already
tested — `app/server/ruby/test/` has 134 test files), supercollider's
`testsuite/server/supernova/server_node_graph_test.cpp` (432 lines of C++ testing node
graphs with no device).

If we adopt a mechanism, we must be able to write that test. That constraint is the
single best filter this audit produced, and it points the same direction as our existing
Python-on-WAV-file test posture.

---

## Concrete next moves, in order

1. **Write the SPSC ring first** (~30 lines, our own, Python or C). It is the only
   primitive everything else sits on, and it is the only one where a bug is silent.
   Test: concurrent producer/consumer, verify no loss, verify no torn reads, verify the
   full-queue policy.
2. **Write the event heap second** (~40 lines). Add the stability counter on the first
   day — both mature projects needed it, and it is untestable-by-accident.
3. **Then** port `EventHistory`'s bounded store (MIT), with the clock-mixing bug fixed.
4. **Then** seed the stochastic layer with PTS's public-domain MWC + PTS's `cleanString`.
5. **Decide the control rate explicitly** and move analysis to it, per `SC_Rate.cpp:33`.
   This is likely the largest single win available and the smallest change.
6. **Only then** consider whether any synthesis is needed — and if so, ZzFX under MIT.

## What I could not verify

- **Sonic Pi's actual real-time timing engine.** It is a git submodule
  (`app/external/supersonic` → `samaaron/supersonic`) and was absent from both `dev` and
  `stable` at the commits audited. AGPL. Not in my assigned repo list, not audited. **If
  we want to learn from a modern audio engine, this is the gap.**
- **Any claim about audible quality or measured latency.** No audio hardware; nothing ran.
- **Glicol's traversal order** (`graph.rs:143-165`). The one-block-latency reading is
  inference from `Reversed` + post-order, not a measurement, and no test pins it.
- **Whether Pizzicato's Travis CI still runs.** Travis.org is effectively dead and the
  config targets EOL versions.
- **PTS's dependency count.** Inferred from the absence of vendored source outside
  `demo/` and `docs/`; `package.json` was not read.
- **SuperCollider's `set_control_bus`** (`common/server_shm.hpp:69-71`) is an
  unimplemented stub. The read path exists; the write path was never finished. If we
  build a shared-memory control-bus publish, do **not** treat SC's as a working reference.
