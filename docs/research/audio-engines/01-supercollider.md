# SuperCollider — the reference for glitch-free continuous operation

- **Repo**: https://github.com/supercollider/supercollider
- **Branch audited**: `develop` @ `73181de6af445f7e1031f609fa7290265ef75bbc`
- **License**: **GPL-3.0** (`COPYING`, 674 lines; graphql `licenseInfo.spdxId = GPL-3.0`)
- **Stars / forks**: 6796 / 851 · **disk**: 173 MB
- **Last push**: 2026-10-18 (actively developed)

## Inventory

| | |
|---|---|
| blobs in tree | 7331 |
| source files (c/cpp/h/ipp/sc/scd/rs/py/js) | 5736 |
| markdown/rst/txt/html | 79 (926 KB) |
| source bytes | 53.3 MB |
| test-ish source files | 245 under `testsuite/` |
| CI | `.github/workflows/{test_linux,test_macos,test_windows,build_linux,build_macos,build_windows,build_wasm,build_flatpak,sclang_error_regression}.yml` + `lint_non_code.yml` |

Source-to-markdown ratio **~73:1**. This is a code project with documentation as an
afterthought — which is exactly what you want in an audit: the docs describe intent,
the code describes reality, and here they agree.

Substantial chunks of the byte count are vendored Boost (`external_libraries/boost`,
the top 12 "source files" by size are all Boost preprocessed headers). The genuinely
hand-written server is `server/scsynth/` — about 800 KB across ~40 files.

## THE architecture, in one function

`SC_Bela.cpp` is the newest and cleanest audio backend (CoreAudio/JACK/PulseAudio are
all ported from it). `SC_BelaDriver::BelaAudioCallback` at
`server/scsynth/SC_Bela.cpp:119` is, effectively, the whole answer to "how do they
avoid dropouts". Read it in order:

### 1. Denormal defence, first thing (`:122`)

```cpp
void SC_BelaDriver::BelaAudioCallback(BelaContext* belaContext) {
    struct timespec tspec;
    sc_SetDenormalFlags();
```

Subnormal floats cost hundreds of cycles on x86 and will happily blow a deadline by
being *slow*. SuperCollider disables them at the top of every callback rather than
trying to avoid producing them.

### 2. Continuously re-estimate the host's clock, one-pole smoothed (`:138-146`)

```cpp
double instSampleRate = (sampleTime - mPrevSampleTime) / (hostSecs - mPrevHostSecs);
double smoothSampleRate = mSmoothSampleRate;
smoothSampleRate = smoothSampleRate + 0.002 * (instSampleRate - smoothSampleRate);
if (fabs(smoothSampleRate - mSampleRate) > 10.) {
    smoothSampleRate = mSampleRate;
}
mOSCincrement = static_cast<int64>(mOSCincrementNumerator / smoothSampleRate);
```

It measures the *actual* elapsed samples per wall-clock second — every callback, forever —
smooths with a 0.002 one-pole, and snaps to nominal if it drifts more than 10 Hz. This is
the pattern: **the audio loop is also a sensor loop, and the sensor is continuously
calibrated.** A drift of more than 10 Hz is treated as a bug (a stalled callback), not as
a new sample rate.

### 3. Drain three ring buffers before doing any audio (`:152-154`)

```cpp
mFromEngine.Free();
mToEngine.Perform();
mOscPacketsToEngine.Perform();
```

Non-blocking, best-effort, and *before* the DSP. See §"The ring buffers" below.

### 4. Per control block: compute a wall-clock horizon, then fire everything due (`:187-218`)

```cpp
for (int i = 0; i < numBufs; ++i, mWorld->mBufCounter++) {
    ...
    int64 nextTime = mOSCbuftime + mOSCincrement;
    while ((schedTime = mScheduler.NextTime()) <= nextTime) {
        float diffTime = static_cast<float>(schedTime - mOSCbuftime) * mOSCtoSamples + 0.5;
        float diffTimeFloor = std::floor(diffTime);
        mWorld->mSampleOffset = static_cast<int>(diffTimeFloor);
        mWorld->mSubsampleOffset = diffTime - diffTimeFloor;
        if (mWorld->mSampleOffset < 0)                     mWorld->mSampleOffset = 0;
        else if (mWorld->mSampleOffset >= mWorld->mBufLength) mWorld->mSampleOffset = mWorld->mBufLength - 1;
        SC_ScheduledEvent event = mScheduler.Remove();
        event.Perform();
    }
    mWorld->mSampleOffset = 0;
    mWorld->mSubsampleOffset = 0.f;
    World_Run(mWorld);
```

This is the single most important mechanism in the repository. The scheduler ticks at
the **control rate** (one block, nominally 64 samples), but every event that fires inside
a block gets a **sample offset + a fractional subsample offset**. So scheduling
resolution is one sample — or better — even though the scheduler itself only wakes up
every 64 samples. Events are fired *before* `World_Run`, so a node created by an event
is already in the graph when the block is rendered.

### 5. Copy out only the buses that were actually written (`:224-232`)

```cpp
tch = mWorld->mAudioBusTouched;
for (int k = 0; k < minOutputs; ++k)
    if (*tch++ == mWorld->mBufCounter)
        std::memcpy(belaContext->audioOut + k * bufFrames, outBuses + k * bufFrames, ...);
```

Every bus carries a "last touched block counter". An untouched bus is not copied. Cheap
continuous work-avoidance: if nothing is listening on bus 7, bus 7 costs nothing.

## The ring buffers

`common/MsgFifo.h` is **124 lines total** and is the cleanest real-time handoff primitive
in the whole audit. Two variants:

**`MsgFifoNoFree<T, N>`** (`common/MsgFifo.h:84-117`) — single producer, single consumer:

```cpp
bool Write(MsgType& data) {
    unsigned int next = NextPos(mWriteHead.load(memory_order_relaxed));
    if (next == mReadHead.load(memory_order_relaxed))
        return false;                       // fifo is full
    mItems[next] = data;
    mWriteHead.store(next, memory_order_release);
    return true;
}
void Perform() {                            // get next and advance
    while (HasData()) {
        unsigned int next = NextPos(mReadHead.load(memory_order_relaxed));
        mItems[next].Perform();
        mReadHead.store(next, memory_order_release);
    }
}
int NextPos(int inPos) { return (inPos + 1) & (N - 1); }   // :113
```

Properties that matter:
- `N` is a power of two, so indexing is `(p+1) & (N-1)` — **no modulo, no division**.
- Exactly two atomics. No CAS, no lock, no `compare_exchange`.
- Producer loads relaxed, stores **release**; consumer mirrors. Single-word heads mean
  the standard SPSC release/acquire pair is sufficient and provably correct.
- `Write` **returns false when full**. It never blocks, never allocates, never retries.
  A full queue drops the newest event. That is the backpressure policy, stated in one line.

**`MsgFifo<T, N>`** (`common/MsgFifo.h:34-80`) adds a third head, `mFreeHead`, so the
*producer* never touches a slot the consumer hasn't drained yet:

```cpp
bool Write(MsgType& data) {
    unsigned int next = NextPos(mWriteHead.load(memory_order_relaxed));
    if (next == mFreeHead.load(memory_order_relaxed)) return false;
    mItems[next] = data;
    mWriteHead.store(next, memory_order_release);
    return true;
}
void Free() {                                // reclaim messages
    while (NeedsFree()) {
        unsigned int next = NextPos(mFreeHead.load(memory_order_relaxed));
        mItems[next].Free();
        mFreeHead.store(next, memory_order_release);
    }
}
```

Note the division of labour: `MsgFifo.Free()` is called by the **audio** thread at the top
of the callback (`SC_Bela.cpp:152`), reclaiming messages the control thread has finished
with. Freeing on the consumer's schedule avoids an allocator round-trip on the producer.

### Where they live

`server/scsynth/SC_HiddenWorld.h:91-94`:

```cpp
typedef MsgFifoNoFree<TriggerMsg, 1024> TriggersFifo;
typedef MsgFifoNoFree<NodeReplyMsg, 1024> NodeReplyFifo;
typedef MsgFifoNoFree<NodeEndMsg, 1024> NodeEndsFifo;
typedef MsgFifoNoFree<DeleteGraphDefMsg, 512> DeleteGraphDefsFifo;
```

and the struct fields at `SC_HiddenWorld.h:116-118`. One FIFO **per message class**, so a
pathological producer of one kind cannot starve another. Sizes are fixed at compile time.

## The scheduling queue

`server/scsynth/PriorityQueue.h` is **121 lines** — a fixed-capacity binary min-heap keyed
on an absolute `int64` sample time:

```cpp
template <class Event, int N> class PriorityQueueT {
    bool Add(Event& inEvent) {
        if (mSize >= N) return false;              // :36-37  full -> silently drop
        inEvent.mStabilityCount = mStabilityCounter++;   // :39  FIFO tiebreak
        long mom = mSize++; long me = mom;
        for (; mom > 0;) {                        // :43-50 percolate up
            mom = (mom - 1) >> 1;
            if (inEvent.key() < mEvents[mom].key()) { mEvents[me] = mEvents[mom]; me = mom; }
            else break;
        }
        mEvents[me] = inEvent;
        return true;
    }
    void Perform(int64 inTime) {                  // :57-62
        while (NextTime() <= inTime) { Event event = Remove(); event.Perform(); }
    }
    int64 NextTime() { return mEvents[0].mTime; }
```

Three details worth stealing:
1. **Capacity is a template parameter.** The event array lives inline in the world struct.
   No allocation, ever, on the audio path.
2. **A monotonic stability counter gives FIFO order within one timestamp.** Without it,
   an event at time T could fire before an earlier-submitted event at time T. This is the
   classic ordering bug in audio schedulers, and it is solved in one line.
3. **Empty is encoded as `kMaxInt64`**, so `NextTime()` on an empty queue is safe and the
   `Perform` loop terminates naturally.

## Power-of-two delay buffers

`server/scsynth/SC_World.cpp:1112` and `:1125-1126`:

```cpp
inline int32 BUFMASK(int32 x) { return (1 << (31 - CLZ(x))) - 1; }
...
buf->mask  = BUFMASK(numSamples);   // for delay lines
buf->mask1 = buf->mask - 1;         // for oscillators
```

`BUFMASK` rounds a length *up* to the next power of two and returns the mask, so every
delay-line read/write is `buf[(pos + n) & buf->mask]`. Oscillator phase gets `mask1` so
that modulo-N wraparound of the phase accumulator is a mask too. Every delay line in
SuperCollider is therefore rounded up to a power of two — which is why `BufAlloc` sizes
are effectively quantized. `CLZ` here is a single instruction (`lzcnt`/`bsr`).

## Three clock rates, not one

`server/scsynth/SC_Rate.cpp:26-41`:

```cpp
inRate->mSampleRate  = inSampleRate;
inRate->mBufLength   = inBufLength;
inRate->mBufRate     = inSampleRate / inBufLength;    // :33
inRate->mFilterLoops = inBufLength / 3;
inRate->mFilterRemain = inBufLength % 3;
```

SuperCollider's synthdefs declare every UGen at one of three rates: **sample rate**
(per-sample), **control rate** (once per 64-sample block, i.e. ~750 Hz at 48 kHz), or
**audio rate** (subsample interpolation). Filter coefficient interpolation is even split
into a loop count plus a remainder so the filter coefficient can advance smoothly within
a block.

This is the answer to "measure cheaply and continuously": **you do not run your analysis
at the sample rate.** You run the expensive part at the control rate and only interpolate
at the sample rate. YIN pitch tracking at 48 kHz is unnecessary; YIN at 750 Hz with
linear interpolation of the resulting f0 contour is perceptually equivalent and ~64× cheaper.

## The four-stage command pipeline

`server/scsynth/SC_SequencedCommand.h:22-27`:

```cpp
/*
 * Having SequencedCommands allows performing actions that might otherwise require
 * taking a mutex, which is undesirable in a real time thread.
 * Some commands require several stages of processing at both the real time
 * and non real time levels. This class does the messaging between levels for you
 * so that you only need to write the functions.
 */
```

and the shape at `SC_SequencedCommand.h:73-78`:

```cpp
virtual bool Stage1(); //     real time
virtual bool Stage2(); // non real time
virtual bool Stage3(); //     real time
virtual void Stage4(); // non real time
```

Commands that need to do something expensive (allocate a buffer, decode a file) alternate
between the real-time and non-real-time threads, using **real-time-only memory**
(`World_Alloc`, `SC_SequencedCommand.h:41`) for the payload. A file decode touches no
shared state and only the final `float*` crosses the boundary.

## OSC bundles are the atomic scheduling unit

`server/scsynth/SC_ComPort.cpp:56-145` (`UnrollOSCPacket`): an OSC **bundle** carries a
timetag; that timetag is the logical time. Nested bundles are detected
(`hasNestedBundle`, `:71`) and the server flattens them itself (`:75-116`) rather than
pushing the problem to the client. Atomicity is free: a bundle is one timestamp, so
"create these four synths and start them together" needs no locking because they all
land on the same control block.

## Control buses live in shared memory — and one half is unimplemented

`common/server_shm.hpp:51-52` and `:73`:

```cpp
control_busses_ = (float*)segment.allocate(control_busses * sizeof(float));
...
float* get_control_busses() { return control_busses_.get(); }
```

The audio thread writes a float to a shared-memory page; the language reads it directly.
No OSC round trip, no message, no latency. **This is exactly the publish/subscribe we
want between the speech-analysis engine and the MIDI score renderer.**

But `:69-71`:

```cpp
void set_control_bus(int bus, float value) {
    // TODO: we need to set the control busses via a work queue
}
```

**The write path is an unimplemented stub.** Reading is real; writing was never finished.
Do not model your control-bus design on this — use the `MsgFifo` pattern for inbound
writes instead.

## Nodes are calc functions, groups are calc functions

`server/scsynth/SC_Group.cpp:65-72`:

```cpp
void Group_Calc(Group* inGroup) {
    Node* child = inGroup->mHead;
    while (child) {
        Node* next = child->mNext;
        (*child->mCalcFunc)(child);
        child = next;
    }
}
```

A group is a doubly linked list of children; `World_Run` (`SC_World.cpp:939-943`) calls
the top group's `mCalcFunc`, which walks its children in insertion order. `Group_AddTail`
(`:408-419`) is O(1) append. Ordering is **insertion order**, not topological — the
SynthDef's internal graph is topologically sorted at compile time instead. Node teardown
must therefore preserve order, which is why `NodeEndMsg` carries `mPrevNodeID` and
`mNextNodeID` (`SC_HiddenWorld.h:70-82`).

## What to take, what to skip

**Take:**
- the `MsgFifoNoFree` shape (33 lines) — reimplement, it is a textbook SPSC ring
- the `PriorityQueueT` shape with the stability counter — reimplement
- the sample/subsample offset trick for sub-block scheduling resolution
- the three-rate model (control / audio / sample) as the template for cheap measurement
- the "touch the bus you wrote" pattern
- continuous self-calibration of the host clock

**Skip:**
- `SC_World.cpp` is 1266 lines of a single translation unit doing allocation, graph
  management, buffer management and message dispatch. Not an architecture worth copying;
  it is one that accumulated.
- `SC_SequencedCommand.cpp` is 1687 lines and 52 KB of generated-looking boilerplate
  (`BufAllocCmd`, `BufReadCmd`, `BufReadChannelCmd`, ... all the same four stages).
- GPL-3.0. You can read it and reimplement every mechanism above; you cannot copy a line.

## Verification notes

- `testsuite/server/supernova/server_node_graph_test.cpp` (432 lines) is a genuine C++
  unit test of node-graph behaviour — the C++ side is tested without an audio device,
  which is directly relevant to our WSL constraint.
- `testsuite/classlibrary/` has ~200 more `.sc` tests including
  `TestClockScheduling.sc` and `TestTempoClock.sc`.
- I did **not** build or run any of it. All line numbers are from the `develop` tree at
  the commit above.
