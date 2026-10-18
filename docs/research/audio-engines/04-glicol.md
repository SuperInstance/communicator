# Glicol — a Rust modular synth, and where it gets real-time hygiene wrong

- **Repo**: https://github.com/chaosprint/glicol
- **Branch audited**: `main` @ `0317db2e3f157b911bfc28c72ad5db90db963f24`
- **License**: **MIT** (`LICENSE`, 19 lines, "Copyright (c) 2020-present Qichao Lan")
- **Stars / forks**: 3012 / 103 · **disk**: 114 MB
- **Last push**: 2025-04-06 — **~18 months stale.** No commits since April 2025.

## Inventory

| | |
|---|---|
| blobs | 207 |
| Rust files | 101 (`rs/main`, `rs/synth`, `rs/parser`, `rs/wasm`) |
| JS files | 15 |
| markdown | 6 (**14 KB total** — the worst-documented repo in this audit) |
| Rust source bytes | ~294 KB |
| tests | 3 files: `rs/main/tests/sin.rs`, `rs/parser/tests/sin.rs`, `rs/parser/tests/all_nodes.rs` |
| CI | `.github/workflows/ci.yml` — **clippy + rustfmt only. No `cargo test`.** |

Four separate Cargo packages each carry their own `LICENSE` (MIT): `rs/`, `rs/main/`,
`rs/parser/`, `rs/synth/`, `rs/wasm/`.

## What a Rust modular synth gets right

### StableGraph — node identity survives hot-reload

`rs/synth/src/context.rs:89`:

```rust
pub type GlicolGraph<const N: usize> = petgraph::stable_graph::StableGraph<GlicolNodeData<N>, ()>;
```

`StableGraph`, not `Graph`. Node indices are permanent handles that are never invalidated
by unrelated insertions or removals. For a live-coding engine that rewrites its own graph
while running, this is not a detail — it is the difference between "the signal keeps
flowing while you retype" and "every edit invalidates every reference and everything
crashes". Glicol wires its named nodes through `tags: HashMap<&'static str, NodeIndex>`
(`context.rs:96`), so `#sine` always refers to the same node index across a recompile.

### Fixed capacity, declared once

`rs/synth/src/context.rs:28-29`:

```rust
max_nodes: 1024,
max_edges: 1024,
```

passed to `StableGraph::with_capacity` (`context.rs:104`) and
`Processor::with_capacity` (`context.rs:118`). The intent is stated in a comment at
`context.rs:136-137`:

```rust
/// an alternative to new() specify the estimated max node and edge numbers
/// to avoid dynamic allocation
```

### A reusable DFS stack — the traversal itself does not allocate

`rs/synth/src/graph.rs:41` and `:60-73`:

```rust
pub struct Processor<G, const N: usize> where G: Visitable {
    // State related to the traversal of the audio graph starting from the output node.
    dfs_post_order: DfsPostOrder<G::NodeId, G::Map>,
    // Solely for collecting the inputs of a node in order to apply its `Node::process` method.
    inputs: HashMap<usize, Input<N>>,
}
...
pub fn with_capacity(max_nodes: usize) -> Self where G::Map: Default {
    let dfs_post_order = DfsPostOrder {
        stack: Vec::with_capacity(max_nodes),
        ..Default::default()
    };
```

**A graph traversal normally allocates on every pass.** Glicol holds the traversal state
on the context and reuses it. The `inputs` `HashMap` is also reused, `.clear()`ed per node
rather than reallocated (`graph.rs:146`). This is the single most important real-time
idea in the Rust code.

### The whole audio callback is 19 lines

`rs/main/src/lib.rs:568-586`:

```rust
pub fn next_block(&mut self, buf: Vec<&[f32]>) -> &[Buffer<N>] {
    //  -> &Vec<Buffer<N>>
    if !buf.is_empty() {
        self.context.graph[self.index_info[&"~input".to_string()][0]].buffers[0]
            .copy_from_slice(buf[0]);
    }
    if buf.len() > 1 {
        self.context.graph[self.index_info[&"~input".to_string()][0]].buffers[1]
            .copy_from_slice(buf[1]);
    }
    self.context
        .processor
        .process(&mut self.context.graph, self.context.destination);
    self.clock += N;
    &self.context.graph[self.context.destination].buffers
}
```

Copy input → one graph pass → advance a global sample clock → hand back the destination
buffers. `self.clock += N` is the timing model in its entirety: **a single monotonically
increasing sample counter**, and every time-dependent node compares against it.

## The traversal order — and a latency consequence

`rs/synth/src/graph.rs:133-165`:

```rust
pub fn process<G, T, const N: usize>(...) {
    processor.dfs_post_order.reset(Reversed(&*graph));
    processor.dfs_post_order.move_to(node);
    while let Some(n) = processor.dfs_post_order.next(Reversed(&*graph)) {
        processor.inputs.clear();
        for in_n in (&*graph).neighbors_directed(n, Incoming) {
            // Skip edges that connect the node to itself to avoid aliasing `node`.
            if n == in_n { continue; }
            let input_container = graph.node_weight(in_n).expect(NO_NODE);
            let input = Input::new(&input_container.buffers, (*graph).to_index(in_n));
            processor.inputs.insert((*graph).to_index(in_n), input);
        }
        let data = graph.node_weight_mut(n).expect(NO_NODE);
        data.node.process(&mut processor.inputs, &mut data.buffers);
    }
}
```

`Reversed(&graph)` + DFS **post-order** from the destination. Post-order emits a node
*after* all its descendants; in the reversed graph the destination's descendants are its
ancestors in the original graph — i.e. its **producers**. So the destination is emitted
first and producers are emitted after it. Nodes are push-based (each writes into its own
`buffers`, reads inputs from their buffers), so on an acyclic chain this means each
producer writes the block *after* its consumer has already read — one block of latency.

Also note `:148-151`: self-edges are skipped specifically to avoid a mutable-aliasing
soundness problem, and `:159-162` explains why the raw-pointer version was removed.

**Honest status: I did not run glicol, and I found no test that pins this ordering.** The
one-block-latency reading is inference from the traversal, not a measured result. It is
probably intentional — every node that cares about time (delay, filter, sequencer) already
carries state across blocks — but verify before relying on it.

## THE negative: the JS host allocates in its own audio callback

`js/npm/glicol-engine.js` is an `AudioWorkletProcessor` that runs the entire engine
(`:5`). The architecture around it is genuinely good:

```js
// :17-21
this.useSAB = options.processorOptions.useSAB;
... new RingBuffer(options.processorOptions.codeQueue, Uint8Array));
... new RingBuffer(options.processorProcessorOptions.paramQueue, Uint8Array));
```

Two `SharedArrayBuffer` ring buffers carry live-coding source code and parameter messages
from the main thread into the worklet — the same SPSC-over-shared-memory pattern as
SuperCollider's `MsgFifo`, at the JS/wasm boundary. Block size is fixed at **128** frames.
And errors are surfaced: `this.port.postMessage({type: 'e', info: ...})` at `:169`.

But the `process()` callback (`:120-175`) does this **every 128 frames ≈ every 2.9 ms**:

```js
process(inputs, outputs, _parameters) {
    ...
    if (inputs[0][0]) {
        this._inPtr = this._wasm.exports.alloc(128)                       // :143  wasm heap alloc
        this._inBuf = new Float32Array(                                 // :144  JS object
            this._wasm.exports.memory.buffer, this._inPtr, 128)
        this._inBuf.set(inputs[0][0])
    }
    this._wasm.exports.process(...)
    this._outBuf = new Float32Array(this._wasm.exports.memory.buffer, ...)   // :155
    this._result = new Uint8Array(this._wasm.exports.memory.buffer, ...)     // :161
    ...
    outputs[0][0].set(this._outBuf.slice(0, 128))                        // :172  copies
    outputs[0][1].set(this._outBuf.slice(128, 256))                      // :173  copies
```

Six allocations plus four `slice()` copies per audio block, in the audio thread, on a
`Float32Array` view that is re-wrapped every time. That is GC pressure directly in the
real-time path — the classic cause of the glitches this architecture is supposed to
prevent. It also contradicts the Rust half, which goes out of its way to preallocate the
DFS stack.

Also, the `alloc(128)` on line 143 **leaks**: it allocates a new 128-float block in the
wasm heap on every callback and never frees it. Over an hour of playback that is
~124 million floats of unreachable wasm memory.

**Do not copy the JS host's `process()` body. Copy its shape** (AudioWorklet owns the
engine; SAB rings carry control messages) **and hoist every allocation and view
construction into a setup path.**

Note also `js/npm/ringbuf.js:1-4` — the ring buffer is vendored from
`padenot/ringbuf.js` and is **MPL-2.0** (`"license: MPL-2.0 License"`) — a different
license from the surrounding MIT project, and a file-scoped copyleft. Glicol's *use* of it
is fine under MIT; copying that file into an MIT project is not. There is an MPL-2.0
§3.3 compliant path (publish modifications, keep it as a separate file), but the
simplest answer is to write 30 lines of your own SPSC ring over a `SharedArrayBuffer` — see
SuperCollider's `MsgFifo.h` for the shape.

## THE other negative: the sequencer is O(events × samples)

`rs/synth/src/node/sequencer/seq.rs:45-65`:

```rust
let bar_length = 240.0 / self.bpm as f64 * self.sr as f64 / self.speed as f64;
for out in &mut *output[0] {
    *out = 0.0;
    for event in &self.events {
        if (self.step % (bar_length as usize)) == ((event.0 as f64 * bar_length) as usize) {
            ...
        }
    }
    self.step += 1;
}
```

This is the clearest thing to *not* copy. For **every sample**, it iterates **every
event** and takes two integer modulo operations. That is a per-sample loop nest over the
whole pattern. With 64 events at 44.1 kHz that is 2.8 million modulo pairs per second.

It also has a correctness problem: `(event.0 * bar_length) as usize` truncates. An event
whose beat position does not land on a whole sample after scaling never matches, or
matches at the wrong place. There is no event queue and no absolute-time comparison
anywhere in glicol — the "sample-accurate" claim is brute force.

Sonic Pi's `SC_ScheduledEvent` machinery (SuperCollider) and Sonic Pi's own priority queue
are the correct patterns. Glicol gets sample accuracy by not having an event loop at all.

## Node inventory — what a "complete" small modular synth covers

`rs/synth/src/node/` is a clean taxonomy and is a decent shopping list if we ever want
synthesis at all:

- `oscillator/` — `saw_osc.rs`, `sin_osc.rs`, `squ_osc.rs`, `tri_osc.rs`
- `filter/` — `onepole.rs`, `rlpf.rs`, `rhpf.rs`, `apfmsgain.rs` (one-pole, RBJ LP/HP,
  all-pass for phase)
- `envelope/` — `adsr.rs`, `envperc.rs`
- `delay/` — `delayms.rs`, `delayn.rs` (time- and sample-counted)
- `effect/` — `reverb.rs`, `plate.rs` (8.0 KB, the largest effect), `pan.rs`, `balance.rs`
- `sequencer/` — `seq.rs`, `arrange.rs`, `choose.rs`, `speed.rs`
- `signal/` — `constsig.rs`, `noise.rs`, `imp.rs`, `points.rs`, `phasor.rs` (empty)
- `dynamic/` — `eval.rs`, `expr.rs`, `meta.rs` — **live parameter evaluation as a node**.
  `expr.rs` evaluates an expression on the audio thread to modulate a parameter. Directly
  relevant to "continuous speech drives a parameter".
- `compound/` — `bd.rs`, `hh.rs`, `sn.rs`, `sawsynth.rs`, `trisynth.rs`, `squsynth.rs`
  (drum kits as composites)
- `synth/` — `msgsynth.rs`, `pattern_synth.rs`

`rs/synth/src/node/signal/constsig.rs` is the interesting primitive class for us: a
**constant signal node**. That is a control-rate value expressed as an audio-rate node,
so modulation is just graph structure rather than a separate subsystem.

## The `audiocontext!` macro

`rs/synth/src/context.rs:79-86`:

```rust
#[macro_export]
macro_rules! audiocontext {
    ($size:expr, {$($para: ident = $data:expr),*}) => {
        ( AudioContextBuilder::<$size>::new()$(.$para($data))*.build() )
    }
}
```

`const N: usize` is the **block size, in the type system**. It threads through
`Buffer<N>`, `Node<N>`, `Input<N>`, `Processor<G, N>`. The block size cannot be
misconfigured at runtime because it is not a runtime value. Nice, although it does mean a
compile per block size.

## What to take, what to skip

**Take (MIT):**
- `petgraph::stable_graph::StableGraph` + a name→`NodeIndex` tag map for any graph you
  rewrite at runtime (`context.rs:89,96`)
- a **persistent traversal object** on the context so the graph walk does not allocate
  (`graph.rs:41,64-67`)
- fixed `max_nodes`/`max_edges` declared in one config struct (`context.rs:28-29`)
- `self.clock += N` — one global sample counter as the only notion of time
  (`main_lib.rs:584`)
- `constsig` — constant-signal nodes so modulation is graph structure
- the AudioWorklet + SAB-ring *shape* (`glicol-engine.js:5,17-21`), minus its allocations

**Skip:**
- the per-sample event scan (`seq.rs:45-65`) — use a priority queue
- the allocating `process()` body (`glicol-engine.js:120-175`)
- `ringbuf.js` — MPL-2.0, and trivially replaceable
- the repo generally: 18 months without a commit, 14 KB of docs for a 294 KB Rust DSP
  library. Read the node taxonomy; do not adopt the project.

## Verification notes

- No code was built or run. Rust toolchain, wasm build and AudioWorklet were all out of
  scope in a WSL environment with no audio device.
- The `alloc(128)` leak and the `slice()` copies are read directly from the source at the
  commit above; I have not measured their impact.
- The traversal-order latency claim is explicitly marked as inference.
- `rs/parser/tests/all_nodes.rs` (8.5 KB) exists and would likely catch node-wiring
  regressions — but CI never runs it, so it has almost certainly bit-rotted since 2025.
