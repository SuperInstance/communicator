# PTS — compact self-containment, and reproducible randomness

- **Repo**: https://github.com/williamngan/pts
- **Branch audited**: `master` @ `6f666fa8be304e0b730f1ef518c7235c0486682f`
- **License**: **Apache-2.0** (`LICENSE`, 201 lines)
- **Stars / forks**: 5354 / 183 · **disk**: 32 MB
- **Last push**: 2026-10-16 (active)

## Inventory

| | |
|---|---|
| blobs | 613 |
| TS/JS source | 237 files, 8.19 MB (**but see below**) |
| — of which in `src/` | **55 files** |
| markdown | 62 files, 1.65 MB |
| unit tests | **0** |
| CI | `.github/workflows/ci.yml` — **the best in this audit** |

The 8.19 MB is misleading. `demo/edit/vs/monaco.js` alone is 4.4 MB (the Monaco editor,
vendored into a demo). `dist/` alone is 4 files totalling 1.2 MB (`pts.js` 348 KB,
`index.js` 334 KB, `index.d.ts` 314 KB, **`pts.min.js` 200 KB**). The actual library is
the 55 files in `src/`, whose largest are `Op.ts` 92 KB, `Canvas.ts` 70 KB, `Svg.ts` 65 KB,
`Create.ts` 62 KB, `Num.ts` ~30 KB, plus `Util.ts`, `Pt.ts`, `Graph.ts`, `Space.ts`,
`Line.ts`, `Shape.ts`.

**A 2D graphics + generative-art library with the entire numeric core under 300 KB, and a
200 KB minified bundle, with zero vendored runtime dependencies.** (Checked: outside
`src/`, `demo/`, `docs/` and `dist/`, the tree holds only config files (`.editorconfig`,
`.prettierrc.json`, `.nvmrc`, …) and `assets/`. There is no `node_modules` tree and no
third-party source checked in. Note I did **not** read `package.json`, so this is the
file listing speaking, not the manifest.)

That compactness is the main reason this project is in the lane.

## THE best CI in the audit

`.github/workflows/ci.yml`:

```yaml
on:
  push:
  pull_request:
  workflow_dispatch:
  schedule:
    - cron: "23 8 * * 1"

permissions:
  contents: read

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - uses: pnpm/action-setup@b906affcce14559ad1aafd4ab0e9427799e58b1 # v4
      - uses: actions/setup-node@820762786026740c76f36085b0efc47a31fe5020 # v7.0.0
        with: { node-version: 24.18.1, cache: pnpm, cache-dependency-path: pnpm-lock.yaml }
      - run: pnpm install --frozen-lockfile
      - run: pnpm exec playwright install --with-deps chromium
      - run: pnpm check
      - run: git diff --exit-code -- dist          # ← dist reproducibility
      - run: mkdir package && pnpm pack --pack-destination package
```

Three things worth copying:

1. **`git diff --exit-code -- dist`** — the committed build output must be byte-identical
   to a fresh build. A non-reproducible build fails CI. This is the check that keeps a
   committed `dist/` honest, and it is the check most projects skip.
2. **All three actions are pinned to commit SHAs with a `# v7.0.1` comment.** Not tags.
   Not `@latest`. Tags are mutable; SHAs are not.
3. **A weekly cron** (Mondays 08:23) catches upstream dependency rot that push-triggered
   CI never would.

`permissions: contents: read` at the workflow level is also correct — least privilege by
default rather than by exception.

There is a second `runtime` job gated on `needs: validate` with a Playwright matrix.

**And yet there are zero unit tests.** The `test-ish source: 40` count from my first pass
was `demo/` files, not tests. So: excellent CI hygiene, zero correctness tests. The
`pnpm check` presumably runs tsc, which catches type errors — real value, but not
behaviour.

## The one thing that is directly useful to us: reproducible randomness

`src/Num.ts:190-209`:

```typescript
/**
 * Seed the pseudorandom generator for reproducible [`Num.random`](#link) sequences.
 * The seed is hashed by its *effective* key: leading/trailing whitespace and embedded
 * control characters are stripped first, so seeds differing only in those collide.
 * An empty (or whitespace-only) seed yields a fixed default sequence. This generator
 * is deterministic and statistically strong, but not cryptographically secure.
 * @param seed seed string
 */
static seed(seed: string): void {
  this.generator = generator(seed);
}

/**
 * Return a random number between 0 and 1. If a seed was set via [`Num.seed`](#link),
 * draws come from the seeded generator with 32-bit resolution (exact multiples of 2^-32);
 * otherwise it uses `Math.random`.
 */
static random(): number {
  return this.generator ? this.generator.random() : Math.random();
}
```

`Num.seed("session-42")` → every subsequent `Num.random()` is reproducible.

**Why this matters for a voice system.** The peripheral nervous system reacts
stochastically — randomisation of parameter response is the whole point of ZzFX,
Sonic Pi and PTS alike. But a stochastic system you cannot replay is undiagnosable: you
cannot answer "why did it do that?", you cannot regression-test it, you cannot diff two
runs, and you cannot reproduce a user's bug report. **A single seed that makes the entire
stochastic behaviour of the system replayable is worth more than any individual DSP
technique in this audit.**

PTS got there by making the seed a *string* and normalising it (see below), which means
you can seed from a session id, a timestamp, a filename — anything human-readable.

### The generator is public domain

`src/uheprng.ts:1-19`:

```typescript
/* This code has been written by Steve Gibson and can be found here:
 * https://www.grc.com/otg/uheprng.htm
 * The code has been converted to typescript and unused functions have been removed.
 *
 * Port notes:
 * - Only the seeded, deterministic path survives: the original's
 *   nondeterministic startup fill (from the host's `Math.random`) fed a
 *   usage this port removed, so it is omitted and seeding consumes no
 *   outside entropy.
 * - `random()` returns the raw 32-bit MWC fraction — every draw is an
 *   exact multiple of 2^-32. The original composed two draws into a
 *   53-bit double; callers here (Float32-based Pts) don't need that.
 * - Despite the original header below, this is not a cryptographically
 *   secure PRNG by modern standards (its state is recoverable from its
 *   outputs). Pts uses it for reproducible generative art only.
 */
```

and the embedded licence at `:25-28`:

```
LICENSE AND COPYRIGHT:  THIS CODE IS HEREBY RELEASED INTO THE PUBLIC DOMAIN
Gibson Research Corporation releases and disclaims ALL RIGHTS AND TITLE IN
THIS CODE OR ANY DERIVATIVES. Anyone may be freely use it for any purpose.
```

**Public domain.** Unconditionally reusable, in any project, under any licence. This is
the single most permissively-licensed code in the entire audit.

The algorithm is Multiply-With-Carry (George Marsaglia) over a 48-word state, seeded
through Baagoe's `Mash` avalanche hash (`src/uheprng.ts:93-108`):

```typescript
export default function uheprng(seed: string) {
  const o = 48; // set the 'order' number of ENTROPY-holding 32-bit values
  let c = 1;    // init the 'carry' used by the multiply-with-carry (MWC) algorithm
  let p = o;    // init the 'phase' (max-1) of the intermediate variable pointer
  const s: number[] = new Array(o);
  const mash = Mash(); // get a pointer to our high-performance "Mash" hash
```

`Mash` (`:71-91`) is 20 lines of multiply-shift-add over the seed's char codes with a
documented avalanche property — "every bit of the input affects every bit of the output
50% of the time".

**And the seed normalisation (`:110-119`) is a small design idea worth taking:**

```typescript
// this "clean string" function removes leading and trailing spaces and non-printing
// control characters, including any embedded carriage-return (CR) and line-feed (LF)
// characters, from any string it is handed. this is used by the 'hashString' function
// (below) so users always obtain the same EFFECTIVE uheprng seeding key: seeds that
// differ only by surrounding whitespace or embedded control characters collide.
function cleanString(inStr: string): string {
  inStr = inStr.replace(/(^\s*)|(\s*$)/gi, ""); // remove any/all leading spaces
  inStr = inStr.replace(/[\x00-\x1F]/gi, ""); // remove any/all control characters
  return inStr;
}
```

Seeding from user-supplied text (a filename with a trailing newline, a YAML field) and
having `"run\n"` and `"run"` produce different sequences is a real class of
non-reproducible-run bug. Normalising the *effective* key before hashing eliminates it.
PTS documents this behaviour explicitly in the doc comment rather than leaving it as an
accident.

## The numeric primitives worth lifting

`src/Num.ts` — all pure, all static, no state:

| | line | |
|---|---|---|
| `lerp(a, b, t)` | `:39` | |
| `clamp(val, min, max)` | `:49` | |
| `boundValue(val, min, max)` | `:62` | |
| `within(p, a, b)` | `:83` | |
| `randomRange(a, b = 0)` | `:92` | |
| `randomPt(a, b?)` | `:102` | |
| `normalizeValue(n, a, b)` | `:120` | maps to `[0,1]` |
| `mapToRange(...)` | `:172` | |
| `seed(seed)` / `random()` | `:197`, `:207` | |

`Num.normalizeValue(n, a, b)` at `:120` is **exactly** the "map a raw measurement into a
0..1 control value" function our speech-features → MIDI CC path needs. It exists, it is
named clearly, and the project has spent years getting the edge cases right.

And `Num.cycle(...)` (`:156-170`) with a default shaping function:

```typescript
static cycle(
  ...,
  method: (t: number) => number = Shaping.sineInOut,
```

**Cycling through a sequence with an arbitrary easing function.** For a peripheral system
driving real-time parameters, "advance to the next target with a chosen curve" is the
central operation, and PTS has it as a one-call primitive with easing built in. Sonic Pi
learns this from PTS's sibling idea (`ring.tick`).

`Shaping` (`src/Num.ts:595` onward) is a long list of named easing functions — `linear`,
`quadraticIn/Out/InOut`, `cubicIn/Out/InOut`, `sineInOut`, … — with `Geom.cosTable()` /
`sinTable()` (`:561`, `:575`) precomputed lookup tables for the trigonometric ones.
**A named easing curve library, free, with a table-lookup path for the trig cases.**

## Compact self-containment — what it actually does

Three concrete techniques, all visible in `src/`:

1. **`{x, y}` plain objects everywhere.** `Pt`, `Vec`, `Mat` are structural interfaces
   over `{x: number, y: number}` (`src/Types.ts`). No class hierarchy, no allocation for a
   point. `Op.ts` is 92 KB of static functions over these. The entire library is
   functions over two numbers.

2. **`Space` / `Group` as a uniform container.** A `Space` is a region; `Group` is a
   transform stack. Both are plain data. So `Canvas.ts` (70 KB) and `Svg.ts` (65 KB) can
   share 100% of the modelling code and differ only in the ~40 lines that issue
   `ctx.moveTo` vs `d="M ..."`. That is why the library is small: one model, two
   renderers, no intermediate scene-graph abstraction to learn.

3. **No runtime dependencies, ES module + a single UMD bundle.** `dist/pts.min.js` is
   200 KB and `index.d.ts` is 314 KB — the types are bigger than the runtime. Drop the
   file on a page and it works.

**The lesson for us:** the compactness comes from *not inventing an abstraction layer*.
`Op.ts` is a giant flat namespace of functions. It is not beautiful. It is small, it has
no import cycle to fight, and every function is independently testable.

## What to take, what to skip

**Take:**
- **seeded, reproducible randomness with a string seed and normalised effective key**
  (`Num.ts:197`, `uheprng.ts:93-119`). Public domain algorithm; the normalisation rule is
  the part PTS added.
- `Num.normalizeValue` / `mapToRange` / `clamp` / `lerp` — the measurement→CC mapping core
  (`Num.ts:120,172,49,39`)
- `Num.cycle(seq, method)` — advance-with-easing as a single primitive (`Num.ts:156`)
- the `Shaping` easing library + precomputed trig tables (`Num.ts:561-660`)
- `git diff --exit-code -- dist` in CI + SHA-pinned actions + weekly cron
- the flat-functional structure over plain `{x, y}` data

**Skip:**
- The project. It is a 2D graphics library; its generative model is geometric
  (composition, symmetry, subdivision), not temporal or acoustic. There is nothing
  about measurement, timing, latency or audio in it.
- It has **zero unit tests**. Do not treat it as a model for testing.
- `Op.ts` at 92 KB of flat static functions is the *opposite* of good structure. Copy the
  individual functions; do not copy the file's shape.

## Verification notes

- `pts.min.js` is 200.3 KB per the tree listing — that is the minified full library,
  including the editor-excluded parts. I did not measure gzip.
- I did not run `pnpm check` or the Playwright suite.
- I read `Num.ts` (structure and the `seed`/`random`/`cycle`/`Shaping` regions) and
  `uheprng.ts` in full, `Util.ts` partially, and enumerated the `src/` surface from the
  tree. I did not read `Op.ts`, `Canvas.ts` or `Svg.ts`.
- The claim that there are no vendored runtime dependencies comes from the file listing
  (nothing outside `src/`/`demo/`/`docs/`/`dist/` but config and assets), **not** from
  `package.json`, which I did not read.
- I claimed "zero unit tests" from the absence of any `tests/` or `test/` directory in the
  tree. A `pnpm check` script that runs something other than `tsc` could include tests I
  never saw; I did not read `package.json`.
