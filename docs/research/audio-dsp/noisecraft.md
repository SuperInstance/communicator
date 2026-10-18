# noisecraft — the model is gone; what remains is a browser IDE

Investigated 2026-10-18. This one has the most consequential finding in the set, and it is bad news.

## The assigned repo is not the generative model

`maximecb/noisecraft` is a **browser-based visual programming language** for sound synthesis —
Web Audio + Web MIDI, plain JS/HTML/CSS, PureData/Max-MSP inspired. `README.md:1-11` says so, and
`README.md:3` opens with "NoiseCraft is a work in progress".

- 30 `.ncft`, 24 `.js`, 4 `.html`, 2 `.yml`, 2 `.sh`, 2 `.json`, 1 `.ico`, 1 `.png`
- **0 Python files** (`find . -name "*.py"` → 0)
- last commit 2026-09-23, but that is docs/maintenance ("Minor upgrades to stats page")

**Correction to an assumption I made mid-investigation:** I initially took the 30 `.ncft` files
for model weights. They are not. `.ncft` = NoiseCraft project files, and they are JSON node
graphs — all 20 of a sample set parse as JSON:

```json
// examples/x_fm_guitar.ncft
{"title":"X.FM v.1.0","nodes":{"0":{"type":"MonoSeq","name":"ModIndex",
 "ins":[["18",0],null],"inNames":["clock","gateT"],"outNames":["freq","gate"],
 "params":{},"scaleName":"chromatic","scaleRoot":"A3",
 "patterns":[[[0,0,0,0,0,0,0,0,1,0,...]]]}}}
```

Zero machine learning. Node types like `MonoSeq`, pattern arrays, gain/gate wiring.

## The actual generative model is deleted

The brief's description — *can learn a sound synthesizer from raw audio without labels* — is
`maximecb/mincraft`, a **separate repository**. It does not exist:

```
git clone https://github.com/maximecb/mincraft.git    →  Repository not found
```

I did not stop at the 404. I checked the owner's full account via the GraphQL API:
**`maximecb` (Maxime Chevalier-Boisvert) has 39 public repositories, and none is named `mincraft`.**
The top of the account by stars: `noisecraft` (1,228), `babyai` (770), `zetavm` (645), `uvm` (625),
`Turing-Drawings` (457), `atari-representation-learning` (259).

So `mincraft` is **deleted or made private — not archived**. The distinction matters: an archived
repo remains readable. This one is gone.

I also searched for surviving forks (`search(query:"mincraft in:name fork:true")`) in case someone
had mirrored the ML code. Every hit is the video game (`eladjmc/Mincraft-Game`, `ipe1014/mincraft`,
etc.). No mirror of the model.

And the noisecraft IDE repo never even references it — `grep -rniE "mincraft"` across the entire
noisecraft tree returns **zero hits**. The two projects were developed in parallel and share only
an author and an ecosystem.

**Consequence: the brief's central question — "What are its measured results and its real
capabilities?" — cannot be answered, because the artifact is not obtainable.** I am not going to
reconstruct its paper results from memory and present them as findings. Anyone who quotes
"NoiseCraft achieved X% on Minecraft sounds" to us is quoting something we cannot verify from
primary sources, and should say so.

## What the assigned repo actually is, and whether it helps us

| fact | value |
|---|---|
| license | **GPL-2.0** (`LICENSE`) |
| stars | 1,228 |
| language | JavaScript (Web Audio / Web MIDI) |
| tests | `tests.js` exists but is a Node/browser harness, no Python test suite |
| CI | `.github/workflows/test.yml` only |
| runnable here | **No** — needs a browser with Web Audio; headless WSL has no audio |

Its design principles (`README.md`, "Design" section) are worth quoting because they are the
reason it is interesting at all: no web frameworks, plain JS/HTML/CSS, minimum dependencies, no
feature creep.

The one thing genuinely relevant to our "learn our own tables" ambition: the node-graph format
above. `.ncft` is a readable JSON graph of typed nodes with named inputs/outputs, parameter
dictionaries, scale/root info, and step-sequenced pattern arrays. If we ever want a declarative,
inspectable, human-editable description of a synth voice or a sound table, this is a compact and
legible precedent for the format. **But it is GPL-2.0, and it is a data format, not an algorithm
— so: format ideas only, and they are not urgent.**

## Verdict: dead end, and an actively misleading one

Three separate reasons, in descending order of importance:

1. **The thing we were asked to evaluate does not exist any more.** The ML repo is gone. There is
   no measured result to assess, no capability to try, and nothing to learn from directly.
2. **What we were given is not that thing.** The noisecraft repo is a browser IDE with no ML and
   no Python.
3. **Even the shell is GPL-2.0 and browser-bound.** Unrunnable headless, unborrowable as code.

If someone on the team is holding a belief like "noisecraft shows you can learn a synth from raw
audio, so we could too" — that belief rests on a repository that no longer exists and results we
cannot check. It should not be used to justify a plan. The honest status is **unknown**, not
"validated".

The one live thread: `maximecb/atari-representation-learning` ("Code for Unsupervised State
Representation Learning in Atari") is by the same author, is public, and is genuinely on-topic for
"learn a compact representation of acoustic states" — the thing the brief speculated figaro might
help with. It is outside my assigned scope and I have not read it, but it is the more promising
lead of the two and someone should look at it.

## Unverified

- I cannot confirm *when* or *why* `mincraft` was removed, or whether it moved to a private or
  differently-named location. The author's account shows no rename.
- I did not read `noisecraft/tests.js` or `server.js` in depth.
- All claims about the noisecraft IDE's runtime behaviour are from source and README, not execution.