# Audio LLMs — learning journey

*Written incrementally during the research pass, 2026-10-18. Lane:
`docs/research/audio-llms/`. Repos: AIGC-Audio/AudioGPT, OpenBMB/VoxCPM,
modelscope/FunASR.*

This is the honest log — what I expected, what surprised me, what I got wrong
and had to correct. Per-repo findings are in `audio-gpt.md`, `voxcpm.md`,
`funasr.md`. Cross-cutting conclusions are in `SYNTHESIS.md`.

---

## Round 1 — what I expected

I went in expecting three "audio language models" that share a family
resemblance: an audio front-end that compresses audio into tokens, an LLM that
conditions on those tokens, and a decoder that renders audio back out. I
expected the three repos to be points on one curve — a token rate, a
parameter count, a benchmark table — and that the useful question would be
"which point sits on our latency/quality frontier?"

That expectation was wrong in a specific and useful way. **These are not three
implementations of one idea. They are three different *categories* of thing**,
and the category difference matters more than any individual number:

| repo | what it actually is |
|---|---|
| AudioGPT | a LangChain tool-calling agent with an OpenAI LLM on top |
| VoxCPM | a single monolithic TTS model (LLM + tokenizer + DiT, one thing) |
| FunASR | a production ASR *toolkit* — dozens of models, a C++ runtime, CI |

Once I saw that, the useful question changed from "which is best" to "which
mechanisms transfer to a continuous, cheap, single-mic peripheral".

## Round 2 — the premise I was given was wrong (and this mattered)

My brief said AudioGPT uses **ATST** as one of its audio front-ends. I went
looking for it and could not find it anywhere:

```
$ grep -ril "atst" . --exclude-dir=.git
./text_to_audio/.../vggsound_train.txt
./text_to_audio/.../vggsound.csv
```

Two hits. I assumed that was a real partial match and nearly wrote it up as
"ATST is vendored". Checking what those two files actually contain:

```
aATSTp9B9Q4_43000_53000
aATSTp9B9Q4_31000_41000
```

Those are **YouTube video IDs** in the VGGSound file list, coincidentally
containing the letters `atst`. Not the model. ATST appears in the AudioGPT
*paper* as part of the unified token-based architecture, but **the released
code contains no ATST at all**.

This is the single most valuable correction in the whole pass, and it is worth
recording *how* I nearly got it wrong. A grep hit count of 2 looks like
evidence. It was noise. Had I written "ATST is partially vendored" I would have
sent a confident falsehood downstream, and the file:line convention I was
given is only as good as the discipline of checking what a matching line
actually *is*. Substring matches inside data files are the classic trap —
`grep` does not know the difference between an identifier and a word that
happens to contain one.

**Rule I took from this:** for every claim about model identity, I resolved it
to a file:line and read the surrounding context, rather than trusting a match.

## Round 3 — AudioGPT: the architecture question was answered by the prompt

My brief asked: "Is the LLM actually conditioning audio, or is it a text LLM
bolted to audio front-ends?"

I expected to have to trace the tensor flow to answer this. I did not. The
system prompt handed to the LLM states it outright, at
`audio-chatgpt.py:38`:

> AudioGPT can not directly read audios, but it has a list of tools to finish
> different speech, audio, and singing voice tasks. Each audio will have a file
> name formed as "audio/xxx.wav".

So the LLM's entire channel for audio information is **a filename string and a
text description**. Combined with `audio-chatgpt.py:16` (`from
langchain.llms.openai import OpenAI`) and `audio-chatgpt.py:1054` (`self.llm
= OpenAI(temperature=0)`), the picture is complete without any further
tracing: a text-only LLM calling audio models as LangChain tools. It is
**not** an audio-conditioned LLM. The interesting audio research (Make-An-Audio,
GenerSpeech, DiffSinger) all lives in the *tools*, not in the reasoning layer.

This changed my verdict on AudioGPT from "evaluate the model" to "evaluate the
glue" — and the glue is where the value question actually lies.

## Round 4 — where I wasted effort

Honest accounting of time that bought nothing:

1. **Counting AudioGPT's 260 `.py` files as a quality signal.** They look
   impressive. 125 are a vendored copy of NeuralSeq and 78 are a vendored
   Make-An-Audio/LDM tree. AudioGPT's own surface is ~57 files, one of which
   (`audio-chatgpt.py`, 1442 lines) is the actual program. The "260 py" number
   is worth almost nothing as a health metric in a repo that vendors upstream
   trees. The number that matters is *files the maintainer wrote*.

2. **Hunting for a hidden benchmark in AudioGPT.** I expected a paper-grade
   evaluation table somewhere. There is none — `grep -icE "cer|wer|benchmark|
   mos|score|FID" README.md` returns **0**. The README is a capability table
   with hand-waving statuses ("Yes (WIP)"). There is nothing to extract and no
   point looking further.

3. **Looking for a `Cargo.toml`-style dependency contradiction.** The brief
   cited a prior finding of "a README naming a model with no dependency on it".
   In AudioGPT the equivalent is milder: the README's model table is
   *honest* about not shipping things ("Currently not every model has
   repository", README.md:21), so the search for a contradiction came up empty.
   AudioGPT is not a lying repo — it is a thin repo. Those need different
   treatments and I initially reached for the wrong one.

## Round 5 — the surprise that actually reframed the task

I came in with a "which model is fastest" framing. FunASR contains a document
that argues that framing is unsound,
`docs/benchmark/rtf_reproducibility.md`:

> Do not compare an offline batch `RTFx` result with streaming first-token
> latency or end-to-end product latency. They measure different things.

...and then supplies a **required-fields table** (data, model revision, runtime,
hardware, software versions, pipeline, batching, *timing scope*, quality
method) that any speed claim must fill in. That is more useful to us than any
single RTF number in any of the three repos, because our stated constraint is
*"latency matters more than quality; we need to know RTF numbers"* — and
documents like this are exactly what stop us from buying a fast number that
measures the wrong interval.

I did not go looking for this. It surfaced only because I checked whether CI
actually ran anything, then followed the docs it referenced.

The same doc also warns against trusting a headline throughput: Fun-ASR-Nano
is `RTFx 340` under vLLM but `RTFx 21` on the plain PyTorch baseline — a
**16× swing from the serving stack alone**. Any RTF figure we see quoted at
someone is meaningless without its runtime.

## Round 6 — what changed my mind about the ranking

My mid-pass ranking, before I read the code, was roughly "FunASR for ASR,
VoxCPM for speech out, AudioGPT ignore". What the evidence produced:

- **AudioGPT** — dead end for us, and not because the code is bad. Because it
  needs an OpenAI key, two CUDA devices, 2022-era LangChain, and ships an
  **empty LICENSE file (0 bytes)**. Not even legally reusable.
- **VoxCPM** — the best *mechanisms* of the three, but the worst news for our
  latency constraint: its RTF is excellent on an RTX 4090 and **1.76 on an M4
  Pro via llama.cpp — slower than real time**. The number that matters for a
  cheap continuous peripheral is the bad one, and the README leads with the
  good one.
- **FunASR** — by far the most real repo, and it contains the single mechanism
  that directly solves our problem: **2pass** streaming.

The 2pass discovery is the payoff of doing this properly. Streaming ASR is
measurably worse than offline (Paraformer Aishell1 dev CER **2.37 online vs
1.76 offline**). The naive response is "streaming is too lossy". FunASR's
answer is to run the streaming model for latency and the offline model as an
end-of-sentence corrector — getting offline accuracy at streaming latency. It
is a `-mode 2pass` flag and a prebuilt server binary.

## What I would tell my past self

- Establish *category* before evaluating. I lost time trying to compare three
  things that were never on the same axis.
- Verify a grep hit by reading it. Two lines of `aATSTp9B9Q4_43000_53000`
  nearly became a false finding.
- Count authored files, not tracked files. Vendored trees inflate everything.
- When a repo ships a measurement-methodology doc, that doc is often worth
  more than the model. FunASR's is the best writing in this whole pass.

See `SYNTHESIS.md` for what we should actually do with all of this.