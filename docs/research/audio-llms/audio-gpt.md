# AudioGPT — AIGC-Audio/AudioGPT

*Audited 2026-10-18 at commit `a674543c537bb3a60b8f521df76b97a627c1c379`
(`main`). All claims below verified against the working tree, not the README.*

## Verdict up front

**NOT USEFUL TO US.** This is not primarily a quality judgment — it is a
*category* mismatch. AudioGPT is not an audio language model; it is a
LangChain tool-calling agent wrapped around a dozen unrelated audio models, and
the reasoning layer never touches audio. It also cannot run locally without an
OpenAI key, and its LICENSE file is empty.

## What it actually is, end to end

The system prompt the LLM receives states the architecture directly
(`audio-chatgpt.py:38`):

> AudioGPT can not directly read audios, but it has a list of tools to finish
> different speech, audio, and singing voice tasks. Each audio will have a file
> name formed as "audio/xxx.wav". When talking about audios, AudioGPT is very
> strict to the file name and will never fabricate nonexistent files.

That is the whole story. The LLM's entire channel for audio information is a
**filename string**. Supporting evidence:

| Evidence | Location | What it proves |
|---|---|---|
| `from langchain.llms.openai import OpenAI` | `audio-chatgpt.py:16` | The reasoning LLM is OpenAI's |
| `self.llm = OpenAI(temperature=0)` | `audio-chatgpt.py:1054` | No local LLM; temperature pinned to 0 |
| `agent="conversational-react-description"` | `audio-chatgpt.py:1145`, `:1189` | ReAct tool-loop, not an audio decoder |
| `export OPENAI_API_KEY=...` | `run.md` | An API key is a hard requirement |

`run.md` is explicit: "prepare your private openAI private key".

**So: text LLM bolted to audio front-ends.** The parent's suspicion is
confirmed, and it is confirmed *by the project's own prompt*, not by inference.
Audio conditioning happens inside each tool model, never at the reasoning
layer. There is no audio embedding path into the LLM anywhere in the file.

### Answering "is it a text LLM bolted to front-ends?" precisely

The subtlety worth recording: this is **not** a criticism of the tools. The
tool models are real and mostly substantial — Make-An-Audio for text-to-audio,
GenerSpeech for style transfer, DiffSinger/VISinger for singing, Whisper and
Conformer for recognition, plus enhancement/separation/detection heads
(`audio-chatgpt.py:104-1050`). The *composition* is what is thin: a
filename-passing ReAct loop over ~20 tools. The intelligence is in the
vendored third-party code, not in AudioGPT.

## REAL vs described

### The README's model table does not describe this repo

README.md:21 says "Currently not every model has repository" — the repo is
**honest** that the table overstates what ships. Scoring it on
scaffold-vs-execution framing is therefore the wrong lens: this is not a repo
that lies, it is a repo that is thin and says so.

### ATST is absent — and the near-miss is the lesson

The brief listed ATST as an audio front-end. It is not in the code:

```
$ grep -ril "atst" . --exclude-dir=.git
./text_to_audio/Make_An_Audio/ldm/modules/losses_audio/vggishish/data/vggsound_train.txt
./text_to_audio/Make_An_Audio/ldm/modules/losses_audio/vggishish/data/vggsound.csv
```

Both hits are substring matches inside VGGSound **YouTube video IDs**:

```
aATSTp9B9Q4_43000_53000
aATSTp9B9Q4_31000_41000
```

ATST is part of the AudioGPT *paper's* unified token architecture. **Zero ATST
in the released code.** If we had taken the grep hit count at face value we
would have propagated a false claim.

### No measurements of any kind

```
$ grep -icE "cer|wer|benchmark|mos|score|FID" README.md
0
```

There are **no quality benchmarks, no parameter counts, no RTF figures**
anywhere in the README. For a project whose sibling claims are benchmarked
heavily, this is a meaningful absence: there is no evidence base to evaluate.

### File census — most of the code is vendored

| Bucket | `.py` files | Note |
|---|---|---|
| `NeuralSeq/` | 125 | vendored upstream (DiffSinger-family seq encoder) |
| `text_to_audio/Make_An_Audio/` | 78 | vendored LDM tree |
| AudioGPT-authored | ~57 | including the 1442-line `audio-chatgpt.py` |

260 tracked `.py` files sounds substantial; ~57 are theirs.

### Zero tests, zero CI

```
$ ls .github          → No such file or directory
```

There is no `.github/` directory at all — **no CI whatsoever**. The six
`test`-matching tracked files are a vendored emotion script, a weak-label CSV,
a vendored data script, a PNG, and a text file. Not one asserts AudioGPT
behaviour.

### The LICENSE is 0 bytes

```
$ wc -c LICENSE
0 LICENSE
```

An empty license file means **no grant of any kind**. Combined with the
acknowledgement section listing ESPNet / Visual ChatGPT / Stable Diffusion
upstream work, this is a hard stop on code reuse: we could not legally adopt
any of it, only independently reimplement ideas.

## Does it handle speech / music / sound separately?

Yes — but by **tool routing**, not by modality-aware modelling. There are
distinct tools for TTS (`Synthesize Speech Given the User Input Text`),
singing (`Generate Singing Voice From User Input Text, Note and Duration
Sequence`), text-to-audio, and sound separation/detection
(`audio-chatgpt.py:1075-1195`). The separation is *taxonomic in the tool
namespace*, decided by the LLM reading a text request. There is no unified
audio representation spanning the categories.

## Parameter counts

**None published, and none derivable.** The README contains no parameter counts
and the code loads ~20 heterogeneous models from external checkpoints. The
LLM itself is an off-the-shelf OpenAI model whose identity and size are never
specified (`OpenAI(temperature=0)` takes no `model_name`). Anyone quoting an
"AudioGPT parameter count" is quoting the paper, not this repo.

## Does it run locally?

**No.** Four independent blockers:

1. **OpenAI API key required** — `audio-chatgpt.py:1054`, `run.md`. The
   reasoning layer is not local and cannot be made local without replacing it.
2. **Two CUDA devices hardcoded** — `audio-chatgpt.py:1055` pins `T2I` to
   `cuda:1`, `cuda:0` for ~14 others. Literal device indices, not `auto`.
3. **Dead dependency API** — `requirements.txt` pins `langchain==0.0.101` and
   `audio-chatgpt.py:13` imports `from langchain.agents.initialize import
   initialize_agent`, a module path removed in modern LangChain. This code
   cannot install on a current environment.
4. **`download.sh` is partly broken** — many `wget` lines use HF `/blob/`
   URLs (the HTML page) rather than `/resolve/` (the file), and several use
   `wget -i <single-url>` where `-i` expects a *file* of URLs. Weight fetching
   is unreliable by construction.

Weights are not committed (`0` files matching `.pt|.ckpt|.pth|.bin|.safetensors`)
and are pulled from third-party Spaces, not AudioGPT's own org.

## STEAL THIS

Thin, but three things are real:

1. **The tool-namespace-as-modality-router idea.** Listing capabilities as
   discrete, explicitly-described tools with explicit input contracts
   (`audio-chatgpt.py:1075-1195`) is a legitimately good pattern for a
   peripheral that must decide "is this speech, a musical event, or a noise?"
   before acting. The description strings double as dispatch documentation.
   Our trigger layer could borrow the shape without the OpenAI dependency.

2. **The "never fabricate a filename" prompt rule** (`audio-chatgpt.py:38`).
   When an agent's output is a *handle* to a media artifact, constraining it to
   emit only handles that came from a tool observation is a cheap hallucination
   guard. Directly applicable to a system that returns audio to a voice
   peripheral.

3. **The honest capability table** (README.md:21) is worth copying as a
   *norm*: state what is not shipped. We would rather have this than the
   alternative.

## NOT USEFUL TO US

- **The LLM architecture — zero transfer.** Filename-as-audio-channel is not a
  pattern we want; it is an admission of no audio conditioning.
- **Nothing runs locally or CPU.** Two hardcoded CUDA indices plus a mandatory
  OpenAI key is disqualifying against "continuous and cheap" and against a
  WSL box with no GPU budget worth burning.
- **Empty LICENSE** — legally unusable for reuse.
- **No benchmarks, no tests, no CI** — there is no regression signal and no
  measured basis for any claim. Adopting it would mean adopting ~57 files of
  unverified orchestration.
- **Latency is unmeasured and structurally poor** for us: a chat-completion
  round trip sits between the user's speech and any tool call. Nothing here is
  built for a continuous peripheral with a sub-second budget.

## Honest closing note

I went in ready to find this repo deceptive and found the opposite: it
overstates in its README table but *discloses* the overstatement inline, and
its README makes no benchmark claims to begin with. The finding here is not
"caught it lying" — it is "it is a thin integration layer with no license and
no local path". That is a less dramatic and more accurate result.