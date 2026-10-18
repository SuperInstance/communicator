/**
 * Model request payload construction.
 *
 * This module exists because it holds the one assumption in this directory that
 * I could NOT verify, and an unverified assumption should be isolated in one
 * small file where it is obvious, not scattered through request handling.
 *
 * WHAT IS VERIFIED (read-only, 2026-10-18):
 *   - `@cf/pipecat-ai/smart-turn-v2` exists in the catalog and is priced
 *     $0.000338 per audio minute; its OUTPUT schema is exactly
 *     {is_complete: boolean, probability: number}. No transcript field exists,
 *     which is why the gate cannot use it for text.
 *   - Its INPUT schema accepts either an audio stream
 *     ({body, contentType}) or a base64 string, plus an optional `dtype`.
 *   - clef-omni's input requires `model`, `state` and `questions`; `audio` is
 *     an ARRAY of at most 4 embedded clips, 8 MiB and 300 s each, and "Remote
 *     URLs are not accepted."
 *
 * WHAT IS NOT VERIFIED:
 *   The exact byte encoding smart-turn-v2 wants for PCM. The schema accepts
 *   `dtype` of uint8/float32/float64 but does not document the container
 *   (raw array vs WAV) or the sample rate it expects. The watcher's native grid
 *   is 16 kHz / 16-bit mono (src/watcher.py FRAME_LEN 800 at 16 kHz).
 *
 *   I did not resolve this by running an inference, because that spends neurons
 *   and this task forbids mutating spend. So the encoder is isolated here and
 *   defaults to the documented base64 form; fixing it is a change to this file
 *   alone. Until it is measured against a real call, treat the smart-turn audio
 *   path as UNVERIFIED and do not rely on `probability` being calibrated.
 */

/** Clef question set asked of every promoted clip.
 *
 * Deliberately small and typed. clef-omni accepts 1 to 64 questions
 * (schema-input.json); every question costs input tokens against a budget that
 * is genuinely finite, so this starts at two and grows only when a measurement
 * says a question earns its tokens.
 *
 * `type: "choice"` returns probabilities per option plus a confidence, which is
 * what the existing local judge consumes (src/judged.py). `type: "noul"` returns
 * the probability of yes. Both are the typed primitives the System One API
 * documents.
 */
export function buildClefPayload(clip) {
  return {
    state: describeContext(clip),
    questions: {
      is_speech: {
        type: "noul",
        instructions: "Is this recognisable human speech?",
        criteria: {
          true: "It is recognisable human speech.",
          false: "It is not recognisable human speech.",
        },
      },
      is_directive: {
        type: "choice",
        instructions: "Is the speaker asking for something to be done?",
        criteria: {
          yes: "The speaker asks for something to be done.",
          no: "The speaker is not asking for anything.",
        },
      },
    },
  };
}

/**
 * Textual context accompanying the audio.
 *
 * clef-omni requires a `state`, and the schema notes "Long text state is
 * truncated to fit the model's token limit" -- so this is kept deliberately
 * terse. Spending tokens restating what the audio already contains would buy
 * nothing and would quietly inflate every judgement's cost.
 */
function describeContext(clip) {
  return [
    `sample_rate_hz: ${clip.sample_rate ?? 16000}`,
    `duration_s: ${(clip.duration_s ?? 0).toFixed(3)}`,
    `origin: ${clip.origin ?? "acoustic"}`,
  ].join("\n");
}

/**
 * smart-turn-v2 request body.
 *
 * Base64 string form, which the published input schema lists first. See the
 * module docstring: the exact PCM encoding is UNVERIFIED and isolated here.
 */
export function buildSmartTurnPayload(clip) {
  if (typeof clip.audio_b64 !== "string" || !clip.audio_b64.length) {
    throw new Error(`clip ${clip.event_id} has no audio_b64`);
  }
  return {
    audio: clip.audio_b64,
    // The watcher's native grid. Passed explicitly because the schema offers no
    // sample-rate field and an implicit default would be a guess.
    dtype: "uint8",
  };
}