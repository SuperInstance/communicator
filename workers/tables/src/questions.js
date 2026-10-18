/**
 * The clef question set for the tables plane.
 *
 * WHY THIS IS DUPLICATED rather than imported from ../../hotpath/src/media.js:
 * hotpath and tables are SEPARATE wrangler projects with separate
 * wrangler.toml entries, and wrangler bundles each project from its own
 * root. A relative import that crosses the project boundary would not survive
 * bundling, so sharing the source would be a false economy that breaks on
 * deploy. The duplication is four constants and a comment; the shared thing
 * that actually must not drift is costs.json, which both projects bind.
 *
 * If these questions are ever centralised properly, the right home is a small
 * shared module included by both projects' bundler config -- not an import
 * that reaches outside the project.
 *
 * clef-omni accepts 1 to 64 typed questions (its published input schema). Every
 * question costs input tokens against a genuinely finite neuron budget, so
 * this stays at two until a measurement justifies a third.
 */

/** clef-omni requires this field; the schema pins the value. */
export const CLEF_MODEL_SELECTOR = "clef-omni";

export function clefQuestions() {
  return {
    // noul -> the probability of "yes". Used as a scalar speech-legibility
    // probe that the gate does not need to make itself.
    is_speech: {
      type: "noul",
      instructions: "Is this recognisable human speech?",
      criteria: {
        true: "It is recognisable human speech.",
        false: "It is not recognisable human speech.",
      },
    },
    // choice -> per-option probabilities plus a confidence. This is the
    // primitive the local judge in src/judged.py already consumes, so the
    // cloud path returns a shape the existing table code can read.
    is_directive: {
      type: "choice",
      instructions: "Is the speaker asking for something to be done?",
      criteria: {
        yes: "The speaker asks for something to be done.",
        no: "The speaker is not asking for anything.",
      },
    },
  };
}

/**
 * The textual `state` clef-omni requires alongside the questions.
 *
 * Deliberately terse. The schema warns "Long text state is truncated to fit the
 * model's token limit", and restating what the audio already contains would buy
 * nothing while inflating the cost of every single judgement.
 */
export function clefState({ sample_rate, duration_s, origin }) {
  return [
    `sample_rate_hz: ${sample_rate ?? 16000}`,
    `duration_s: ${Number(duration_s ?? 0).toFixed(3)}`,
    `origin: ${origin ?? "acoustic"}`,
  ].join("\n");
}