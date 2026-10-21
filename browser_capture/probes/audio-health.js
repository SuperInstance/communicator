/* audio-health -- patrol the measured WAV health of recent sessions.
 * The SILENT_AUDIO class, patrolled forever, not remembered: the liaison
 * already alerts at save time; this probe re-reads the measured truth on
 * a schedule from the page, so a silent-input regression is visible from
 * the lab rows even if nobody is watching the outbox.
 *
 * Bump meta.version to hot-swap into already-open pages (no reload).
 */
export const meta = { name: "audio-health", version: 1 };

export async function run(ctx) {
  const r = await fetch("/api/sessions");
  const j = await r.json();
  const recent = (j.sessions || []).slice(0, 5);
  if (!recent.length)
    return { summary: "no sessions on disk yet" };
  const silent = recent.filter(s => s.audio_silent).length;
  const worst = recent[recent.length - 1];
  const newest = recent[0];
  const summary = recent.length + " recent: " +
    (silent === 0 ? "all measure audio" :
      silent + " SILENT (newest silent: " +
      (newest.audio_silent ? newest.dir : "no") + ")") +
    " | newest " + newest.dir + " peak=" +
    (newest.audio_measured ? newest.audio_measured.peak_abs : "?") +
    " rms=" + (newest.audio_measured ? newest.audio_measured.rms : "?");
  await ctx.result({ summary: summary, silent_recent: silent,
    newest_peak: newest.audio_measured &&
      newest.audio_measured.peak_abs });
  if (silent > 0)
    throw new Error(silent + " of the last " + recent.length +
      " sessions measure SILENT audio (" + worst.dir + " era)");
  return { summary: summary };
}
