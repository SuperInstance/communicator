/**
 * Bounded-concurrency map.
 *
 * Exists for one reason: the platform allows six simultaneous open connections
 * per invocation (developers.cloudflare.com/workers/platform/limits/
 * #simultaneous-open-connections), and a Workers AI call occupies exactly one
 * for the whole inference. `Promise.all(clips.map(...))` would open one
 * connection per clip at once and get the invocation killed at the seventh --
 * a failure that looks like a random provider error rather than a limit, which
 * is what makes it expensive to debug later.
 *
 * Note this is NOT the subrequest limit (50/request on free). Count and
 * concurrency are different limits: 50 calls issued sequentially is fine, and 7
 * issued simultaneously is not.
 *
 * Rejections are captured per item rather than aborting the batch. One failed
 * clip must not cost the whole batch its budget accounting or lose the clips
 * that succeeded.
 */
export async function mapLimit(items, limit, fn) {
  if (!Number.isInteger(limit) || limit < 1) {
    throw new Error(`mapLimit: limit must be a positive integer, got ${limit}`);
  }
  const results = new Array(items.length);
  let cursor = 0;

  async function worker() {
    for (;;) {
      const index = cursor++;
      if (index >= items.length) return;
      try {
        results[index] = await fn(items[index], index);
      } catch (err) {
        results[index] = { ok: false, error: String(err?.message ?? err) };
      }
    }
  }

  const workers = [];
  const width = Math.min(limit, items.length);
  for (let i = 0; i < width; i++) workers.push(worker());
  await Promise.all(workers);
  return results;
}