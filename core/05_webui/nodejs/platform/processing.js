'use strict';

/**
 * Processing: turning a department's stored files into its chatbot's passages.
 * Started by a person (the Process button), never by an upload.
 *
 * One file at a time, in upload order: the bot's Python service reads it, chunks
 * it with the bots' chunker and embeds it (dept_service.py /api/dept/extract);
 * the passages are saved with their vectors and the file becomes 'done'. A file
 * whose text has to come from images (a scan) becomes 'needs_ocr' — OCR is its
 * own step, run while the bots are not answering — and an unreadable one
 * 'failed', each with the reason. A restart puts an interrupted file back in the
 * queue.
 */

const EXTRACT_TIMEOUT_MS = 60 * 60 * 1000;     // a large document on a busy GPU

function createProcessor({ store, pathOf, py, log = console }) {
  let running = false;
  let wanted = false;
  let current = null;

  async function work() {
    if (running) { wanted = true; return; }
    running = true;
    try {
      do {
        wanted = false;
        for (;;) {
          const f = await store.claimNext();
          if (!f) break;
          current = f.id;
          await one(f);
          current = null;
        }
      } while (wanted);
    } catch (err) {
      log.error(`[platform] processing stopped: ${err.message}`);
    } finally {
      running = false;
      current = null;
    }
  }

  async function one(f) {
    const t0 = Date.now();
    let out;
    try {
      out = await py('/api/dept/extract', { path: pathOf(f), kind: f.kind, name: f.name }, EXTRACT_TIMEOUT_MS);
    } catch (err) {
      await store.setOutcome(f.id, 'failed', err.message);
      log.error(`[platform] processing ${f.name}: ${err.message}`);
      return;
    }
    if (out.status === 'done' && out.chunks && out.chunks.length) {
      const saved = await store.saveChunks(f.id, out.chunks, out.pages);
      log.log(`[platform] processed ${f.name}: ${saved ? `${out.chunks.length} passages` : 'deleted meanwhile'} `
        + `in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    } else {
      await store.setOutcome(f.id, out.status === 'needs_ocr' ? 'needs_ocr' : 'failed', out.reason || 'No text.');
      log.log(`[platform] ${f.name}: ${out.status} — ${out.reason}`);
    }
  }

  return {
    /** Queue files (ids, or every new file) and start working. Returns how many were queued. */
    async queue(departmentId, fileIds) {
      const n = await store.queueFiles(departmentId, fileIds);
      if (n) work();
      return n;
    },
    /** After a restart: put interrupted files back and carry on with the queue. */
    async resume() {
      const n = await store.requeueInterrupted();
      if (n) log.log(`[platform] ${n} interrupted file(s) back in the processing queue`);
      work();
    },
    busy: () => running,
    current: () => current,
    whenIdle: async () => { while (running) await new Promise((r) => setTimeout(r, 20)); },
  };
}

module.exports = { createProcessor };
