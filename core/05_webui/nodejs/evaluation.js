'use strict';

// Review snapshots are independent of chat history, so deleting a conversation
// does not discard a question already saved for evaluation. Each admin/bot
// can access only its own queue; ordinary user accounts cannot access Evaluation.
const crypto = require('crypto');
const fs = require('fs/promises');
const path = require('path');
const express = require('express');
const failure = (status, message) => Object.assign(new Error(message), { status });

function createStore(root, pipeline, history) {
  if (!/^[a-z0-9_-]+$/i.test(pipeline)) throw new Error('Invalid evaluation pipeline');
  let chain = Promise.resolve();
  const lock = fn => {
    const result = chain.then(fn);
    chain = result.catch(() => {});
    return result;
  };
  const directory = owner => path.join(root, pipeline, 'evaluation',
    crypto.createHash('sha256').update(owner).digest('hex'));
  const filename = (owner, id) => {
    if (!/^[a-f0-9]{64}$/.test(id)) throw failure(400, 'Invalid evaluation ID.');
    return path.join(directory(owner), id + '.json');
  };
  async function read(owner, id) {
    try { return JSON.parse(await fs.readFile(filename(owner, id), 'utf8')); }
    catch (err) { if (err.code === 'ENOENT') throw failure(404, 'Evaluation not found.'); throw err; }
  }
  async function write(owner, item) {
    const file = filename(owner, item.id);
    await fs.mkdir(path.dirname(file), { recursive: true, mode: 0o700 });
    const tmp = file + '.' + crypto.randomUUID() + '.tmp';
    try {
      await fs.writeFile(tmp, JSON.stringify(item), { mode: 0o600 });
      await fs.rename(tmp, file);
    } finally { await fs.unlink(tmp).catch(() => {}); }
    return item;
  }
  return {
    async list(owner, chatId) {
      await chain;
      let files;
      try { files = await fs.readdir(directory(owner)); }
      catch (err) { if (err.code === 'ENOENT') return []; throw err; }
      const items = await Promise.all(files.filter(f => /^[a-f0-9]{64}\.json$/.test(f))
        .map(f => read(owner, f.slice(0, -5))));
      return items.filter(item => !chatId || item.chatId === chatId)
        .sort((a, b) => b.savedAt.localeCompare(a.savedAt))
        .map(({ sources, trace, ...item }) => item);
    },
    async get(owner, id) { await chain; return read(owner, id); },
    save(owner, chatId, messageIndex) {
      return lock(async () => {
        if (typeof chatId !== 'string' || !Number.isInteger(messageIndex) || messageIndex < 0) {
          throw failure(400, 'Select a completed question and answer.');
        }
        // Stable ID makes double-clicks/retries idempotent, even across restarts.
        const id = crypto.createHash('sha256').update(`${chatId}:${messageIndex}`).digest('hex');
        try { return await read(owner, id); }
        catch (err) { if (err.status !== 404) throw err; }
        const chat = await history.get(owner, chatId);
        const question = chat.messages[messageIndex];
        const answer = chat.messages[messageIndex + 1];
        if (question?.role !== 'user' || answer?.role !== 'assistant' || !answer.text?.trim()) {
          throw failure(400, 'Wait for the answer to finish before saving it for review.');
        }
        return write(owner, { id, chatId, messageIndex, chatTitle: chat.title,
          question: question.text, answer: answer.text, sources: answer.results || [],
          trace: answer.trace || [], standalone: answer.standalone || '', debugId: answer.debugId || '',
          savedAt: new Date().toISOString(), status: 'pending', reviewedAt: null });
      });
    },
    review(owner, id, reviewed) {
      return lock(async () => {
        if (typeof reviewed !== 'boolean') throw failure(400, 'Reviewed must be true or false.');
        const item = await read(owner, id);
        item.status = reviewed ? 'reviewed' : 'pending';
        item.reviewedAt = reviewed ? new Date().toISOString() : null;
        return write(owner, item);
      });
    },
  };
}

function createRouter(store) {
  const router = express.Router();
  router.use((req, res, next) => {
    res.setHeader('Cache-Control', 'no-store');
    if (!req.gadUser?.username) {
      return res.status(401).json({ success: false, error: 'Sign in to access your evaluation queue.' });
    }
    if (req.gadUser.role !== 'admin') {
      return res.status(403).json({ success: false, error: 'Admin access required for Evaluation.' });
    }
    next();
  });
  router.use(express.json({ limit: '8kb' }));
  const handle = fn => (req, res, next) => Promise.resolve(fn(req, res)).catch(next);
  router.get('/', handle(async (req, res) => res.json({ success: true,
    items: await store.list(req.gadUser.username, req.query.chatId) })));
  router.post('/', handle(async (req, res) => res.json({ success: true,
    item: await store.save(req.gadUser.username, req.body?.chatId, req.body?.messageIndex) })));
  router.get('/:id', handle(async (req, res) => res.json({ success: true,
    item: await store.get(req.gadUser.username, req.params.id) })));
  router.patch('/:id', handle(async (req, res) => res.json({ success: true,
    item: await store.review(req.gadUser.username, req.params.id, req.body?.reviewed) })));
  router.use((err, _req, res, _next) => {
    const status = err.status || 500;
    if (status >= 500) console.error('[evaluation]', err.message);
    res.status(status).json({ success: false,
      error: status >= 500 ? 'The evaluation could not be saved or loaded. Please try again.' : err.message });
  });
  return router;
}

module.exports = { createStore, createRouter };
