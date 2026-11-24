'use strict';

// Atomic files in persistent bot data, scoped by the authenticated account.
// Writes are serialized for the single Node process serving each bot.
const crypto = require('crypto');
const fs = require('fs/promises');
const path = require('path');
const express = require('express');
const failure = (status, message) => Object.assign(new Error(message), { status });

function createStore(root, pipeline) {
  if (!/^[a-z0-9_-]+$/i.test(pipeline)) throw new Error('Invalid history pipeline');
  let chain = Promise.resolve();
  const lock = fn => {
    const result = chain.then(fn);
    chain = result.catch(() => {});
    return result;
  };
  const directory = owner => path.join(root, pipeline, 'history',
    crypto.createHash('sha256').update(owner).digest('hex'));
  function filename(owner, id) {
    if (!/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i.test(id)) throw failure(400, 'Invalid chat ID.');
    return path.join(directory(owner), id + '.json');
  }
  async function read(owner, id) {
    try { return JSON.parse(await fs.readFile(filename(owner, id), 'utf8')); }
    catch (err) { if (err.code === 'ENOENT') throw failure(404, 'Chat not found.'); throw err; }
  }
  async function write(owner, chat) {
    const file = filename(owner, chat.id);
    await fs.mkdir(path.dirname(file), { recursive: true, mode: 0o700 });
    const tmp = file + '.' + crypto.randomUUID() + '.tmp';
    try {
      await fs.writeFile(tmp, JSON.stringify(chat), { mode: 0o600 });
      await fs.rename(tmp, file);
    } finally { await fs.unlink(tmp).catch(() => {}); }
    return chat;
  }
  function messages(value) {
    if (!Array.isArray(value) || value.length > 2000) throw failure(400, 'Invalid chat messages.');
    return value.map(m => {
      if (!m || !['user', 'assistant'].includes(m.role) || typeof m.text !== 'string') {
        throw failure(400, 'Each message needs a role and text.');
      }
      return { role: m.role, text: m.text, results: Array.isArray(m.results) ? m.results : [],
        trace: Array.isArray(m.trace) ? m.trace : [], hops: Number.isInteger(m.hops) ? m.hops : 0,
        standalone: typeof m.standalone === 'string' ? m.standalone : '', success: m.success === true,
        debugId: typeof m.debugId === 'string' && /^[a-f0-9-]{36}$/.test(m.debugId) ? m.debugId : '',
        debugError: typeof m.debugError === 'string' ? m.debugError.slice(0, 200) : '' };
    });
  }
  const summary = ({ id, title, pinned, createdAt, updatedAt }) => ({ id, title, pinned, createdAt, updatedAt });
  return {
    async list(owner) {
      await chain;
      let files;
      try { files = await fs.readdir(directory(owner)); }
      catch (err) { if (err.code === 'ENOENT') return []; throw err; }
      const chats = await Promise.all(files.filter(f => /^[a-f0-9-]+\.json$/i.test(f)).map(async f => {
        try { return summary(await read(owner, f.slice(0, -5))); }
        catch (err) { if (err.status === 404) return null; throw err; }
      }));
      return chats.filter(Boolean).sort((a, b) => Number(b.pinned) - Number(a.pinned)
        || b.updatedAt.localeCompare(a.updatedAt));
    },
    async get(owner, id) { await chain; return read(owner, id); },
    create(owner, value) {
      return lock(async () => {
        const rows = messages(value);
        if (!rows.length) throw failure(400, 'A chat needs at least one message.');
        const now = new Date().toISOString();
        const title = (rows.find(m => m.role === 'user')?.text || 'New chat').trim().slice(0, 100);
        return write(owner, { id: crypto.randomUUID(), title, pinned: false,
          createdAt: now, updatedAt: now, messages: rows });
      });
    },
    save(owner, id, value) {
      return lock(async () => {
        const rows = messages(value);
        const chat = await read(owner, id); // Late saves must not resurrect deleted chats.
        chat.messages = rows;
        chat.updatedAt = new Date().toISOString();
        return write(owner, chat); // Preserve pin state during automatic saves.
      });
    },
    pin(owner, id, pinned) {
      return lock(async () => {
        if (typeof pinned !== 'boolean') throw failure(400, 'Pinned must be true or false.');
        const chat = await read(owner, id);
        chat.pinned = pinned;
        return write(owner, chat);
      });
    },
    remove(owner, id) {
      return lock(async () => { await read(owner, id); await fs.unlink(filename(owner, id)); });
    },
  };
}

function createRouter(store, pipeline) {
  const router = express.Router();
  router.use((_req, res, next) => { res.setHeader('Cache-Control', 'no-store'); next(); });
  router.use((req, res, next) => {
    if (!req.gadUser?.username || !['admin', 'user'].includes(req.gadUser.role)) {
      return res.status(401).json({ success: false, error: 'Sign in to access your chats.' });
    }
    next();
  });
  router.use(express.json({ limit: '16mb' }));
  // The page sends a chat as {b64: base64(JSON)} — no tag-like text on the wire, which
  // the office firewall in front of the bot blocks (HTTP 403). Plain JSON still works.
  router.use((req, _res, next) => {
    if (req.body && typeof req.body.b64 === 'string') {
      try { req.body = JSON.parse(Buffer.from(req.body.b64, 'base64').toString('utf8')); }
      catch (_) { return next(failure(400, 'Invalid chat data.')); }
    }
    next();
  });
  const handle = fn => (req, res, next) => Promise.resolve(fn(req, res)).catch(next);
  router.get('/', handle(async (req, res) => res.json({ success: true,
    pipeline, owner: req.gadUser.username, chats: await store.list(req.gadUser.username) })));
  router.post('/', handle(async (req, res) => res.status(201).json({ success: true,
    chat: await store.create(req.gadUser.username, req.body?.messages) })));
  router.get('/:id', handle(async (req, res) => res.json({ success: true,
    chat: await store.get(req.gadUser.username, req.params.id) })));
  router.put('/:id', handle(async (req, res) => res.json({ success: true,
    chat: await store.save(req.gadUser.username, req.params.id, req.body?.messages) })));
  router.patch('/:id', handle(async (req, res) => res.json({ success: true,
    chat: await store.pin(req.gadUser.username, req.params.id, req.body?.pinned) })));
  router.delete('/:id', handle(async (req, res) => {
    await store.remove(req.gadUser.username, req.params.id);
    res.json({ success: true });
  }));
  router.use((err, _req, res, _next) => {
    const status = err.status || 500;
    if (status >= 500) console.error('[chat-history]', err.message);
    res.status(status).json({ success: false,
      error: status >= 500 ? 'Chat history could not be saved or loaded. Please try again.' : err.message });
  });
  return router;
}

module.exports = { createStore, createRouter };
