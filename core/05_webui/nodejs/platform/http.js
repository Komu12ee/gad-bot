'use strict';

// Small helpers shared by the platform's routers.

const express = require('express');

const failure = (status, message, extra) => Object.assign(new Error(message), { status }, extra || {});
const EXTRA_FIELDS = ['freeBytes', 'usedBytes', 'received', 'retryAfter'];

/** An async handler whose errors go to the error handler. */
const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res)).catch(next);

/** An async check run before the routes after it (throws to refuse). */
const before = (fn) => (req, res, next) => Promise.resolve(fn(req, res)).then(() => next(), next);

/**
 * JSON bodies (room for one 4 MB upload piece in base64). Bodies other than upload
 * pieces come as {b64: base64(JSON)}: names and passwords are typed text, and the
 * office firewall in front of the bots refuses tag-like text. Plain JSON works too.
 */
function jsonBody() {
  const parse = express.json({ limit: '8mb' });
  return [parse, (req, _res, next) => {
    if (req.body && typeof req.body.b64 === 'string') {
      try { req.body = JSON.parse(Buffer.from(req.body.b64, 'base64').toString('utf8')); }
      catch (_) { return next(failure(400, 'Invalid request data.')); }
    }
    next();
  }];
}

// eslint-disable-next-line no-unused-vars
function errorHandler(err, _req, res, _next) {
  if (res.headersSent) return;
  const status = err.status || err.statusCode || 500;
  const known = !!(err.status || err.statusCode);
  if (!known) console.error(`[platform] ${err.stack || err.message}`);
  const body = { success: false, error: known ? err.message : 'Something went wrong on the server.' };
  for (const k of EXTRA_FIELDS) if (err[k] !== undefined) body[k] = err[k];
  res.status(status).json(body);
}

module.exports = { failure, wrap, before, jsonBody, errorHandler };
