'use strict';

/**
 * Sign-in sessions for department accounts (the department panel, later the
 * department's chatbot) — separate from the bot's own login (gad/auth.js).
 *
 * A department session must never pass as a session of the bot that hosts the
 * panel, and the other way round. Both are signed from the same secret, so the key
 * here is DERIVED from it (HMAC(secret, "department-session-v1")): a token made
 * here fails the bot's check and a bot token fails this one. The token also names
 * its department; the cookie name carries the department id, so one browser can be
 * signed in to two departments.
 *
 * The token holds only ids and the account's session epoch. Every request reads the
 * account again (routes.js), so a block, a role change or a deleted account takes
 * effect at once, and a password reset (new epoch) ends older sessions.
 */

const crypto = require('crypto');

function key() {
  const s = process.env.GAD_SESSION_SECRET || process.env.SESSION_SECRET || process.env.JWT_SECRET;
  if (!s) throw new Error('[platform] No signing secret set (GAD_SESSION_SECRET/SESSION_SECRET/JWT_SECRET).');
  return crypto.createHmac('sha256', s).update('department-session-v1').digest();
}

function ttlSeconds() {
  return parseInt(process.env.GAD_SESSION_TTL_SECONDS || '', 10) || 8 * 60 * 60;
}

const sign = (data) => crypto.createHmac('sha256', key()).update(data).digest('base64url');
const cookieName = (departmentId) => `dept${Number(departmentId)}_session`;

function issue(user, ttl = ttlSeconds()) {
  const now = Math.floor(Date.now() / 1000);
  const body = { kind: 'department', pipeline: `dept:${user.departmentId}`,
    d: user.departmentId, u: user.id, e: user.sessionEpoch, iat: now, exp: now + ttl };
  const data = Buffer.from(JSON.stringify(body)).toString('base64url');
  return `${data}.${sign(data)}`;
}

/** The payload when the token is genuine, unexpired and for this department; else null. */
function verify(token, departmentId) {
  if (!token || typeof token !== 'string') return null;
  const dot = token.indexOf('.');
  if (dot <= 0) return null;
  const data = token.slice(0, dot);
  const a = Buffer.from(token.slice(dot + 1));
  const b = Buffer.from(sign(data));
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
  let body;
  try { body = JSON.parse(Buffer.from(data, 'base64url').toString('utf8')); }
  catch (_) { return null; }
  if (body.kind !== 'department' || body.d !== Number(departmentId)) return null;
  if (typeof body.exp !== 'number' || body.exp < Math.floor(Date.now() / 1000)) return null;
  return body;
}

function cookieOptions() {
  const opts = {
    httpOnly: true,
    sameSite: 'lax',
    path: '/',
    secure: process.env.NODE_ENV === 'production' && process.env.COOKIE_INSECURE !== '1',
    maxAge: ttlSeconds() * 1000,
  };
  if (process.env.COOKIE_DOMAIN) opts.domain = process.env.COOKIE_DOMAIN;
  return opts;
}

function read(req, departmentId) {
  const name = cookieName(departmentId);
  for (const part of String((req.headers && req.headers.cookie) || '').split(';')) {
    const i = part.indexOf('=');
    if (i > -1 && part.slice(0, i).trim() === name) {
      try { return decodeURIComponent(part.slice(i + 1).trim()); } catch (_) { return null; }
    }
  }
  return null;
}

function set(res, user) {
  res.cookie(cookieName(user.departmentId), issue(user), cookieOptions());
}

function clear(res, departmentId) {
  const { maxAge, ...opts } = cookieOptions();
  res.clearCookie(cookieName(departmentId), opts);
}

module.exports = { issue, verify, read, set, clear, cookieName };
