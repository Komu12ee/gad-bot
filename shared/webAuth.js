'use strict';

/**
 * Modular session auth — shared by the selection gateway and every pipeline UI.
 *
 * ONE signing key (SESSION_SECRET, falling back to the existing JWT_SECRET) means
 * a session issued by the gateway login verifies on any pipeline server, so a
 * pipeline can refuse anyone who did not sign in at the gateway first.
 *
 * Deliberately uses ONLY Node's built-in `crypto` (no npm dependency) so this one
 * file can be `require`d from both services despite their separate node_modules.
 *
 * ── Swapping auth method later ──────────────────────────────────────────────
 * The rest of the app only depends on this contract:
 *     issueToken(payload)        -> signed string  (gateway login sets it as a cookie)
 *     verifyToken(token)         -> payload | null
 *     requireAuth({ loginUrl })  -> express middleware (sets req.user or rejects)
 *     COOKIE_NAME, cookieOptions()
 * Replace the internals (OAuth, SSO, mTLS, opaque sessions in Redis, …) and every
 * caller keeps working. Nothing else in the codebase knows how a session is made.
 */

const crypto = require('crypto');

// Read lazily so require-order / dotenv timing never matters: by the time a
// request is served the env is loaded (the pipeline also inherits it on spawn).
function secret() {
  const s = process.env.SESSION_SECRET || process.env.JWT_SECRET;
  if (!s) throw new Error('[webAuth] SESSION_SECRET/JWT_SECRET is not set — cannot sign sessions.');
  return s;
}

const COOKIE_NAME = process.env.SESSION_COOKIE_NAME || 'chips_rag_session';

function ttlSeconds() {
  return parseInt(process.env.SESSION_TTL_SECONDS || '', 10) || 8 * 60 * 60; // 8h
}

function hmac(data) {
  return crypto.createHmac('sha256', secret()).update(data).digest('base64url');
}

/** Sign a payload into a compact `<body>.<sig>` token (JWT-like, HS256). */
function issueToken(payload, ttl = ttlSeconds()) {
  const now = Math.floor(Date.now() / 1000);
  const body = { ...payload, iat: now, exp: now + ttl };
  const data = Buffer.from(JSON.stringify(body)).toString('base64url');
  return `${data}.${hmac(data)}`;
}

/** Verify a token's signature + expiry. Returns the payload, or null. */
function verifyToken(token) {
  if (!token || typeof token !== 'string') return null;
  const dot = token.indexOf('.');
  if (dot <= 0) return null;
  const data = token.slice(0, dot);
  const sig = token.slice(dot + 1);
  let a, b;
  try {
    a = Buffer.from(sig);
    b = Buffer.from(hmac(data));
  } catch (_) {
    return null;
  }
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
  let body;
  try {
    body = JSON.parse(Buffer.from(data, 'base64url').toString('utf8'));
  } catch (_) {
    return null;
  }
  if (typeof body.exp !== 'number' || body.exp < Math.floor(Date.now() / 1000)) return null;
  return body;
}

/** Cookie attributes for the session cookie. Secure in prod; overridable via env. */
function cookieOptions() {
  const opts = {
    httpOnly: true,
    sameSite: 'lax',
    path: '/',
    secure: process.env.NODE_ENV === 'production' && process.env.COOKIE_INSECURE !== '1',
    maxAge: ttlSeconds() * 1000,
  };
  // For subdomain-per-pipeline setups: COOKIE_DOMAIN=.rag.example.gov
  if (process.env.COOKIE_DOMAIN) opts.domain = process.env.COOKIE_DOMAIN;
  return opts;
}

// Read the session token from the cookie (no cookie-parser dependency required),
// falling back to an Authorization: Bearer header for programmatic/internal calls.
function getToken(req) {
  if (req.cookies && req.cookies[COOKIE_NAME] != null) return req.cookies[COOKIE_NAME];
  const raw = req.headers && req.headers.cookie;
  if (raw) {
    for (const part of raw.split(';')) {
      const i = part.indexOf('=');
      if (i > -1 && part.slice(0, i).trim() === COOKIE_NAME) {
        return decodeURIComponent(part.slice(i + 1).trim());
      }
    }
  }
  const h = (req.headers && req.headers['authorization']) || '';
  return h.startsWith('Bearer ') ? h.slice(7) : null;
}

/**
 * Express middleware: allow only requests carrying a valid session.
 *  - Unauthenticated top-level HTML navigation → redirect to the gateway login.
 *  - Everything else (XHR/API) → 401 JSON so the frontend can react.
 *
 * `loginUrl` may be a string or a function(req) → string, so a pipeline server can
 * compute the gateway URL from the request host in local dev (different port).
 */
function requireAuth(opts = {}) {
  const resolveLoginUrl = (req) => {
    const u = opts.loginUrl || process.env.LOGIN_URL || '/';
    return typeof u === 'function' ? u(req) : u;
  };
  return (req, res, next) => {
    const user = verifyToken(getToken(req));
    if (user) {
      req.user = user;
      return next();
    }
    const wantsHtml = req.method === 'GET' && req.accepts && req.accepts('html') && !req.xhr;
    if (wantsHtml) return res.redirect(resolveLoginUrl(req));
    return res.status(401).json({ success: false, error: 'Unauthorized — please sign in.' });
  };
}

module.exports = { COOKIE_NAME, cookieOptions, issueToken, verifyToken, getToken, requireAuth };
