'use strict';

/**
 * File-backed persistence for the GAD account store — the fallback used when NO
 * PostgreSQL is configured (no DATABASE_URL / PG* vars). It restores the original
 * per-bot JSON store so a fresh clone boots with zero infrastructure: one file per
 * pipeline at <RAG_DATA_ROOT|repo>/data/<pipeline>/auth/users.json, exactly the
 * layout gad/migrate-users-to-pg.js reads, so an existing file keeps working and can
 * still be migrated into Postgres later.
 *
 * This module holds NO business logic — validation, normalisation, the government
 * email gate and the seed-admin policy all live in store.js, which calls these
 * primitives. store-pg.js is the Postgres twin; the two expose the same surface and
 * return the same internal record shape:
 *   { username, role, email, mobile, passwordHash, emailVerified, createdAt }
 * where emailVerified is undefined (legacy/verified) | true | false.
 *
 * Uniqueness (username within the file, plus non-empty email and non-empty mobile)
 * is enforced here to match the Postgres constraints; a violation throws an Error
 * tagged with `.uniqueField` so store.js can render the same friendly message.
 *
 * Writes are serialised through an in-process promise chain and committed
 * atomically (temp file + rename), which is sufficient for the single-process,
 * single-writer model each pipeline runs under.
 */

const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');

// Same resolution as gad/migrate-users-to-pg.js: RAG_DATA_ROOT wins, else <repo>/data.
// gad/ sits at core/05_webui/nodejs/gad, so four levels up is the repo root.
function dataRoot() {
  const repoRoot = path.join(__dirname, '..', '..', '..', '..');
  return process.env.RAG_DATA_ROOT || path.join(repoRoot, 'data');
}
function fileFor(pipeline) {
  return path.join(dataRoot(), pipeline, 'auth', 'users.json');
}

function uniqueError(field) {
  const e = new Error(`duplicate ${field}`);
  e.uniqueField = field; // 'username' | 'email' | 'mobile'
  return e;
}

// ── record ↔ stored-JSON mapping ─────────────────────────────────────────────
// Stored objects omit emailVerified when it is "verified/legacy" (tri-state NULL),
// matching the pre-migration files exactly; present-but-false means pending signup.
function jsonToRecord(u) {
  if (!u) return null;
  return {
    username: u.username,
    role: u.role === 'admin' ? 'admin' : 'user',
    email: u.email || '',
    mobile: u.mobile || '',
    passwordHash: u.passwordHash,
    emailVerified: Object.prototype.hasOwnProperty.call(u, 'emailVerified') && u.emailVerified !== null
      ? !!u.emailVerified : undefined,
    createdAt: u.createdAt || null,
  };
}
function recordToJson(rec) {
  const out = {
    username: rec.username,
    role: rec.role,
    email: rec.email || '',
    mobile: rec.mobile || '',
    passwordHash: rec.passwordHash,
    createdAt: rec.createdAt || new Date().toISOString(),
  };
  if (rec.emailVerified === false || rec.emailVerified === true) out.emailVerified = rec.emailVerified;
  return out;
}

// ── file IO (serialised) ─────────────────────────────────────────────────────
async function readAll(pipeline) {
  try {
    const parsed = JSON.parse(await fsp.readFile(fileFor(pipeline), 'utf8'));
    return Array.isArray(parsed.users) ? parsed.users : [];
  } catch (_) {
    return []; // missing/corrupt file → empty store (first run)
  }
}

async function writeAll(pipeline, users) {
  const file = fileFor(pipeline);
  await fsp.mkdir(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp`;
  await fsp.writeFile(tmp, JSON.stringify({ users }, null, 2), { mode: 0o600 });
  await fsp.rename(tmp, file);
}

// Serialise every read-modify-write so concurrent requests can't clobber the file.
let _chain = Promise.resolve();
function withLock(fn) {
  const run = _chain.then(fn, fn);
  // Keep the chain alive regardless of this op's outcome.
  _chain = run.then(() => {}, () => {});
  return run;
}

function findByUsername(users, username) {
  return users.find((u) => u.username === username);
}

// ── primitives (mirror store-pg.js) ──────────────────────────────────────────
async function init() {
  // Nothing to create up front — the file is written lazily on first insert. Just
  // make sure the directory can exist for the configured pipeline is unnecessary.
}

async function list(pipeline) {
  const users = await readAll(pipeline);
  return users
    .map(jsonToRecord)
    .sort((a, b) => String(a.createdAt || '').localeCompare(String(b.createdAt || '')));
}

async function get(pipeline, username) {
  const users = await readAll(pipeline);
  return jsonToRecord(findByUsername(users, username));
}

async function getByIdentifier(pipeline, { username, email, mobile }) {
  const users = await readAll(pipeline);
  const hit = users.find((u) =>
    u.username === username ||
    (email && (u.email || '') !== '' && (u.email || '') === email) ||
    (mobile && (u.mobile || '') !== '' && (u.mobile || '') === mobile));
  return jsonToRecord(hit);
}

async function countAdmins(pipeline) {
  const users = await readAll(pipeline);
  return users.filter((u) => u.role === 'admin').length;
}

async function insert(pipeline, rec) {
  return withLock(async () => {
    const users = await readAll(pipeline);
    if (findByUsername(users, rec.username)) throw uniqueError('username');
    if (rec.email && users.some((u) => (u.email || '') === rec.email)) throw uniqueError('email');
    if (rec.mobile && users.some((u) => (u.mobile || '') === rec.mobile)) throw uniqueError('mobile');
    const stored = recordToJson({ ...rec, createdAt: rec.createdAt || new Date().toISOString() });
    users.push(stored);
    await writeAll(pipeline, users);
    return jsonToRecord(stored);
  });
}

async function update(pipeline, username, patch) {
  return withLock(async () => {
    const users = await readAll(pipeline);
    const u = findByUsername(users, username);
    if (!u) return null;
    if (patch.email !== undefined && patch.email !== (u.email || '') && patch.email
        && users.some((o) => o.username !== username && (o.email || '') === patch.email)) {
      throw uniqueError('email');
    }
    if (patch.mobile !== undefined && patch.mobile !== (u.mobile || '') && patch.mobile
        && users.some((o) => o.username !== username && (o.mobile || '') === patch.mobile)) {
      throw uniqueError('mobile');
    }
    if (patch.role !== undefined) u.role = patch.role === 'admin' ? 'admin' : 'user';
    if (patch.email !== undefined) u.email = patch.email || '';
    if (patch.mobile !== undefined) u.mobile = patch.mobile || '';
    if (patch.passwordHash !== undefined) u.passwordHash = patch.passwordHash;
    if (patch.emailVerified !== undefined) {
      if (patch.emailVerified === null) delete u.emailVerified;
      else u.emailVerified = !!patch.emailVerified;
    }
    await writeAll(pipeline, users);
    return jsonToRecord(u);
  });
}

async function remove(pipeline, username) {
  return withLock(async () => {
    const users = await readAll(pipeline);
    const next = users.filter((u) => u.username !== username);
    if (next.length !== users.length) await writeAll(pipeline, next);
  });
}

// Insert a fresh admin, or promote an existing same-named account to admin. Mirrors
// the Postgres upsert used to seed the first admin; emailVerified stays unset (NULL).
async function upsertAdmin(pipeline, { username, email, mobile, passwordHash }) {
  return withLock(async () => {
    const users = await readAll(pipeline);
    const u = findByUsername(users, username);
    if (u) {
      u.role = 'admin';
    } else {
      users.push(recordToJson({
        username, role: 'admin', email: email || '', mobile: mobile || '',
        passwordHash, emailVerified: null, createdAt: new Date().toISOString(),
      }));
    }
    await writeAll(pipeline, users);
  });
}

module.exports = {
  init, list, get, getByIdentifier, countAdmins,
  insert, update, remove, upsertAdmin,
};
