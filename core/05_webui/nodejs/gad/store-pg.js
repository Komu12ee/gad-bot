'use strict';

/**
 * PostgreSQL persistence for the GAD account store — the backend used when a
 * connection is configured (DATABASE_URL / PG* vars). store-file.js is the
 * zero-infrastructure twin; both expose the same primitive surface and return the
 * same internal record shape:
 *   { username, role, email, mobile, passwordHash, emailVerified, createdAt }
 * where emailVerified is undefined (NULL/legacy/verified) | true | false.
 *
 * This module holds NO business logic — validation, normalisation and the
 * seed-admin policy live in store.js, which calls these primitives. All rows live
 * in one shared `gad_users` table isolated by the `pipeline` column (see gad/db.js);
 * uniqueness of username/email/mobile is enforced by DB constraints, and a violation
 * is re-thrown as an Error tagged with `.uniqueField` so store.js renders the same
 * friendly message the file backend produces.
 */

const db = require('./db');

/** DB row (snake_case) → internal record (camelCase, incl. hash). */
function rowToRecord(row) {
  if (!row) return null;
  return {
    username: row.username,
    role: row.role,
    email: row.email || '',
    mobile: row.mobile || '',
    passwordHash: row.password_hash,
    // NULL → undefined so store.isEmailVerified() reads it as verified.
    emailVerified: row.email_verified === null || row.email_verified === undefined
      ? undefined : row.email_verified,
    createdAt: row.created_at instanceof Date ? row.created_at.toISOString() : row.created_at,
  };
}

// Postgres unique-violation (23505) → the neutral tag both backends share.
function tagUnique(err) {
  if (!err || err.code !== '23505') return err;
  const c = err.constraint || '';
  const field = c.includes('email') ? 'email' : c.includes('mobile') ? 'mobile' : 'username';
  const e = new Error(`duplicate ${field}`);
  e.uniqueField = field;
  return e;
}

async function init() {
  return db.init();
}

async function list(pipeline) {
  const { rows } = await db.query(
    'SELECT * FROM gad_users WHERE pipeline = $1 ORDER BY created_at ASC', [pipeline]);
  return rows.map(rowToRecord);
}

async function get(pipeline, username) {
  const { rows } = await db.query(
    'SELECT * FROM gad_users WHERE pipeline = $1 AND username = $2', [pipeline, username]);
  return rowToRecord(rows[0]);
}

async function getByIdentifier(pipeline, { username, email, mobile }) {
  const { rows } = await db.query(
    `SELECT * FROM gad_users
       WHERE pipeline = $1
         AND ( username = $2
               OR (email  <> '' AND email  = $3)
               OR (mobile <> '' AND mobile = $4) )
       LIMIT 1`,
    [pipeline, username, email || '', mobile || '']);
  return rowToRecord(rows[0]);
}

async function countAdmins(pipeline) {
  const { rows } = await db.query(
    `SELECT COUNT(*)::int AS n FROM gad_users WHERE pipeline = $1 AND role = 'admin'`, [pipeline]);
  return rows[0].n;
}

async function insert(pipeline, rec) {
  try {
    const { rows } = await db.query(
      `INSERT INTO gad_users (pipeline, username, role, email, mobile, password_hash, email_verified)
         VALUES ($1, $2, $3, $4, $5, $6, $7)
         RETURNING *`,
      [pipeline, rec.username, rec.role, rec.email || '', rec.mobile || '',
       rec.passwordHash, rec.emailVerified === undefined ? null : rec.emailVerified]);
    return rowToRecord(rows[0]);
  } catch (err) {
    throw tagUnique(err);
  }
}

async function update(pipeline, username, patch) {
  // Build a SET clause from only the provided fields — mirrors the file backend,
  // which likewise touches just the keys present in `patch`.
  const sets = [];
  const vals = [pipeline, username];
  const add = (col, val) => { vals.push(val); sets.push(`${col} = $${vals.length}`); };
  if (patch.role !== undefined) add('role', patch.role);
  if (patch.email !== undefined) add('email', patch.email || '');
  if (patch.mobile !== undefined) add('mobile', patch.mobile || '');
  if (patch.passwordHash !== undefined) add('password_hash', patch.passwordHash);
  if (patch.emailVerified !== undefined) add('email_verified', patch.emailVerified);
  if (!sets.length) return get(pipeline, username);
  try {
    const { rows } = await db.query(
      `UPDATE gad_users SET ${sets.join(', ')}
         WHERE pipeline = $1 AND username = $2 RETURNING *`, vals);
    return rowToRecord(rows[0]);
  } catch (err) {
    throw tagUnique(err);
  }
}

async function remove(pipeline, username) {
  await db.query('DELETE FROM gad_users WHERE pipeline = $1 AND username = $2', [pipeline, username]);
}

async function upsertAdmin(pipeline, { username, email, mobile, passwordHash }) {
  await db.query(
    `INSERT INTO gad_users (pipeline, username, role, email, mobile, password_hash, email_verified)
       VALUES ($1, $2, 'admin', $3, $4, $5, NULL)
       ON CONFLICT (pipeline, username)
       DO UPDATE SET role = 'admin'`,
    [pipeline, username, email || '', mobile || '', passwordHash]);
}

module.exports = {
  init, list, get, getByIdentifier, countAdmins,
  insert, update, remove, upsertAdmin,
};
