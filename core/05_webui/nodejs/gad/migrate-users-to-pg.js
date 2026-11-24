'use strict';

/**
 * One-time migration: import existing JSON user stores into PostgreSQL.
 *
 * The old store kept one file per bot at <RAG_DATA_ROOT>/<pipeline>/auth/users.json
 * (password hashes included). This reads each of chips/fg/fg2 that exists and upserts
 * its accounts into the shared gad_users table under the matching pipeline.
 *
 *   Run once, after Postgres is reachable:
 *     node gad/migrate-users-to-pg.js
 *
 * Idempotent: ON CONFLICT (pipeline, username) DO NOTHING, so re-running never
 * clobbers an account already in the DB. Existing bcrypt hashes are copied verbatim
 * — nobody has to reset a password.
 *
 * Point at a single explicit file instead with:
 *     GAD_USERS_FILE=/path/users.json  PIPELINE=fg2  node gad/migrate-users-to-pg.js
 */

const fs = require('fs');
const path = require('path');

// Load env the way the server does. config.js reads the repo-root .env (the single
// source of truth); this standalone script is run by hand, so it must load it too.
// The local nodejs/.env is loaded as a gap-filler for standalone runs. dotenv never
// overrides an already-set process.env value, so the shell's env still wins.
try {
  const dotenv = require('dotenv');
  dotenv.config({ path: path.join(__dirname, '..', '..', '..', '..', '.env') }); // repo-root .env
  dotenv.config({ path: path.join(__dirname, '..', '.env') });                   // nodejs/.env
} catch (_) { /* dotenv optional */ }

const db = require('./db');

const PIPELINES = ['chips', 'fg', 'fg2'];

function dataRoot() {
  const repoRoot = path.join(__dirname, '..', '..', '..', '..');
  return process.env.RAG_DATA_ROOT || path.join(repoRoot, 'data');
}

/** Old per-bot file location, mirroring the retired store.storeFile() logic. */
function legacyFileFor(pipeline) {
  if (process.env.GAD_USERS_FILE) return process.env.GAD_USERS_FILE;
  return path.join(dataRoot(), pipeline, 'auth', 'users.json');
}

function readUsers(file) {
  try {
    const parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
    return Array.isArray(parsed.users) ? parsed.users : [];
  } catch (_) {
    return [];
  }
}

// tri-state: missing → NULL (verified/legacy), else the stored boolean.
function verifiedFlag(u) {
  return u.emailVerified === undefined ? null : !!u.emailVerified;
}

async function importPipeline(pipeline) {
  const file = legacyFileFor(pipeline);
  const users = readUsers(file);
  if (!users.length) return { pipeline, file, found: 0, inserted: 0, skipped: 0 };

  let inserted = 0, skipped = 0;
  for (const u of users) {
    if (!u || !u.username || !u.passwordHash) { skipped += 1; continue; }
    const res = await db.query(
      `INSERT INTO gad_users (pipeline, username, role, email, mobile, password_hash, email_verified, created_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, COALESCE($8::timestamptz, now()))
         ON CONFLICT (pipeline, username) DO NOTHING`,
      [pipeline, String(u.username).toLowerCase(), u.role === 'admin' ? 'admin' : 'user',
       u.email || '', u.mobile || '', u.passwordHash, verifiedFlag(u), u.createdAt || null]);
    if (res.rowCount === 1) inserted += 1; else skipped += 1;
  }
  return { pipeline, file, found: users.length, inserted, skipped };
}

(async () => {
  await db.init();
  // With GAD_USERS_FILE set, only the named PIPELINE is migrated (single explicit file).
  const targets = process.env.GAD_USERS_FILE
    ? [(process.env.PIPELINE || 'chips').toLowerCase()]
    : PIPELINES;

  let total = 0;
  for (const p of targets) {
    const r = await importPipeline(p);
    total += r.inserted;
    if (r.found) {
      console.log(`[migrate] ${r.pipeline}: ${r.inserted} imported, ${r.skipped} already present (from ${r.file})`);
    } else {
      console.log(`[migrate] ${r.pipeline}: no legacy file at ${r.file} — nothing to import`);
    }
  }
  console.log(`[migrate] Done. ${total} account(s) imported into PostgreSQL.`);
  await db.close();
})().catch((err) => {
  console.error(`[migrate] Failed: ${err.message}`);
  process.exit(1);
});
