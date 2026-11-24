'use strict';

/**
 * PostgreSQL connection + schema for the GAD (chips/fg/fg2) account store.
 *
 * ONE database is shared by all three bots; rows are isolated by a `pipeline`
 * column, so the composite primary key (pipeline, username) reproduces exactly the
 * per-bot separation the old per-file JSON store gave. See store.js for the callers.
 *
 * Connection comes from the environment, in this order of preference:
 *   · DATABASE_URL            postgres://user:pass@host:5432/dbname   (preferred)
 *   · PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE   (standard libpq vars)
 * If neither is set the process fails fast at startup — an auth store that silently
 * has no backing DB is worse than not booting.
 *
 * The pool is a lazy singleton so requiring this module has no side effects; the
 * first query (or init()) opens connections.
 */

const { Pool } = require('pg');

let _pool = null;

function haveConfig() {
  return !!(process.env.DATABASE_URL || process.env.PGHOST || process.env.PGUSER);
}

function pool() {
  if (_pool) return _pool;
  if (!haveConfig()) {
    throw new Error(
      '[gad/db] No PostgreSQL connection configured. Set DATABASE_URL ' +
      '(postgres://user:pass@host:5432/dbname) or the PGHOST/PGUSER/PGPASSWORD/PGDATABASE vars.'
    );
  }
  const cfg = process.env.DATABASE_URL
    ? { connectionString: process.env.DATABASE_URL }
    : {}; // otherwise pg reads PG* vars itself
  // TLS is opt-in via PGSSLMODE=require (managed Postgres); local installs use none.
  if (process.env.PGSSLMODE && process.env.PGSSLMODE !== 'disable') {
    cfg.ssl = { rejectUnauthorized: process.env.PGSSLMODE === 'verify-full' };
  }
  cfg.max = parseInt(process.env.PGPOOL_MAX || '', 10) || 10;
  _pool = new Pool(cfg);
  // A pool-level error (e.g. the DB restarts) must not crash the process.
  _pool.on('error', (err) => console.error(`[gad/db] idle client error: ${err.message}`));
  return _pool;
}

/** Run a parameterised query. Returns the pg result. */
function query(text, params) {
  return pool().query(text, params);
}

/**
 * Create the users table and its uniqueness constraints if they do not exist.
 * Idempotent — safe to call on every boot. Must complete before the server accepts
 * requests (server.js awaits it during startup).
 *
 * email_verified is TRI-STATE, matching the JSON store's semantics exactly:
 *   NULL  → verified (legacy + admin-provisioned accounts an admin vouched for)
 *   TRUE  → verified (confirmed the emailed link)
 *   FALSE → pending  (self-signup, cannot sign in until the link is opened)
 */
async function init() {
  await query(`
    CREATE TABLE IF NOT EXISTS gad_users (
      pipeline       TEXT        NOT NULL,
      username       TEXT        NOT NULL,
      role           TEXT        NOT NULL DEFAULT 'user',
      email          TEXT        NOT NULL DEFAULT '',
      mobile         TEXT        NOT NULL DEFAULT '',
      password_hash  TEXT        NOT NULL,
      email_verified BOOLEAN,
      created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (pipeline, username)
    )
  `);
  // A contact may repeat as '' (no email/mobile on file), so uniqueness is enforced
  // only over NON-empty values, and only within a pipeline.
  await query(`
    CREATE UNIQUE INDEX IF NOT EXISTS gad_users_pipeline_email_uq
      ON gad_users (pipeline, email) WHERE email <> ''
  `);
  await query(`
    CREATE UNIQUE INDEX IF NOT EXISTS gad_users_pipeline_mobile_uq
      ON gad_users (pipeline, mobile) WHERE mobile <> ''
  `);
}

/** Close the pool (tests / graceful shutdown). */
async function close() {
  if (_pool) { const p = _pool; _pool = null; await p.end(); }
}

module.exports = { query, init, close, pool, haveConfig };
