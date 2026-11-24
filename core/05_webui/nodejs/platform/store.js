'use strict';

/**
 * Departments, their uploaded files and their accounts, in PostgreSQL (the same
 * database as the bot accounts, gad/db.js).
 *
 *   departments        one row per department: name, short name, storage allowed
 *   department_files   one row per uploaded file
 *   department_users   the department's own sign-ins: admins (the department panel)
 *                      and users (the department's chatbot)
 *   department_chunks  a processed file's passages with their bge-m3 vectors
 *                      (pgvector) — what the department's chatbot searches
 *   department_queries every question asked of a department's chatbot, its answer
 *                      and sources (for the search history / review screens)
 *
 * Storage is counted as the size of the files a department has uploaded (and is
 * uploading: a file holds its space from the moment its upload starts, so two
 * uploads at once cannot both squeeze into the last free space). The check runs
 * with the department row locked (SELECT … FOR UPDATE), so concurrent uploads are
 * counted one after another — and the space is summed in a SEPARATE statement
 * after the lock is held: under READ COMMITTED a statement that waited for a lock
 * still reads with the snapshot it started with, so a sum in the locking statement
 * itself would miss the file the other upload just added.
 *
 * Files are only stored here. Turning them into a department's bot (OCR, chunks,
 * search index) is started separately, never by an upload — `processing` records
 * how far that has got ('not_started' until then).
 */

const db = require('../gad/db');

const failure = (status, message, extra) => Object.assign(new Error(message), { status }, extra || {});

async function init() {
  await db.query(`
    CREATE TABLE IF NOT EXISTS departments (
      id           BIGSERIAL   PRIMARY KEY,
      slug         TEXT        NOT NULL UNIQUE,
      name         TEXT        NOT NULL,
      quota_bytes  BIGINT      NOT NULL CHECK (quota_bytes > 0),
      status       TEXT        NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
      created_by   TEXT        NOT NULL DEFAULT '',
      created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
      updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `);
  await db.query(`CREATE UNIQUE INDEX IF NOT EXISTS departments_name_uq ON departments (lower(name))`);
  await db.query(`
    CREATE TABLE IF NOT EXISTS department_files (
      id             UUID        PRIMARY KEY,
      department_id  BIGINT      NOT NULL REFERENCES departments (id) ON DELETE RESTRICT,
      original_name  TEXT        NOT NULL,
      kind           TEXT        NOT NULL,
      size_bytes     BIGINT      NOT NULL CHECK (size_bytes > 0),
      sha256         TEXT,
      status         TEXT        NOT NULL DEFAULT 'uploading'
                                 CHECK (status IN ('uploading', 'uploaded', 'deleted')),
      scan           TEXT        NOT NULL DEFAULT 'not_scanned',
      processing     TEXT        NOT NULL DEFAULT 'not_started',
      uploaded_by    TEXT        NOT NULL DEFAULT '',
      created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
      uploaded_at    TIMESTAMPTZ,
      deleted_by     TEXT,
      deleted_at     TIMESTAMPTZ
    )
  `);
  await db.query(`CREATE INDEX IF NOT EXISTS department_files_dept_idx ON department_files (department_id, status)`);
  // The same file twice in one department would only use its storage twice.
  await db.query(`
    CREATE UNIQUE INDEX IF NOT EXISTS department_files_same_file_uq
      ON department_files (department_id, sha256) WHERE status = 'uploaded'
  `);
  // session_epoch: raised when the password is reset or the account is blocked, so
  // sessions issued before that stop working (they carry the epoch they were made in).
  await db.query(`
    CREATE TABLE IF NOT EXISTS department_users (
      id             BIGSERIAL   PRIMARY KEY,
      department_id  BIGINT      NOT NULL REFERENCES departments (id) ON DELETE CASCADE,
      username       TEXT        NOT NULL,
      role           TEXT        NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user')),
      password_hash  TEXT        NOT NULL,
      status         TEXT        NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'blocked')),
      session_epoch  INTEGER     NOT NULL DEFAULT 0,
      created_by     TEXT        NOT NULL DEFAULT '',
      created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
      last_login_at  TIMESTAMPTZ,
      UNIQUE (department_id, username)
    )
  `);
  // Whether the CHiPS admin has put the department's chatbot on the launcher page.
  // Off until switched on: a new department is never shown by itself.
  await db.query(`ALTER TABLE departments ADD COLUMN IF NOT EXISTS listed BOOLEAN NOT NULL DEFAULT false`);
  // Processing: 'not_started' → 'queued' → 'processing' → 'done' | 'needs_ocr' | 'failed'.
  await db.query(`ALTER TABLE department_files ADD COLUMN IF NOT EXISTS processing_error TEXT`);
  await db.query(`ALTER TABLE department_files ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ`);
  await db.query(`ALTER TABLE department_files ADD COLUMN IF NOT EXISTS chunk_count INTEGER NOT NULL DEFAULT 0`);
  await db.query(`ALTER TABLE department_files ADD COLUMN IF NOT EXISTS page_count INTEGER`);
  await db.query(`
    CREATE TABLE IF NOT EXISTS department_queries (
      id             BIGSERIAL   PRIMARY KEY,
      department_id  BIGINT      NOT NULL REFERENCES departments (id) ON DELETE CASCADE,
      user_id        BIGINT      REFERENCES department_users (id) ON DELETE SET NULL,
      username       TEXT        NOT NULL DEFAULT '',
      question       TEXT        NOT NULL,
      answer         TEXT,
      sources        JSONB       NOT NULL DEFAULT '[]',
      status         TEXT        NOT NULL DEFAULT 'answered',
      error          TEXT,
      seconds        REAL,
      created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `);
  await db.query(`CREATE INDEX IF NOT EXISTS department_queries_dept_idx ON department_queries (department_id, created_at DESC)`);
  await initVectors();
}

// pgvector: the passages table needs it. deploy/setup-postgres.sh installs it and
// creates the extension; without it departments, files and accounts still work and
// processing / the chatbot say why they cannot run.
let vectorState = { ready: false, error: 'not checked' };

async function initVectors() {
  try {
    await db.query('CREATE EXTENSION IF NOT EXISTS vector');
  } catch (err) {
    // No right to create it (only a superuser may): fine if it is there already.
    const { rows } = await db.query(`SELECT 1 FROM pg_extension WHERE extname = 'vector'`);
    if (!rows.length) {
      vectorState = { ready: false, error: `pgvector is not installed in PostgreSQL (${err.message}). Run deploy/setup-postgres.sh.` };
      return;
    }
  }
  // Exact search per department (no ANN index): a department's passages are found
  // through department_id, then every one is compared — exact, and fast at the
  // size of one department. An HNSW index would filter AFTER its approximate scan
  // and could miss a small department's passages among the others.
  await db.query(`
    CREATE TABLE IF NOT EXISTS department_chunks (
      id             BIGSERIAL   PRIMARY KEY,
      department_id  BIGINT      NOT NULL REFERENCES departments (id) ON DELETE CASCADE,
      file_id        UUID        NOT NULL REFERENCES department_files (id) ON DELETE CASCADE,
      seq            INTEGER     NOT NULL,
      page_from      INTEGER,
      page_to        INTEGER,
      text           TEXT        NOT NULL,
      context_line   TEXT,
      section        TEXT,
      embedding      vector(1024) NOT NULL,
      sparse         JSONB       NOT NULL DEFAULT '{}'
    )
  `);
  await db.query(`CREATE INDEX IF NOT EXISTS department_chunks_dept_idx ON department_chunks (department_id)`);
  await db.query(`CREATE INDEX IF NOT EXISTS department_chunks_file_idx ON department_chunks (file_id)`);
  vectorState = { ready: true, error: null };
}

const vectors = () => ({ ...vectorState });

// Space a department is using: uploaded files plus uploads in progress.
const USED = `COALESCE((SELECT SUM(size_bytes) FROM department_files f
                        WHERE f.department_id = d.id AND f.status IN ('uploading', 'uploaded')), 0)`;
const FILES = `(SELECT COUNT(*) FROM department_files f WHERE f.department_id = d.id AND f.status = 'uploaded')`;
const USERS = `(SELECT COUNT(*) FROM department_users u WHERE u.department_id = d.id)`;
const READY = `(SELECT COUNT(*) FROM department_files f WHERE f.department_id = d.id AND f.status = 'uploaded' AND f.processing = 'done')`;
const DEPT_COLUMNS = `d.*, ${USED} AS used_bytes, ${FILES} AS file_count, ${USERS} AS user_count, ${READY} AS ready_count`;

function department(r) {
  return {
    id: Number(r.id),
    slug: r.slug,
    name: r.name,
    quotaBytes: Number(r.quota_bytes),
    usedBytes: Number(r.used_bytes || 0),
    fileCount: Number(r.file_count || 0),
    userCount: Number(r.user_count || 0),
    readyCount: Number(r.ready_count || 0),
    status: r.status,
    listed: !!r.listed,
    createdBy: r.created_by,
    createdAt: r.created_at,
  };
}

function file(r) {
  return {
    id: r.id,
    departmentId: Number(r.department_id),
    name: r.original_name,
    kind: r.kind,
    sizeBytes: Number(r.size_bytes),
    sha256: r.sha256 || null,
    status: r.status,
    scan: r.scan,
    processing: r.processing,
    processingError: r.processing_error || null,
    processedAt: r.processed_at || null,
    chunkCount: Number(r.chunk_count || 0),
    pageCount: r.page_count == null ? null : Number(r.page_count),
    uploadedBy: r.uploaded_by,
    createdAt: r.created_at,
    uploadedAt: r.uploaded_at,
  };
}

/** Lock the department row, then (new statement, fresh snapshot) count its space. */
async function lockDepartment(c, id) {
  const { rows } = await c.query(`SELECT * FROM departments WHERE id = $1 FOR UPDATE`, [id]);
  if (!rows.length) throw failure(404, 'Department not found.');
  const used = await c.query(
    `SELECT COALESCE(SUM(size_bytes), 0) AS used FROM department_files
      WHERE department_id = $1 AND status IN ('uploading', 'uploaded')`, [id]);
  return { row: rows[0], used: Number(used.rows[0].used) };
}

async function transaction(fn) {
  const client = await db.pool().connect();
  try {
    await client.query('BEGIN');
    const out = await fn(client);
    await client.query('COMMIT');
    return out;
  } catch (err) {
    await client.query('ROLLBACK').catch(() => {});
    throw err;
  } finally {
    client.release();
  }
}

function uniqueError(err, messages) {
  if (err && err.code === '23505') {
    const hit = Object.keys(messages).find((k) => String(err.constraint || '').includes(k));
    return failure(409, hit ? messages[hit] : 'That already exists.');
  }
  return err;
}

// ── Departments ───────────────────────────────────────────────────────────────

async function listDepartments() {
  const { rows } = await db.query(`SELECT ${DEPT_COLUMNS} FROM departments d ORDER BY lower(d.name)`);
  return rows.map(department);
}

async function getDepartment(id) {
  const { rows } = await db.query(`SELECT ${DEPT_COLUMNS} FROM departments d WHERE d.id = $1`, [id]);
  if (!rows.length) throw failure(404, 'Department not found.');
  return department(rows[0]);
}

async function getDepartmentBySlug(slug) {
  const { rows } = await db.query(`SELECT ${DEPT_COLUMNS} FROM departments d WHERE d.slug = $1`, [slug]);
  if (!rows.length) throw failure(404, 'Department not found.');
  return department(rows[0]);
}

const DEPT_UNIQUE = {
  departments_slug_key: 'Another department already has that short name.',
  departments_name_uq: 'A department with that name already exists.',
};

async function createDepartment({ name, slug, quotaBytes, by }) {
  try {
    const { rows } = await db.query(
      `INSERT INTO departments (name, slug, quota_bytes, created_by) VALUES ($1, $2, $3, $4) RETURNING id`,
      [name, slug, quotaBytes, by || '']);
    return getDepartment(rows[0].id);
  } catch (err) {
    throw uniqueError(err, DEPT_UNIQUE);
  }
}

/** Change name / storage / status. Storage cannot go below what is already used. */
async function updateDepartment(id, { name, quotaBytes, status, listed }) {
  try {
    await transaction(async (c) => {
      const { used } = await lockDepartment(c, id);
      if (quotaBytes !== undefined && quotaBytes < used) {
        throw failure(409, 'The department already uses more than that.', { usedBytes: used });
      }
      await c.query(
        `UPDATE departments SET name = COALESCE($2, name), quota_bytes = COALESCE($3, quota_bytes),
                status = COALESCE($4, status), listed = COALESCE($5, listed), updated_at = now() WHERE id = $1`,
        [id, name === undefined ? null : name, quotaBytes === undefined ? null : quotaBytes,
          status === undefined ? null : status, listed === undefined ? null : listed]);
    });
  } catch (err) {
    throw uniqueError(err, DEPT_UNIQUE);
  }
  return getDepartment(id);
}

/** Remove a department that holds no files. Returns the ids of its old file rows. */
async function deleteDepartment(id) {
  return transaction(async (c) => {
    const { used } = await lockDepartment(c, id);
    if (used > 0) throw failure(409, 'Delete the department’s files first.');
    await c.query(`DELETE FROM department_files WHERE department_id = $1`, [id]);
    await c.query(`DELETE FROM departments WHERE id = $1`, [id]);
  });
}

// ── Files ─────────────────────────────────────────────────────────────────────

async function listFiles(departmentId) {
  const { rows } = await db.query(
    `SELECT * FROM department_files WHERE department_id = $1 AND status IN ('uploading', 'uploaded')
      ORDER BY created_at DESC`, [departmentId]);
  return rows.map(file);
}

async function getFile(id) {
  const { rows } = await db.query(`SELECT * FROM department_files WHERE id = $1`, [id]);
  if (!rows.length || rows[0].status === 'deleted') throw failure(404, 'File not found.');
  return file(rows[0]);
}

/**
 * Start an upload: the file's size is taken from the department's free storage
 * now, or the upload is refused. `id` is chosen by the caller (it names the file
 * on disk).
 */
async function reserveFile(departmentId, { id, name, kind, sizeBytes, by }) {
  return transaction(async (c) => {
    const { row: d, used } = await lockDepartment(c, departmentId);
    if (d.status !== 'active') throw failure(409, 'This department is disabled; files cannot be added.');
    const free = Number(d.quota_bytes) - used;
    if (sizeBytes > free) {
      throw failure(413, 'Not enough storage left for this file.', { freeBytes: Math.max(0, free) });
    }
    const ins = await c.query(
      `INSERT INTO department_files (id, department_id, original_name, kind, size_bytes, uploaded_by)
       VALUES ($1, $2, $3, $4, $5, $6) RETURNING *`,
      [id, departmentId, name, kind, sizeBytes, by || '']);
    return file(ins.rows[0]);
  });
}

/** The whole file arrived and passed its checks. */
async function markUploaded(id, { sha256, scan }) {
  try {
    const { rows } = await db.query(
      `UPDATE department_files SET status = 'uploaded', sha256 = $2, scan = $3, uploaded_at = now()
        WHERE id = $1 AND status = 'uploading' RETURNING *`, [id, sha256, scan || 'not_scanned']);
    if (!rows.length) throw failure(409, 'This upload is no longer open.');
    return file(rows[0]);
  } catch (err) {
    if (err && err.code === '23505') {
      const { rows } = await db.query(
        `SELECT g.original_name FROM department_files f JOIN department_files g
            ON g.department_id = f.department_id AND g.sha256 = $2 AND g.status = 'uploaded'
          WHERE f.id = $1`, [id, sha256]);
      throw failure(409, `The same file is already uploaded${rows.length ? ` as “${rows[0].original_name}”` : ''}.`);
    }
    throw err;
  }
}

/** Forget an unfinished upload (its storage is free again). */
async function dropUpload(id) {
  await db.query(`DELETE FROM department_files WHERE id = $1 AND status = 'uploading'`, [id]);
}

/** Delete an uploaded file: the row stays as a record ('deleted'); its storage is free
 *  again and its passages leave the department's search at once. */
async function markDeleted(id, by) {
  return transaction(async (c) => {
    const { rows } = await c.query(
      `UPDATE department_files SET status = 'deleted', deleted_by = $2, deleted_at = now(), chunk_count = 0
        WHERE id = $1 AND status = 'uploaded' RETURNING *`, [id, by || '']);
    if (!rows.length) throw failure(404, 'File not found.');
    if (vectorState.ready) await c.query(`DELETE FROM department_chunks WHERE file_id = $1`, [id]);
    return file(rows[0]);
  });
}

/** Uploads started more than `hours` ago and never finished. */
async function staleUploads(hours) {
  const { rows } = await db.query(
    `SELECT * FROM department_files WHERE status = 'uploading' AND created_at < now() - make_interval(hours => $1)`,
    [hours]);
  return rows.map(file);
}

// ── Accounts ─────────────────────────────────────────────────────────────────

function account(r) {
  return {
    id: Number(r.id),
    departmentId: Number(r.department_id),
    username: r.username,
    role: r.role,
    status: r.status,
    sessionEpoch: Number(r.session_epoch),
    createdBy: r.created_by,
    createdAt: r.created_at,
    lastLoginAt: r.last_login_at,
  };
}

async function listUsers(departmentId) {
  const { rows } = await db.query(
    `SELECT * FROM department_users WHERE department_id = $1 ORDER BY role, username`, [departmentId]);
  return rows.map(account);
}

async function getUser(departmentId, userId) {
  const { rows } = await db.query(
    `SELECT * FROM department_users WHERE department_id = $1 AND id = $2`, [departmentId, userId]);
  if (!rows.length) throw failure(404, 'Account not found.');
  return account(rows[0]);
}

/** The account with its password hash, for signing in; null when there is none. */
async function getLogin(departmentId, username) {
  const { rows } = await db.query(
    `SELECT * FROM department_users WHERE department_id = $1 AND username = $2`, [departmentId, username]);
  return rows.length ? { ...account(rows[0]), passwordHash: rows[0].password_hash } : null;
}

async function createUser(departmentId, { username, role, passwordHash, by }) {
  try {
    const { rows } = await db.query(
      `INSERT INTO department_users (department_id, username, role, password_hash, created_by)
       VALUES ($1, $2, $3, $4, $5) RETURNING *`, [departmentId, username, role, passwordHash, by || '']);
    return account(rows[0]);
  } catch (err) {
    if (err && err.code === '23505') throw failure(409, 'That username is already taken in this department.');
    if (err && err.code === '23503') throw failure(404, 'Department not found.');
    throw err;
  }
}

/**
 * The rules a department's own admin works under (`actor` = that admin; the CHiPS
 * admin passes none): not their own account, and never the last active admin —
 * the department would be locked out of its panel.
 */
async function guardChange(c, departmentId, target, after, actor) {
  if (!actor) return;
  const stillAdmin = after.status === 'active' && after.role === 'admin';
  if (target.id === actor && !stillAdmin) throw failure(409, 'You cannot block, delete or demote your own account.');
  if (target.status === 'active' && target.role === 'admin' && !stillAdmin) {
    const { rows } = await c.query(
      `SELECT COUNT(*) AS n FROM department_users
        WHERE department_id = $1 AND role = 'admin' AND status = 'active' AND id <> $2`, [departmentId, target.id]);
    if (Number(rows[0].n) === 0) throw failure(409, 'The department needs at least one active admin account.');
  }
}

async function lockUser(c, departmentId, userId) {
  await lockDepartment(c, departmentId);           // one account change at a time per department
  const { rows } = await c.query(
    `SELECT * FROM department_users WHERE department_id = $1 AND id = $2 FOR UPDATE`, [departmentId, userId]);
  if (!rows.length) throw failure(404, 'Account not found.');
  return account(rows[0]);
}

/** Change role / status / password. A new password or a block ends existing sessions. */
async function updateUser(departmentId, userId, { role, status, passwordHash }, { actor } = {}) {
  return transaction(async (c) => {
    const target = await lockUser(c, departmentId, userId);
    const after = { role: role || target.role, status: status || target.status };
    await guardChange(c, departmentId, target, after, actor);
    const endSessions = !!passwordHash || (after.status === 'blocked' && target.status !== 'blocked');
    const { rows } = await c.query(
      `UPDATE department_users SET role = $3, status = $4, password_hash = COALESCE($5, password_hash),
              session_epoch = session_epoch + $6
        WHERE department_id = $1 AND id = $2 RETURNING *`,
      [departmentId, userId, after.role, after.status, passwordHash || null, endSessions ? 1 : 0]);
    return account(rows[0]);
  });
}

async function deleteUser(departmentId, userId, { actor } = {}) {
  return transaction(async (c) => {
    const target = await lockUser(c, departmentId, userId);
    await guardChange(c, departmentId, target, { role: 'user', status: 'blocked' }, actor);
    await c.query(`DELETE FROM department_users WHERE department_id = $1 AND id = $2`, [departmentId, userId]);
  });
}

async function touchLogin(userId) {
  await db.query(`UPDATE department_users SET last_login_at = now() WHERE id = $1`, [userId]);
}

// ── Processing ───────────────────────────────────────────────────────────────

/**
 * Queue a department's files for processing: `fileIds` (any state but in progress),
 * or, without ids, every file not processed yet. Returns how many were queued.
 */
async function queueFiles(departmentId, fileIds) {
  const { rowCount } = fileIds && fileIds.length
    ? await db.query(
      `UPDATE department_files SET processing = 'queued', processing_error = NULL
        WHERE department_id = $1 AND status = 'uploaded' AND id = ANY($2::uuid[])
          AND processing NOT IN ('queued', 'processing')`, [departmentId, fileIds])
    : await db.query(
      `UPDATE department_files SET processing = 'queued', processing_error = NULL
        WHERE department_id = $1 AND status = 'uploaded' AND processing = 'not_started'`, [departmentId]);
  return rowCount;
}

/** The oldest queued file, marked 'processing' (null when none). */
async function claimNext() {
  const { rows } = await db.query(
    `UPDATE department_files SET processing = 'processing'
      WHERE id = (SELECT id FROM department_files WHERE processing = 'queued' AND status = 'uploaded'
                  ORDER BY uploaded_at, created_at LIMIT 1 FOR UPDATE SKIP LOCKED)
      RETURNING *`);
  return rows.length ? file(rows[0]) : null;
}

/** Files left 'processing' by a restart go back to the queue. */
async function requeueInterrupted() {
  const { rowCount } = await db.query(`UPDATE department_files SET processing = 'queued' WHERE processing = 'processing'`);
  return rowCount;
}

/** A file could not be indexed: 'needs_ocr' or 'failed', with the reason. Its old passages go. */
async function setOutcome(fileId, state, reason) {
  await transaction(async (c) => {
    await c.query(
      `UPDATE department_files SET processing = $2, processing_error = $3, processed_at = now(), chunk_count = 0
        WHERE id = $1`, [fileId, state, reason || null]);
    if (vectorState.ready) await c.query(`DELETE FROM department_chunks WHERE file_id = $1`, [fileId]);
  });
}

const vec = (arr) => `[${arr.map((x) => (Number.isFinite(x) ? x : 0)).join(',')}]`;

/**
 * A processed file's passages replace any it had; the file becomes 'done'. Only if
 * the file is still uploaded and still being processed (a delete meanwhile wins).
 */
async function saveChunks(fileId, chunks, pageCount) {
  return transaction(async (c) => {
    const { rows } = await c.query(
      `SELECT * FROM department_files WHERE id = $1 AND status = 'uploaded' AND processing = 'processing' FOR UPDATE`, [fileId]);
    if (!rows.length) return false;
    const f = rows[0];
    await c.query(`DELETE FROM department_chunks WHERE file_id = $1`, [fileId]);
    for (let i = 0; i < chunks.length; i += 50) {
      const part = chunks.slice(i, i + 50);
      const values = [];
      const params = [];
      part.forEach((ch, j) => {
        const b = j * 10;
        values.push(`($${b + 1}, $${b + 2}, $${b + 3}, $${b + 4}, $${b + 5}, $${b + 6}, $${b + 7}, $${b + 8}, $${b + 9}::vector, $${b + 10}::jsonb)`);
        params.push(f.department_id, fileId, ch.seq, ch.page_from ?? null, ch.page_to ?? null, ch.text,
          ch.context_line ?? null, ch.section ?? null, vec(ch.dense), JSON.stringify(ch.sparse || {}));
      });
      await c.query(
        `INSERT INTO department_chunks (department_id, file_id, seq, page_from, page_to, text, context_line, section, embedding, sparse)
         VALUES ${values.join(', ')}`, params);
    }
    await c.query(
      `UPDATE department_files SET processing = 'done', processing_error = NULL, processed_at = now(),
              chunk_count = $2, page_count = $3 WHERE id = $1`, [fileId, chunks.length, pageCount || null]);
    return true;
  });
}

// ── Search ───────────────────────────────────────────────────────────────────

const CHUNK_COLUMNS = `c.id, c.file_id, c.seq, c.page_from, c.page_to, c.text, c.context_line, c.section, c.sparse,
                       f.original_name`;
const chunkRow = (r) => ({
  id: Number(r.id), fileId: r.file_id, seq: r.seq, pageFrom: r.page_from, pageTo: r.page_to, text: r.text,
  contextLine: r.context_line, section: r.section, sparse: r.sparse || {}, fileName: r.original_name,
  score: r.score == null ? null : Number(r.score),
});

/** The department's passages nearest to a query vector (cosine), best first. */
async function searchDense(departmentId, dense, limit) {
  const { rows } = await db.query(
    `SELECT ${CHUNK_COLUMNS}, 1 - (c.embedding <=> $2::vector) AS score
       FROM department_chunks c JOIN department_files f ON f.id = c.file_id
      WHERE c.department_id = $1 AND f.status = 'uploaded' AND f.processing = 'done'
      ORDER BY c.embedding <=> $2::vector LIMIT $3`, [departmentId, vec(dense), limit]);
  return rows.map(chunkRow);
}

/** Passages that contain any of these exact strings (reference numbers and the like). */
async function searchExact(departmentId, terms, limit) {
  if (!terms.length) return [];
  const patterns = terms.map((t) => `%${t.replace(/[\\%_]/g, (m) => '\\' + m)}%`);
  const { rows } = await db.query(
    `SELECT ${CHUNK_COLUMNS}, NULL::real AS score
       FROM department_chunks c JOIN department_files f ON f.id = c.file_id
      WHERE c.department_id = $1 AND f.status = 'uploaded' AND f.processing = 'done'
        AND c.text ILIKE ANY($2::text[])
      ORDER BY c.file_id, c.seq LIMIT $3`, [departmentId, patterns, limit]);
  return rows.map(chunkRow);
}

async function countChunks(departmentId) {
  const { rows } = await db.query(
    `SELECT COUNT(*) AS n FROM department_chunks c JOIN department_files f ON f.id = c.file_id
      WHERE c.department_id = $1 AND f.status = 'uploaded' AND f.processing = 'done'`, [departmentId]);
  return Number(rows[0].n);
}

async function logQuery(q) {
  await db.query(
    `INSERT INTO department_queries (department_id, user_id, username, question, answer, sources, status, error, seconds)
     VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9)`,
    [q.departmentId, q.userId || null, q.username || '', q.question, q.answer || null,
      JSON.stringify(q.sources || []), q.status, q.error || null, q.seconds == null ? null : q.seconds]);
}

/** The launcher's list: departments the CHiPS admin put on it, while they are active.
 *  Processed documents are not required — until there are some, the chatbot says
 *  the department's documents are not ready yet. */
async function openDepartments() {
  const { rows } = await db.query(
    `SELECT ${DEPT_COLUMNS} FROM departments d WHERE d.status = 'active' AND d.listed ORDER BY lower(d.name)`);
  return rows.map(department);
}

module.exports = {
  init, vectors, listDepartments, getDepartment, getDepartmentBySlug, createDepartment, updateDepartment, deleteDepartment,
  listFiles, getFile, reserveFile, markUploaded, dropUpload, markDeleted, staleUploads,
  listUsers, getUser, getLogin, createUser, updateUser, deleteUser, touchLogin,
  queueFiles, claimNext, requeueInterrupted, setOutcome, saveChunks,
  searchDense, searchExact, countChunks, logQuery, openDepartments,
};
