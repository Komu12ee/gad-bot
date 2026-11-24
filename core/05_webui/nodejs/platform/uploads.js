'use strict';

/**
 * Adding, checking, keeping and deleting a department's files — used by the CHiPS
 * admin's routes and by the department panel's.
 *
 * Every call that names a file by id takes a `scope`: the department the caller
 * may touch. The department panel passes its own department, so a file id from
 * another department is "not found" there; the CHiPS admin passes none.
 *
 * Uploads go in pieces of CHUNK_BYTES, base64-encoded (the office firewall in front
 * of the bots refuses request bodies holding tag-like text, which a document's raw
 * bytes may contain). A piece that fails can be sent again; the first piece is
 * checked at once, so a wrong file stops before the rest is sent; the whole file is
 * checked at the end.
 *
 * On disk: <root>/platform/departments/<id>/files/<file id>.<kind> (named by id,
 * never by the uploaded name). Unfinished uploads sit in …/incoming/ and are
 * cleared after STALE_HOURS.
 */

const crypto = require('crypto');
const fs = require('fs/promises');
const { createReadStream } = require('fs');
const { execFile } = require('child_process');
const path = require('path');
const check = require('./filecheck');
const { failure } = require('./http');

const CHUNK_BYTES = 4 * 1024 * 1024;
const STALE_HOURS = 24;
const MAX_FILE_BYTES = (parseInt(process.env.PLATFORM_MAX_FILE_MB || '', 10) || 200) * 1024 * 1024;

function fileId(raw) {
  const id = String(raw || '').toLowerCase();
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(id)) throw failure(400, 'Invalid file.');
  return id;
}

async function sizeOf(file) {
  try { return (await fs.stat(file)).size; }
  catch (err) { if (err.code === 'ENOENT') return 0; throw err; }
}

function sha256Of(file) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    createReadStream(file).on('error', reject).on('data', (d) => hash.update(d))
      .on('end', () => resolve(hash.digest('hex')));
  });
}

/**
 * PLATFORM_SCAN_CMD, e.g. "clamdscan --no-summary --fdpass": run with the file path
 * appended. Exit 0 = clean, 1 = a virus was found, anything else = the scan failed.
 */
function scanFile(file, cmd) {
  if (!cmd) return Promise.resolve('not_scanned');
  const [bin, ...args] = cmd.split(/\s+/).filter(Boolean);
  return new Promise((resolve, reject) => {
    execFile(bin, [...args, file], { timeout: 5 * 60 * 1000 }, (err) => {
      if (!err) return resolve('clean');
      if (err.code === 1) return reject(failure(400, 'The virus scanner flagged this file. It was not kept.'));
      console.error(`[platform] virus scan failed: ${err.message}`);
      reject(failure(503, 'The virus scan could not run. Try the upload again later.'));
    });
  });
}

function createUploads({ store, root, scanCmd }) {
  const base = path.join(root, 'platform', 'departments');
  const deptDir = (id) => path.join(base, String(Number(id)));
  const partPath = (f) => path.join(deptDir(f.departmentId), 'incoming', `${f.id}.part`);
  const filePath = (f) => path.join(deptDir(f.departmentId), 'files', `${f.id}.${f.kind}`);

  // One request at a time per upload (pieces arrive in order; finish waits for them).
  const locks = new Map();
  function withLock(key, fn) {
    const run = (locks.get(key) || Promise.resolve()).then(fn);
    const tail = run.catch(() => {});
    locks.set(key, tail);
    tail.then(() => { if (locks.get(key) === tail) locks.delete(key); });
    return run;
  }

  async function owned(fid, scope) {
    const f = await store.getFile(fileId(fid));
    if (scope !== undefined && f.departmentId !== Number(scope)) throw failure(404, 'File not found.');
    return f;
  }

  async function discard(f) {
    await fs.unlink(partPath(f)).catch(() => {});
    await store.dropUpload(f.id);
  }

  /** Start an upload: {name, size}. The size is taken from the free storage now. */
  async function start(departmentId, body, by) {
    const b = body || {};
    const name = check.cleanName(b.name);
    const kind = check.kindOf(name);
    if (!name || !kind) {
      const allowed = check.describe().flatMap((t) => t.ext).map((e) => '.' + e).join(', ');
      throw failure(400, `This type of file cannot be uploaded. Allowed: ${allowed}.`);
    }
    const sizeBytes = Number(b.size);
    if (!Number.isSafeInteger(sizeBytes) || sizeBytes <= 0) throw failure(400, 'The file is empty.');
    if (sizeBytes > MAX_FILE_BYTES) {
      throw failure(413, `The file is larger than ${Math.round(MAX_FILE_BYTES / 1024 / 1024)} MB.`);
    }
    const file = await store.reserveFile(departmentId, { id: crypto.randomUUID(), name, kind, sizeBytes, by });
    return { file, chunkBytes: CHUNK_BYTES };
  }

  /** One piece: {offset, data: base64}. A resend after a lost reply gets 409 + received. */
  async function chunk(fid, body, scope) {
    const { offset, data } = body || {};
    if (!Number.isSafeInteger(offset) || offset < 0 || typeof data !== 'string' || !data
        || !/^[A-Za-z0-9+/]+={0,2}$/.test(data)) {
      throw failure(400, 'Invalid upload data.');
    }
    const buf = Buffer.from(data, 'base64');
    if (!buf.length || buf.length > CHUNK_BYTES) throw failure(400, 'Invalid upload data.');
    const id = fileId(fid);
    return withLock(id, async () => {
      const f = await owned(id, scope);
      if (f.status !== 'uploading') throw failure(409, 'This upload is already finished.');
      const part = partPath(f);
      const have = await sizeOf(part);
      if (offset !== have) throw failure(409, 'The upload is out of step.', { received: have });
      if (offset + buf.length > f.sizeBytes) throw failure(400, 'More data arrived than the file’s size.');
      if (offset === 0) {
        const bad = check.checkHeader(buf.subarray(0, check.HEAD_BYTES), f.kind);
        if (bad) { await discard(f); throw failure(400, bad); }
      }
      await fs.mkdir(path.dirname(part), { recursive: true });
      const fh = await fs.open(part, offset === 0 ? 'w' : 'r+');
      try { await fh.write(buf, 0, buf.length, offset); }
      finally { await fh.close(); }
      return offset + buf.length;
    });
  }

  /** All pieces sent: check the whole file, then keep it. */
  async function finish(fid, scope) {
    const id = fileId(fid);
    return withLock(id, async () => {
      const f = await owned(id, scope);
      if (f.status === 'uploaded') return f;               // a repeated finish
      const part = partPath(f);
      const have = await sizeOf(part);
      if (have !== f.sizeBytes) {
        throw failure(409, `The upload is incomplete (${have} of ${f.sizeBytes} bytes arrived).`, { received: have });
      }
      const bad = await check.checkFile(part, f.kind);
      if (bad) { await discard(f); throw failure(400, bad); }
      let scan;
      try { scan = await scanFile(part, scanCmd); }
      catch (err) { await discard(f); throw err; }
      const sha256 = await sha256Of(part);
      const final = filePath(f);
      await fs.mkdir(path.dirname(final), { recursive: true });
      await fs.rename(part, final);
      try {
        return await store.markUploaded(id, { sha256, scan });
      } catch (err) {
        await fs.unlink(final).catch(() => {});
        await store.dropUpload(id).catch(() => {});
        throw err;
      }
    });
  }

  /** Cancel an upload, or delete an uploaded file (its storage is free again). */
  async function remove(fid, by, scope) {
    const id = fileId(fid);
    await withLock(id, async () => {
      const f = await owned(id, scope);
      if (f.status === 'uploading') return discard(f);
      await store.markDeleted(id, by);
      await fs.unlink(filePath(f)).catch((err) => {
        if (err.code !== 'ENOENT') console.error(`[platform] could not remove ${filePath(f)}: ${err.message}`);
      });
    });
  }

  async function download(res, fid, scope) {
    const f = await owned(fid, scope);
    if (f.status !== 'uploaded') throw failure(404, 'File not found.');
    res.setHeader('Content-Type', check.TYPES[f.kind].mime);
    res.setHeader('Content-Disposition',
      `attachment; filename="${f.name.replace(/[^\x20-\x7e]|["\\]/g, '_')}"; filename*=UTF-8''${encodeURIComponent(f.name)}`);
    res.setHeader('Cache-Control', 'no-store');
    await new Promise((resolve, reject) => res.sendFile(filePath(f), (err) => (err ? reject(err) : resolve())));
  }

  async function cleanup() {
    for (const f of await store.staleUploads(STALE_HOURS)) {
      await fs.unlink(partPath(f)).catch(() => {});
      await store.dropUpload(f.id);
    }
  }

  async function removeDepartmentFolder(id) {
    await fs.rm(deptDir(id), { recursive: true, force: true }).catch((err) =>
      console.error(`[platform] could not remove ${deptDir(id)}: ${err.message}`));
  }

  async function disk() {
    if (typeof fs.statfs !== 'function') return null;
    try {
      await fs.mkdir(base, { recursive: true });
      const s = await fs.statfs(base);
      return { freeBytes: s.bavail * s.bsize, totalBytes: s.blocks * s.bsize };
    } catch (_) { return null; }
  }

  return { start, chunk, finish, remove, download, cleanup, removeDepartmentFolder, disk, pathOf: filePath };
}

module.exports = { createUploads, CHUNK_BYTES, MAX_FILE_BYTES, STALE_HOURS };
