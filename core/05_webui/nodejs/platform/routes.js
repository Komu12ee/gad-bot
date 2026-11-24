'use strict';

/**
 * Department platform — HTTP routes.
 *
 * Two routers over the same departments:
 *
 *   router       /api/platform/…   the CHiPS admin: create departments, give them
 *                storage, add or delete their files, keep their accounts. Mounted
 *                behind the bot's own admin login (server.js).
 *
 *   deptRouter   /dept/<short name>/…   the department panel: the department's own
 *                accounts sign in (session.js). Admins add or delete files within
 *                their storage, process them and keep the accounts; every account
 *                asks the department's chatbot. Mounted BEFORE the bot's login — it
 *                has its own. GET /dept/ lists the open department bots (launcher).
 *
 * Uploads past a department's storage are refused. Files are only stored on
 * upload; processing them into the chatbot's passages is started by a person
 * (processing.js), and answering goes through chat.js.
 *
 * Served by one bot only (PLATFORM_PIPELINE, default "chips"). Needs PostgreSQL
 * (DATABASE_URL); without it the pages say so and the bot itself runs as before.
 */

const path = require('path');
const express = require('express');
const check = require('./filecheck');
const session = require('./session');
const limiter = require('../gad/ratelimit');
const history = require('../chat-history');
const { createUploads, CHUNK_BYTES, MAX_FILE_BYTES } = require('./uploads');
const { createAccounts, publicAccount } = require('./accounts');
const { createProcessor } = require('./processing');
const { createChat } = require('./chat');
const pyservice = require('./pyservice');
const { failure, wrap, before, jsonBody, errorHandler } = require('./http');

const MAX_QUOTA_BYTES = 100 * 1024 ** 4;     // 100 TB: only guards against a typo
// Short names live under /dept/<name>/, so no route can clash; these (plus the bots'
// own names) are refused only because they would read like system pages.
const RESERVED = ['platform', 'api', 'admin', 'select', 'pipeline', 'static', 'dept'];
const PUBLIC_DIR = path.join(__dirname, 'public');
const LOGIN_MAX = () => parseInt(process.env.LOGIN_MAX_ATTEMPTS || '', 10) || 10;
const LOGIN_WINDOW = () => parseInt(process.env.LOGIN_WINDOW_SECONDS || '', 10) || 900;
const SIGNUP_COOLDOWN = () => parseInt(process.env.SIGNUP_COOLDOWN_SECONDS || '', 10) || 30;
const SIGNUP_DAILY_CAP = () => parseInt(process.env.SIGNUP_DAILY_CAP || '', 10) || 10;

function pipelineNames() {
  try { return Object.keys(require(path.join(__dirname, '..', '..', '..', '..', 'pipelines.json'))); }
  catch (_) { return ['chips', 'fg', 'fg2']; }
}

/** Short name for links and the future bot: "Health & Family Welfare" -> "health-family-welfare". */
function slugify(s) {
  return String(s || '').toLowerCase().normalize('NFKD').replace(/[^\x00-\x7f]/g, '')
    .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 31).replace(/-+$/, '');
}
const SLUG_RE = /^[a-z][a-z0-9-]{1,30}$/;

function cleanText(s, max) {
  return String(s == null ? '' : s).replace(/[\u0000-\u001f\u007f<>]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, max);
}

function idParam(raw, what) {
  const id = Number(raw);
  if (!Number.isSafeInteger(id) || id <= 0) throw failure(400, `Invalid ${what}.`);
  return id;
}

function quotaFrom(raw) {
  const n = Number(raw);
  if (!Number.isSafeInteger(n) || n <= 0) throw failure(400, 'Give the storage as a size above zero.');
  if (n > MAX_QUOTA_BYTES) throw failure(400, 'That storage size is too large.');
  return n;
}

/**
 * @param {object} o
 * @param {object} o.store         platform/store.js (or a stand-in with the same functions)
 * @param {string} o.root          RAG_DATA_ROOT
 * @param {boolean} o.enabled      this bot serves the platform
 * @param {function} o.haveDatabase  () => is PostgreSQL configured
 * @param {string} [o.scanCmd]     PLATFORM_SCAN_CMD
 * @param {string} [o.pyUrl]       the bot's Python service (Flask), e.g. http://127.0.0.1:5001
 * @param {function} [o.py]        or a stand-in for its client (tests)
 */
function createPlatform({ store, root, enabled, haveDatabase, scanCmd, pyUrl, py, corrections, pipeline = 'chips' }) {
  const uploads = createUploads({ store, root, scanCmd });
  const accounts = createAccounts({ store });
  const callPy = py || pyservice.createClient(pyUrl || 'http://127.0.0.1:5000');
  const processor = createProcessor({ store, pathOf: uploads.pathOf, py: callPy });
  const ask = createChat({ store, py: callPy, corrections, correctionScope: id => `${pipeline}:dept:${id}` });
  const search = () => (store.vectors ? store.vectors() : { ready: false, error: 'not available' });
  const reserved = new Set([...RESERVED, ...pipelineNames()]);
  const departmentHistories = new Map();

  let ready = false;
  let initError = null;
  let lastTry = 0;
  let timer = null;

  const cleanup = () => uploads.cleanup();

  async function init() {
    if (!enabled) return;
    lastTry = Date.now();
    if (!haveDatabase()) {
      initError = 'PostgreSQL is not configured for this bot (DATABASE_URL), so departments cannot be stored.';
      return;
    }
    try {
      await store.init();
      ready = true;
      initError = null;
      await cleanup().catch((err) => console.error(`[platform] cleanup: ${err.message}`));
      if (!timer) {
        timer = setInterval(() => cleanup().catch((err) => console.error(`[platform] cleanup: ${err.message}`)), 60 * 60 * 1000);
        if (timer.unref) timer.unref();
      }
      console.log('[platform] departments ready');
      if (search().ready) await processor.resume();
      else console.warn(`[platform] department search off: ${search().error}`);
    } catch (err) {
      initError = `The department tables could not be set up: ${err.message}`;
      console.error(`[platform] ${initError}`);
    }
  }

  // Everything but /status needs this to be the platform bot with its tables in place.
  const needReady = before(async () => {
    if (!enabled) throw failure(404, 'Not found.');
    if (!ready && Date.now() - lastTry > 30000) await init();
    if (!ready) throw failure(503, initError || 'Departments are not available.');
  });

  // ══ The CHiPS admin ════════════════════════════════════════════════════════
  const router = express.Router();
  router.use(jsonBody());
  const chipsAdmin = (req) => `${(req.gadUser && req.gadUser.username) || 'admin'} (CHiPS)`;

  router.get('/status', wrap(async (_req, res) => {
    if (enabled && !ready && Date.now() - lastTry > 30000) await init();
    res.json({
      success: true, enabled, ready, error: enabled ? initError : null,
      types: check.describe(), maxFileBytes: MAX_FILE_BYTES, chunkBytes: CHUNK_BYTES,
      virusScan: !!scanCmd, disk: enabled ? await uploads.disk() : null,
      search: enabled && ready ? search() : { ready: false, error: null },
    });
  }));

  router.use(needReady);

  // ── Departments
  router.get('/departments', wrap(async (_req, res) => {
    res.json({ success: true, departments: await store.listDepartments() });
  }));

  router.post('/departments', wrap(async (req, res) => {
    const b = req.body || {};
    const name = cleanText(b.name, 120);
    if (name.length < 2) throw failure(400, 'Enter the department’s name.');
    const slug = slugify(b.slug || name);
    if (!SLUG_RE.test(slug)) throw failure(400, 'Give a short name in English letters (for example “health”).');
    if (reserved.has(slug)) throw failure(400, `“${slug}” is already used by the system. Choose another short name.`);
    const department = await store.createDepartment({
      name, slug, quotaBytes: quotaFrom(b.quotaBytes), by: (req.gadUser && req.gadUser.username) || '',
    });
    res.status(201).json({ success: true, department });
  }));

  router.patch('/departments/:id', wrap(async (req, res) => {
    const id = idParam(req.params.id, 'department');
    const b = req.body || {};
    const patch = {};
    if (b.name !== undefined) {
      patch.name = cleanText(b.name, 120);
      if (patch.name.length < 2) throw failure(400, 'Enter the department’s name.');
    }
    if (b.quotaBytes !== undefined) patch.quotaBytes = quotaFrom(b.quotaBytes);
    if (b.status !== undefined) {
      if (!['active', 'disabled'].includes(b.status)) throw failure(400, 'Invalid status.');
      patch.status = b.status;
    }
    if (b.listed !== undefined) {               // shown on the launcher page, or not
      if (typeof b.listed !== 'boolean') throw failure(400, 'Invalid launcher setting.');
      patch.listed = b.listed;
    }
    if (!Object.keys(patch).length) throw failure(400, 'Nothing to change.');
    res.json({ success: true, department: await store.updateDepartment(id, patch) });
  }));

  router.delete('/departments/:id', wrap(async (req, res) => {
    const id = idParam(req.params.id, 'department');
    await store.deleteDepartment(id);
    await uploads.removeDepartmentFolder(id);
    res.json({ success: true });
  }));

  // ── A department's files
  router.get('/departments/:id/files', wrap(async (req, res) => {
    const id = idParam(req.params.id, 'department');
    const department = await store.getDepartment(id);
    res.json({ success: true, department, files: await store.listFiles(id) });
  }));
  router.post('/departments/:id/files', wrap(async (req, res) => {
    const out = await uploads.start(idParam(req.params.id, 'department'), req.body, chipsAdmin(req));
    res.status(201).json({ success: true, ...out });
  }));
  router.put('/files/:fid/chunk', wrap(async (req, res) => {
    res.json({ success: true, received: await uploads.chunk(req.params.fid, req.body) });
  }));
  router.post('/files/:fid/finish', wrap(async (req, res) => {
    res.json({ success: true, file: await uploads.finish(req.params.fid) });
  }));
  router.delete('/files/:fid', wrap(async (req, res) => {
    await uploads.remove(req.params.fid, chipsAdmin(req));
    res.json({ success: true });
  }));
  router.get('/files/:fid/download', wrap(async (req, res) => uploads.download(res, req.params.fid)));

  // ── Processing (files → the chatbot's passages): {fileIds?}; none = every new file
  router.post('/departments/:id/process', wrap(async (req, res) => {
    const id = idParam(req.params.id, 'department');
    await store.getDepartment(id);
    res.json({ success: true, queued: await queueProcessing(id, req.body) });
  }));

  // ── A department's accounts
  router.get('/departments/:id/users', wrap(async (req, res) => {
    const id = idParam(req.params.id, 'department');
    await store.getDepartment(id);
    res.json({ success: true, users: (await accounts.list(id)).map(publicAccount) });
  }));
  router.post('/departments/:id/users', wrap(async (req, res) => {
    const user = await accounts.create(idParam(req.params.id, 'department'), req.body, chipsAdmin(req));
    res.status(201).json({ success: true, user: publicAccount(user) });
  }));
  router.patch('/departments/:id/users/:uid', wrap(async (req, res) => {
    const user = await accounts.update(idParam(req.params.id, 'department'), idParam(req.params.uid, 'account'), req.body);
    res.json({ success: true, user: publicAccount(user) });
  }));
  router.delete('/departments/:id/users/:uid', wrap(async (req, res) => {
    await accounts.remove(idParam(req.params.id, 'department'), idParam(req.params.uid, 'account'));
    res.json({ success: true });
  }));

  router.use(errorHandler);

  async function queueProcessing(departmentId, body) {
    if (!search().ready) throw failure(503, search().error || 'Processing is not available.');
    const ids = body && body.fileIds;
    if (ids !== undefined && (!Array.isArray(ids) || ids.length > 1000
        || !ids.every((x) => typeof x === 'string' && /^[0-9a-f-]{36}$/i.test(x)))) {
      throw failure(400, 'Invalid files.');
    }
    return processor.queue(departmentId, ids && ids.length ? ids : null);
  }

  // ══ The department panel ═══════════════════════════════════════════════════
  // Pages and API live under /dept/<short name>/; the pages use relative links
  // (api/…, ../../platform-assets/…), so they work under any proxy prefix.
  const dept = express.Router();
  dept.use(jsonBody());

  async function loadDepartment(req) {
    const slug = String(req.params.slug || '');
    if (!SLUG_RE.test(slug)) throw failure(404, 'Department not found.');
    req.dept = await store.getDepartmentBySlug(slug);
  }

  /** The signed-in account of this department, read fresh (a block works at once). */
  async function signedIn(req) {
    await loadDepartment(req);
    const token = session.verify(session.read(req, req.dept.id), req.dept.id);
    if (!token) return null;
    const user = await store.getUser(req.dept.id, token.u).catch((err) => {
      if (err.status === 404) return null;
      throw err;
    });
    if (!user || user.status !== 'active' || user.sessionEpoch !== token.e) return null;
    return user;
  }

  const needSession = before(async (req) => {
    req.deptUser = await signedIn(req);
    if (!req.deptUser) throw failure(401, 'Please sign in again.');
  });
  const needAdmin = before(async (req) => {
    if (req.deptUser.role !== 'admin') throw failure(403, 'This account cannot use the department panel.');
  });

  // Pages. "/dept/health" → "/dept/health/" so relative links resolve inside it.
  const page = (file) => (_req, res) => {
    res.setHeader('Cache-Control', 'no-cache');
    res.sendFile(path.join(PUBLIC_DIR, file));
  };
  // The department bots the CHiPS admin put on the launcher: name and short name only.
  dept.get('/', needReady, wrap(async (_req, res) => {
    const list = await store.openDepartments();
    res.json({ success: true, departments: list.map((d) => ({ name: d.name, slug: d.slug })) });
  }));

  dept.get('/:slug', needReady, wrap(async (req, res) => {
    if (!req.path.endsWith('/')) return res.redirect(`${encodeURIComponent(req.params.slug)}/`);
    if (!(await signedIn(req))) return res.redirect('login');
    return page('panel.html')(req, res);
  }));
  dept.get('/:slug/login', needReady, wrap(async (req, res) => {
    await loadDepartment(req);
    return page('login.html')(req, res);
  }));
  dept.get('/:slug/signup', needReady, wrap(async (req, res) => {
    await loadDepartment(req);
    return page('signup.html')(req, res);
  }));
  dept.get('/:slug/corrections', needReady, wrap(async (req,res) => {
    req.deptUser = await signedIn(req);
    if (!req.deptUser) return res.redirect('login');
    if (req.deptUser.role !== 'admin') throw failure(403,'Admin access required.');
    return page('corrections.html')(req,res);
  }));

  // Public: which department this is, for the sign-in page.
  dept.get('/:slug/api/info', needReady, wrap(async (req, res) => {
    await loadDepartment(req);
    res.json({ success: true, department: { name: req.dept.name, slug: req.dept.slug, status: req.dept.status } });
  }));

  // Self-registration belongs only to the department named in the URL. The
  // existing account creation validates and hashes passwords; clients cannot
  // choose a role, status, creator or a different department.
  dept.post('/:slug/api/signup', needReady, wrap(async (req, res) => {
    await loadDepartment(req);
    if (req.dept.status !== 'active') throw failure(403, 'Registration is unavailable while this department is disabled.');
    const key = `dept-signup:cool:${req.ip}`;
    const gate = limiter.cooldown(key, SIGNUP_COOLDOWN());
    if (!gate.ok) throw failure(429, `Please wait ${gate.retryAfter}s before registering again.`, { retryAfter: gate.retryAfter });
    const cap = limiter.counter(`dept-signup:day:${req.ip}`, SIGNUP_DAILY_CAP(), 24 * 3600);
    if (!cap.ok) throw failure(429, 'Too many registrations from this network today. Contact your department admin.', { retryAfter: cap.retryAfter });
    const body = req.body || {};
    const user = await accounts.create(req.dept.id,
      { username: body.username, password: body.password, role: 'user' }, 'self-signup');
    limiter.mark(key);
    console.log(`[platform] department signup: ${user.username} (user) @ ${req.dept.slug}`);
    res.status(201).json({ success: true, user: publicAccount(user),
      message: 'Your account has been created. You can sign in now.' });
  }));

  dept.post('/:slug/api/login', needReady, wrap(async (req, res) => {
    await loadDepartment(req);
    const { username, password } = req.body || {};
    if (typeof username !== 'string' || !username.trim() || typeof password !== 'string' || !password) {
      throw failure(400, 'Enter your username and password.');
    }
    const idKey = `dept-login:${req.dept.id}:${username.trim().toLowerCase()}`;
    const byId = limiter.counter(idKey, LOGIN_MAX(), LOGIN_WINDOW());
    const byIp = limiter.counter(`dept-login:ip:${req.ip}`, LOGIN_MAX() * 5, LOGIN_WINDOW());
    if (!byId.ok || !byIp.ok) {
      const retryAfter = Math.max(byId.retryAfter || 0, byIp.retryAfter || 0);
      throw failure(429, `Too many sign-in attempts. Try again in ${Math.ceil(retryAfter / 60)} min.`, { retryAfter });
    }
    const user = await accounts.verify(req.dept.id, username, password);
    if (!user) throw failure(401, 'Wrong username or password.');
    limiter.reset(idKey);
    // Said only after the right password, so it never tells which names exist.
    if (user.status !== 'active') throw failure(403, 'This account is blocked. Ask your department admin or CHiPS.');
    await store.touchLogin(user.id);
    session.set(res, user);
    console.log(`[platform] department sign-in: ${user.username} (${user.role}) @ ${req.dept.slug}`);
    res.json({ success: true, user: publicAccount(user) });
  }));

  dept.post('/:slug/api/logout', needReady, wrap(async (req, res) => {
    await loadDepartment(req);
    session.clear(res, req.dept.id);
    res.json({ success: true });
  }));

  dept.use('/:slug/api', needReady, needSession);

  // Reuse the bots' persistent history API. Department ID scopes the directory;
  // the fresh department session's account ID scopes the owner within it.
  dept.use('/:slug/api/history', (req, res, next) => {
    const key = String(req.dept.id);
    if (!departmentHistories.has(key)) {
      departmentHistories.set(key, history.createRouter(
        history.createStore(path.join(root, 'departments'), key), key));
    }
    req.gadUser = { username: String(req.deptUser.id), role: req.deptUser.role };
    departmentHistories.get(key)(req, res, next);
  });

  dept.get('/:slug/api/session', wrap(async (req, res) => {
    res.json({ success: true, user: publicAccount(req.deptUser), department: req.dept });
  }));
  dept.post('/:slug/api/password', wrap(async (req, res) => {
    const b = req.body || {};
    const user = await accounts.changeOwn(req.deptUser, b.current, b.next);
    session.set(res, user);                    // this browser stays signed in; others are signed out
    res.json({ success: true });
  }));

  // The chatbot: every account of the department (users and admins).
  dept.post('/:slug/api/ask', wrap(async (req, res) => {
    if (!search().ready) throw failure(503, 'The assistant is not available yet.');
    res.json({ success: true, ...(await ask(req.dept, req.deptUser, req.body)) });
  }));
  // A source document named in an answer, to read it.
  dept.get('/:slug/api/sources/:fid', wrap(async (req, res) => uploads.download(res, req.params.fid, req.dept.id)));

  // Files, processing and accounts: the department's admins, their own department only.
  dept.use('/:slug/api', needAdmin);
  if (corrections) dept.use('/:slug/api/corrections', require('../verified-corrections').createRouter(corrections, {
    scope: req => `${pipeline}:dept:${req.dept.id}`, actor: req => req.deptUser,
  }));
  const own = (req) => req.dept.id;
  const by = (req) => req.deptUser.username;

  dept.get('/:slug/api/files', wrap(async (req, res) => {
    res.json({
      success: true, department: await store.getDepartment(own(req)), files: await store.listFiles(own(req)),
      limits: { types: check.describe(), maxFileBytes: MAX_FILE_BYTES }, search: search(),
    });
  }));
  dept.post('/:slug/api/files', wrap(async (req, res) => {
    res.status(201).json({ success: true, ...(await uploads.start(own(req), req.body, by(req))) });
  }));
  dept.put('/:slug/api/files/:fid/chunk', wrap(async (req, res) => {
    res.json({ success: true, received: await uploads.chunk(req.params.fid, req.body, own(req)) });
  }));
  dept.post('/:slug/api/files/:fid/finish', wrap(async (req, res) => {
    res.json({ success: true, file: await uploads.finish(req.params.fid, own(req)) });
  }));
  dept.delete('/:slug/api/files/:fid', wrap(async (req, res) => {
    await uploads.remove(req.params.fid, by(req), own(req));
    res.json({ success: true });
  }));
  dept.get('/:slug/api/files/:fid/download', wrap(async (req, res) => uploads.download(res, req.params.fid, own(req))));
  dept.post('/:slug/api/process', wrap(async (req, res) => {
    res.json({ success: true, queued: await queueProcessing(own(req), req.body) });
  }));

  dept.get('/:slug/api/users', wrap(async (req, res) => {
    res.json({ success: true, users: (await accounts.list(own(req))).map(publicAccount), me: req.deptUser.id });
  }));
  dept.post('/:slug/api/users', wrap(async (req, res) => {
    const user = await accounts.create(own(req), req.body, by(req));
    res.status(201).json({ success: true, user: publicAccount(user) });
  }));
  dept.patch('/:slug/api/users/:uid', wrap(async (req, res) => {
    const user = await accounts.update(own(req), idParam(req.params.uid, 'account'), req.body, req.deptUser.id);
    res.json({ success: true, user: publicAccount(user) });
  }));
  dept.delete('/:slug/api/users/:uid', wrap(async (req, res) => {
    await accounts.remove(own(req), idParam(req.params.uid, 'account'), req.deptUser.id);
    res.json({ success: true });
  }));

  // Anything else under /dept is answered here, never by the bot behind it.
  dept.use((_req, _res, next) => next(failure(404, 'Not found.')));
  dept.use((err, req, res, next) => {
    // Pages get a plain message; the API gets JSON.
    if (!/\/api\//.test(req.path) && !res.headersSent && req.method === 'GET') {
      const status = err.status || 500;
      if (!err.status) console.error(`[platform] ${err.stack || err.message}`);
      return res.status(status).type('text/plain').send(err.status ? err.message : 'Something went wrong.');
    }
    return errorHandler(err, req, res, next);
  });

  // Shared scripts and styles of the platform pages (no secrets; public so the
  // department sign-in page can load them).
  const assets = express.static(path.join(PUBLIC_DIR, 'assets'), {
    etag: true, setHeaders: (res) => res.setHeader('Cache-Control', 'no-cache'),
  });

  return {
    router, deptRouter: dept, assets, init, cleanup, processor,
    stop: () => { if (timer) clearInterval(timer); timer = null; },
  };
}

module.exports = { createPlatform, slugify, CHUNK_BYTES, MAX_FILE_BYTES };
