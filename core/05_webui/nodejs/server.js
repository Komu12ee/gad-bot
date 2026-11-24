'use strict';

const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const morgan = require('morgan');
const path = require('path');

const PORT = process.env.PORT || 3002;
const FLASK_PORT = process.env.FLASK_PORT || 5002;
const FLASK_URL = `http://127.0.0.1:${FLASK_PORT}`;
const IS_PROD = process.env.NODE_ENV === 'production';

const app = express();

// Behind nginx (same machine, or another VM on the private network): believe
// X-Forwarded-For only from those hops, so req.ip is the visitor's real address —
// the per-network signup / login limits key on it. A visitor's own X-Forwarded-For
// is not trusted. TRUST_PROXY overrides (Express syntax, e.g. "10.0.0.5").
app.set('trust proxy', process.env.TRUST_PROXY || 'loopback, linklocal, uniquelocal');

// CORS for the pipeline UIs. In the supported topologies the browser only ever
// talks to its OWN origin (same-origin behind the reverse proxy, or the pipeline's
// own subdomain), so NO cross-origin request is needed — hence CORS is DISABLED by
// default. Reflecting the request origin with credentials:true (the old default)
// let any website make credentialed requests to a reachable pipeline port and read
// the response. Only enable CORS by explicitly listing trusted origins in
// CORS_ORIGIN (e.g. https://rag.example.gov).
const CORS_ORIGIN = process.env.CORS_ORIGIN
    ? process.env.CORS_ORIGIN.split(',').map((s) => s.trim()).filter(Boolean)
    : false;
if (CORS_ORIGIN) {
    app.use(require('cors')({ origin: CORS_ORIGIN, credentials: true }));
}

// ─── Logging ────────────────────────────────────────────────────────────────
app.use(morgan(IS_PROD ? 'combined' : 'dev'));

// ─── Security headers ───────────────────────────────────────────────────────
app.use((_req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  // SAMEORIGIN (not DENY) so the Documents page can render the source PDF in a
  // same-origin <iframe>; still blocks cross-origin framing (clickjacking).
  res.setHeader('X-Frame-Options', 'SAMEORIGIN');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  next();
});

// ─── Health check (UNAUTHENTICATED) ──────────────────────────────────────────
// The launcher polls this to know the server is up, so it must stay public. It
// exposes nothing sensitive. Registered before the auth gate below.
app.get('/health', (_req, res) => res.status(200).json({ success: true, status: 'ok' }));

// ─── Branding (PUBLIC) ───────────────────────────────────────────────────────
// Which department this instance serves. Display strings only — no secrets — so
// it is exposed before the auth gate, letting the login page brand itself.
// Registered before the /api proxy so it is answered here, not forwarded to Flask.
const { BRANDING } = require('./config');
app.get('/api/branding', (_req, res) => res.json(BRANDING));
const correctionModule = require('./verified-corrections');
const corrections = correctionModule.createService({ py: require('./platform/pyservice').createClient(FLASK_URL) });
app.use('/_internal/verified-corrections', correctionModule.internalRouter(corrections,
  (process.env.PIPELINE || 'chips').toLowerCase()));

// ─── Departments (PUBLIC part: the department panel) ─────────────────────────
// Departments, their storage, files and accounts — served by one bot only
// (PLATFORM_PIPELINE, default chips), in PostgreSQL. The department panel
// (/dept/<short name>/) has its own sign-in with DEPARTMENT accounts, so it is
// mounted before this bot's login gate; the CHiPS admin's side (/api/platform) is
// mounted behind it, below. On the other bots both answer "not enabled".
const platformRoot = process.env.RAG_DATA_ROOT || path.resolve(__dirname, '../../../data');
const platformPipeline = (process.env.PIPELINE || 'chips').toLowerCase();
const platform = require('./platform/routes').createPlatform({
  store: require('./platform/store'),
  root: platformRoot,
  enabled: platformPipeline === (process.env.PLATFORM_PIPELINE || 'chips').toLowerCase(),
  haveDatabase: () => require('./gad/db').haveConfig(),
  scanCmd: (process.env.PLATFORM_SCAN_CMD || '').trim(),
  pyUrl: FLASK_URL,
  corrections, pipeline: platformPipeline,
});
app.use('/platform-assets', platform.assets);
app.use('/dept', platform.deptRouter);

// ─── Auth gate ────────────────────────────────────────────────────────────────
// EVERY bot (chips/fg/fg2) now runs its OWN standalone login with admin/user roles
// (see gad/*.js) — independent of the selection gateway, which is just a public
// launcher. Admins get the document-verification workspace, users get the chatbot
// only. No bot UI, asset, or proxied API/PDF call is served without a valid session
// for THIS bot. CORS preflight (OPTIONS) is answered above, so it never reaches here.
{
  const botRoutes = require('./gad/routes');
  const { requireGadAuth, requireGadAdmin } = require('./gad/auth');

  // The account store is PostgreSQL-backed; the schema is created and the first
  // admin seeded in the async bootstrap below, before the server accepts requests.

  // Standalone login page (PUBLIC) and the auth/account API (login/session are
  // public; the account-management routes gate themselves to admins).
  app.get('/login', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'gad-login.html')));
  // Self-service signup is public by design: government staff register with their
  // official address and confirm it by email before the account can sign in.
  app.get('/signup', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'gad-signup.html')));
  app.use('/gad-auth', botRoutes);

  // The login page is public, so the handful of NON-SENSITIVE assets it references
  // must be public too. Without this, the auth gate below treats each CSS/JS/img
  // request as HTML navigation (Accept: */*) and redirects it to the login page —
  // so the login renders unstyled with a broken emblem. Only branding assets are
  // exposed; the app shell (index.html), documents.html and gad-admin.html stay
  // gated (served by express.static AFTER the gate).
  const publicDir = path.join(__dirname, 'public');
  const assetOpts = { maxAge: IS_PROD ? '7d' : 0, etag: true };
  app.get('/design-system.css', (_req, res) => res.sendFile(path.join(publicDir, 'design-system.css')));
  // Non-sensitive shared UI assets are also used by department sessions, which
  // authenticate independently of this bot's account gate.
  app.get('/chat-ui.css', (_req, res) => res.sendFile(path.join(publicDir, 'style.css')));
  app.get('/chat-ui.js', (_req, res) => res.sendFile(path.join(publicDir, 'chat-ui.js')));
  app.get('/branding.js', (_req, res) => res.sendFile(path.join(publicDir, 'branding.js')));
  app.get('/graph.html', (_req, res) => res.sendFile(path.join(publicDir, 'graph.html')));
  app.get('/graph.js', (_req, res) => res.sendFile(path.join(publicDir, 'graph.js')));
  app.use('/vendor', express.static(path.join(publicDir, 'vendor'), assetOpts));
  app.use('/img', express.static(path.join(publicDir, 'img'), assetOpts));

  // Forward public graph, retriever & officer API requests directly to Flask
  app.use('/api/graph', createProxyMiddleware({
    target: FLASK_URL,
    changeOrigin: true,
    pathRewrite: (p) => '/api/graph' + p
  }));
  app.use('/api/retriever', createProxyMiddleware({
    target: FLASK_URL,
    changeOrigin: true,
    pathRewrite: (p) => '/api/retriever' + p
  }));
  app.use('/api/officer', createProxyMiddleware({
    target: FLASK_URL,
    changeOrigin: true,
    pathRewrite: (p) => '/api/officer' + (p.startsWith('/') ? p : '/' + p)
  }));
  app.use('/api/officers', createProxyMiddleware({
    target: FLASK_URL,
    changeOrigin: true,
    pathRewrite: (p) => {
      if (p === '/' || p === '') return '/api/officers';
      if (p.startsWith('/?')) return '/api/officers' + p.slice(1);
      return '/api/officers' + (p.startsWith('/') ? p : '/' + p);
    }
  }));

  // Gate everything below on a valid session for this bot (or an internal service token).
  app.use(requireGadAuth({ loginUrl: '/login' }));

  // Admin-only surfaces, BEFORE static so they can't be served to a plain user:
  //  · the document-management pages (HTML)
  //  · every /api/documents route (verify / edit / discard — the write surface)
  //  · the batch test runs (many questions at once, answers downloadable)
  //  · /api/admin — e.g. switching the answer model (it serves every bot)
  // Chat citations use /01_preprocessing/used_files, which stays open to any
  // signed-in user so answers can still show their source PDFs.
  app.get(['/documents.html', '/gad-admin.html', '/batch-test.html', '/evaluation.html', '/departments.html', '/gpu.html'],
    requireGadAdmin({ redirectUrl: '/' }));
  app.use('/api/documents', requireGadAdmin());
  app.use('/api/batch', requireGadAdmin());
  app.use(['/api/eval', '/eval'], requireGadAdmin());
  app.use('/api/admin', requireGadAdmin());
  app.use('/api/platform', requireGadAdmin());   // departments and their files (CHiPS admin)
}

// Save conversations in this bot's persistent data, behind its account gate.
const history = require('./chat-history');
const historyPipeline = (process.env.PIPELINE || 'chips').toLowerCase();
const historyRoot = process.env.RAG_DATA_ROOT || path.resolve(__dirname, '../../../data');
const chatHistoryStore = history.createStore(historyRoot, historyPipeline);
app.use('/api/history', history.createRouter(chatHistoryStore, historyPipeline));
const evaluation = require('./evaluation');
app.use('/api/evaluation', evaluation.createRouter(evaluation.createStore(historyRoot, historyPipeline, chatHistoryStore)));
app.use('/api/corrections', correctionModule.createRouter(corrections, {
  scope: historyPipeline, actor: req => req.gadUser,
}));

// The CHiPS admin's side of the departments (created above; admin-gated above).
// Answered here on every bot so /api/platform never reaches Flask.
app.use('/api/platform', platform.router);

// ─── Static assets ──────────────────────────────────────────────────────────
// Code (html / js / css) is revalidated on every load — a 304 when unchanged — so an
// update reaches everyone at once. With the old 7-day cache a browser that had
// visited kept the previous app.js for a week (index.html's ?v= tag is bumped by
// hand, and was not for the follow-up-question release). Images keep 7 days.
app.use(
  express.static(path.join(__dirname, 'public'), {
    etag: true,
    setHeaders: (res, filePath) => {
      res.setHeader('Cache-Control', /\.(html|js|css)$/i.test(filePath) || !IS_PROD
        ? 'no-cache' : 'public, max-age=604800');
    },
  })
);

// ─── Proxies ─────────────────────────────────────────────────────────────────
// hpm v3 strips the mount path before forwarding (e.g. /api/query → /query).
// pathRewrite as a FUNCTION (not object) re-adds the prefix.
// Object form { '^/api': '/api' } is silently ignored in hpm v3.

const apiProxy = createProxyMiddleware({
  target: FLASK_URL,
  changeOrigin: true,
  selfHandleResponse: false,
  pathRewrite: (path) => '/api' + path,   // /query → /api/query
  on: {
    error: (err, _req, res) => {
      console.error('[proxy] API error:', err.message);
      if (!res.headersSent) {
        res.status(502).json({
          success: false,
          error: 'RAG backend is unreachable. Is Flask running?',
        });
      }
    },
    proxyReq: (proxyReq, req) => {
      // Who is asking — a batch test run records who started it. Set here from the
      // session, never taken from the browser.
      proxyReq.removeHeader('x-gad-user');
      proxyReq.removeHeader('x-gad-role');
      if (req.gadUser && /^[a-z0-9._-]{1,64}$/i.test(req.gadUser.username || '')) {
        proxyReq.setHeader('X-Gad-User', req.gadUser.username);
        proxyReq.setHeader('X-Gad-Role', req.gadUser.role);
      }
      if (!IS_PROD) {
        console.log(`[proxy] → ${req.method} ${FLASK_URL}${req.url}`);
      }
    },
  },
});

const pdfProxy = createProxyMiddleware({
  target: FLASK_URL,
  changeOrigin: true,
  pathRewrite: (path) => '/01_preprocessing' + path,
  on: {
    error: (err, _req, res) => {
      console.error('[proxy] PDF error:', err.message);
      if (!res.headersSent) {
        res.status(502).json({ error: 'PDF service unavailable' });
      }
    },
  },
});

// The department bots' internal routes in Flask (dept_service.py) are for this
// server only (platform/pyservice.js, with a key) — never forwarded from a browser.
app.use('/api/dept', (_req, res) => res.status(404).json({ success: false, error: 'Not found.' }));

const evalProxy = createProxyMiddleware({
  target: FLASK_URL,
  changeOrigin: true,
  selfHandleResponse: false,
  pathRewrite: (path) => '/eval' + path,
  on: {
    error: (err, _req, res) => {
      console.error('[proxy] Eval API error:', err.message);
      if (!res.headersSent) {
        res.status(502).json({ success: false, error: 'RAG backend is unreachable.' });
      }
    },
  },
});

// Guarded by the gateway-session auth gate mounted above.
app.use('/api', apiProxy);
app.use('/eval', evalProxy);
app.use('/01_preprocessing', pdfProxy);

// ─── SPA fallback ───────────────────────────────────────────────────────────
app.get('*', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ─── Start ───────────────────────────────────────────────────────────────────
// The account store must be ready and an admin seeded BEFORE the first request is
// served. The store picks its backend from the environment: PostgreSQL when
// DATABASE_URL / PG* vars are set, otherwise a per-bot JSON file (zero infra, so a
// fresh clone just boots). Both are async, so startup waits before app.listen. A
// configured-but-unreachable store is fatal — an auth gate with no backing store
// must not serve.
const botStore = require('./gad/store');

(async () => {
  try {
    await botStore.init();            // create PG schema if missing; no-op for file store
    await botStore.ensureSeedAdmin(); // guarantee an admin exists on first run
  } catch (err) {
    console.error(`[startup] Account store (${botStore.backendName()}) failed to initialise — refusing to start: ${err.message}`);
    if (botStore.backendName() === 'postgres') {
      console.error('[startup] Check DATABASE_URL (or PGHOST/PGUSER/PGPASSWORD/PGDATABASE) and that the server is running, or unset them to use the file store.');
    }
    process.exit(1);
  }
  // Not fatal: without its tables the Departments page says why; the bot runs on.
  await platform.init();
  await corrections.init();

  app.listen(PORT, '0.0.0.0', () => {
    console.log('');
    console.log('═══════════════════════════════════════════════════════════');
    console.log('  CHiPS-RAG  –  Express UI Server');
    console.log('═══════════════════════════════════════════════════════════');
    console.log(`  UI      →  http://0.0.0.0:${PORT}`);
    console.log(`  Flask   →  ${FLASK_URL}  (proxied, gateway-session required)`);
    console.log(`  Mode    →  ${IS_PROD ? 'production' : 'development'}`);
    console.log('═══════════════════════════════════════════════════════════');
    console.log('');
  });
})();
