/**
 * CHiPS-RAG  –  Frontend application
 *
 * Single sign-on lives at the selection gateway. This pipeline UI trusts the
 * gateway session (an httpOnly cookie sent automatically on same-origin /api/*
 * requests); there is no separate pipeline login. On any 401 the app reloads so
 * the server's auth gate redirects to the gateway login.
 * PDFs are fetched as blobs (same-origin cookie) and shown in a half-screen panel.
 */

'use strict';
// Same-origin: the pipeline's Node server serves this page and proxies /api/* to
// its Flask backend. Empty base works for every pipeline (chips/fg/fg2) when hit
// directly (:3001/:3002/:3003). Set to "/pipeline/<name>" only behind a path-based reverse proxy.
// Empty for direct access; '/pipeline/<name>' when served under the reverse proxy
// (set by the inline snippet in the page <head>). Every same-origin call below is
// prefixed with it, so the app works in both topologies unchanged.
const BASE = (typeof window !== 'undefined' && window.__APP_BASE__) || '';

const THEME_KEY = 'chips_rag_theme';

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  const toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.textContent = theme === 'light' ? 'Dark mode' : 'Light mode';
    toggle.setAttribute('aria-label', theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode');
  }
}

function initTheme() {
  applyTheme('light');

  const toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const nextTheme = document.body.dataset.theme === 'light' ? 'dark' : 'light';
      localStorage.setItem(THEME_KEY, nextTheme);
      applyTheme(nextTheme);
    });
  }
}

// ── Token storage (sessionStorage: cleared on tab/browser close) ──────────
const TOKEN_KEY = 'chips_rag_token';
const USER_KEY = 'chips_rag_user';

const session = {
  save(token, user) {
    sessionStorage.setItem(TOKEN_KEY, token);
    sessionStorage.setItem(USER_KEY, JSON.stringify(user));
  },
  token() { return sessionStorage.getItem(TOKEN_KEY); },
  user() { try { return JSON.parse(sessionStorage.getItem(USER_KEY)); } catch { return null; } },
  clear() { sessionStorage.removeItem(TOKEN_KEY); sessionStorage.removeItem(USER_KEY); },
  exists() { return !!sessionStorage.getItem(TOKEN_KEY); },
};

// ── State ─────────────────────────────────────────────────────────────────
const state = {
  initialized: false,
  loading: false,
  lastResults: [],
  pdfBlobUrl: null,   // current blob URL so we can revoke it on close
  // Thinking mode is sticky for the session but deliberately NOT persisted — it is
  // slower (several model calls) and not what the retrieval eval measures, so each
  // visit starts on the single-shot path; the toggle switches it on per session.
  thinkingMode: false,
  // Only this conversation's recent successful turns go to the answer model.
  turns: [],
  messages: [],
  chats: [],
  chatId: null,
  historyKey: null,
  historyBusy: false,
  dirty: false,
  evaluationIndexes: new Set(),
  reviewButtons: [],
  isAdmin: false,
};
const CHAT_TURNS_SENT = 3;

// ── DOM refs ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const ui = {
  // login
  loginScreen: $('login-screen'),
  loginForm: $('login-form'),
  loginUsername: $('login-username'),
  loginPassword: $('login-password'),
  loginError: $('login-error'),
  loginSubmit: $('login-submit'),
  loginBtnText: $('login-btn-text'),
  loginBtnSpinner: $('login-btn-spinner'),

  // app shell
  appShell: $('app-shell'),
  userDisplay: $('user-display'),
  btnLogout: $('btn-logout'),

  // rag ui
  btnSend: $('btn-send'),
  btnThinking: $('btn-thinking'),
  btnNewChat: $('btn-new-chat'),
  queryInput: $('query-input'),
  chatEmpty: $('chat-empty'),
  chatMessages: $('chat-messages'),
  chatList: $('chat-list'),

  // source chunk drawer (right)
  drawerOverlay: $('drawer-overlay'),
  sourceDrawer: $('source-drawer'),
  drawerBody: $('drawer-body'),
  drawerClose: $('drawer-close'),

  // pdf panel (left half)
  pdfPanel: $('pdf-panel'),
  pdfOverlay: $('pdf-overlay'),
  pdfIframe: $('pdf-iframe'),
  pdfTitle: $('pdf-title'),
  pdfClose: $('pdf-close'),
  pdfLoading: $('pdf-loading'),

  toastContainer: $('toast-container'),
};

// ── API client ────────────────────────────────────────────────────────────
const api = {
  async _request(method, path, body) {
    // Auth is the same-origin gateway session cookie (sent automatically); no
    // Authorization header needed. The server's auth gate enforces it.
    const headers = { 'Content-Type': 'application/json', 'Accept': 'application/json' };
    const opts = { method, headers };
    if (body !== undefined) opts.body = JSON.stringify(body);

    const res = await fetch(path, opts);
    const json = await res.json().catch(() => ({}));

    if (res.status === 401) {
      // Gateway session missing/expired — bounce to the single sign-on gateway.
      goToGatewayLogin();
      throw new Error('UNAUTHENTICATED');
    }

    return { ok: res.ok, status: res.status, data: json };
  },

  // Fetch a PDF as a blob (relies on the same-origin session cookie).
  async fetchPdf(path) {
    const res = await fetch(path);
    if (res.status === 401) {
      goToGatewayLogin();
      throw new Error('UNAUTHENTICATED');
    }
    if (!res.ok) throw new Error(`PDF fetch failed: ${res.status}`);
    return res.blob();
  },

  login: (username, password) =>
    api._request('POST', `${BASE}/auth/login`, { username, password }, false),
  logout: () =>
    api._request('POST', `${BASE}/auth/logout`, undefined, false),
  health: () => api._request('GET', `${BASE}/api/health`),
  init: () => api._request('POST', `${BASE}/api/init`),
  dbStatus: () => api._request('GET', `${BASE}/api/db-status`),
  examples: () => api._request('GET', `${BASE}/api/examples`),
  settings: (body) => api._request(body ? 'POST' : 'GET', `${BASE}/api/settings`, body),
  query: (q, n, thinking) => api._request('POST', `${BASE}/api/query`,
    { query: q, num_results: n, thinking: !!thinking, history: state.turns.slice(-CHAT_TURNS_SENT) }),
};

// ── Session ───────────────────────────────────────────────────────────────
// Single sign-on lives at the gateway. When the pipeline session is missing or
// expired, a full-page reload hits the server's auth gate, which 302-redirects
// to the gateway login — so we never show a (now removed) local login screen.
let _redirecting = false;
function goToGatewayLogin() {
  if (_redirecting) return;
  _redirecting = true;
  window.location.reload();
}

// ── Screen switching ──────────────────────────────────────────────────────
function showLogin(errorMsg) {
  ui.appShell.classList.add('hidden');
  ui.loginScreen.classList.remove('hidden');
  ui.loginUsername.value = '';
  ui.loginPassword.value = '';
  if (errorMsg) showLoginError(errorMsg);
  else hideLoginError();
  ui.loginUsername.focus();
}

// Shown in the sidebar when no real account is signed in (login is skipped in
// this deployment). Placeholder name of a Chhattisgarh government user.
const PLACEHOLDER_USER = 'Lalit Kumar';

function showApp() {
  ui.loginScreen.classList.add('hidden');
  ui.appShell.classList.remove('hidden');
  const user = session.user();
  ui.userDisplay.textContent = (user && user.username) ? user.username : PLACEHOLDER_USER;
}

function showLoginError(msg) {
  ui.loginError.textContent = msg;
  ui.loginError.classList.remove('hidden');
}
function hideLoginError() {
  ui.loginError.classList.add('hidden');
}

// ── Login form ────────────────────────────────────────────────────────────
async function handleLogin(e) {
  e.preventDefault();
  hideLoginError();

  const username = ui.loginUsername.value.trim();
  const password = ui.loginPassword.value;

  if (!username || !password) {
    showLoginError('Please enter both username and password.');
    return;
  }

  ui.loginSubmit.disabled = true;
  ui.loginBtnText.classList.add('hidden');
  ui.loginBtnSpinner.classList.remove('hidden');

  try {
    const { ok, data } = await api.login(username, password);
    if (ok && data.success) {
      session.save(data.token, data.user);
      showApp();
      bootRagUI();
    } else {
      showLoginError(data.error || 'Invalid credentials.');
    }
  } catch (err) {
    if (err.message !== 'UNAUTHENTICATED') {
      showLoginError('Cannot reach server. Please try again.');
    }
  }

  ui.loginSubmit.disabled = false;
  ui.loginBtnText.classList.remove('hidden');
  ui.loginBtnSpinner.classList.add('hidden');
}

// ── Logout ────────────────────────────────────────────────────────────────
async function handleLogout() {
  closePdfPanel();
  closeDrawer();
  if (gadMode) {
    // GAD has its own session; clear it and return to the standalone GAD login.
    await fetch(`${BASE}/gad-auth/logout`, { method: 'POST', credentials: 'same-origin' }).catch(() => { });
    window.location.href = `${BASE}/login`;
    return;
  }
  // Other pipelines: clear the shared gateway session, then a reload lands on the
  // gateway login via the auth gate.
  await api.logout().catch(() => { });
  goToGatewayLogin();
}

// ── Toast ─────────────────────────────────────────────────────────────────
function toast(message, type = 'info', duration = 3500) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = message;
  ui.toastContainer.appendChild(el);
  setTimeout(() => el.remove(), duration);
}

// ── Readiness ─────────────────────────────────────────────────────────────
// The pipeline is auto-initialised by the launcher, so the end user never sees
// or triggers any of this. We just confirm it's ready and, if not, warm it up
// silently before enabling the question box.
async function ensureReady() {
  try {
    const { ok, data } = await api.health();
    if (ok && data.pipeline_initialized) {
      state.initialized = true;
      enableQueryBar();
      return;
    }
  } catch (_) { /* fall through to warm-up */ }

  // Not ready yet — warm it up in the background.
  disableQueryBar('Preparing…');
  try {
    const { ok, data } = await api.init();
    if (ok && data.success) {
      state.initialized = true;
      showChatEmptyDefault();
      enableQueryBar();
      return;
    }
    // Distinguish an EMPTY knowledge base (no documents indexed yet) from a
    // transient warm-up failure. An empty KB is a real, explainable state — not a
    // reason to leave the user staring at a disabled "Preparing…" box forever.
    const details = (data && data.details) || {};
    const err = ((data && data.error) || '').toLowerCase();
    if (details.collection_exists === false || err.includes('does not exist')) {
      showNoDocumentsState();
    }
  } catch (_) { /* handled below */ }

  // Keep polling: the moment a document is verified/indexed the collection
  // appears and the next pass enables the bar automatically.
  setTimeout(ensureReady, 6000);
}

// Shown when the pipeline is up but its vector store has no indexed documents.
// Replaces the misleading permanent "Preparing…" with an honest, actionable note.
function showNoDocumentsState() {
  disableQueryBar('No documents indexed yet');
  const empty = ui.chatEmpty;
  if (empty && !empty.dataset.noDocs) {
    empty.dataset.noDocs = '1';
    const p = empty.querySelector('p');
    if (p) {
      p.innerHTML = 'No documents have been indexed yet, so there is nothing to search. ' +
        'Documents join the knowledge base when the ingest run indexes them — ' +
        'this assistant activates automatically once they are in.';
    }
  }
}

function showChatEmptyDefault() {
  const empty = ui.chatEmpty;
  if (empty && empty.dataset.noDocs) {
    delete empty.dataset.noDocs;
    const p = empty.querySelector('p');
    if (p) {
      p.innerHTML = 'Ask a question about your department&rsquo;s documents and letters. ' +
        'Type your question below and press <strong>Send</strong>.';
    }
  }
}

// ── Query bar ─────────────────────────────────────────────────────────────
function enableQueryBar() {
  ui.btnSend.disabled = state.loading || state.historyBusy;
  ui.queryInput.placeholder = 'Type your question here…';
}
function disableQueryBar(msg = 'One moment…') {
  ui.btnSend.disabled = true;
  ui.queryInput.placeholder = msg;
}
function autoResize() { window.ChatUI.resizeInput(ui.queryInput); }

// ── Chat ──────────────────────────────────────────────────────────────────
function hideChatEmpty() { ui.chatEmpty.style.display = 'none'; }

const { escapeHtml, renderInline, renderMarkdown, splitThinking, historyText, renderAnswer } = window.ChatUI;

// Avatar images beside each message. The assistant is represented by the
// Government of Chhattisgarh emblem; the user by a neutral person mark.
const AVATARS = {
  user: 'img/avatar-user.svg',
  assistant: 'img/cg-emblem.png',
  thinking: 'img/cg-emblem.png',
};

// ── Reasoning trace (thinking mode) ───────────────────────────────────────
// Renders what the assistant actually did: which sub-questions it searched, what
// came back, what it judged missing, and how it ordered the documents by date. The
// point is that a precedence answer ("notice B supersedes notice A") can be checked
// rather than taken on faith, so each step shows its evidence — including the exact
// date string and the exact phrase that decided precedence.
const TRACE_LABEL = {
  plan: 'Plan', retrieve: 'Search', assess: 'Check',
  timeline: 'Dates', synthesize: 'Answer',
};

function buildTrace(steps, hops) {
  const wrap = document.createElement('div');
  wrap.className = 'trace';

  const btn = document.createElement('button');
  btn.className = 'trace-toggle';
  btn.type = 'button';
  btn.setAttribute('aria-expanded', 'false');
  const hopTxt = hops ? `${hops} search round${hops === 1 ? '' : 's'}` : `${steps.length} steps`;
  btn.innerHTML = `<span class="chev">▸</span> How this answer was reached (${escapeHtml(hopTxt)})`;

  const list = document.createElement('ul');
  list.className = 'trace-steps';
  list.hidden = true;
  steps.forEach(s => list.appendChild(buildTraceStep(s)));

  btn.addEventListener('click', () => {
    const open = btn.getAttribute('aria-expanded') === 'true';
    btn.setAttribute('aria-expanded', String(!open));
    list.hidden = open;
  });

  wrap.appendChild(btn);
  wrap.appendChild(list);
  return wrap;
}

function buildTraceStep(step) {
  const li = document.createElement('li');
  li.className = 'trace-step';
  li.dataset.kind = step.kind;

  const head = document.createElement('div');
  head.className = 'trace-head';
  head.innerHTML =
    `<span class="trace-kind">${escapeHtml(TRACE_LABEL[step.kind] || step.kind)}</span>` +
    `<span class="trace-title">${escapeHtml(step.title || '')}</span>` +
    `<span class="trace-time">${Number(step.elapsed_s || 0).toFixed(1)}s</span>`;
  li.appendChild(head);

  const detail = renderTraceDetail(step);
  if (detail) li.appendChild(detail);
  return li;
}

function renderTraceDetail(step) {
  const d = step.detail || {};
  const box = document.createElement('div');
  box.className = 'trace-detail';
  let html = '';

  if (step.kind === 'plan' && d.sub_questions?.length) {
    if (d.reasoning) html += `<div>${escapeHtml(d.reasoning)}</div>`;
    html += '<ul>' + d.sub_questions.map(q => `<li class="q">${escapeHtml(q)}</li>`).join('') + '</ul>';
  }

  if (step.kind === 'retrieve') {
    if (d.queries?.length) {
      html += '<ul>' + d.queries.map(q =>
        `<li class="q">${escapeHtml(q.question)} <span class="trace-time">${q.new ?? 0} new</span></li>`
      ).join('') + '</ul>';
    }
    (d.new_chunks || []).slice(0, 6).forEach(c => {
      html += `<div class="trace-chunk"><span class="src">${escapeHtml(c.source)}</span>` +
        (c.supporting ? ' <em>(adjacent)</em>' : '') +
        ` <span class="trace-time">${c.score}</span>` +
        `<span class="snip">${escapeHtml(c.snippet || '')}</span></div>`;
    });
  }

  if (step.kind === 'assess') {
    if (d.answered?.length) {
      html += '<ul>' + d.answered.map(a => `<li>✓ ${escapeHtml(a)}</li>`).join('') + '</ul>';
    }
    if (d.gaps?.length) {
      html += '<ul>' + d.gaps.map(g => `<li class="gap">Missing: ${escapeHtml(g)}</li>`).join('') + '</ul>';
    }
  }

  // The precedence evidence. Showing the matched date string and the matched
  // supersession phrase is the whole point — it is what makes the ruling checkable.
  if (step.kind === 'timeline' && d.documents?.length) {
    html += '<ul>' + d.documents.map(doc => {
      const when = doc.latest_date
        ? `${escapeHtml(doc.latest_date)}${doc.date_evidence ? ` <span class="trace-time">“${escapeHtml(doc.date_evidence)}”</span>` : ''}`
        : '<span class="trace-time">undated</span>';
      const rels = (doc.relations || [])
        .map(r => ` <span class="trace-rel">${escapeHtml(r.kind)}: “${escapeHtml(r.phrase)}”</span>`).join('');
      return `<li><span class="q">${escapeHtml(doc.source)}</span> — ${when}${rels}</li>`;
    }).join('') + '</ul>';
  }

  if (step.kind === 'synthesize' && d.sources?.length) {
    html += '<ul>' + d.sources.map(s => `<li>${escapeHtml(s)}</li>`).join('') + '</ul>';
  }

  if (d.error) html += `<div class="gap">${escapeHtml(d.error)}</div>`;
  if (!html) return null;
  box.innerHTML = html;
  return box;
}

function refreshReviewButtons() {
  for (const { index, button } of state.reviewButtons) {
    const saved = state.evaluationIndexes.has(index);
    const complete = state.messages[index]?.role === 'user'
      && state.messages[index + 1]?.role === 'assistant' && !!state.messages[index + 1].text?.trim();
    button.disabled = !state.isAdmin || saved || !complete || !state.chatId || state.dirty || state.loading
      || state.historyBusy || button.dataset.saving === '1';
    button.classList.toggle('is-saved', saved);
    button.setAttribute('aria-pressed', String(saved));
    const label = saved ? 'Saved for review in Evaluation' : 'Save question and answer for review';
    button.title = label;
    button.setAttribute('aria-label', label);
  }
}

function makeReviewButton(index) {
  const button = document.createElement('button');
  button.type = 'button'; button.className = 'review-button'; button.disabled = true;
  button.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 22V3m0 1c5-4 11 4 16 0v11c-5 4-11-4-16 0" /></svg>';
  button.title = 'Save question and answer for review';
  button.setAttribute('aria-label', button.title);
  button.addEventListener('click', () => saveForReview(index, button));
  state.reviewButtons.push({ index, button });
  return button;
}

async function saveForReview(index, button) {
  if (!state.isAdmin) return;
  const chatId = state.chatId;
  if (state.loading || state.historyBusy || !chatId || state.messages[index]?.role !== 'user'
    || state.messages[index + 1]?.role !== 'assistant' || !state.messages[index + 1].text?.trim()) return;
  button.dataset.saving = '1'; refreshReviewButtons();
  try {
    await saveCurrentChat();
    const { ok, data } = await api._request('POST', `${BASE}/api/evaluation`, { chatId, messageIndex: index });
    if (!ok || !data.success) throw new Error(data.error || 'Could not save this answer for review.');
    if (state.chatId === chatId) state.evaluationIndexes.add(index);
    toast('Question and answer saved in Evaluation.', 'info');
  } catch (err) { toast(err.message, 'error', 6000); }
  finally { delete button.dataset.saving; refreshReviewButtons(); }
}

async function loadEvaluationFlags(chatId) {
  if (!state.isAdmin) return;
  try {
    const { ok, data } = await api._request('GET', `${BASE}/api/evaluation?chatId=${encodeURIComponent(chatId)}`);
    if (ok && data.success && state.chatId === chatId) {
      state.evaluationIndexes = new Set(data.items.map(item => item.messageIndex));
    }
  } catch (_) { /* Saving remains idempotent if review status cannot be loaded. */ }
}

function appendMessage(role, text, meta = {}) {
  hideChatEmpty();
  const { msg, body } = window.ChatUI.message(role, text, { avatars: AVATARS, meta });
  // Reasoning trace (thinking mode only), above the sources control. Collapsed
  // by default — the answer is the product, this is the audit trail.
  if (role === 'assistant' && meta.trace?.length) {
    body.appendChild(buildTrace(meta.trace, meta.hops));
  }
  // Sources control lives inside the bubble, under a hairline, so it reads as
  // part of the assistant's message rather than a detached button beneath it.
  if (role === 'assistant' && meta.results?.length) {
    const foot = document.createElement('div');
    foot.className = 'msg-sources';
    const btn = document.createElement('button');
    btn.className = 'sources-btn';
    btn.innerHTML = `<span class="source-count">${meta.results.length}</span> View sources`;
    btn.addEventListener('click', () => openDrawer(meta.results));
    foot.appendChild(btn);
    body.appendChild(foot);
  }
  if (role === 'assistant' && state.isAdmin) {
    body.appendChild(buildQueryDebug(meta.debugId, BASE, meta.debugError));
  }

  if (role === 'user' && state.isAdmin) msg.prepend(makeReviewButton(meta.messageIndex ?? state.messages.length));
  ui.chatMessages.appendChild(msg);
  msg.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return msg;
}

// ── Streaming (SSE) ─────────────────────────────────────────────────────────
// Consume /api/query/stream. The backend pushes: status (coarse phase, normal
// mode), step (a completed thinking-mode step), token (one answer delta), done
// (final results/trace/sources), error. We parse the SSE frames off a fetch body
// stream — EventSource is GET-only, and we POST a JSON body.
async function streamQuery(text, deep, h) {
  const res = await fetch(`${BASE}/api/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: text, num_results: 8, thinking: deep,
                           history: state.turns.slice(-CHAT_TURNS_SENT) }),
    credentials: 'same-origin',
  });
  if (res.status === 401) throw new Error('UNAUTHENTICATED');
  if (!res.ok || !res.body) {
    let data = {}; try { data = await res.json(); } catch (_) { }
    h.onError?.(data.error || `HTTP ${res.status}`, data); return;
  }

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let sep;
    while ((sep = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, sep); buf = buf.slice(sep + 2);
      const line = frame.split('\n').find(l => l.startsWith('data:'));
      if (!line) continue;                       // ": stream open" keep-alive comment
      let ev; try { ev = JSON.parse(line.slice(5).trim()); } catch { continue; }
      if (ev.type === 'status') h.onStatus?.(ev.stage, ev);
      else if (ev.type === 'step') h.onStep?.(ev.step);
      else if (ev.type === 'token') h.onToken?.(ev.text);
      else if (ev.type === 'done') h.onDone?.(ev);
      else if (ev.type === 'error') h.onError?.(ev.error, ev);
    }
  }
}

// Present-tense label for the live status line — what the model is doing NOW.
// Steps arrive on completion, but each phase takes seconds, so surfacing the
// step as it lands reads as live progress (and beats a static "…").
function liveStepLabel(step) {
  const hop = step?.detail?.hop || (step?.title?.match(/Hop (\d+)/)?.[1]);
  switch (step.kind) {
    case 'plan':       return 'Planning the search…';
    case 'retrieve':   return `Searching documents${hop ? ` (round ${hop})` : ''}…`;
    case 'assess':     return /sufficient|ready/i.test(step.title) ? 'Writing the answer…' : 'Checking what\'s covered…';
    case 'timeline':   return 'Ordering documents by date…';
    case 'synthesize': return 'Writing the answer…';
    default:           return 'Working…';
  }
}

// An assistant bubble that we mutate as events arrive: a live status line on top,
// the streaming answer below it.
function appendStreamingMessage() {
  hideChatEmpty();
  const msg = document.createElement('div');
  msg.className = 'msg msg-assistant';
  const avatar = document.createElement('img');
  avatar.className = 'msg-avatar'; avatar.src = AVATARS.assistant; avatar.alt = 'Assistant';
  const content = document.createElement('div'); content.className = 'msg-content';
  const body = document.createElement('div'); body.className = 'msg-body assistant-body';
  body.innerHTML =
    '<div class="live-status"><span class="live-spinner"></span>' +
    '<span class="live-text">Understanding your question…</span></div>' +
    '<div class="stream-body"></div>';
  content.appendChild(body);
  msg.appendChild(avatar); msg.appendChild(content);
  ui.chatMessages.appendChild(msg);
  msg.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return {
    msg,
    statusEl: body.querySelector('.live-status'),
    textEl: body.querySelector('.live-text'),
    bodyEl: body.querySelector('.stream-body'),
    body,
  };
}

// ── Send query ────────────────────────────────────────────────────────────
async function sendQuery() {
  const text = ui.queryInput.value.trim();
  if (!text || state.loading || state.historyBusy) return;

  state.loading = true;
  disableQueryBar('Finding the answer…');
  const userMessage = appendMessage('user', text);
  state.messages.push({ role: 'user', text });
  state.dirty = true;
  // Saved now; if the server cannot take it, the question is still asked — the chat
  // stays in this browser and is saved on the next try.
  try { await saveCurrentChat(); }
  catch (_) { keepDraft(); }
  ui.queryInput.value = '';
  autoResize();

  const deep = !!state.thinkingMode;
  const el = appendStreamingMessage();
  let answer = '';
  let done = null;
  let savedReply = null;
  let raf = 0;

  // Re-render the accumulated markdown, throttled to one paint per frame so a
  // fast token stream does not thrash the DOM. A blinking caret marks liveness.
  const paint = () => {
    raf = 0;
    const thinkingOpen = !!el.bodyEl.querySelector('.model-thinking')?.open;
    const parts = splitThinking(answer, true);
    el.statusEl.classList.toggle('hidden', parts.hasThinking || !!parts.answer);
    el.bodyEl.innerHTML = renderAnswer(answer, { streaming: true, thinkingOpen });
    el.msg.scrollIntoView({ behavior: 'smooth', block: 'end' });
  };
  const schedulePaint = () => { if (!raf) raf = requestAnimationFrame(paint); };

  try {
    await streamQuery(text, deep, {
      onStatus: (stage, ev) => {
        el.textEl.textContent = stage !== 'retrieving' ? 'Writing the answer…'
          : (ev && ev.standalone ? `Searching for: ${ev.standalone}` : 'Searching documents…');
      },
      onStep: (step) => { el.textEl.textContent = liveStepLabel(step); },
      onToken: (t) => {
        answer += t;
        schedulePaint();
      },
      onDone: (data) => { done = data; },
      onError: (err, event) => {
        done = { ...event, success: false, error: err };
        if (!answer) { el.statusEl.classList.add('hidden'); el.bodyEl.innerHTML = renderAnswer(`Sorry, something went wrong: ${err}`); }
        toast(err || 'Query failed', 'error');
      },
    });

    if (raf) cancelAnimationFrame(raf);
    el.statusEl.classList.add('hidden');

    // Finalize: render the clean answer and attach trace + sources, matching the
    // non-streaming bubble exactly (buildTrace / sources control).
    const finalText = (done && done.answer) || answer || (done?.error ? `Sorry, something went wrong: ${done.error}` : '');
    if (finalText) {
      savedReply = { role: 'assistant', text: finalText, success: !!done?.success,
        results: done?.results || [], trace: done?.trace || [], hops: done?.hops || 0,
        standalone: done?.standalone || '', debugId: done?.debugId || '', debugError: done?.debugError || '' };
      el.bodyEl.innerHTML = renderAnswer(finalText);
      if (done && done.standalone) {
        // How a follow-up was understood — the user can see (and rephrase) it.
        const note = document.createElement('div'); note.className = 'search-note';
        note.textContent = `Searched for: ${done.standalone}`;
        el.bodyEl.prepend(note);
      }
      if (done && done.success) {
        state.turns.push({ question: text, answer: historyText(done.answer || finalText) });
        if (state.turns.length > 10) state.turns.shift();
      }
      state.lastResults = (done && done.results) || [];
      if (done && done.trace?.length) el.bodyEl.appendChild(buildTrace(done.trace, done.hops));
      if (state.lastResults.length) {
        const foot = document.createElement('div'); foot.className = 'msg-sources';
        const btn = document.createElement('button'); btn.className = 'sources-btn';
        btn.innerHTML = `<span class="source-count">${state.lastResults.length}</span> View sources`;
        const results = state.lastResults;
        btn.addEventListener('click', () => openDrawer(results));
        foot.appendChild(btn); el.bodyEl.appendChild(foot);
      }
    } else if (!el.bodyEl.textContent.trim()) {
      el.bodyEl.innerHTML = renderAnswer('Sorry, no answer was produced. Please try again.');
    }
    if (!savedReply) savedReply = { role: 'assistant', text: el.bodyEl.textContent.trim(), success: false };
    if (state.isAdmin) {
      el.bodyEl.appendChild(buildQueryDebug(done?.debugId, BASE, done?.debugError));
      savedReply.debugId = done?.debugId || '';
      savedReply.debugError = done?.debugError || '';
    }
  } catch (err) {
    if (raf) cancelAnimationFrame(raf);
    if (err.message !== 'UNAUTHENTICATED') {
      el.msg.remove();
      appendMessage('assistant', 'Sorry, the assistant could not be reached. Please try again in a moment.');
      savedReply = { role: 'assistant', text: 'Sorry, the assistant could not be reached. Please try again in a moment.', success: false };
      toast('Network error', 'error');
    } else {
      el.msg.remove();
    }
  }

  if (savedReply) {
    state.messages.push(savedReply);
    state.dirty = true;
    try { await saveCurrentChat(); }
    catch (_) {
      keepDraft();
      toast('This chat could not be saved to the server yet — it is kept in your browser and saved on the next try.', 'info', 6000);
    }
  }
  state.loading = false;
  enableQueryBar();
  renderChats();
  refreshReviewButtons();
}

// ── PDF panel ─────────────────────────────────────────────────────────────
async function openPdfPanel(fname) {
  // Show panel immediately with loading state
  ui.pdfTitle.textContent = fname;
  ui.pdfIframe.src = '';
  ui.pdfLoading.classList.remove('hidden');
  ui.pdfIframe.classList.add('hidden');
  ui.pdfPanel.classList.remove('hidden');
  ui.pdfOverlay.classList.remove('hidden');

  try {
    const pdfPath = `${BASE}/01_preprocessing/used_files/${encodeURIComponent(fname)}`;
    const blob = await api.fetchPdf(pdfPath);

    // Revoke previous blob URL to free memory
    if (state.pdfBlobUrl) {
      URL.revokeObjectURL(state.pdfBlobUrl);
    }

    state.pdfBlobUrl = URL.createObjectURL(blob);
    ui.pdfIframe.src = state.pdfBlobUrl;

    // Show iframe once loaded
    ui.pdfIframe.onload = () => {
      ui.pdfLoading.classList.add('hidden');
      ui.pdfIframe.classList.remove('hidden');
    };
  } catch (err) {
    if (err.message !== 'UNAUTHENTICATED') {
      ui.pdfLoading.classList.add('hidden');
      ui.pdfPanel.innerHTML +=
        `<div class="pdf-error">Could not load PDF: ${escapeHtml(err.message)}</div>`;
      toast('Could not load PDF', 'error');
    }
  }
}

function closePdfPanel() {
  ui.pdfPanel.classList.add('hidden');
  ui.pdfOverlay.classList.add('hidden');
  ui.pdfIframe.src = '';
  if (state.pdfBlobUrl) {
    URL.revokeObjectURL(state.pdfBlobUrl);
    state.pdfBlobUrl = null;
  }
}

// Chunks carry an internal header ("# Chunk 191\nDocument: …\nPages: …\n---\n").
// Strip it so the reader sees only the passage from the document.
function cleanExcerpt(text) {
  const marker = text.indexOf('---');
  let body = marker !== -1 ? text.slice(marker + 3) : text;
  body = body.replace(/^#\s*Chunk.*$/im, '').trim();
  return body.slice(0, 320) + (body.length > 320 ? '…' : '');
}

// ── Source drawer (right panel) ───────────────────────────────────────────
function openDrawer(results) {
  ui.drawerBody.innerHTML = '';

  // De-duplicate by document — the end user cares which documents the answer
  // came from, not how many internal chunks matched.
  const seen = new Set();
  results.forEach(r => {
    // Dedupe by the document itself (doc_dir), not its display name — several
    // distinct budget pages legitimately share the label "Main Budget".
    const key = r.source || r.actual_pdf || r.label || 'Document';
    if (seen.has(key)) return;
    seen.add(key);

    const label = r.label || r.actual_pdf || r.source || 'Document';
    const card = document.createElement('div');
    card.className = 'source-card';
    card.innerHTML = `
      <div class="source-card-header">
        <span class="source-filename">${escapeHtml(label)}</span>
      </div>
      <div class="source-excerpt">${escapeHtml(cleanExcerpt(r.excerpt || r.text || ''))}</div>
    `;

    // Most sources are scraped web pages with no local PDF — open the live source in
    // a new tab. Only fall back to the in-panel PDF viewer for uploaded documents
    // that actually have a used_files/<hash>.pdf on disk (no source_url).
    const pdfBtn = document.createElement('button');
    pdfBtn.className = 'pdf-open-btn';
    pdfBtn.textContent = r.source_url ? 'Open source' : 'Open document';
    pdfBtn.addEventListener('click', () => {
      if (r.source_url) window.open(r.source_url, '_blank', 'noopener');
      else openPdfPanel(r.actual_pdf || r.source);
    });
    card.appendChild(pdfBtn);

    ui.drawerBody.appendChild(card);
  });

  ui.drawerOverlay.classList.remove('hidden');
  ui.sourceDrawer.classList.remove('hidden');
}

function closeDrawer() {
  ui.drawerOverlay.classList.add('hidden');
  ui.sourceDrawer.classList.add('hidden');
}

// ── Persistent conversations ────────────────────────────────────────
// A chat is saved on the server after every question and answer. Two things kept
// that from sticking: the office firewall in front of the bot answers HTTP 403 to a
// request body holding tag-like text — an OCR table ("<td>") in a source, a
// "<think>" block — so every save of such a chat failed; and a failed save also
// refused the next question. Now the save travels base64-encoded (no tag-like text
// on the wire; chat-history.js decodes it), a failed save never stops a question,
// and the chat is kept in this browser until the server has it — a refresh restores
// it and saves it again.
function toBase64(text) {
  const bytes = new TextEncoder().encode(text);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

async function historyRequest(method, suffix = '', body) {
  const packed = body === undefined ? undefined : { b64: toBase64(JSON.stringify(body)) };
  const { ok, data } = await api._request(method, `${BASE}/api/history${suffix}`, packed);
  if (!ok || !data.success) throw new Error(data.error || 'Chat history is unavailable. Please try again.');
  return data;
}

function draftKey() { return state.historyKey ? `${state.historyKey}:unsaved` : null; }
function keepDraft() {   // the chat as it stands, until the server confirms it
  try {
    const key = draftKey();
    if (key && state.messages.length) {
      localStorage.setItem(key, JSON.stringify({ chatId: state.chatId, messages: state.messages, at: Date.now() }));
    }
  } catch (_) { /* full or blocked storage: the server save still runs */ }
}
function dropDraft() { try { const key = draftKey(); if (key) localStorage.removeItem(key); } catch (_) { } }
function readDraft() {
  try { const key = draftKey(); return key ? JSON.parse(localStorage.getItem(key) || 'null') : null; }
  catch (_) { return null; }
}

// Saves that failed are tried again while the page is open.
setInterval(() => {
  if (state.dirty && !state.loading && !state.historyBusy) saveCurrentChat().catch(() => { });
}, 20000);

function rememberChat() {
  try {
    if (!state.historyKey) return;
    if (state.chatId) localStorage.setItem(state.historyKey, state.chatId);
    else localStorage.removeItem(state.historyKey);
  } catch (_) { /* The conversations themselves are saved on the server. */ }
}

function updateChatSummary(chat) {
  const { messages, ...summary } = chat;
  state.chats = state.chats.filter(c => c.id !== summary.id).concat(summary);
  renderChats();
}

async function saveCurrentChat() {
  if (!state.dirty || !state.messages.length) return;
  keepDraft();                         // a refresh keeps the chat even if this save fails
  const saving = state.messages.length;
  const data = await historyRequest(state.chatId ? 'PUT' : 'POST',
    state.chatId ? `/${state.chatId}` : '', { messages: state.messages });
  state.chatId = data.chat.id;
  state.dirty = state.messages.length !== saving;   // a message added meanwhile still needs saving
  if (!state.dirty) dropDraft();
  else keepDraft();
  rememberChat();
  updateChatSummary(data.chat);
}

// Put a conversation on screen (saved or restored from this browser).
function showMessages(messages) {
  state.messages = messages;
  let question = null;
  for (const [messageIndex, message] of state.messages.entries()) {
    appendMessage(message.role, message.text, { ...message, messageIndex });
    if (message.role === 'user') question = message.text;
    else if (question && message.success) {
      state.turns.push({ question, answer: historyText(message.text) });
      question = null;
    }
  }
  state.turns = state.turns.slice(-10);
}

function clearCurrentChat() {
  state.chatId = null;
  state.messages = [];
  state.turns = [];
  state.lastResults = [];
  state.dirty = false;
  state.evaluationIndexes = new Set();
  state.reviewButtons = [];
  ui.chatMessages.innerHTML = '';
  ui.queryInput.value = '';
  autoResize();
  closeDrawer();
  closePdfPanel();
  showChatEmptyDefault();
  ui.chatEmpty.style.display = '';
  rememberChat();
  renderChats();
}

async function openChat(id) {
  if (state.loading || state.historyBusy || id === state.chatId) return;
  state.historyBusy = true;
  disableQueryBar('Opening chat…');
  try {
    await saveCurrentChat();
    const { chat } = await historyRequest('GET', `/${id}`);
    clearCurrentChat();
    state.chatId = chat.id;
    showMessages(chat.messages);
    await loadEvaluationFlags(chat.id);
    rememberChat();
    renderChats();
  } catch (err) { toast(err.message, 'error', 6000); }
  finally {
    state.historyBusy = false;
    if (state.initialized) enableQueryBar();
    renderChats();
    refreshReviewButtons();
  }
}

async function chatAction(chat, action) {
  if (state.loading || state.historyBusy) return;
  if (action === 'delete' && !confirm(`Delete “${chat.title}”? This removes the saved conversation.`)) return;
  state.historyBusy = true;
  disableQueryBar('Updating chats…');
  try {
    await saveCurrentChat();
    if (action === 'delete') {
      await historyRequest('DELETE', `/${chat.id}`);
      state.chats = state.chats.filter(c => c.id !== chat.id);
      if (state.chatId === chat.id) clearCurrentChat();
    } else {
      const { chat: updated } = await historyRequest('PATCH', `/${chat.id}`, { pinned: !chat.pinned });
      updateChatSummary(updated);
    }
  } catch (err) { toast(err.message, 'error', 6000); }
  finally {
    state.historyBusy = false;
    if (state.initialized) enableQueryBar();
    renderChats();
  }
}

function renderChats() {
  window.ChatUI.renderChatList(ui.chatList, state.chats, {
    activeId: state.chatId, disabled: state.loading || state.historyBusy,
    onOpen: openChat, onAction: chatAction,
  });
}

// A chat this browser kept because its last save did not reach the server: shown
// again, and saved (into its chat when the server has it, else as a new chat).
async function restoreDraft(draft) {
  const known = draft.chatId && state.chats.some(c => c.id === draft.chatId);
  if (known) {
    await openChat(draft.chatId);
    if (state.chatId !== draft.chatId || draft.messages.length <= state.messages.length) {
      dropDraft();                     // the server already has all of it
      return;
    }
    clearCurrentChat();
  } else {
    clearCurrentChat();
  }
  state.chatId = known ? draft.chatId : null;
  showMessages(draft.messages);
  hideChatEmpty();
  state.dirty = true;
  try { await saveCurrentChat(); }
  catch (_) { toast('This chat is kept in your browser and will be saved when the server can be reached.', 'info', 6000); }
  renderChats();
}

async function loadChats(restore = true) {
  try {
    const data = await historyRequest('GET');
    state.chats = data.chats;
    state.historyKey = `chips_rag_active_chat:${data.pipeline}:${data.owner}`;
    renderChats();
    let active;
    try { active = localStorage.getItem(state.historyKey); } catch (_) { }
    const draft = restore ? readDraft() : null;
    if (draft && Array.isArray(draft.messages) && draft.messages.length) {
      await restoreDraft(draft);
    } else if (restore && active && state.chats.some(c => c.id === active)) {
      await openChat(active);
    }
  } catch (err) {
    ui.chatList.replaceChildren();
    const retry = document.createElement('button'); retry.type = 'button'; retry.className = 'history-retry';
    retry.textContent = 'Could not load chats. Retry'; retry.addEventListener('click', () => loadChats());
    const li = document.createElement('li'); li.appendChild(retry); ui.chatList.appendChild(li);
    toast(err.message, 'error', 6000);
  }
}


// ── RAG UI boot ───────────────────────────────────────────────────────────
async function bootRagUI() {
  disableQueryBar('Preparing…');
  await applyGadRoleGating();
  await loadChats();
  await ensureReady();

  ui.btnSend.addEventListener('click', sendQuery);

  // New chat: forget this chat's turns (the context follow-ups are read in) and
  // clear the screen.
  if (ui.btnNewChat) {
    ui.btnNewChat.addEventListener('click', async () => {
      if (state.loading || state.historyBusy) return;
      state.historyBusy = true;
      try {
        await saveCurrentChat();
        clearCurrentChat();
        ui.queryInput.focus();
      } catch (err) { toast(err.message, 'error', 6000); }
      finally { state.historyBusy = false; if (state.initialized) enableQueryBar(); renderChats(); }
    });
  }

  // Thinking mode toggle. The placeholder changes too, so it is obvious which
  // path the next question will take without reading the button.
  if (ui.btnThinking) {
    ui.btnThinking.addEventListener('click', () => {
      state.thinkingMode = !state.thinkingMode;
      ui.btnThinking.setAttribute('aria-pressed', String(state.thinkingMode));
      ui.queryInput.placeholder = state.thinkingMode
        ? 'Ask something that needs digging — it will search several times…'
        : 'Type your question here…';
      toast(state.thinkingMode
        ? 'Thinking mode on — slower, shows its reasoning'
        : 'Thinking mode off', 'info');
    });
    // Reflect the default in the button + placeholder on load.
    ui.btnThinking.setAttribute('aria-pressed', String(state.thinkingMode));
    ui.queryInput.placeholder = state.thinkingMode
      ? 'Ask something that needs digging — it will search several times…'
      : 'Type your question here…';
  }

  ui.queryInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!ui.btnSend.disabled) sendQuery();
    }
  });
  ui.queryInput.addEventListener('input', autoResize);

  // Drawer close
  ui.drawerClose.addEventListener('click', closeDrawer);
  ui.drawerOverlay.addEventListener('click', closeDrawer);

  // PDF panel close
  ui.pdfClose.addEventListener('click', closePdfPanel);
  ui.pdfOverlay.addEventListener('click', closePdfPanel);

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      closeDrawer(); closePdfPanel();
      ui.chatList.querySelectorAll('details[open]').forEach(d => { d.open = false; });
    }
  });
  document.addEventListener('click', e => {
    ui.chatList.querySelectorAll('details[open]').forEach(d => { if (!d.contains(e.target)) d.open = false; });
  });
}

// ── GAD role gating ─────────────────────────────────────────────────────────
// GAD (fg2) has its own standalone login with admin/user roles. On other
// pipelines the /gad-auth/session endpoint does not exist, so this is a no-op and
// the UI behaves as before. When it IS the GAD pipeline:
//   · show the real signed-in username
//   · role "user"  → hide the Documents workspace (chatbot only)
//   · role "admin" → keep Documents and reveal the Accounts and Batch test links
let gadMode = false;
async function applyGadRoleGating() {
  let data;
  try {
    const res = await fetch(`${BASE}/gad-auth/session`, { credentials: 'same-origin' });
    if (!res.ok) return;              // chips/fg: endpoint absent → leave UI as-is
    data = await res.json();
  } catch (_) { return; }
  if (!data || !data.authenticated || !data.user) return;

  gadMode = true;
  const { username, role } = data.user;
  state.isAdmin = role === 'admin';
  if (ui.userDisplay) ui.userDisplay.textContent = username;

  const docs = document.getElementById('nav-documents');
  const accts = document.getElementById('nav-accounts');
  const batch = document.getElementById('nav-batch');
  const gpu = document.getElementById('nav-gpu');
  const evaluation = document.getElementById('nav-evaluation');
  const evalContainer = document.getElementById('nav-eval-container');
  if (evaluation) evaluation.classList.toggle('hidden', !state.isAdmin);
  if (evalContainer) evalContainer.classList.toggle('hidden', !state.isAdmin);
  if (role === 'admin') {
    if (docs) docs.classList.remove('hidden');
    if (accts) accts.classList.remove('hidden');
    if (batch) batch.classList.remove('hidden');
    if (gpu) gpu.classList.remove('hidden');
    showDepartmentsLink();
  } else {
    if (docs) docs.classList.add('hidden');       // users: chatbot only
    if (accts) accts.classList.add('hidden');
    if (batch) batch.classList.add('hidden');
    if (gpu) gpu.classList.add('hidden');
  }
}

// Departments are managed on one bot only (the CHiPS bot by default).
async function showDepartmentsLink() {
  const link = document.getElementById('nav-departments');
  if (!link) return;
  try {
    const res = await fetch(`${BASE}/api/platform/status`, { credentials: 'same-origin' });
    const data = res.ok ? await res.json() : null;
    link.classList.toggle('hidden', !(data && data.enabled));
  } catch (_) { /* leave it hidden */ }
}

// ── Entry point ───────────────────────────────────────────────────────────
function boot() {
  ui.btnLogout.addEventListener('click', handleLogout);
  // Skip login UI for FG deployment (no auth). Boot the app immediately.
  // Keep logout handler available if someone wants to clear state.
  try { ui.loginForm.removeEventListener && ui.loginForm.removeEventListener('submit', handleLogin); } catch (_) { }
  try { ui.loginPassword.removeEventListener && ui.loginPassword.removeEventListener('keydown', handleLogin); } catch (_) { }

  showApp();
  bootRagUI();
}

initTheme();
boot();
