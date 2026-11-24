'use strict';

// Batch test panel (admins): many questions answered by the bot one after another,
// on the server (core/05_webui/batch_runs.py), so a run goes on when this page is
// closed. Every request rides the same-origin session cookie; the server lets only
// admins through.

const $ = (id) => document.getElementById(id);
const toast = (msg) => {
  const t = $('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove('show'), 3000);
};

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function fmtDate(iso) {
  if (!iso) return '';
  try { return new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }); }
  catch { return iso; }
}

// '/pipeline/<name>' behind the reverse proxy, '' when hit directly.
const BASE = (typeof window !== 'undefined' && window.__APP_BASE__) || '';

async function api(method, path, body) {
  const opts = { method, headers: {}, credentials: 'same-origin' };
  if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  const res = await fetch(BASE + path, opts);
  if (res.status === 401) { window.location.href = `${BASE}/login`; throw new Error('unauth'); }
  if (res.status === 403) { window.location.href = `${BASE}/`; throw new Error('forbidden'); }
  return res.json().catch(() => ({ success: false, error: `HTTP ${res.status}` }));
}

const ACTIVE = new Set(['queued', 'running']);
const SECONDS_EACH = { normal: 20, thinking: 60 };   // rough, for the estimate only

// ── The question list ─────────────────────────────────────────────────────────
// A loaded .csv may carry expected answers; they are used while the text box still
// holds exactly the questions that came from the file.
let loaded = null;   // { text, items: [{question, expected}] }

function linesOf(text) {
  return text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
}

function currentItems() {
  const text = $('questions').value;
  if (loaded && loaded.text === text) return loaded.items;
  return linesOf(text).map((question) => ({ question }));
}

function updateCount() {
  const n = currentItems().length;
  const mins = Math.max(1, Math.round((n * SECONDS_EACH[$('run-mode').value]) / 60));
  const withExp = currentItems().filter((i) => i.expected).length;
  $('q-count').textContent = n
    ? `${n} question${n === 1 ? '' : 's'} · about ${mins} min` + (withExp ? ` · ${withExp} with an expected answer` : '')
    : '0 questions';
}

// RFC 4180 CSV: quoted cells may hold commas, quotes ("") and line breaks.
function parseCsv(text) {
  const rows = []; let row = []; let cell = ''; let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(cell); cell = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(cell); rows.push(row); row = []; cell = '';
    } else cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}

function itemsFromFile(name, text) {
  text = text.replace(/^﻿/, '');
  if (!/\.csv$/i.test(name)) return linesOf(text).map((question) => ({ question }));
  const rows = parseCsv(text).filter((r) => (r[0] || '').trim());
  if (rows.length && /^(question|questions|prompt|query|प्रश्न|सवाल)s?$/i.test((rows[0][0] || '').trim())) rows.shift();
  return rows.map((r) => ({
    question: r[0].replace(/\s+/g, ' ').trim(),
    expected: (r[1] || '').trim() || undefined,
  }));
}

$('questions').addEventListener('input', updateCount);
$('run-mode').addEventListener('change', updateCount);
$('load-file').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', async () => {
  const f = $('file').files[0];
  $('file').value = '';
  if (!f) return;
  const items = itemsFromFile(f.name, await f.text());
  if (!items.length) { toast('No questions found in that file.'); return; }
  const text = items.map((i) => i.question).join('\n');
  $('questions').value = text;
  loaded = { text, items };
  if (!$('run-name').value.trim()) $('run-name').value = f.name.replace(/\.(csv|txt)$/i, '');
  updateCount();
  toast(`Loaded ${items.length} questions from ${f.name}.`);
});

$('start').addEventListener('click', async () => {
  const items = currentItems();
  if (!items.length) { toast('Write or load at least one question.'); return; }
  const btn = $('start'); btn.disabled = true;
  let res;
  try {
    res = await api('POST', '/api/batch/runs',
      { name: $('run-name').value, mode: $('run-mode').value, questions: items });
  } catch (_) { btn.disabled = false; return; }
  btn.disabled = false;
  if (!res.success) { toast(res.error || 'Could not start the run.'); return; }
  toast(`Started: ${res.run.total} questions.`);
  $('questions').value = ''; $('run-name').value = ''; loaded = null; updateCount();
  await loadRuns();
  openRun(res.run.id);
});

// ── Runs ─────────────────────────────────────────────────────────────────────
let runs = [];

function progressHtml(r) {
  const pct = r.total ? Math.round((100 * (r.answered + r.errors)) / r.total) : 0;
  return `<div class="bar"><i class="${r.errors && !r.answered ? 'err' : ''}" style="width:${pct}%"></i></div>
    <div class="hint" style="margin-top:4px">${r.answered}/${r.total} answered${r.errors ? ` · <span style="color:var(--red)">${r.errors} failed</span>` : ''}</div>`;
}

function downloadUrl(id, format) {
  return `${BASE}/api/batch/runs/${encodeURIComponent(id)}/download?format=${format}`;
}

async function loadRuns() {
  let data;
  try { data = await api('GET', '/api/batch/runs'); } catch (_) { return; }
  runs = data.runs || [];
  const rows = $('run-rows');
  if (!runs.length) { rows.innerHTML = '<tr><td colspan="6" class="note">No test runs yet.</td></tr>'; return; }
  rows.innerHTML = runs.map((r) => {
    const active = ACTIVE.has(r.status);
    const unanswered = r.total - r.answered;
    return `<tr class="${r.id === openId ? 'sel' : ''}">
      <td><b>${esc(r.name)}</b>${r.started_by ? `<div class="hint">by ${esc(r.started_by)}</div>` : ''}</td>
      <td>${esc(fmtDate(r.created_at))}</td>
      <td>${esc(r.mode)}</td>
      <td style="min-width:150px">${progressHtml(r)}</td>
      <td><span class="chip ${esc(r.status)}">${esc(r.status)}</span></td>
      <td><div class="row-actions">
        <button class="mini" data-act="view" data-id="${esc(r.id)}">View</button>
        <a class="mini" href="${downloadUrl(r.id, 'csv')}">CSV</a>
        ${active ? `<button class="mini" data-act="stop" data-id="${esc(r.id)}">Stop</button>` : ''}
        ${!active && unanswered ? `<button class="mini" data-act="resume" data-id="${esc(r.id)}">Answer the rest (${unanswered})</button>` : ''}
        ${!active ? `<button class="mini danger" data-act="delete" data-id="${esc(r.id)}">Delete</button>` : ''}
      </div></td>
    </tr>`;
  }).join('');
}

$('refresh').addEventListener('click', () => { loadRuns(); if (openId) openRun(openId); });

$('run-rows').addEventListener('click', async (e) => {
  const btn = e.target.closest('button[data-act]');
  if (!btn) return;
  const { act, id } = btn.dataset;
  const r = runs.find((x) => x.id === id);
  if (act === 'view') { openRun(id); return; }
  if (act === 'stop') {
    const res = await api('POST', `/api/batch/runs/${encodeURIComponent(id)}/stop`);
    toast(res.success ? 'Stopping after the current question.' : (res.error || 'Failed.'));
  } else if (act === 'resume') {
    const res = await api('POST', `/api/batch/runs/${encodeURIComponent(id)}/resume`);
    toast(res.success ? 'Answering the rest.' : (res.error || 'Failed.'));
    if (res.success) openRun(id);
  } else if (act === 'delete') {
    if (!confirm(`Delete the test run "${r ? r.name : id}" and its answers? Download it first if you need it.`)) return;
    const res = await api('DELETE', `/api/batch/runs/${encodeURIComponent(id)}`);
    toast(res.success ? 'Deleted.' : (res.error || 'Failed.'));
    if (res.success && openId === id) closeRun();
  }
  loadRuns();
});

// ── One run's answers ────────────────────────────────────────────────────────
let openId = null;
let filter = 'all';
let pollTimer = null;
const shown = new Map();   // item n → what it showed, so a refresh redraws only what changed

function closeRun() {
  openId = null; clearTimeout(pollTimer);
  $('detail').classList.add('hidden');
}

// The answer as plain text, with the model's **bold** kept.
function answerHtml(text) {
  return esc(text).replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>');
}

function itemHtml(it, answering) {
  const status = answering ? 'running' : it.status;
  const label = answering ? 'answering…' : it.status === 'pending' ? 'waiting' : it.status === 'error' ? 'failed' : 'answered';
  let html = `<div class="q"><span class="n">${it.n}.</span><b>${esc(it.question)}</b>
    <span class="chip ${status}">${label}</span>${it.seconds != null ? `<span class="secs">${it.seconds} s</span>` : ''}</div>`;
  if (it.expected) html += `<div class="expected"><span>Expected</span> ${esc(it.expected)}</div>`;
  if (it.status === 'done') {
    html += `<div class="answer">${answerHtml(it.answer || '(empty answer)')}</div>`;
    if ((it.sources || []).length) {
      html += '<div class="sources"><span class="label">Source links</span>' + it.sources.map((s) =>
        s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.label)}</a>` : `<span>${esc(s.label)}</span>`).join('') + '</div>';
    }
    if ((it.retrieved || []).length) {
      html += `<details class="retrieved"><summary>Documents the search found (${it.retrieved.length})</summary><ol>` +
        it.retrieved.map((l) => `<li>${esc(l)}</li>`).join('') + '</ol></details>';
    }
  } else if (it.status === 'error') {
    html += `<div class="err-text">${esc(it.error || 'failed')}</div>`;
  }
  return html;
}

function renderRun(run) {
  $('detail').classList.remove('hidden');
  $('d-title').textContent = run.name;
  $('d-csv').href = downloadUrl(run.id, 'csv');
  $('d-json').href = downloadUrl(run.id, 'json');
  const pct = run.total ? Math.round((100 * (run.answered + run.errors)) / run.total) : 0;
  $('d-bar').style.width = `${pct}%`;
  $('d-meta').innerHTML = [
    `<span class="chip ${esc(run.status)}">${esc(run.status)}</span>`,
    `${run.answered}/${run.total} answered`,
    run.errors ? `<span style="color:var(--red)">${run.errors} failed</span>` : '',
    `mode: ${esc(run.mode)}`,
    (run.models || []).length ? `answered by: ${esc(run.models.join(', '))}` : '',
    run.avg_seconds != null ? `${run.avg_seconds} s per answer` : '',
    run.started_by ? `by ${esc(run.started_by)}` : '',
    `created ${esc(fmtDate(run.created_at))}`,
    run.finished_at ? `finished ${esc(fmtDate(run.finished_at))}` : '',
  ].filter(Boolean).map((s) => `<span>${s}</span>`).join('');
  $('d-reason').textContent = run.stop_reason ? `Stopped: ${run.stop_reason}` : '';
  $('d-reason').classList.toggle('hidden', !run.stop_reason || run.status === 'done');

  const box = $('d-items');
  // the worker goes through the unanswered questions in order (failed ones too, on a retry)
  const answeringN = run.status === 'running' ? (run.items.find((i) => i.status !== 'done') || {}).n : null;
  const visible = run.items.filter((i) => filter === 'all' || i.status === 'error');
  if (box.dataset.run !== run.id || box.dataset.filter !== filter) {
    box.innerHTML = ''; shown.clear();
    box.dataset.run = run.id; box.dataset.filter = filter;
  }
  if (!visible.length) {
    box.innerHTML = `<div class="note">${filter === 'error' ? 'No failed questions.' : 'No questions.'}</div>`;
    shown.clear();
    return;
  }
  if (box.querySelector('.note')) box.innerHTML = '';
  const wanted = new Set(visible.map((i) => i.n));
  for (const [n, el] of shown) if (!wanted.has(n)) { el.remove(); shown.delete(n); }
  let prev = null;
  for (const it of visible) {
    const key = `${it.status}|${it.seconds}|${it.n === answeringN}`;
    let el = shown.get(it.n);
    if (!el || el.dataset.key !== key) {
      const fresh = document.createElement('div');
      fresh.className = 'item';
      fresh.dataset.key = key;
      fresh.innerHTML = itemHtml(it, it.n === answeringN);
      if (el) el.replaceWith(fresh);
      el = fresh;
      shown.set(it.n, el);
    }
    if (prev ? prev.nextSibling !== el : box.firstChild !== el) {
      box.insertBefore(el, prev ? prev.nextSibling : box.firstChild);
    }
    prev = el;
  }
}

async function openRun(id) {
  const switching = id !== openId;
  openId = id;
  clearTimeout(pollTimer);
  let data;
  try { data = await api('GET', `/api/batch/runs/${encodeURIComponent(id)}`); } catch (_) { return; }
  if (openId !== id) return;
  if (!data.success) { toast(data.error || 'Could not load the run.'); closeRun(); return; }
  renderRun(data.run);
  if (switching) {
    document.querySelectorAll('#run-rows tr').forEach((tr) =>
      tr.classList.toggle('sel', !!tr.querySelector(`[data-id="${CSS.escape(id)}"]`)));
    $('detail').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  if (ACTIVE.has(data.run.status)) {
    pollTimer = setTimeout(() => { openRun(id); loadRuns(); }, 4000);
  } else if (!switching) {
    loadRuns();   // it just finished: the table's row too
  }
}

document.querySelectorAll('#detail [data-filter]').forEach((b) => b.addEventListener('click', () => {
  filter = b.dataset.filter;
  document.querySelectorAll('#detail [data-filter]').forEach((x) => x.classList.toggle('on', x === b));
  if (openId) openRun(openId);
}));

// ── Boot ──────────────────────────────────────────────────────────────────────
(async function boot() {
  updateCount();
  await loadRuns();
  const running = runs.find((r) => ACTIVE.has(r.status));
  if (running) openRun(running.id);
})();
