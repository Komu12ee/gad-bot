'use strict';

// Document explorer (admins): browse every document the bot has, filter it, and see
// the source next to the text that was read from it. View only — nothing here
// verifies, discards or edits a document.

// Same-origin: this page's Node server proxies /api/* to the pipeline's Flask backend.
const BASE = (typeof window !== 'undefined' && window.__APP_BASE__) || '';

// Per-viewer conveniences (list collapsed, filters); the page works without them.
const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch (_) { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch (_) { /* private window */ } },
};

// ── Collapsible document list ────────────────────────────────────────────────
const toggleListBtn = document.getElementById('toggle-list');
if (store.get('doc_list_collapsed') === '1') document.body.classList.add('list-collapsed');
toggleListBtn.addEventListener('click', () => {
  const collapsed = document.body.classList.toggle('list-collapsed');
  store.set('doc_list_collapsed', collapsed ? '1' : '0');
});

// ── State ────────────────────────────────────────────────────────────────────
let docs = [];
let docsSig = '';
let selectedId = null;
let lastDetailSig = '';   // state of the currently-rendered detail bar (avoids needless re-render/PDF reload)
let searchTerm = '';
let visibleFlat = [];     // docs in rendered order — for keyboard nav
let procById = {};        // id -> processing summary (how the text was produced)
const PAGE = 300;         // rows drawn at a time; "Show more" draws the next lot
let shownLimit = PAGE;
let readerMode = store.get('doc_markdown_reader') === '1';

const FILTER_KEYS = ['section', 'kind', 'text', 'index', 'state'];
const filters = { section: 'all', kind: 'all', text: 'all', index: 'all', state: 'all', sort: 'new' };
try { Object.assign(filters, JSON.parse(store.get('doc_filters') || '{}')); } catch (_) { /* defaults */ }

const $ = (id) => document.getElementById(id);
const toast = (msg) => {
  const t = $('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove('show'), 2500);
};

// ── API (read only) ──────────────────────────────────────────────────────────
const api = {
  list:   () => fetch(`${BASE}/api/documents`).then(r => r.json()),
  md:     (id) => fetch(`${BASE}/api/documents/${id}/md`).then(r => r.json()),
  processing: (id) => fetch(`${BASE}/api/documents/${id}/processing`).then(r => r.json()),
  spellcheck: (id) => fetch(`${BASE}/api/documents/${id}/spellcheck`).then(r => r.json()),
};

// ── What each document is (for filters, badges and the detail bar) ───────────
function stateOf(d) {
  if (d.status === 'error') return 'error';
  if (d.status === 'processing') return 'processing';
  return 'ready';
}
// A chip only where something is wrong or unfinished; a normal document has none.
const STATE_CHIP = { error: { cls: 'error', label: 'Error' }, processing: { cls: 'processing', label: 'Being read…' } };

function kindOf(d) {
  if (d.source_type === 'web' || d.raw_ext === 'html' || d.raw_ext === 'htm') return 'web';
  return 'pdf';
}

// Derive website folder hierarchy from source_url (e.g. gad.cg.gov.in/circulars/2024/order.pdf -> "circulars/2024")
function sectionOf(d) {
  if (!d.source_url) return 'unlinked';
  try {
    const u = new URL(d.source_url);
    const path = u.pathname.replace(/^\/+|\/+$/g, '');
    if (!path) return 'home';
    const parts = path.split('/');
    const folderParts = parts.length > 1 ? parts.slice(0, -1) : [parts[0].replace(/\.[^.]+$/, '')];
    return folderParts.slice(0, 2).join('/') || 'home';
  } catch (_) {
    return 'other';
  }
}

// How the text was read, as a filter key: GPU OCR per model, Sarvam, the older
// local OCR, or text taken straight from the PDF / web page.
function textKeyOf(d) {
  if (d.ocr_method === 'extract') return 'extract';
  if (d.ocr_method === 'ocr_sarvam' || d.ocr_method === 'auto') return 'sarvam';
  if (d.ocr_method === 'ocr') {
    const m = /OCR — (.+?) on GPU/.exec(d.ocr_label || '');
    return m ? `gpu:${m[1]}` : 'local';
  }
  return d.status === 'processing' ? 'pending' : 'unknown';
}
function shortModel(model) {
  return model.replace(/^.*\//, '').replace(/-Instruct/i, '').replace(/-FP8$/i, '');
}
function textLabel(key) {
  if (key.startsWith('gpu:')) return `GPU OCR · ${shortModel(key.slice(4))}`;
  return { extract: 'Text extraction (no OCR)', sarvam: 'Sarvam OCR', local: 'Docling / EasyOCR',
           pending: 'Not read yet', unknown: 'Unknown' }[key] || key;
}

// Short badge for the list row; the full method is on hover and in the detail bar.
function methodBadge(d) {
  const key = textKeyOf(d);
  if (key === 'pending' || key === 'unknown') return '';
  const cls = key === 'extract' ? 'm-extract' : key === 'sarvam' ? 'm-sarvam' : 'm-ocr';
  const label = key.startsWith('gpu:') ? shortModel(key.slice(4)) : textLabel(key);
  return `<span class="badge method ${cls}" title="${escapeHtml(d.ocr_label || label)}">${escapeHtml(label)}</span>`;
}

function matches(d, skip) {
  if (skip !== 'section' && filters.section !== 'all' && sectionOf(d) !== filters.section) return false;
  if (skip !== 'kind' && filters.kind !== 'all' && kindOf(d) !== filters.kind) return false;
  if (skip !== 'text' && filters.text !== 'all' && textKeyOf(d) !== filters.text) return false;
  if (skip !== 'index' && filters.index !== 'all' && (!!d.indexed) !== (filters.index === 'yes')) return false;
  if (skip !== 'state' && filters.state !== 'all' && stateOf(d) !== filters.state) return false;
  if (searchTerm) {
    const hay = `${d.original_name || ''} ${d.source_url || ''} ${d.doc_dir || ''} ${d.id || ''}`.toLowerCase();
    if (!hay.includes(searchTerm)) return false;
  }
  return true;
}

const SORTS = {
  new: (a, b) => String(b.uploaded_at || '').localeCompare(String(a.uploaded_at || '')),
  old: (a, b) => String(a.uploaded_at || '').localeCompare(String(b.uploaded_at || '')),
  name: (a, b) => String(a.original_name || '').localeCompare(String(b.original_name || ''), 'hi'),
  passages: (a, b) => (b.chunk_count || 0) - (a.chunk_count || 0),
};

// ── Filter dropdowns: options with counts (given the other filters) ──────────
function optionsFor(key) {
  const pool = docs.filter(d => matches(d, key));
  const count = (fn) => pool.filter(fn).length;
  if (key === 'section') {
    const bySec = {};
    for (const d of pool) {
      const s = sectionOf(d);
      bySec[s] = (bySec[s] || 0) + 1;
    }
    const sortedSecs = Object.keys(bySec).sort((a, b) => bySec[b] - bySec[a] || a.localeCompare(b));
    return [['all', 'All sections', pool.length], ...sortedSecs.map(s => [s, `📁 ${s}`, bySec[s]])];
  }
  if (key === 'kind') {
    return [['all', 'All types', pool.length], ['pdf', 'PDF', count(d => kindOf(d) === 'pdf')],
            ['web', 'Web page', count(d => kindOf(d) === 'web')]];
  }
  if (key === 'index') {
    return [['all', 'All', pool.length], ['yes', 'Searched by the bot', count(d => d.indexed)],
            ['no', 'Not in the search index', count(d => !d.indexed)]];
  }
  if (key === 'state') {
    return [['all', 'All', pool.length], ['ready', 'Read', count(d => stateOf(d) === 'ready')],
            ['processing', 'Being read', count(d => stateOf(d) === 'processing')],
            ['error', 'Errors', count(d => stateOf(d) === 'error')]];
  }
  // text: one option per way of reading found in the documents, biggest first
  const byKey = {};
  for (const d of docs) byKey[textKeyOf(d)] = 0;
  for (const d of pool) byKey[textKeyOf(d)] += 1;
  if (filters.text !== 'all' && !(filters.text in byKey)) byKey[filters.text] = 0;
  const keys = Object.keys(byKey).sort((a, b) => byKey[b] - byKey[a] || textLabel(a).localeCompare(textLabel(b)));
  return [['all', 'All', pool.length], ...keys.map(k => [k, textLabel(k), byKey[k]])];
}

// Updated in place so an open dropdown is not rebuilt under the pointer.
function renderFilterOptions() {
  for (const key of FILTER_KEYS) {
    const sel = $(`f-${key}`);
    const opts = optionsFor(key);
    if (!opts.some(([v]) => v === filters[key])) filters[key] = 'all';   // a saved value that no longer exists
    const sameValues = sel.options.length === opts.length && opts.every(([v], i) => sel.options[i].value === v);
    if (!sameValues) {
      sel.innerHTML = opts.map(([v]) => `<option value="${escapeHtml(v)}"></option>`).join('');
    }
    opts.forEach(([v, label, n], i) => {
      const text = `${label} (${n.toLocaleString()})`;
      const o = sel.options[i];
      if (o.textContent !== text) o.textContent = text;
      o.disabled = n === 0 && v !== 'all' && v !== filters[key];
    });
    sel.value = filters[key];
    sel.classList.toggle('set', filters[key] !== 'all');
  }
  if (!SORTS[filters.sort]) filters.sort = 'new';
  $('f-sort').value = filters.sort;
  $('clear-filters').hidden = !FILTER_KEYS.some(k => filters[k] !== 'all') && !searchTerm;
}

// ── List ─────────────────────────────────────────────────────────────────────
function renderList() {
  renderFilterOptions();
  const flat = docs.filter(d => matches(d)).sort(SORTS[filters.sort] || SORTS.new);
  visibleFlat = flat;
  const list = $('list');
  list.innerHTML = '';
  const total = docs.length;
  $('count-text').textContent = flat.length === total
    ? `${total.toLocaleString()} documents`
    : `${flat.length.toLocaleString()} of ${total.toLocaleString()} documents`;

  if (!total) {
    list.innerHTML = '<div class="list-empty">No documents yet — they appear here after the ingest run (scrape → OCR → import).</div>';
    return;
  }
  if (!flat.length) {
    list.innerHTML = '<div class="list-empty">No documents match — change a filter or the search.</div>';
    return;
  }
  // Keep the selected document drawn even when it sits past the first page.
  const selIdx = flat.findIndex(d => d.id === selectedId);
  if (selIdx >= shownLimit) shownLimit = Math.ceil((selIdx + 1) / PAGE) * PAGE;
  const frag = document.createDocumentFragment();
  for (const d of flat.slice(0, shownLimit)) {
    const el = document.createElement('div');
    el.className = 'doc' + (d.id === selectedId ? ' active' : '');
    el.dataset.id = d.id;
    const chip = STATE_CHIP[stateOf(d)];
    const sec = sectionOf(d);
    const secBadge = (sec && sec !== 'unlinked')
      ? `<span class="sec-badge" title="Website Section: ${escapeHtml(sec)}">📁 ${escapeHtml(sec)}</span>`
      : '';
    const srcLine = d.source_url
      ? `<div class="meta url" title="${escapeHtml(d.source_url)}">${secBadge}🌐 ${escapeHtml(d.source_url)}</div>`
      : (secBadge ? `<div class="meta url">${secBadge}</div>` : '');
    const facts = [`Added ${fmtDate(d.uploaded_at)}`];
    if (d.indexed && d.chunk_count) facts.push(`${d.chunk_count} passage${d.chunk_count === 1 ? '' : 's'} in the bot`);
    el.innerHTML =
      `<div class="name">${escapeHtml(d.original_name)}</div>` +
      srcLine +
      `<div class="meta">${escapeHtml(facts.join(' · '))}</div>` +
      (chip ? `<span class="badge ${chip.cls}">${chip.label}</span>` : '') + methodBadge(d);
    frag.appendChild(el);
  }
  list.appendChild(frag);
  if (flat.length > shownLimit) {
    const more = document.createElement('button');
    more.className = 'more-btn';
    more.type = 'button';
    more.textContent = `Show ${Math.min(PAGE, flat.length - shownLimit)} more (${(flat.length - shownLimit).toLocaleString()} not shown)`;
    more.addEventListener('click', () => { shownLimit += PAGE; renderList(); });
    list.appendChild(more);
  }
}

$('list').addEventListener('click', (e) => {
  const row = e.target.closest('.doc');
  if (row) selectDoc(row.dataset.id);
});

// ── Detail view ──────────────────────────────────────────────────────────────
async function selectDoc(id) {
  selectedId = id;
  for (const el of $('list').querySelectorAll('.doc')) el.classList.toggle('active', el.dataset.id === id);
  renderDetail();

  const d = docs.find(x => x.id === id);
  if (!d || d.status === 'error') return;
  const res = await api.md(id);
  if (selectedId !== id) return; // switched away
  const ta = $('md-text');
  if (ta) ta.value = res.success ? res.markdown : `(${res.error || 'not ready'})`;
  updateTextView();
}

// A short signature of the state that affects the detail *bar* (not the PDF/text).
function detailSig(d) { return d ? `${d.status}|${d.indexed}|${d.chunk_count}|${!!d.error}|${d.ocr_label}` : ''; }

// The top bar: the title (with a chip only for an error / still being read), and
// where the document came from — source · how its text was read · index · folder.
function detailBarHTML(d) {
  const chip = STATE_CHIP[stateOf(d)];
  const meta = [];
  const sec = sectionOf(d);
  if (d.source_url) {
    try {
      const u = new URL(d.source_url);
      const host = u.hostname;
      meta.push(`<span class="meta-note" title="Website hierarchy">📁 <strong>${escapeHtml(host)}</strong> &gt; ${escapeHtml(sec)}</span>`);
    } catch (_) {
      if (sec !== 'unlinked') meta.push(`<span class="sec-badge">📁 ${escapeHtml(sec)}</span>`);
    }
    meta.push(`<a class="meta-src" href="${escapeHtml(d.source_url)}" target="_blank" rel="noopener"
      title="Open the original source: ${escapeHtml(d.source_url)}">🌐 ${escapeHtml(d.source_url)}</a>`);
  } else if (sec && sec !== 'unlinked') {
    meta.push(`<span class="sec-badge">📁 ${escapeHtml(sec)}</span>`);
  }
  meta.push(methodNoteHTML(d));
  meta.push(`<span class="meta-note">${d.indexed
    ? `In the bot's search index${d.chunk_count ? ` · ${d.chunk_count} passage${d.chunk_count === 1 ? '' : 's'}` : ''}`
    : 'Not in the bot\'s search index'}</span>`);
  if (d.doc_dir) {
    meta.push(`<span class="meta-id" id="copy-id" title="Folder id — click to copy (the chunk files and reports use it)">${escapeHtml(d.doc_dir)}</span>`);
  }
  return `
    <div class="bar-top">
      <div class="title-group">
        <span class="title" title="${escapeHtml(d.original_name)}">${escapeHtml(d.original_name)}</span>
        ${chip ? `<span class="status-chip ${chip.cls}">${chip.label}</span>` : ''}
      </div>
    </div>
    <div class="bar-meta">
      ${meta.join('<span class="meta-sep">·</span>')}
      <span class="bar-spacer"></span>
    </div>`;
}

function wireBar(d) {
  const id = $('copy-id');
  if (id) id.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(d.doc_dir); toast('Folder id copied.'); }
    catch (_) { toast(d.doc_dir); }
  });
}

// "How this text was produced" — a quiet note in the meta row. The method comes
// from the /processing endpoint (cached in procById); until it loads: checking…
function methodNoteHTML(d) {
  const p = procById[d.id];
  const label = p && p.label ? p.label : (d.ocr_label || 'checking method…');
  return `<span class="meta-note" title="How this document's text was produced">⚙ ${escapeHtml(label)}</span>`;
}

// Fetch (once) how a doc was processed, cache it, and refresh the bar in place.
async function loadProcessing(id, force) {
  if (!force && procById[id]) return;
  try {
    const res = await api.processing(id);
    if (res && res.success) {
      procById[id] = res.processing;
      if (selectedId === id) updateDetailBar();
    }
  } catch (_) { /* leave the note as it is */ }
}

// Left pane: the SOURCE the text was read from. For a scraped document that is the
// raw fetched content (pdf/html) — or, when the scraper kept no snapshot, a link to
// the source URL; otherwise the stored PDF.
function sourcePaneHTML(d) {
  if (d.source === 'scrape') {
    const url = d.source_url || '';
    const open = url
      ? `<a class="pane-link" href="${escapeHtml(url)}" target="_blank" rel="noopener" title="${escapeHtml(url)}">Open original ↗</a>` : '';
    const label = `<div class="pane-label"><span>Source</span>${open}</div>`;
    if (d.raw_ext) {
      const sandbox = (d.raw_ext === 'html' || d.raw_ext === 'htm') ? 'sandbox=""' : '';
      return label + `<iframe src="${BASE}/api/documents/${d.id}/raw" ${sandbox}></iframe>`;
    }
    return label + `<div class="no-raw">No stored copy of the source for this item —
      it was taken from a listing/API response.${url
        ? `<br><br><a href="${escapeHtml(url)}" target="_blank" rel="noopener">Open source URL ↗</a>` : ''}</div>`;
  }
  if (!d.pdf_path) {
    return `<div class="pane-label"><span>Source PDF</span></div>` +
      `<div class="no-raw">No source PDF is stored for this document — only the text read from it (right).</div>`;
  }
  return `<div class="pane-label"><span>Source PDF</span></div>` +
         `<iframe src="${BASE}/api/documents/${d.id}/pdf#toolbar=1"></iframe>`;
}

// Update ONLY the top bar in place — never touches the PDF iframe or the text.
function updateDetailBar() {
  const d = docs.find(x => x.id === selectedId);
  const bar = $('detail-bar');
  if (!d || !bar) return;
  bar.innerHTML = detailBarHTML(d);
  wireBar(d);
  lastDetailSig = detailSig(d);
}

// Full render — builds the whole detail incl. the iframe. Called only when the
// selected document CHANGES (or moves to/from the error view).
function renderDetail() {
  exitTextFullscreen();
  exitReview();
  const d = docs.find(x => x.id === selectedId);
  const detail = $('detail');
  if (!d) { detail.innerHTML = '<div class="placeholder">Select a document to view.</div>'; lastDetailSig = ''; return; }
  lastDetailSig = detailSig(d);

  if (d.status === 'error') {
    detail.innerHTML =
      `<div class="detail-bar" id="detail-bar">${detailBarHTML(d)}</div>` +
      `<div class="err-box">Reading this document failed:\n\n${escapeHtml(d.error || 'unknown error')}</div>`;
    wireBar(d);
    loadProcessing(d.id);
    return;
  }

  detail.innerHTML = `
    <div class="detail-bar" id="detail-bar">${detailBarHTML(d)}</div>
    <div class="panes">
      <div class="pane pdf">${sourcePaneHTML(d)}</div>
      <div class="pane md" id="text-pane">
        <div class="pane-label text-pane-label">
          <span>Text read from the document</span>
          <div class="pane-tools">
            <button class="check-words" id="btn-reader" type="button" aria-pressed="false"
                    aria-controls="md-reader" title="Read the extracted text as formatted Markdown">
              Markdown reader
            </button>
            <button class="check-words" id="btn-check" type="button"
                    title="Mark words the OCR may have misread">
              Mark doubtful words
            </button>
            <button class="icon-btn text-fullscreen" id="btn-text-fullscreen" type="button"
                    title="View document text fullscreen" aria-label="View document text fullscreen"
                    aria-controls="text-pane" aria-pressed="false">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                   stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path class="expand-icon" d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5" />
                <path class="collapse-icon" d="M3 8h5V3m8 0v5h5M8 21v-5H3m13 5v-5h5" />
              </svg>
            </button>
          </div>
        </div>
        <textarea id="md-text" readonly spellcheck="false" placeholder="${d.status === 'processing' ? 'Not read yet…' : ''}"></textarea>
        <div id="md-reader" class="md-reader hidden" tabindex="0" role="region" aria-label="Formatted document text"></div>
        <div id="md-review" class="md-review hidden" tabindex="0"></div>
      </div>
    </div>`;

  const checkBtn = $('btn-check');
  if (checkBtn) checkBtn.addEventListener('click', () => (reviewMode ? exitReview() : enterReview(d.id)));
  $('btn-reader').addEventListener('click', () => {
    readerMode = reviewMode || !readerMode;
    store.set('doc_markdown_reader', readerMode ? '1' : '0');
    exitReview();
  });
  $('btn-text-fullscreen').addEventListener('click', toggleTextFullscreen);
  updateTextView();
  wireBar(d);
  loadProcessing(d.id);
}

function updateTextFullscreenButton(pane) {
  const btn = pane.querySelector('#btn-text-fullscreen');
  if (!btn) return;
  const active = pane.classList.contains('is-fullscreen');
  const label = active ? 'Exit fullscreen (Esc)' : 'View document text fullscreen';
  btn.title = label;
  btn.setAttribute('aria-label', label);
  btn.setAttribute('aria-pressed', String(active));
}

async function toggleTextFullscreen() {
  const pane = $('text-pane');
  if (!pane) return;
  if (pane.classList.contains('is-fullscreen')) { exitTextFullscreen(); return; }
  // The viewport overlay also works when browser fullscreen is unavailable or
  // denied (for example, inside an embedded browser).
  pane.classList.add('is-fullscreen');
  updateTextFullscreenButton(pane);
  $('btn-text-fullscreen').focus();
  if (pane.requestFullscreen && document.fullscreenEnabled) {
    try {
      await pane.requestFullscreen();
      if (!pane.classList.contains('is-fullscreen') && document.fullscreenElement === pane) {
        await document.exitFullscreen();
      }
    }
    catch (_) { /* keep the viewport overlay */ }
  }
}

function exitTextFullscreen() {
  const pane = $('text-pane');
  if (!pane || !pane.classList.contains('is-fullscreen')) return;
  pane.classList.remove('is-fullscreen');
  updateTextFullscreenButton(pane);
  if (document.fullscreenElement === pane) {
    document.exitFullscreen().catch(() => { /* already exited */ });
  }
  const btn = $('btn-text-fullscreen');
  if (btn) btn.focus();
}

document.addEventListener('fullscreenchange', () => {
  const pane = $('text-pane');
  if (pane && pane.classList.contains('is-fullscreen') && document.fullscreenElement !== pane) {
    exitTextFullscreen();
  }
});

// All three views use the same unchanged Markdown returned by the backend.
function updateTextView() {
  const ta = $('md-text'), reader = $('md-reader'), rv = $('md-review');
  if (!ta || !reader || !rv) return;
  const reading = readerMode && !reviewMode;
  ta.classList.toggle('hidden', reading || reviewMode);
  reader.classList.toggle('hidden', !reading);
  rv.classList.toggle('hidden', !reviewMode);
  const btn = $('btn-reader');
  btn.setAttribute('aria-pressed', String(reading));
  btn.textContent = reading ? 'Raw Markdown' : 'Markdown reader';
  const check = $('btn-check');
  if (!check.disabled) check.textContent = reviewMode
    ? (readerMode ? 'Back to reader' : 'Raw Markdown') : 'Mark doubtful words';
  if (reading && reader.markdown !== ta.value) {
    try {
      const html = marked.parse(ta.value, { gfm: true, breaks: true, async: false });
      // OCR emits HTML tables with merged cells as well as Markdown. Keep that
      // formatting without allowing document HTML to execute or alter the page.
      const content = DOMPurify.sanitize(html, {
        RETURN_DOM_FRAGMENT: true,
        ALLOWED_TAGS: ['p', 'br', 'hr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
          'strong', 'b', 'em', 'i', 'del', 's', 'ul', 'ol', 'li', 'blockquote',
          'pre', 'code', 'a', 'table', 'caption', 'thead', 'tbody', 'tfoot',
          'tr', 'th', 'td', 'sup', 'sub'],
        ALLOWED_ATTR: ['href', 'title', 'colspan', 'rowspan', 'align', 'start'],
        ALLOW_DATA_ATTR: false,
        ALLOW_ARIA_ATTR: false,
      });
      const d = docs.find(x => x.id === selectedId);
      content.querySelectorAll('a[href]').forEach(link => {
        try {
          const url = new URL(link.getAttribute('href'), (d && d.source_url) || location.href);
          if (!['http:', 'https:', 'mailto:'].includes(url.protocol)) {
            link.removeAttribute('href');
            return;
          }
          link.href = url.href;
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
        } catch (_) { link.removeAttribute('href'); }
      });
      reader.replaceChildren(content);
      if (!reader.textContent.trim()) reader.textContent = ta.placeholder || 'No text.';
      reader.markdown = ta.value;
    } catch (_) {
      // If a static asset failed to load, the source text is still available.
      reader.textContent = ta.value || ta.placeholder || 'No text.';
    }
  }
}

// ── Doubtful words: highlight words the OCR may have misread ─────────────────
// A read-only view of the same text with flagged words marked; hovering one shows
// why it was flagged and the dictionary's nearest words. Nothing is changed.
let reviewMode = false;
let reviewIssues = [];        // [{word,start,end,reason,suggestions}], offsets into the text
let popEl = null;

const REASON_TEXT = {
  broken_encoding: 'Broken character encoding — the PDF font mapped this to non-text codes',
  orphan_matra: 'Starts with a vowel mark — the letter it attaches to was lost',
  unknown_word: 'Not found in the Hindi dictionary',
};

async function enterReview(docId) {
  const btn = $('btn-check');
  const readerBtn = $('btn-reader');
  if (btn) { btn.disabled = true; btn.textContent = 'Checking…'; }
  if (readerBtn) readerBtn.disabled = true;
  let res;
  try {
    res = await api.spellcheck(docId);
  } catch (_) {
    res = { success: false, error: 'Could not reach the spellchecker.' };
  }
  if (btn) btn.disabled = false;
  if (readerBtn) readerBtn.disabled = false;
  if (selectedId !== docId) return;

  if (!res || !res.success) {
    if (btn) btn.textContent = 'Mark doubtful words';
    toast((res && res.error) || 'Spellcheck failed');
    return;
  }
  reviewIssues = res.issues || [];
  reviewMode = true;
  renderReview();

  // Past roughly one doubtful word in eight, the page was misread rather than
  // mis-spelled — say so instead of leaving a wall of highlights.
  const n = reviewIssues.length;
  const words = countDevanagariWords($('md-text').value);
  const ratio = words ? n / words : 0;
  if (!n) {
    toast('No doubtful words found');
  } else if (ratio > 0.125) {
    toast(`${n} of ~${words} words look wrong (${Math.round(ratio * 100)}%) — the page was probably misread.`);
  } else {
    toast(`${n} doubtful word(s)${res.truncated ? ' (first 400 shown)' : ''}`);
  }
}

/** Devanagari words in the text — the denominator for "how bad is this page?". */
function countDevanagariWords(text) {
  return ((text || '').match(/[ऀ-ॿ]{3,}/g) || []).length;
}

function exitReview() {
  reviewMode = false;
  closePopover();
  updateTextView();
}

/** Build the highlighted read-only view from the text. */
function renderReview() {
  const ta = $('md-text'), rv = $('md-review');
  if (!ta || !rv) return;
  const text = ta.value;

  let html = '', cursor = 0;
  reviewIssues.forEach((iss, idx) => {
    if (iss.start < cursor) return;                 // overlapping — skip
    html += escapeHtml(text.slice(cursor, iss.start));
    const has = (iss.suggestions || []).length;
    html += `<mark class="sus ${iss.reason}${has ? '' : ' nofix'}" data-idx="${idx}"` +
            ` title="${escapeHtml(REASON_TEXT[iss.reason] || '')}">` +
            `${escapeHtml(text.slice(iss.start, iss.end))}</mark>`;
    cursor = iss.end;
  });
  html += escapeHtml(text.slice(cursor));

  rv.innerHTML = html || '<em class="empty">No text.</em>';
  updateTextView();

  rv.querySelectorAll('mark.sus').forEach((el) => {
    el.addEventListener('mouseenter', () => openPopover(el));
    el.addEventListener('mouseleave', () => scheduleClose());
    el.addEventListener('click', () => openPopover(el));   // touch/keyboard friendly
  });
}

let closeTimer = null;
function scheduleClose() { clearTimeout(closeTimer); closeTimer = setTimeout(closePopover, 260); }
function cancelClose() { clearTimeout(closeTimer); }

function closePopover() {
  if (popEl) { popEl.remove(); popEl = null; }
}

function openPopover(markEl) {
  cancelClose();
  closePopover();
  const iss = reviewIssues[Number(markEl.dataset.idx)];
  if (!iss) return;

  const pop = document.createElement('div');
  pop.className = 'sus-pop';
  const sugs = iss.suggestions || [];
  pop.innerHTML =
    `<div class="sus-why">${escapeHtml(REASON_TEXT[iss.reason] || 'Check this word')}</div>` +
    (sugs.length
      ? `<div class="sus-why">Nearest dictionary words:</div><div class="sus-list">` + sugs.map((s) =>
          `<div class="sus-opt"><span class="t">${escapeHtml(s.term)}</span>` +
          `<span class="d">edit ${s.distance} · seen ${s.frequency}×</span></div>`).join('') + `</div>`
      : `<div class="sus-none">No close match in the dictionary.</div>`);

  document.body.appendChild(pop);
  const r = markEl.getBoundingClientRect();
  const top = r.bottom + 6 + window.scrollY;
  pop.style.left = `${Math.max(8, Math.min(r.left + window.scrollX, window.innerWidth - pop.offsetWidth - 8))}px`;
  pop.style.top = (top + pop.offsetHeight > window.scrollY + window.innerHeight)
    ? `${r.top + window.scrollY - pop.offsetHeight - 6}px` : `${top}px`;
  pop.addEventListener('mouseenter', cancelClose);
  pop.addEventListener('mouseleave', scheduleClose);
  popEl = pop;
}

// ── Refresh / polling ────────────────────────────────────────────────────────
async function refresh() {
  let res;
  try { res = await api.list(); } catch (_) { return; }
  if (!res.success) return;
  docs = res.documents || [];
  // Redraw the list only when something in it changed (thousands of rows).
  const sig = docs.map(d => `${d.id}|${d.status}|${d.indexed}|${d.chunk_count}|${d.ocr_label}`).join('\n');
  if (sig !== docsSig) { docsSig = sig; renderList(); }

  const d = docs.find(x => x.id === selectedId);
  const ta = $('md-text');
  if (d && ta && d.md_path && ta.value.startsWith('(')) {   // the text has just been read
    const md = await api.md(d.id);
    if (md.success && selectedId === d.id) {
      ta.value = md.markdown;
      updateTextView();
      if (reviewMode) exitReview(); // spellcheck offsets belong to the previous text
    }
  }
  // Update ONLY the bar when the doc's state changed — never rebuild the panes,
  // so the PDF (and its scroll position) stays.
  if (d && detailSig(d) !== lastDetailSig) {
    if (d.status === 'error' || !$('detail-bar')) renderDetail();
    else updateDetailBar();
    loadProcessing(d.id, true);
  }
}

// ── Search + filters ─────────────────────────────────────────────────────────
function saveFilters() { store.set('doc_filters', JSON.stringify(filters)); }

const searchEl = $('doc-search');
let searchTimer = null;
searchEl.addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    searchTerm = searchEl.value.trim().toLowerCase();
    shownLimit = PAGE;
    renderList();
  }, 120);
});
for (const key of [...FILTER_KEYS, 'sort']) {
  $(`f-${key}`).addEventListener('change', (e) => {
    filters[key] = e.target.value;
    shownLimit = PAGE;
    saveFilters();
    renderList();
  });
}
$('clear-filters').addEventListener('click', () => {
  for (const k of FILTER_KEYS) filters[k] = 'all';
  searchEl.value = ''; searchTerm = '';
  shownLimit = PAGE;
  saveFilters();
  renderList();
});

// ── Keyboard: ↑/↓ (or j/k) move through the visible list ─────────────────────
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && $('text-pane')?.classList.contains('is-fullscreen')) {
    e.preventDefault();
    exitTextFullscreen();
    return;
  }
  const el = document.activeElement;
  if (el && /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName)) return;   // typing, choosing, or scrolling the text
  if (el && el.closest('#md-reader, #md-review')) return;
  const dir = (e.key === 'ArrowDown' || e.key === 'j') ? 1
            : (e.key === 'ArrowUp' || e.key === 'k') ? -1 : 0;
  if (!dir || !visibleFlat.length) return;
  e.preventDefault();
  const idx = visibleFlat.findIndex(d => d.id === selectedId);
  const next = idx < 0 ? (dir > 0 ? 0 : visibleFlat.length - 1)
                       : Math.min(visibleFlat.length - 1, Math.max(0, idx + dir));
  const target = visibleFlat[next];
  if (target && target.id !== selectedId) {
    if (next >= shownLimit) { shownLimit += PAGE; renderList(); }
    selectDoc(target.id);
    const active = $('list').querySelector('.doc.active');
    if (active) active.scrollIntoView({ block: 'nearest' });
  }
});

// ── Utils ────────────────────────────────────────────────────────────────────
function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function fmtDate(iso) {
  if (!iso) return '';
  try { return new Date(iso).toLocaleDateString([], { dateStyle: 'medium' }); }
  catch { return iso; }
}

// ── Boot ─────────────────────────────────────────────────────────────────────
refresh();
setInterval(refresh, 4000);
