'use strict';

// ── Client Application for Component Evaluation ──────────────────────────────
const $ = (id) => document.getElementById(id);
const BASE = (typeof window !== 'undefined' && window.__APP_BASE__) || '';

const toast = (msg) => {
  const t = $('toast');
  if (!t) return;
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove('show'), 3500);
};

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function fmtDate(iso) {
  if (!iso) return '—';
  try { return new Date(iso).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' }); }
  catch { return iso; }
}

async function api(method, path, body, isFormData = false) {
  const opts = { method, headers: {}, credentials: 'same-origin' };
  if (body !== undefined) {
    if (isFormData) {
      opts.body = body;
    } else {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
  }
  const res = await fetch(BASE + path, opts);
  if (res.status === 401) { window.location.href = `${BASE}/login`; throw new Error('unauth'); }
  if (res.status === 403) { window.location.href = `${BASE}/`; throw new Error('forbidden'); }
  return res.json().catch(() => ({ success: false, error: `HTTP ${res.status}` }));
}

// ── State ────────────────────────────────────────────────────────────────────
let activeComponent = 'hybrid_retrieval';
let componentsMeta = {};
let stagedFile = null;  // { path, filename, row_count }
let currentRunId = null;
let pollTimer = null;
let lastRunData = null;

// ── Component Labels & Descriptions ──────────────────────────────────────────
const COMPONENT_TITLES = {
  ingestion_ocr: '1. Ingestion & OCR Evaluation',
  query_preprocessing: '2. Query Preprocessing Evaluation',
  hybrid_retrieval: '3. Hybrid Retrieval Evaluation',
  reranking: '4. Second-Stage Reranking Evaluation',
  context_budgeting: '5. Context Budgeting Evaluation',
  generation: '6. Generation (LLM) Evaluation',
  multi_hop: '7. Multi-Hop Loop Evaluation',
  grounding_guardrail: '8. Grounding Guardrails Evaluation',
};

// ── Navigation ───────────────────────────────────────────────────────────────
function switchMainTab(tab) {
  $('tab-btn-corrections')?.classList.toggle('active', tab === 'corrections');
  $('tab-btn-components')?.classList.toggle('active', tab === 'components');
  $('tab-btn-history')?.classList.toggle('active', tab === 'history');
  $('view-corrections')?.classList.toggle('hidden', tab !== 'corrections');
  $('view-components')?.classList.toggle('hidden', tab !== 'components');
  $('view-history')?.classList.toggle('hidden', tab !== 'history');
  if (tab === 'history') loadRunHistory();
  if (tab === 'corrections') loadCorrectionsQueue();
}

// ── Flagged Questions & Verified Corrections (Re3) ───────────────────────────
let queueItems = [];
let correctionsMounted = null;

function initCorrections() {
  const vcRoot = document.getElementById('verified-corrections');
  if (!vcRoot || !window.VerifiedCorrections || correctionsMounted) return;
  correctionsMounted = window.VerifiedCorrections.mount(vcRoot, {
    api: `${BASE}/api/corrections`,
    login: `${BASE}/login`,
    onPublished: async ({ evaluationId }) => {
      if (!evaluationId) return;
      try {
        const res = await requestEval('PATCH', `/${evaluationId}`, { reviewed: true });
        if (res && res.item) {
          queueItems = queueItems.map(row => row.id === res.item.id ? res.item : row);
          renderQueue();
          subSection('queue');
          toast('Verified correction published and question marked reviewed.');
        }
      } catch (_) {}
    },
  });
}

function subSection(name) {
  const show = name === 'corrections';
  const qSec = document.getElementById('review-queue');
  const cSec = document.getElementById('verified-corrections');
  const vqBtn = document.getElementById('view-queue');
  const vcBtn = document.getElementById('view-verified');
  if (qSec) qSec.hidden = show;
  if (cSec) cSec.hidden = !show;
  if (vqBtn) vqBtn.setAttribute('aria-pressed', String(!show));
  if (vcBtn) vcBtn.setAttribute('aria-pressed', String(show));
}

async function requestEval(method, suffix = '', body) {
  const response = await fetch(`${BASE}/api/evaluation${suffix}`, {
    method,
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) })
  });
  if (response.status === 401) { window.location.href = `${BASE}/login`; throw new Error('Please sign in.'); }
  const data = await response.json();
  if (!response.ok || !data.success) throw new Error(data.error || 'Could not load evaluations.');
  return data;
}

function renderQueue() {
  const list = document.getElementById('evaluation-list');
  const filter = document.getElementById('filter');
  const status = document.getElementById('status');
  if (!list || !filter || !status) return;

  list.replaceChildren();
  const pending = queueItems.filter(item => item.status === 'pending').length;
  const countEl = document.getElementById('counts');
  if (countEl) countEl.textContent = `${pending} need review · ${queueItems.length - pending} reviewed`;

  const shown = queueItems.filter(item => filter.value === 'all' || item.status === filter.value);
  status.className = '';
  status.textContent = shown.length ? '' : queueItems.length ? 'No items in this view.'
    : 'Nothing saved yet. Use the flag button beside a chat question to save its answer for review.';

  for (const item of shown) {
    const card = document.createElement('article');
    card.className = 'evaluation-item';

    const header = document.createElement('div');
    header.className = 'item-header';
    const stBadge = document.createElement('span');
    stBadge.className = `review-status ${item.status}`;
    stBadge.textContent = item.status === 'pending' ? 'Needs review' : 'Reviewed';
    header.appendChild(stBadge);
    const dateSpan = document.createElement('span');
    dateSpan.textContent = `Saved ${new Date(item.savedAt).toLocaleString()}`;
    header.appendChild(dateSpan);
    card.appendChild(header);

    const qLbl = document.createElement('div');
    qLbl.className = 'field-label';
    qLbl.textContent = 'Question';
    card.appendChild(qLbl);

    const qH2 = document.createElement('h2');
    qH2.textContent = item.question;
    card.appendChild(qH2);

    const aLbl = document.createElement('div');
    aLbl.className = 'field-label';
    aLbl.textContent = 'Bot answer';
    card.appendChild(aLbl);

    const answer = (item.answer || '').replace(/<think>[\s\S]*?<\/think>\s*/g, '').replace(/<think>[\s\S]*$/, '').trim();
    const aDiv = document.createElement('div');
    aDiv.className = 'saved-answer';
    aDiv.textContent = answer;
    card.appendChild(aDiv);

    const footer = document.createElement('div');
    footer.className = 'item-footer';

    // Button: Correct answer (opens Re3 verified corrections editor)
    const correctBtn = document.createElement('button');
    correctBtn.className = 'secondary';
    correctBtn.type = 'button';
    correctBtn.textContent = 'Correct answer';
    correctBtn.addEventListener('click', () => {
      subSection('corrections');
      if (correctionsMounted) {
        correctionsMounted.open({
          question: item.standalone || item.question,
          originalAnswer: answer,
          evaluationId: item.id
        });
      }
    });
    footer.appendChild(correctBtn);

    // Button: Mark reviewed / Needs review toggle
    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'secondary';
    toggleBtn.type = 'button';
    toggleBtn.textContent = item.status === 'pending' ? 'Mark reviewed' : 'Needs review';
    toggleBtn.addEventListener('click', async () => {
      toggleBtn.disabled = true;
      try {
        const { item: updated } = await requestEval('PATCH', `/${item.id}`, { reviewed: item.status !== 'reviewed' });
        queueItems = queueItems.map(row => row.id === updated.id ? updated : row);
        renderQueue();
      } catch (err) {
        toast(err.message);
        toggleBtn.disabled = false;
      }
    });
    footer.appendChild(toggleBtn);

    card.appendChild(footer);
    list.appendChild(card);
  }
}

async function loadCorrectionsQueue() {
  const refreshBtn = document.getElementById('refresh');
  const status = document.getElementById('status');
  if (refreshBtn) refreshBtn.disabled = true;
  if (status) { status.className = ''; status.textContent = 'Loading saved questions…'; }
  try {
    initCorrections();
    const res = await requestEval('GET');
    queueItems = (res && res.items) || [];
    renderQueue();
    if (correctionsMounted) correctionsMounted.load();
  } catch (err) {
    if (status) { status.className = 'error'; status.textContent = err.message; }
  } finally {
    if (refreshBtn) refreshBtn.disabled = false;
  }
}

// Wire up corrections queue events
document.addEventListener('DOMContentLoaded', () => {
  const vqBtn = document.getElementById('view-queue');
  const vcBtn = document.getElementById('view-verified');
  const rBtn = document.getElementById('refresh');
  const fSel = document.getElementById('filter');
  if (vqBtn) vqBtn.addEventListener('click', () => subSection('queue'));
  if (vcBtn) vcBtn.addEventListener('click', () => {
    subSection('corrections');
    if (correctionsMounted) correctionsMounted.load();
  });
  if (rBtn) rBtn.addEventListener('click', () => {
    loadCorrectionsQueue();
    if (correctionsMounted) correctionsMounted.load();
  });
  if (fSel) fSel.addEventListener('change', renderQueue);
});

function selectComponent(comp) {
  activeComponent = comp;
  document.querySelectorAll('.comp-pill').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.comp === comp);
  });

  const title = COMPONENT_TITLES[comp] || comp;
  $('comp-title').textContent = title;
  const meta = componentsMeta[comp];
  if (meta) {
    $('comp-desc').textContent = meta.description || '';
  }

  // Reset upload area
  stagedFile = null;
  $('staged-card').classList.add('hidden');
  $('results-section').classList.add('hidden');
  $('progress-box').classList.add('hidden');
  $('file-input').value = '';
}

// ── Templates & Upload ───────────────────────────────────────────────────────
function downloadTemplate() {
  window.location.href = `${BASE}/api/eval/component/${encodeURIComponent(activeComponent)}/template`;
}

async function loadSampleDataset() {
  toast('Loading sample dataset for ' + activeComponent + '...');
  try {
    const res = await api('GET', `/api/eval/component/${activeComponent}/template`);
    // Alternatively upload the sample path directly
    const stageRes = await api('POST', `/api/eval/component/${activeComponent}/upload`, {
      rows: [
        {
          query_id: 'Q001',
          query: 'स्थानांतरण नीति 2026-27 के मुख्य प्रावधान क्या हैं?',
          gold_passage_ids: ['circular_transfer_2026.pdf#1', 'circular_transfer_2026.pdf#2'],
          query_language: 'hi',
          doc_language: 'hi',
          is_exact_match_query: false,
        },
        {
          query_id: 'Q002',
          query: 'What is the maximum age relaxation for OBC candidates in government recruitment?',
          gold_passage_ids: ['gad_recruitment_rules_2024.pdf#4', 'age_relaxation_memo.pdf#1'],
          query_language: 'en',
          doc_language: 'hi',
          is_exact_match_query: false,
        },
        {
          query_id: 'Q003',
          query: 'अनुकंपा नियुक्ति के लिए आवेदन की समय सीमा क्या है?',
          gold_passage_ids: ['compassionate_appointment_2023.pdf#2'],
          query_language: 'hi',
          doc_language: 'hi',
          is_exact_match_query: false,
        },
        {
          query_id: 'Q004',
          query: 'आदेश क्रमांक एफ 2-4/2026/1-3 में क्या निर्देश दिए गए हैं?',
          gold_passage_ids: ['gad_order_f2_4_2026.pdf#1'],
          query_language: 'hi',
          doc_language: 'hi',
          is_exact_match_query: true,
        },
        {
          query_id: 'Q005',
          query: 'Dearness Allowance rates with effect from 1st July 2025',
          gold_passage_ids: ['finance_da_order_2025.pdf#1'],
          query_language: 'en',
          doc_language: 'en',
          is_exact_match_query: false,
        },
      ],
    });

    if (stageRes.success) {
      stagedFile = {
        path: stageRes.staged_path,
        filename: stageRes.filename || 'sample_batch.json',
        row_count: stageRes.row_count,
      };
      showStagedFile(stagedFile);
      toast(`Loaded ${stageRes.row_count} sample queries ready for evaluation.`);
    } else {
      toast(stageRes.error || 'Could not load sample.');
    }
  } catch (err) {
    toast('Error loading sample dataset.');
  }
}

async function handleFileSelect(evt) {
  const file = evt.target.files && evt.target.files[0];
  if (!file) return;

  toast(`Validating ${file.name}...`);
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await api('POST', `/api/eval/component/${activeComponent}/upload`, formData, true);
    if (!res.success) {
      toast(res.error || 'Validation failed.');
      return;
    }

    stagedFile = {
      path: res.staged_path,
      filename: res.filename,
      row_count: res.row_count,
      errors: res.errors || [],
    };

    showStagedFile(stagedFile);
    if (res.error_count > 0) {
      toast(`Validation warning: ${res.error_count} row errors found.`);
    } else {
      toast(`Validated: ${res.row_count} rows ready.`);
    }
  } catch (err) {
    toast('Failed to upload file.');
  }
}

function showStagedFile(staged) {
  $('staged-card').classList.remove('hidden');
  $('staged-name').textContent = staged.filename;
  $('staged-rows').textContent = `${staged.row_count} rows valid`;

  const errHint = $('staged-err-hint');
  if (staged.errors && staged.errors.length) {
    errHint.classList.remove('hidden');
    errHint.innerHTML = `⚠ ${staged.errors.length} row issues detected (e.g. Row ${staged.errors[0].row}: ${esc(staged.errors[0].error)})`;
  } else {
    errHint.classList.add('hidden');
  }
}

// ── Run Evaluation & Polling ─────────────────────────────────────────────────
async function startEvaluation() {
  if (!stagedFile || !stagedFile.path) {
    toast('Please upload a valid test batch first.');
    return;
  }

  $('btn-run-eval').disabled = true;
  $('progress-box').classList.remove('hidden');
  $('results-section').classList.add('hidden');
  $('progress-fill').style.width = '0%';
  $('progress-pct').textContent = '0%';
  $('progress-msg').textContent = 'Starting evaluation job...';

  try {
    const res = await api('POST', `/api/eval/component/${activeComponent}/run`, {
      staged_path: stagedFile.path,
      dataset_name: stagedFile.filename,
      config: { alpha: 0.6, k: 60, mock: false },
    });

    if (!res.success) {
      $('btn-run-eval').disabled = false;
      $('progress-box').classList.add('hidden');
      toast(res.error || 'Could not start evaluation run.');
      return;
    }

    currentRunId = res.run_id;
    toast(`Evaluation run started (${currentRunId}).`);
    pollRun(currentRunId);
  } catch (err) {
    $('btn-run-eval').disabled = false;
    $('progress-box').classList.add('hidden');
    toast('Error starting evaluation.');
  }
}

function pollRun(runId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const res = await api('GET', `/api/eval/component/${activeComponent}/runs/${runId}`);
      if (!res.success) {
        clearInterval(pollTimer);
        $('btn-run-eval').disabled = false;
        toast('Evaluation polling failed.');
        return;
      }

      const total = res.total || stagedFile?.row_count || 1;
      const prog = res.progress || 0;
      const pct = Math.min(100, Math.round((prog / total) * 100));

      $('progress-fill').style.width = `${pct}%`;
      $('progress-pct').textContent = `${pct}%`;
      $('progress-msg').textContent = res.message || `Processing item ${prog} of ${total}...`;

      if (res.status === 'done' || res.status === 'failed' || res.status === 'stopped') {
        clearInterval(pollTimer);
        $('btn-run-eval').disabled = false;
        $('progress-box').classList.add('hidden');
        if (res.status === 'done') {
          toast('Evaluation completed!');
          renderResults(res);
        } else {
          toast(`Run finished with status: ${res.status}`);
        }
      }
    } catch (_) {
      clearInterval(pollTimer);
      $('btn-run-eval').disabled = false;
    }
  }, 1000);
}

async function cancelEvaluation() {
  if (!currentRunId) return;
  try {
    await api('POST', `/api/eval/runs/${currentRunId}/cancel`);
    toast('Cancelling evaluation run...');
  } catch (_) {}
}

// ── Render Results ───────────────────────────────────────────────────────────
function renderResults(data) {
  lastRunData = data;
  $('results-section').classList.remove('hidden');
  $('res-run-id').textContent = data.run_id;
  $('res-timestamp').textContent = fmtDate(data.run?.finished_at || data.run?.created_at);

  const kpis = data.run?.kpis || {};
  const prevKpis = data.previous_run_kpis || {};

  // 1. KPI Cards
  const kpiGrid = $('kpi-grid');
  kpiGrid.innerHTML = Object.entries(kpis).map(([k, item]) => {
    const valStr = typeof item.value === 'number'
      ? (item.formula.includes('%') || k.includes('recall') || k.includes('hit') || k.includes('lift') ? `${(item.value * (k.includes('lift') ? 1 : 100)).toFixed(1)}%` : item.value.toFixed(3))
      : String(item.value);

    const targetStr = typeof item.target === 'number'
      ? (item.formula.includes('%') || k.includes('recall') || k.includes('hit') ? `${(item.target * 100).toFixed(0)}%` : String(item.target))
      : String(item.target);

    // Delta vs previous run
    let deltaHtml = '';
    if (prevKpis[k] && typeof prevKpis[k].value === 'number' && typeof item.value === 'number') {
      const diff = item.value - prevKpis[k].value;
      const isUp = diff >= 0;
      const diffStr = (diff >= 0 ? '+' : '') + (diff * (k.includes('lift') ? 1 : 100)).toFixed(1) + (k.includes('lift') ? '%' : '%');
      deltaHtml = `<span class="kpi-delta ${isUp ? 'up' : 'down'}">${isUp ? '↑' : '↓'} ${diffStr}</span>`;
    }

    return `
      <div class="kpi-card">
        <div class="kpi-head">
          <span>${esc(k.replace(/_/g, ' '))}</span>
          <span class="kpi-formula-tip" data-tooltip="Formula: ${esc(item.formula)}">ⓘ</span>
        </div>
        <div class="kpi-val">
          ${esc(valStr)}
        </div>
        <div class="kpi-sub">
          <span>Target: <b>${esc(targetStr)}</b></span>
          <span class="chip ${esc(item.status)}">${esc(item.status)}</span>
        </div>
        ${deltaHtml ? `<div style="font-size:11px; margin-top:2px;">${deltaHtml} <small style="color:var(--text-3);">vs prev</small></div>` : ''}
      </div>
    `;
  }).join('');

  // 2. Ablations (for Hybrid Retrieval)
  const ablations = data.run?.ablations || {};
  const ablSec = $('ablation-section');
  if (ablations && ablations.dense_only) {
    ablSec.classList.remove('hidden');
    const rows = [
      { name: 'Dense Retrieval (BGE-M3 1024-dim)', r10: ablations.dense_only.recall_at_10, desc: 'Semantic vectors via Qdrant HNSW' },
      { name: 'Sparse Retrieval (Lexical weights)', r10: ablations.sparse_only.recall_at_10, desc: 'Learned lexical terms via Qdrant sparse index' },
      { name: 'Exact Match (MatchText)', r10: ablations.exact_match_only.recall_at_10, desc: 'Literal full-text on codes and identifiers' },
      { name: '<b>RRF Fused Hybrid (Dense + Sparse)</b>', r10: ablations.rrf_fused.recall_at_10, desc: '<b>Reciprocal Rank Fusion (k=60, α=0.6)</b>', isRrf: true },
    ];
    $('ablation-rows').innerHTML = rows.map((r) => {
      const pct = (r.r10 * 100).toFixed(1) + '%';
      const lift = r.isRrf ? `<b style="color:var(--green)">+${ablations.rrf_lift_percent?.toFixed(1)}%</b>` : '—';
      return `
        <tr ${r.isRrf ? 'style="background:var(--navy-tint); font-weight:600;"' : ''}>
          <td>${r.name}</td>
          <td><b>${pct}</b></td>
          <td>${lift}</td>
          <td style="color:var(--text-2);">${r.desc}</td>
        </tr>
      `;
    }).join('');
  } else {
    ablSec.classList.add('hidden');
  }

  // 3. Slices
  const slices = data.run?.slices || {};
  const sliceRows = Object.entries(slices).map(([name, s]) => `
    <tr>
      <td><b>${esc(name.replace(/_/g, ' '))}</b></td>
      <td>${s.count || '—'}</td>
      <td>${(s.recall_at_10 * 100).toFixed(1)}%</td>
      <td>${(s.mrr || 0).toFixed(3)}</td>
    </tr>
  `).join('');
  $('slice-rows').innerHTML = sliceRows || '<tr><td colspan="4" style="color:var(--text-3);">No slice data available.</td></tr>';

  // 4. Failures
  const failures = data.failures || [];
  $('failure-count').textContent = failures.length;
  renderFailuresTable(failures);
}

function renderFailuresTable(failures) {
  const tbody = $('failure-rows');
  if (!failures.length) {
    tbody.innerHTML = '<tr><td colspan="5" style="color:var(--green); text-align:center; padding:18px;">✓ All test cases met the acceptance criteria! Zero failure rows.</td></tr>';
    return;
  }
  tbody.innerHTML = failures.map((f) => `
    <tr>
      <td><code>${esc(f.row_id)}</code></td>
      <td style="font-weight:500;">${esc(f.input)}</td>
      <td><span style="color:var(--green); font-size:11.5px;">${esc(JSON.stringify(f.expected))}</span></td>
      <td><span style="color:var(--text-2); font-size:11.5px;">${esc(JSON.stringify(f.actual))}</span></td>
      <td><span class="chip fail" style="font-size:10px;">FAIL</span> <small style="color:var(--red);">${esc(f.reason)}</small></td>
    </tr>
  `).join('');
}

function filterFailures() {
  const query = $('failure-filter').value.toLowerCase().trim();
  const allFailures = lastRunData?.failures || [];
  if (!query) {
    renderFailuresTable(allFailures);
    return;
  }
  const filtered = allFailures.filter((f) =>
    String(f.row_id).toLowerCase().includes(query) ||
    String(f.input).toLowerCase().includes(query) ||
    String(f.reason).toLowerCase().includes(query)
  );
  renderFailuresTable(filtered);
}

function exportResultsCsv() {
  if (!currentRunId) return;
  window.location.href = `${BASE}/api/eval/runs/${encodeURIComponent(currentRunId)}/export`;
}

// ── Run History Tab ──────────────────────────────────────────────────────────
async function loadRunHistory() {
  const tbody = $('history-rows');
  tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:20px;">Loading history...</td></tr>';
  try {
    const res = await api('GET', '/api/eval/runs');
    if (!res.success || !res.runs.length) {
      tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:20px; color:var(--text-3);">No evaluation runs found.</td></tr>';
      return;
    }
    tbody.innerHTML = res.runs.map((r) => {
      const kpiBrief = Object.entries(r.kpis || {})
        .slice(0, 3)
        .map(([k, v]) => `${k.replace(/_/g, '')}: <b>${(v.value * 100).toFixed(0)}%</b>`)
        .join(' · ');

      return `
        <tr>
          <td><code>${esc(r.run_id)}</code></td>
          <td><b>${esc(r.component)}</b></td>
          <td>${esc(r.dataset_name || '—')}</td>
          <td>${esc(fmtDate(r.created_at))}</td>
          <td><span class="chip ${esc(r.status)}">${esc(r.status)}</span></td>
          <td>${r.n_rows || 0}</td>
          <td style="font-size:11.5px;">${kpiBrief || '—'}</td>
          <td>
            <a class="secondary" style="padding:3px 8px; font-size:11px;" href="${BASE}/api/eval/runs/${encodeURIComponent(r.run_id)}/export">CSV</a>
          </td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    tbody.innerHTML = '<tr><td colspan="8" style="color:var(--red); text-align:center; padding:20px;">Failed to load history.</td></tr>';
  }
}

// ── Initialize ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const metaRes = await api('GET', '/api/eval/components');
    if (metaRes.success) {
      componentsMeta = metaRes.components;
    }
  } catch (_) {}

  selectComponent('hybrid_retrieval');

  function handleNavHash() {
    const hash = window.location.hash.replace(/^#/, '');
    if (hash === 'history') {
      switchMainTab('history');
    } else if (hash === 'components') {
      switchMainTab('components');
    } else {
      switchMainTab('corrections');
    }
  }

  handleNavHash();
  window.addEventListener('hashchange', handleNavHash);
});
