'use strict';

// Departments page (CHiPS admins): create departments with an amount of storage,
// keep their accounts, and add or delete their files. The server
// (platform/routes.js) refuses files past a department's storage and files that
// are not what their name says; this page checks the same things first only to
// answer sooner. Shared pieces (requests, uploads, accounts) are in
// platform-assets/platform.js.

const P = window.Platform;
const $ = (id) => document.getElementById(id);
const { esc, fmtBytes, sizeParts, UNITS, usageHtml } = P;
const toast = P.makeToast($('toast'));

// '/pipeline/<name>' behind the reverse proxy, '' when hit directly.
const BASE = (typeof window !== 'undefined' && window.__APP_BASE__) || '';
const api = P.makeApi(BASE, {
  onUnauth: () => { window.location.href = `${BASE}/login`; },
  onForbidden: () => { window.location.href = `${BASE}/`; },
});

let status = null;          // /api/platform/status
let departments = [];
let current = null;         // department being managed
let editing = null;         // department id whose storage is being changed
let accounts = null;        // accounts panel of `current`
let activeSection = 'overview';
let available = false;
let uploading = false;
let creating = false;
let fileRequest = 0;
let lastFiles = [];
const pages = { dept: 1, file: 1, account: 1 };
const PAGE_SIZE = 10;
const sectionScroll = {};
const SECTIONS = {
  overview: ['Overview', 'Storage, files and accounts at a glance.'],
  departments: ['Departments', 'Find a department and open its files or accounts.'],
  new: ['New department', 'Create a department and allocate its storage.'],
  files: ['Files', 'Upload and process documents for the selected department.'],
  accounts: ['Accounts', 'Manage access to the selected department.'],
};
const SECTION_NAMES = Object.keys(SECTIONS);

function saveLocation(replace = false) {
  const params = new URLSearchParams({ section: activeSection });
  if (current) params.set('department', current.id);
  const hash = `#${params}`;
  if (location.hash !== hash) history[replace ? 'replaceState' : 'pushState'](null, '', hash);
}

function showSection(name, { focus = true, writeUrl = true, replace = false } = {}) {
  if (!SECTION_NAMES.includes(name)) name = 'overview';
  if (!current && ['files', 'accounts'].includes(name)) name = 'departments';
  const workspace = $('workspace');
  sectionScroll[activeSection] = workspace.scrollTop;
  activeSection = name;
  document.querySelectorAll('[data-section-panel]').forEach((el) => {
    el.classList.toggle('hidden', el.dataset.sectionPanel !== name);
  });
  document.querySelectorAll('.section-button').forEach((button) => {
    const selected = button.dataset.section === name;
    button.classList.toggle('on', selected);
    if (selected) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  $('section-title').textContent = SECTIONS[name][0];
  $('section-description').textContent = SECTIONS[name][1];
  workspace.scrollTop = sectionScroll[name] || 0;
  if (writeUrl) saveLocation(replace);
  if (focus) $('section-title').focus({ preventScroll: true });
}

async function openSection(name, options) {
  showSection(name, options);
  if (!current) return;
  if (activeSection === 'files') await showFiles();
  if (activeSection === 'accounts') await loadAccounts(current);
}

document.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-section], button[data-expand]');
  if (!button || button.disabled) return;
  if (button.dataset.section) openSection(button.dataset.section);
  else {
    const details = $(button.dataset.expand);
    details.open = !details.open;
    if (details.open) details.querySelector('input:not([type="file"])')?.focus({ preventScroll: true });
  }
});
document.querySelectorAll('button[data-expand]').forEach((button) => {
  const details = $(button.dataset.expand);
  button.setAttribute('aria-controls', details.id);
  button.setAttribute('aria-expanded', String(details.open));
  details.addEventListener('toggle', () => button.setAttribute('aria-expanded', String(details.open)));
});

function pageRows(prefix, rows, total = rows.length) {
  const count = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  pages[prefix] = Math.max(1, Math.min(pages[prefix], count));
  const start = (pages[prefix] - 1) * PAGE_SIZE;
  const end = Math.min(start + PAGE_SIZE, rows.length);
  $(prefix + '-range').textContent = rows.length
    ? `${start + 1}–${end} of ${rows.length}${total !== rows.length ? ` matching (${total} total)` : ''}`
    : total ? `No matches (${total} total)` : 'No records';
  $(prefix + '-page').textContent = `Page ${pages[prefix]} of ${count}`;
  $(prefix + '-prev').disabled = pages[prefix] <= 1;
  $(prefix + '-next').disabled = pages[prefix] >= count;
  return rows.slice(start, end);
}

function wireList(prefix, render) {
  for (const suffix of ['search', 'filter']) $(prefix + '-' + suffix).addEventListener(suffix === 'search' ? 'input' : 'change', () => {
    pages[prefix] = 1;
    render();
  });
  for (const [suffix, delta] of [['prev', -1], ['next', 1]]) $(prefix + '-' + suffix).addEventListener('click', () => {
    pages[prefix] += delta;
    render();
    $(prefix === 'dept' ? 'departments-panel' : prefix === 'file' ? 'files-panel' : 'accounts-panel').querySelector('.table-wrap').scrollTop = 0;
  });
}
wireList('dept', () => { editing = null; renderDepartments(); });
wireList('file', renderFiles);
wireList('account', () => accounts?.render());

function syncSelection() {
  $('department-select').disabled = uploading || !available;
  $('department-select').value = current ? String(current.id) : '';
  $('create').disabled = !available || uploading || creating;
  document.querySelectorAll('[data-needs-department]').forEach((el) => { el.disabled = !current; });
  $('nav-dept-name').textContent = current ? current.name : 'Choose a department';
  $('nav-file-count').textContent = current ? current.fileCount : '—';
  $('nav-account-count').textContent = current ? current.userCount : '—';
  $('department-context').classList.toggle('hidden', !current);
  $('selected-overview').classList.toggle('hidden', !current);
  $('choose-department').classList.toggle('hidden', !!current);
  if (!current) return;
  $('context-name').textContent = current.name;
  $('context-status').className = `chip ${current.status === 'active' ? 'active' : 'disabled'}`;
  $('context-status').textContent = current.status === 'active' ? 'Active' : 'Disabled';
  $('a-link').href = panelLink(current);
  $('overview-chat-link').href = panelLink(current);
  $('overview-dept-name').textContent = current.name;
  $('overview-usage').innerHTML = usageHtml(current);
  $('overview-launcher').textContent = !current.listed ? 'Hidden from the launcher'
    : current.status !== 'active' ? 'Hidden while the department is disabled'
    : 'Shown on the launcher';
  $('overview-file-count').textContent = `${current.fileCount} (${current.readyCount} ready)`;
  $('overview-account-count').textContent = current.userCount;
}

async function restoreLocation() {
  const params = new URLSearchParams(location.hash.slice(1));
  const section = SECTION_NAMES.includes(params.get('section')) ? params.get('section') : 'overview';
  const d = departments.find((item) => String(item.id) === params.get('department'));
  if (uploading && current?.id !== d?.id) { saveLocation(true); return; }
  if (d) await manage(d, section, { writeUrl: false, focus: false });
  else {
    clearSelection();
    showSection(section, { writeUrl: false, focus: false });
  }
}
window.addEventListener('popstate', () => { if (!location.hash || location.hash.startsWith('#section=')) restoreLocation(); });
$('department-select').addEventListener('change', async () => {
  const d = departments.find((item) => String(item.id) === $('department-select').value);
  if (d) await manage(d, ['files', 'accounts'].includes(activeSection) ? activeSection : 'overview');
  else { clearSelection(); showSection('overview'); }
});

function clearSelection() {
  current = null; accounts = null; lastFiles = []; fileRequest++;
  clearTimeout(showFiles.poll);
  syncSelection();
  renderDepartments();
}

// ── Status ───────────────────────────────────────────────────────────────────
async function loadStatus() {
  status = await api('GET', '/api/platform/status');
  const problem = !status.success ? (status.error || 'Departments are not available.')
    : !status.enabled ? 'Departments are managed on the CHiPS bot, not on this one.'
    : !status.ready ? (status.error || 'Departments are not available yet.') : null;
  $('unavailable').classList.toggle('hidden', !problem);
  $('unavailable-text').textContent = problem || '';
  available = !problem;
  syncSelection();
  if (status.success) {
    const types = status.types || [];
    $('type-hint').textContent = `${P.extList(types)} — up to ${fmtBytes(status.maxFileBytes)} each.`
      + (status.virusScan ? ' Every file is virus-scanned.' : '');
    $('file-input').accept = types.flatMap((t) => t.ext).map((e) => '.' + e).join(',');
  }
  return !problem;
}

function renderDiskHint() {
  const given = departments.reduce((s, d) => s + d.quotaBytes, 0);
  const used = departments.reduce((s, d) => s + d.usedBytes, 0);
  let text = departments.length
    ? `Given to ${departments.length} department${departments.length === 1 ? '' : 's'}: ${fmtBytes(given)} (${fmtBytes(used)} used).`
    : '';
  if (status && status.disk) {
    text += ` Free on the server disk: ${fmtBytes(status.disk.freeBytes)}.`;
    if (given - used > status.disk.freeBytes) text += ' More storage has been given out than the disk has free.';
  }
  $('disk-hint').textContent = text.trim();
  $('new-disk-hint').textContent = text.trim();
  $('nav-dept-count').textContent = departments.length;
  $('total-departments').textContent = departments.length;
  $('active-departments').textContent = `${departments.filter((d) => d.status === 'active').length} active`;
  $('total-quota').textContent = fmtBytes(given);
  $('total-used').textContent = `${fmtBytes(used)} used`;
  $('total-files').textContent = departments.reduce((sum, d) => sum + d.fileCount, 0);
  $('total-ready').textContent = `${departments.reduce((sum, d) => sum + d.readyCount, 0)} ready for the chatbot`;
  $('total-accounts').textContent = departments.reduce((sum, d) => sum + d.userCount, 0);
}

// ── Departments ──────────────────────────────────────────────────────────────
function storageCell(d) {
  if (editing !== d.id) return usageHtml(d);
  const p = sizeParts(d.quotaBytes);
  return `<div class="edit-size"><div class="size-input">
      <input type="number" min="1" step="any" value="${p.value}" data-edit-size aria-label="Storage" />
      <select data-edit-unit aria-label="Unit">${Object.keys(UNITS).map((u) =>
        `<option${u === p.unit ? ' selected' : ''}>${u}</option>`).join('')}</select></div>
    <button class="mini" data-act="save-size" data-id="${d.id}" type="button">Save</button>
    <button class="mini" data-act="cancel-size" type="button">Cancel</button></div>`;
}

// The launcher page (the bot list everyone sees first) shows a department's chatbot
// when the CHiPS admin has switched it on and the department is active — processed
// documents or not (until there are some, its chatbot says they are not ready yet).
function launcherCell(d) {
  let state;
  if (!d.listed) state = '<span class="chip disabled">Hidden</span>';
  else if (d.status !== 'active') state = '<span class="chip needs-ocr">On — hidden</span><span class="sub">the department is disabled</span>';
  else if (!d.readyCount) state = '<span class="chip active">Shown</span><span class="sub">no documents processed yet</span>';
  else state = '<span class="chip active">Shown</span>';
  const label = d.listed ? 'Hide from launcher' : 'Show on launcher';
  return `${state}<div style="margin-top:6px"><button class="mini" data-act="listed" data-id="${d.id}" type="button">${label}</button></div>`;
}

function renderDepartments() {
  const tbody = $('dept-rows');
  const query = $('dept-search').value.trim().toLowerCase();
  const state = $('dept-filter').value;
  const filtered = departments.filter((d) => (!state || d.status === state)
    && `${d.name} ${d.slug}`.toLowerCase().includes(query));
  const rows = pageRows('dept', filtered, departments.length);
  renderDiskHint();
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="note">${departments.length ? 'No matching departments.' : 'No departments yet. Use New department to create one.'}</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map((d) => `<tr class="${current && current.id === d.id ? 'sel' : ''}">
      <td><button class="department-name" data-act="overview" data-id="${d.id}" type="button" ${uploading ? 'disabled' : ''}>${esc(d.name)}</button><span class="sub">${esc(d.slug)}</span></td>
      <td>${storageCell(d)}</td>
      <td>${d.fileCount}</td>
      <td>${d.userCount}</td>
      <td><span class="chip ${esc(d.status)}">${d.status === 'active' ? 'Active' : 'Disabled'}</span></td>
      <td>${launcherCell(d)}</td>
      <td><div class="row-actions">
        <button class="mini" data-act="manage" data-id="${d.id}" type="button" ${uploading ? 'disabled' : ''}>Files</button>
        <button class="mini" data-act="accounts" data-id="${d.id}" type="button" ${uploading ? 'disabled' : ''}>Accounts</button>
        <details class="row-more"><summary class="mini">More</summary><div class="row-menu">
        <button class="mini" data-act="edit-size" data-id="${d.id}" type="button">Change storage</button>
        <button class="mini" data-act="toggle" data-id="${d.id}" type="button">${d.status === 'active' ? 'Disable' : 'Enable'}</button>
        <button class="mini danger" data-act="delete" data-id="${d.id}" type="button"
          ${d.usedBytes > 0 ? 'disabled title="Delete its files first"' : ''}>Delete</button>
        </div></details>
      </div></td></tr>`).join('');
}

async function loadDepartments() {
  const r = await api('GET', '/api/platform/departments');
  if (!r.success) {
    $('dept-rows').innerHTML = `<tr><td colspan="7" class="note">${esc(r.error || 'Could not load departments.')}</td></tr>`;
    return;
  }
  departments = r.departments;
  if (current) {
    const selected = departments.find((d) => d.id === current.id);
    if (selected) current = selected;
    else {
      clearSelection();
      if (['files', 'accounts'].includes(activeSection)) showSection('departments', { replace: true });
    }
  }
  $('department-select').innerHTML = '<option value="">Choose a department</option>'
    + departments.map((d) => `<option value="${d.id}">${esc(d.name)}</option>`).join('');
  renderDepartments();
  syncSelection();
  if (current) renderUsage();
}

// Short name follows the name until it is typed in by hand.
let slugTouched = false;
function slugify(s) {
  return String(s || '').toLowerCase().normalize('NFKD').replace(/[^\x00-\x7f]/g, '')
    .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 31).replace(/-+$/, '');
}
$('d-name').addEventListener('input', () => { if (!slugTouched) $('d-slug').value = slugify($('d-name').value); });
$('d-slug').addEventListener('input', () => { slugTouched = $('d-slug').value.trim() !== ''; });

$('create').addEventListener('click', async () => {
  const name = $('d-name').value.trim();
  const slug = $('d-slug').value.trim();
  const size = Number($('d-size').value);
  if (name.length < 2) return toast('Enter the department’s name.');
  if (!slug) return toast('Give a short name in English letters (for example “health”).');
  if (!(size > 0)) return toast('Enter the storage for this department.');
  const quotaBytes = Math.round(size * UNITS[$('d-unit').value]);
  if (uploading || creating || !available) return;
  creating = true; syncSelection();
  try {
    const r = await api('POST', '/api/platform/departments', { name, slug, quotaBytes });
    if (!r.success) return toast(r.error || 'Could not create the department.', 6000);
    $('d-name').value = ''; $('d-slug').value = ''; $('d-size').value = ''; slugTouched = false;
    toast(`Created “${r.department.name}” with ${fmtBytes(r.department.quotaBytes)}. Add its first admin account.`, 5000);
    await loadDepartments();
    await manage(departments.find((d) => d.id === r.department.id), 'accounts');
    $('account-create').open = true;
    $('a-form').querySelector('[name="username"]').focus({ preventScroll: true });
  } finally {
    creating = false; syncSelection();
  }
});

$('dept-rows').addEventListener('click', async (e) => {
  const btn = e.target.closest('button[data-act]');
  if (!btn) return;
  const id = Number(btn.dataset.id);
  const d = departments.find((x) => x.id === id);
  switch (btn.dataset.act) {
    case 'manage':
      await manage(d);
      break;
    case 'accounts':
      await manage(d, 'accounts');
      break;
    case 'overview':
      await manage(d, 'overview');
      break;
    case 'edit-size':
      editing = id; renderDepartments();
      break;
    case 'cancel-size':
      editing = null; renderDepartments();
      break;
    case 'save-size': {
      const row = btn.closest('tr');
      const size = Number(row.querySelector('[data-edit-size]').value);
      const unit = row.querySelector('[data-edit-unit]').value;
      if (!(size > 0)) return toast('Enter the storage for this department.');
      const r = await api('PATCH', `/api/platform/departments/${id}`, { quotaBytes: Math.round(size * UNITS[unit]) });
      if (!r.success) {
        return toast(r.usedBytes !== undefined
          ? `It already uses ${fmtBytes(r.usedBytes)} — give at least that, or delete files first.`
          : (r.error || 'Could not change the storage.'), 6000);
      }
      editing = null;
      toast(`${r.department.name}: storage is now ${fmtBytes(r.department.quotaBytes)}.`);
      await loadDepartments();
      break;
    }
    case 'toggle': {
      const next = d.status === 'active' ? 'disabled' : 'active';
      if (next === 'disabled' && !confirm(`Disable “${d.name}”? No files can be added to it while it is disabled.`)) return;
      const r = await api('PATCH', `/api/platform/departments/${id}`, { status: next });
      if (!r.success) return toast(r.error || 'Could not change the department.', 6000);
      await loadDepartments();
      break;
    }
    case 'listed': {
      const show = !d.listed;
      if (show && !confirm(`Show “${d.name}” on the launcher page? Everyone who opens it will see this assistant; `
        + 'using it still needs one of the department’s accounts.')) return;
      const r = await api('PATCH', `/api/platform/departments/${id}`, { listed: show });
      if (!r.success) return toast(r.error || 'Could not change the launcher setting.', 6000);
      const dd = r.department;
      let msg = `“${dd.name}” is now on the launcher.`;
      if (!show) msg = `“${dd.name}” is no longer on the launcher.`;
      else if (dd.status !== 'active') msg = `“${dd.name}” will show on the launcher once it is enabled.`;
      else if (!dd.readyCount) msg = `“${dd.name}” is now on the launcher. No documents are processed yet — its chatbot says so until they are.`;
      toast(msg, 5000);
      await loadDepartments();
      break;
    }
    case 'delete': {
      if (!confirm(`Delete the department “${d.name}” and its accounts? This cannot be undone.`)) return;
      const r = await api('DELETE', `/api/platform/departments/${id}`);
      if (!r.success) return toast(r.error || 'Could not delete the department.', 6000);
      if (current && current.id === id) {
        clearSelection();
        if (['files', 'accounts'].includes(activeSection)) showSection('departments', { replace: true });
        else saveLocation(true);
      }
      toast(`Deleted “${d.name}”.`);
      await loadDepartments();
      break;
    }
  }
});

// ── One department: accounts and files ───────────────────────────────────────
function panelLink(d) {
  return `${location.origin}${BASE}/dept/${encodeURIComponent(d.slug)}/`;
}

async function manage(d, section = 'files', options) {
  if (!d) return;
  if (uploading && current?.id !== d.id) return toast('Wait for the current uploads to finish before changing department.');
  if (current?.id !== d.id) {
    clearTimeout(showFiles.poll);
    fileRequest++;
    lastFiles = [];
    for (const prefix of ['file', 'account']) {
      pages[prefix] = 1;
      $(prefix + '-search').value = ''; $(prefix + '-filter').value = '';
    }
    $('uploads').replaceChildren();
    $('upload-files').open = false; $('account-create').open = false;
    $('file-rows').innerHTML = '<tr><td colspan="6" class="note">Loading files…</td></tr>';
    $('process-new').disabled = true;
    $('process-new').textContent = 'Process new files';
    $('process-hint').textContent = '';
    for (const prefix of ['file', 'account']) {
      pageRows(prefix, []);
      $(prefix + '-range').textContent = '';
    }
  }
  current = d;
  renderDepartments();
  renderUsage();
  showSection(section, options);
  if (activeSection === 'accounts') await loadAccounts(d);
  else if (activeSection === 'files') await showFiles();
}

async function loadAccounts(d) {
  $('a-title').textContent = `Accounts — ${d.name}`;
  if (!accounts || accounts.departmentId !== d.id) {
    // A fresh panel per department (its form keeps no state of another).
    const tbody = $('a-rows');
    const fresh = tbody.cloneNode(false);
    tbody.replaceWith(fresh);
    const form = $('a-form');
    const freshForm = form.cloneNode(true);
    freshForm.querySelectorAll('input').forEach((input) => { input.value = ''; });
    form.replaceWith(freshForm);
    fresh.innerHTML = '<tr><td colspan="5" class="note">Loading accounts…</td></tr>';
    accounts = P.accountsPanel({
      api, usersPath: `/api/platform/departments/${d.id}/users`, form: freshForm, tbody: fresh, toast,
      defaultRole: 'admin', onChange: loadDepartments,
      selectUsers: (users) => {
        if (current?.id !== d.id) return [];
        const query = $('account-search').value.trim().toLowerCase();
        const role = $('account-filter').value;
        return pageRows('account', users.filter((u) => (!role || u.role === role)
          && u.username.toLowerCase().includes(query)), users.length);
      },
    });
    accounts.departmentId = d.id;
  }
  await accounts.load();
}

$('a-copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('a-link').href); toast('Link copied.'); }
  catch (_) { toast('Right-click Open department chatbot and copy its link.'); }
});

function renderUsage() {
  if (!current) return;
  syncSelection();
  $('f-title').textContent = `Files — ${current.name}`;
  $('f-usage').innerHTML = usageHtml(current);
}

async function showFiles() {
  if (!current) return;
  const id = current.id;
  const request = ++fileRequest;
  clearTimeout(showFiles.poll);
  renderUsage();
  const r = await api('GET', `/api/platform/departments/${id}/files`);
  if (current?.id !== id || request !== fileRequest) return;
  if (!r.success) {
    $('file-rows').innerHTML = `<tr><td colspan="6" class="note">${esc(r.error || 'Could not load the files.')}</td></tr>`;
    return;
  }
  current = r.department;
  renderUsage();
  const files = r.files.filter((f) => f.status === 'uploaded');
  lastFiles = files;
  renderFiles();
  const search = status.search || {};
  const fresh = P.newCount(files);
  $('process-new').disabled = !search.ready || !fresh;
  $('process-new').textContent = fresh ? `Process new files (${fresh})` : 'Process new files';
  $('process-hint').textContent = !search.ready
    ? (search.error || 'Processing is not available.')
    : `${current.readyCount} of ${files.length} file${files.length === 1 ? '' : 's'} ready for the chatbot.`;
  // While files are being processed, look again every few seconds.
  if (P.anyBusy(files)) {
    const id = current.id;
    showFiles.poll = setTimeout(async () => {
      if (current && current.id === id) { await showFiles(); if (!P.anyBusy(lastFiles)) await loadDepartments(); }
    }, 3000);
  }
}

function renderFiles() {
  const query = $('file-search').value.trim().toLowerCase();
  const state = $('file-filter').value;
  const filtered = lastFiles.filter((f) => (!state || f.processing === state) && f.name.toLowerCase().includes(query));
  const visible = pageRows('file', filtered, lastFiles.length);
  const labels = Object.fromEntries((status?.types || []).map((t) => [t.kind, t.label]));
  $('file-rows').innerHTML = !visible.length && lastFiles.length
    ? '<tr><td colspan="6" class="note">No matching files.</td></tr>'
    : P.fileRows(visible, { labels, href: (f) => `${BASE}/api/platform/files/${encodeURIComponent(f.id)}/download` });
}

async function processFiles(fileIds) {
  if (!current) return;
  const r = await api('POST', `/api/platform/departments/${current.id}/process`, fileIds ? { fileIds } : {});
  if (!r.success) return toast(r.error || 'Could not start processing.', 6000);
  toast(r.queued ? `${r.queued} file${r.queued === 1 ? '' : 's'} queued for processing.` : 'Nothing new to process.');
  await showFiles();
}
$('process-new').addEventListener('click', () => processFiles(null));

$('file-rows').addEventListener('click', async (e) => {
  const proc = e.target.closest('button[data-process]');
  if (proc) return processFiles([proc.dataset.process]);
  const btn = e.target.closest('button[data-del]');
  if (!btn) return;
  if (!confirm(`Delete “${btn.dataset.name}”? Its storage becomes free again.`)) return;
  const r = await api('DELETE', `/api/platform/files/${encodeURIComponent(btn.dataset.del)}`);
  if (!r.success) return toast(r.error || 'Could not delete the file.', 6000);
  await showFiles();
  await loadDepartments();
});

async function uploadAll(files) {
  if (!current) return;
  if (uploading) return toast('Wait for the current uploads to finish.');
  const dept = current;
  if (dept.status !== 'active') return toast('This department is disabled; enable it to add files.', 5000);
  uploading = true; syncSelection(); renderDepartments();
  $('upload-files').open = true;
  try {
    for (const file of files) {
      await P.uploadOne({
        api, file, status, container: $('uploads'),
        startPath: `/api/platform/departments/${dept.id}/files`, filesPath: '/api/platform/files',
      });
      if (current && current.id === dept.id) await showFiles();
    }
    await loadDepartments();
  } finally {
    uploading = false; syncSelection(); renderDepartments();
  }
}

P.wireDrop({ drop: $('drop'), pick: $('pick'), input: $('file-input'), onFiles: uploadAll });
$('refresh').addEventListener('click', async () => {
  $('refresh').disabled = true;
  $('refresh-status').textContent = 'Refreshing…';
  try {
    if (await loadStatus()) {
      await loadDepartments();
      if (current && activeSection === 'files') await showFiles();
      if (current && activeSection === 'accounts') await loadAccounts(current);
      $('refresh-status').textContent = 'Data refreshed';
    } else $('refresh-status').textContent = 'Unavailable';
  } finally { $('refresh').disabled = false; }
});

(async () => {
  if (await loadStatus()) { await loadDepartments(); await restoreLocation(); }
  else $('dept-rows').innerHTML = '<tr><td colspan="7" class="note">—</td></tr>';
})();
