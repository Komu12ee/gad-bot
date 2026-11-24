'use strict';

// Department panel (/dept/<short name>/). Every account asks the department's
// chatbot (Ask); admins also add, process and delete documents and keep the
// accounts. Every request is relative (api/…) and answered only for the signed-in
// department.

(function () {
  const P = window.Platform;
  const $ = (id) => document.getElementById(id);
  const { esc, fmtBytes, usageHtml } = P;
  const toast = P.makeToast($('toast'));
  const api = P.makeApi('', { onUnauth: () => { window.location.href = 'login'; } });

  let me = null;
  let department = null;
  let status = null;     // file types and limits, from the files list
  let accounts = null;

  // ── Tabs ───────────────────────────────────────────────────────────────────
  let chat = null;
  function closeNavigation() {
    document.body.classList.remove('department-nav-open');
    $('dept-menu').setAttribute('aria-expanded', 'false');
    $('dept-backdrop').classList.add('hidden');
  }
  function selectPane(name) {
    document.querySelectorAll('[data-tab]').forEach(btn => btn.classList.toggle('on', btn.dataset.tab === name));
    document.querySelectorAll('[data-pane]').forEach(pane => pane.classList.toggle('hidden', pane.dataset.pane !== name));
    if (name === 'accounts' && accounts) accounts.load();
    if (name === 'files') loadFiles();
    if (name === 'ask') chat?.focus();
    closeNavigation();
  }
  document.querySelectorAll('[data-tab]').forEach(btn => btn.addEventListener('click', () => selectPane(btn.dataset.tab)));
  $('dept-menu').addEventListener('click', () => {
    const open = document.body.classList.toggle('department-nav-open');
    $('dept-menu').setAttribute('aria-expanded', String(open));
    $('dept-backdrop').classList.toggle('hidden', !open);
  });
  $('dept-backdrop').addEventListener('click', closeNavigation);
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeNavigation(); });

  // ── Ask ────────────────────────────────────────────────────────────────────
  // ── Documents (admins) ─────────────────────────────────────────────────────
  let poll = null;
  async function loadFiles() {
    if (!me || me.role !== 'admin') return;
    const r = await api('GET', 'api/files');
    if (!r.success) {
      $('file-rows').innerHTML = `<tr><td colspan="6" class="note">${esc(r.error || 'Could not load the documents.')}</td></tr>`;
      return;
    }
    department = r.department;
    status = r.limits;
    $('f-usage').innerHTML = usageHtml(department);
    $('disabled-note').classList.toggle('hidden', department.status === 'active');
    $('type-hint').textContent = `${P.extList(status.types)} — up to ${fmtBytes(status.maxFileBytes)} each.`;
    $('file-input').accept = status.types.flatMap((t) => t.ext).map((e) => '.' + e).join(',');
    const labels = Object.fromEntries(status.types.map((t) => [t.kind, t.label]));
    const files = r.files.filter((f) => f.status === 'uploaded');
    $('file-rows').innerHTML = P.fileRows(files, { labels, href: (f) => `api/files/${encodeURIComponent(f.id)}/download` });
    const search = r.search || {};
    const fresh = P.newCount(files);
    $('process-new').disabled = !search.ready || !fresh;
    $('process-new').textContent = fresh ? `Process new files (${fresh})` : 'Process new files';
    $('process-hint').textContent = !search.ready
      ? (search.error || 'Processing is not available yet.')
      : `${department.readyCount} of ${files.length} document${files.length === 1 ? '' : 's'} ready for the chatbot.`;
    clearTimeout(poll);
    if (P.anyBusy(files)) poll = setTimeout(loadFiles, 3000);    // while processing, look again
  }

  async function processFiles(fileIds) {
    const r = await api('POST', 'api/process', fileIds ? { fileIds } : {});
    if (!r.success) return toast(r.error || 'Could not start processing.', 6000);
    toast(r.queued ? `${r.queued} document${r.queued === 1 ? '' : 's'} queued for processing.` : 'Nothing new to process.');
    await loadFiles();
  }
  $('process-new').addEventListener('click', () => processFiles(null));

  $('file-rows').addEventListener('click', async (e) => {
    const proc = e.target.closest('button[data-process]');
    if (proc) return processFiles([proc.dataset.process]);
    const btn = e.target.closest('button[data-del]');
    if (!btn) return;
    if (!confirm(`Delete “${btn.dataset.name}”? Its storage becomes free again and the chatbot stops using it.`)) return;
    const r = await api('DELETE', `api/files/${encodeURIComponent(btn.dataset.del)}`);
    if (!r.success) return toast(r.error || 'Could not delete the document.', 6000);
    await loadFiles();
  });

  async function uploadAll(files) {
    if (!department || !status) return;
    if (department.status !== 'active') return toast('CHiPS has switched this department off: files cannot be added for now.', 5000);
    for (const file of files) {
      await P.uploadOne({ api, file, status, container: $('uploads'), startPath: 'api/files', filesPath: 'api/files' });
      await loadFiles();
    }
  }
  P.wireDrop({ drop: $('drop'), pick: $('pick'), input: $('file-input'), onFiles: uploadAll });

  // ── My password ────────────────────────────────────────────────────────────
  $('pw-save').addEventListener('click', async () => {
    const current = $('pw-current').value;
    const next = $('pw-next').value;
    if (!current || !next) return toast('Enter your current and your new password.');
    if (next !== $('pw-again').value) return toast('The two new passwords are not the same.');
    const r = await api('POST', 'api/password', { current, next });
    if (!r.success) return toast(r.error || 'Could not change the password.', 6000);
    ['pw-current', 'pw-next', 'pw-again'].forEach((id) => { $(id).value = ''; });
    toast('Your password is changed.');
  });

  $('btn-logout').addEventListener('click', async () => {
    await api('POST', 'api/logout');
    window.location.href = 'login';
  });

  // ── Start ──────────────────────────────────────────────────────────────────
  (async () => {
    const r = await api('GET', 'api/session');
    if (!r.success) { window.location.href = 'login'; return; }
    me = r.user;
    department = r.department;
    $('dept-name').textContent = department.name;
    $('mobile-title').textContent = department.name;
    document.title = `${department.name} · assistant`;
    $('who').textContent = me.username;
    $('disabled-note').classList.toggle('hidden', department.status === 'active');
    if (me.role !== 'admin') document.querySelectorAll('[data-admin]').forEach((el) => el.remove());
    else {
      document.querySelectorAll('#sidebar [data-admin]').forEach(el => el.classList.remove('hidden'));
      accounts = P.accountsPanel({ api, usersPath: 'api/users', form: $('a-form'), tbody: $('a-rows'), toast });
    }
    $('dept-loading').classList.add('hidden');
    $('app-shell').classList.remove('hidden');
    chat = await window.DepartmentChat.create({ container: $('dept-chat-surface'), department, user: me,
      api, toast, onSelect: () => selectPane('ask') });
    chat.focus();
  })();
})();
