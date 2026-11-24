'use strict';

// Shared by the platform pages — the CHiPS admin's Departments page and the
// department panel: formatting, requests, uploads in pieces, and the accounts list.
//
// Request bodies go base64-encoded ({b64}) and file contents in base64 pieces: the
// office firewall in front of the bots refuses bodies holding tag-like text.

(function () {
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function fmtDate(iso) {
    if (!iso) return '';
    try { return new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }); }
    catch { return iso; }
  }

  const UNITS = { MB: 1024 ** 2, GB: 1024 ** 3, TB: 1024 ** 4 };
  function fmtBytes(n) {
    n = Number(n) || 0;
    if (n < 1024) return `${n} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let v = n / 1024; let i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return `${v >= 100 ? Math.round(v) : v.toFixed(1).replace(/\.0$/, '')} ${units[i]}`;
  }
  function sizeParts(bytes) {     // largest unit that gives a whole-ish number
    for (const u of ['TB', 'GB', 'MB']) {
      const v = bytes / UNITS[u];
      if (v >= 1 && Math.abs(v * 100 - Math.round(v * 100)) < 1e-6) return { value: +v.toFixed(2), unit: u };
    }
    return { value: +(bytes / UNITS.MB).toFixed(2), unit: 'MB' };
  }

  function toBase64(bytes) {
    let bin = '';
    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(bin);
  }

  function makeToast(el) {
    return (msg, ms = 3500) => {
      el.textContent = msg; el.classList.add('show');
      clearTimeout(el.timer);
      el.timer = setTimeout(() => el.classList.remove('show'), ms);
    };
  }

  /** api(method, path, body, {raw}) → JSON (+ httpStatus); a network failure → {network:true}. */
  function makeApi(prefix, { onUnauth, onForbidden } = {}) {
    return async function api(method, path, body, { raw = false } = {}) {
      const opts = { method, headers: {}, credentials: 'same-origin' };
      if (body !== undefined) {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(raw ? body : { b64: toBase64(new TextEncoder().encode(JSON.stringify(body))) });
      }
      let res;
      try { res = await fetch(prefix + path, opts); }
      catch (_) { return { success: false, network: true, error: 'The server could not be reached.' }; }
      if (res.status === 401 && onUnauth) { onUnauth(); throw new Error('unauth'); }
      if (res.status === 403 && onForbidden) { onForbidden(); throw new Error('forbidden'); }
      const data = await res.json().catch(() => ({ success: false, error: `HTTP ${res.status}` }));
      data.httpStatus = res.status;
      return data;
    };
  }

  // ── Uploads ────────────────────────────────────────────────────────────────
  function allowedExt(name, types) {
    const m = /\.([a-z0-9]{1,5})$/i.exec(name);
    return !!m && types.some((t) => t.ext.includes(m[1].toLowerCase()));
  }
  const extList = (types) => types.flatMap((t) => t.ext).map((e) => '.' + e).join(', ');

  function uploadRow(container, file) {
    const el = document.createElement('div');
    el.className = 'upload';
    el.innerHTML = `<div class="name" title="${esc(file.name)}"><b>${esc(file.name)}</b> · ${esc(fmtBytes(file.size))}</div>
      <div class="bar"><i style="width:0"></i></div>
      <button class="mini" type="button">Cancel</button>
      <div class="msg hidden"></div>`;
    container.prepend(el);
    const btn = el.querySelector('button');
    const done = () => { btn.disabled = false; btn.textContent = 'Dismiss'; btn.onclick = () => el.remove(); };
    const row = {
      cancelled: false,
      progress(p) { el.querySelector('.bar i').style.width = `${(p * 100).toFixed(1)}%`; },
      fail(msg) {
        el.querySelector('.msg').textContent = msg;
        el.querySelector('.msg').classList.remove('hidden');
        el.querySelector('.bar i').classList.add('full');
        done();
      },
      ok() {
        el.classList.add('ok');
        el.querySelector('.msg').textContent = 'Uploaded.';
        el.querySelector('.msg').classList.remove('hidden');
        done();
        setTimeout(() => el.remove(), 4000);
      },
      gone() { el.remove(); },
    };
    btn.onclick = () => { row.cancelled = true; btn.disabled = true; };
    return row;
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  /**
   * Upload one file: start (POST startPath), pieces (PUT filesPath/<id>/chunk),
   * finish. Anything that goes wrong cancels the upload so it holds no storage.
   */
  async function uploadOne({ api, file, startPath, filesPath, status, container }) {
    const row = uploadRow(container, file);
    const types = status.types || [];
    if (!allowedExt(file.name, types)) return row.fail(`This type of file cannot be uploaded. Allowed: ${extList(types)}.`);
    if (!file.size) return row.fail('The file is empty.');
    if (file.size > status.maxFileBytes) return row.fail(`The file is larger than ${fmtBytes(status.maxFileBytes)}.`);

    const start = await api('POST', startPath, { name: file.name, size: file.size });
    if (!start.success) {
      return row.fail(start.freeBytes !== undefined
        ? `Not enough storage left: ${fmtBytes(start.freeBytes)} free, the file is ${fmtBytes(file.size)}.`
        : (start.error || 'The upload could not start.'));
    }
    const at = `${filesPath}/${encodeURIComponent(start.file.id)}`;
    const cancel = () => api('DELETE', at);
    uploadOne.active.add(at);
    try {
      let offset = 0;
      while (offset < file.size) {
        if (row.cancelled) { await cancel(); return row.gone(); }
        const piece = new Uint8Array(await file.slice(offset, offset + start.chunkBytes).arrayBuffer());
        let r;
        for (let attempt = 1; ; attempt++) {
          r = await api('PUT', `${at}/chunk`, { offset, data: toBase64(piece) }, { raw: true });
          if (r.success || r.received !== undefined) break;
          if ((r.network || r.httpStatus >= 502) && attempt < 4) { await sleep(2000 * attempt); continue; }
          break;
        }
        if (!r.success && r.received === undefined) {
          await cancel();                // frees its storage (harmless if already gone)
          return row.fail(r.error || 'The upload failed.');
        }
        offset = r.received;
        row.progress(offset / file.size);
      }
      const fin = await api('POST', `${at}/finish`);
      if (!fin.success) {
        await cancel();
        return row.fail(fin.error || 'The file could not be kept.');
      }
      row.ok();
      return fin.file;
    } finally {
      uploadOne.active.delete(at);
    }
  }
  uploadOne.active = new Set();
  window.addEventListener('beforeunload', (e) => {
    if (uploadOne.active.size) { e.preventDefault(); e.returnValue = ''; }
  });

  /** Wire a drop zone + "Add files…" button to an upload function taking File[]. */
  function wireDrop({ drop, pick, input, onFiles }) {
    pick.addEventListener('click', () => input.click());
    input.addEventListener('change', () => { const files = [...input.files]; input.value = ''; onFiles(files); });
    ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', (e) => onFiles([...((e.dataTransfer && e.dataTransfer.files) || [])]));
  }

  function usageHtml(d) {
    const pct = d.quotaBytes ? Math.min(100, (d.usedBytes / d.quotaBytes) * 100) : 0;
    return `<div class="usage"><div class="bar"><i class="${pct >= 95 ? 'full' : ''}" style="width:${pct.toFixed(1)}%"></i></div>
      <small>${esc(fmtBytes(d.usedBytes))} of ${esc(fmtBytes(d.quotaBytes))}</small></div>`;
  }

  // ── Accounts ───────────────────────────────────────────────────────────────
  /**
   * The accounts of one department: a form to add one and a table to block,
   * change role, reset the password or delete. `usersPath` is the list's URL;
   * `me` (the department panel) marks the signed-in account, whose own row has no
   * actions (the server refuses them too).
   * Optional `selectUsers` controls display-only filtering/paging. Actions still
   * resolve against the full account list and use the existing API checks.
   */
  function accountsPanel({ api, usersPath, form, tbody, toast, defaultRole = 'user', onChange, selectUsers }) {
    let me = null;
    let resetting = null;
    let users = [];

    function render() {
      const visible = selectUsers ? selectUsers(users) : users;
      if (!users.length) {
        tbody.innerHTML = '<tr><td colspan="5" class="note">No accounts yet.</td></tr>';
        return;
      }
      if (!visible.length) {
        tbody.innerHTML = '<tr><td colspan="5" class="note">No matching accounts.</td></tr>';
        return;
      }
      tbody.innerHTML = visible.map((u) => {
        const self = me === u.id;
        const actions = self ? '<span class="hint">your account</span>'
          : resetting === u.id
            ? `<div class="edit-size"><input type="password" data-newpw placeholder="New password" autocomplete="new-password"
                 style="padding:6px 9px;border:1px solid var(--border);border-radius:var(--radius);font-size:13px;width:170px" />
               <button class="mini" data-act="save-pw" data-id="${u.id}" type="button">Save</button>
               <button class="mini" data-act="cancel-pw" type="button">Cancel</button></div>`
            : `<div class="row-actions">
                <button class="mini" data-act="status" data-id="${u.id}" type="button">${u.status === 'active' ? 'Block' : 'Unblock'}</button>
                <button class="mini" data-act="role" data-id="${u.id}" type="button">${u.role === 'admin' ? 'Make user' : 'Make admin'}</button>
                <button class="mini" data-act="reset" data-id="${u.id}" type="button">Reset password</button>
                <button class="mini danger" data-act="delete" data-id="${u.id}" type="button">Delete</button></div>`;
        return `<tr>
          <td><b>${esc(u.username)}</b><span class="sub">added by ${esc(u.createdBy || '—')}</span></td>
          <td><span class="chip ${esc(u.role)}">${u.role === 'admin' ? 'Admin' : 'User'}</span></td>
          <td><span class="chip ${esc(u.status)}">${u.status === 'active' ? 'Active' : 'Blocked'}</span></td>
          <td>${u.lastLoginAt ? esc(fmtDate(u.lastLoginAt)) : '<span class="hint">never</span>'}</td>
          <td>${actions}</td></tr>`;
      }).join('');
    }

    async function load() {
      const r = await api('GET', usersPath);
      if (!r.success) {
        tbody.innerHTML = `<tr><td colspan="5" class="note">${esc(r.error || 'Could not load the accounts.')}</td></tr>`;
        return;
      }
      users = r.users;
      if (r.me !== undefined) me = r.me;
      render();
    }

    const field = (name) => form.querySelector(`[name="${name}"]`);
    field('role').value = defaultRole;
    form.querySelector('[data-add]').addEventListener('click', async () => {
      const body = { username: field('username').value.trim(), password: field('password').value, role: field('role').value };
      if (!body.username || !body.password) return toast('Enter a username and a password.');
      const r = await api('POST', usersPath, body);
      if (!r.success) return toast(r.error || 'Could not add the account.', 6000);
      field('username').value = ''; field('password').value = ''; field('role').value = defaultRole;
      toast(`Added ${r.user.role === 'admin' ? 'admin' : 'user'} “${r.user.username}”. Give them the password yourself.`, 5000);
      await load();
      if (onChange) onChange();
    });

    tbody.addEventListener('click', async (e) => {
      const btn = e.target.closest('button[data-act]');
      if (!btn) return;
      const id = Number(btn.dataset.id);
      const u = users.find((x) => x.id === id);
      const at = `${usersPath}/${id}`;
      let r;
      switch (btn.dataset.act) {
        case 'status':
          if (u.status === 'active' && !confirm(`Block “${u.username}”? They are signed out at once and cannot sign in until unblocked.`)) return;
          r = await api('PATCH', at, { status: u.status === 'active' ? 'blocked' : 'active' });
          break;
        case 'role':
          r = await api('PATCH', at, { role: u.role === 'admin' ? 'user' : 'admin' });
          break;
        case 'reset': resetting = id; render(); tbody.querySelector('[data-newpw]').focus(); return;
        case 'cancel-pw': resetting = null; render(); return;
        case 'save-pw': {
          const pw = btn.closest('td').querySelector('[data-newpw]').value;
          r = await api('PATCH', at, { password: pw });
          if (r.success) { resetting = null; toast(`New password set for “${u.username}”; their other sessions are signed out.`, 5000); }
          break;
        }
        case 'delete':
          if (!confirm(`Delete the account “${u.username}”? This cannot be undone.`)) return;
          r = await api('DELETE', at);
          break;
        default: return;
      }
      if (!r.success) return toast(r.error || 'Could not change the account.', 6000);
      await load();
      if (onChange) onChange();
    });

    return { load, render };
  }

  // ── Processing (files → the chatbot's passages) ────────────────────────────
  const BUSY = new Set(['queued', 'processing']);
  function processingCell(f) {
    const reason = f.processingError ? `<span class="sub">${esc(f.processingError)}</span>` : '';
    switch (f.processing) {
      case 'done': return `<span class="chip uploaded">Ready</span><span class="sub">${f.chunkCount} passage${f.chunkCount === 1 ? '' : 's'}</span>`;
      case 'queued': return '<span class="chip uploading">Waiting…</span>';
      case 'processing': return '<span class="chip uploading">Processing…</span>';
      case 'needs_ocr': return `<span class="chip needs-ocr">Needs OCR</span>${reason}`;
      case 'failed': return `<span class="chip error">Failed</span>${reason}`;
      default: return '<span class="chip not_started">Not processed</span>';
    }
  }
  function processButton(f) {
    if (BUSY.has(f.processing) || f.processing === 'needs_ocr') return '';
    const label = f.processing === 'done' ? 'Process again' : f.processing === 'failed' ? 'Try again' : 'Process';
    return `<button class="mini" data-process="${esc(f.id)}" type="button">${label}</button>`;
  }
  /** The file table's rows, the same on the CHiPS page and the department panel. */
  function fileRows(files, { labels, href }) {
    if (!files.length) return '<tr><td colspan="6" class="note">No files yet.</td></tr>';
    return files.map((f) => `<tr>
      <td><b>${esc(f.name)}</b></td>
      <td>${esc(labels[f.kind] || f.kind)}${f.pageCount ? `<span class="sub">${f.pageCount} page${f.pageCount === 1 ? '' : 's'}</span>` : ''}</td>
      <td>${esc(fmtBytes(f.sizeBytes))}</td>
      <td>${esc(fmtDate(f.uploadedAt))}<span class="sub">${esc(f.uploadedBy)}</span></td>
      <td>${processingCell(f)}</td>
      <td><div class="row-actions">
        ${processButton(f)}
        <a class="mini" href="${href(f)}">Download</a>
        <button class="mini danger" data-del="${esc(f.id)}" data-name="${esc(f.name)}" type="button">Delete</button>
      </div></td></tr>`).join('');
  }
  const anyBusy = (files) => files.some((f) => BUSY.has(f.processing));
  const newCount = (files) => files.filter((f) => f.processing === 'not_started').length;

  // ── Answers ────────────────────────────────────────────────────────────────
  // Compatibility export: chat rendering has one implementation, shared with the bots.
  const renderAnswer = raw => window.ChatUI.renderAnswer(raw);

  window.Platform = {
    esc, fmtDate, fmtBytes, sizeParts, UNITS, toBase64, makeToast, makeApi,
    uploadOne, wireDrop, usageHtml, extList, accountsPanel,
    processingCell, fileRows, anyBusy, newCount, renderAnswer,
  };
})();
