'use strict';

// GAD account-management panel. Every request rides the same-origin gad_session
// cookie; the server enforces admin-only access (a plain user is redirected/403'd).

const $ = (id) => document.getElementById(id);
const toast = (msg) => {
  const t = $('toast'); t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2600);
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
  return res.json().catch(() => ({}));
}

let me = null; // current admin's username (so we can disable self-delete)

async function loadMe() {
  try {
    const s = await api('GET', '/gad-auth/session');
    me = s.authenticated ? s.user.username : null;
  } catch (_) { /* handled by api() */ }
}

async function loadUsers() {
  const rows = $('user-rows');
  let data;
  try { data = await api('GET', '/gad-auth/users'); }
  catch (_) { return; }
  const users = (data.users || []).sort((a, b) =>
    (a.role === b.role ? a.username.localeCompare(b.username) : a.role === 'admin' ? -1 : 1));
  if (!users.length) { rows.innerHTML = '<tr><td colspan="6" class="note">No accounts yet.</td></tr>'; return; }
  rows.innerHTML = users.map((u) => {
    const isSelf = u.username === me;
    const dash = '<span class="note" style="padding:0">—</span>';
    return `<tr>
      <td>${esc(u.username)}${isSelf ? ' <span class="note" style="padding:0">(you)</span>' : ''}</td>
      <td><span class="role ${u.role}">${esc(u.role)}</span></td>
      <td>${u.email ? esc(u.email) : dash}</td>
      <td>${u.mobile ? esc(u.mobile) : dash}</td>
      <td>${esc(fmtDate(u.createdAt))}</td>
      <td><div class="row-actions">
        <button class="mini" data-act="contact" data-user="${esc(u.username)}" data-email="${esc(u.email || '')}" data-mobile="${esc(u.mobile || '')}">Edit contact</button>
        <button class="mini" data-act="password" data-user="${esc(u.username)}">Reset password</button>
        <button class="mini" data-act="role" data-user="${esc(u.username)}" data-role="${u.role}">Make ${u.role === 'admin' ? 'user' : 'admin'}</button>
        <button class="mini danger" data-act="delete" data-user="${esc(u.username)}" ${isSelf ? 'disabled title="You cannot delete your own account"' : ''}>Delete</button>
      </div></td>
    </tr>`;
  }).join('');
}

// ── Create ────────────────────────────────────────────────────────────────────
$('create-btn').addEventListener('click', async () => {
  const username = $('new-username').value.trim();
  const password = $('new-password').value;
  const role = $('new-role').value;
  const email = $('new-email').value.trim();
  const mobile = $('new-mobile').value.trim();
  if (!username || !password) { toast('Username and password are required.'); return; }
  const btn = $('create-btn'); btn.disabled = true;
  const res = await api('POST', '/gad-auth/users', { username, password, role, email, mobile });
  btn.disabled = false;
  if (res.success) {
    toast(`Created ${res.user.username} (${res.user.role}).`);
    ['new-username', 'new-password', 'new-email', 'new-mobile'].forEach((id) => { $(id).value = ''; });
    $('new-role').value = 'user';
    loadUsers();
  } else {
    toast(res.error || 'Could not create account.');
  }
});

// ── Row actions (event delegation) ────────────────────────────────────────────
$('user-rows').addEventListener('click', async (e) => {
  const btn = e.target.closest('button[data-act]');
  if (!btn || btn.disabled) return;
  const user = btn.dataset.user;
  const act = btn.dataset.act;

  if (act === 'contact') {
    const email = prompt(`Email for "${user}" (blank to clear):`, btn.dataset.email || '');
    if (email === null) return;
    const mobile = prompt(`Mobile for "${user}" (10 digits, blank to clear):`, btn.dataset.mobile || '');
    if (mobile === null) return;
    const res = await api('POST', `/gad-auth/users/${encodeURIComponent(user)}/contact`, { email, mobile });
    toast(res.success ? `Updated contact for ${user}.` : (res.error || 'Failed.'));
    loadUsers();
  } else if (act === 'password') {
    const pw = prompt(`New password for "${user}" (min 8 characters):`);
    if (pw == null) return;
    const res = await api('POST', `/gad-auth/users/${encodeURIComponent(user)}/password`, { password: pw });
    toast(res.success ? `Password reset for ${user}.` : (res.error || 'Failed.'));
  } else if (act === 'role') {
    const next = btn.dataset.role === 'admin' ? 'user' : 'admin';
    if (!confirm(`Change ${user}'s role to ${next}?`)) return;
    const res = await api('POST', `/gad-auth/users/${encodeURIComponent(user)}/role`, { role: next });
    toast(res.success ? `${user} is now ${next}.` : (res.error || 'Failed.'));
    loadUsers();
  } else if (act === 'delete') {
    if (!confirm(`Delete account "${user}"? This cannot be undone.`)) return;
    const res = await api('DELETE', `/gad-auth/users/${encodeURIComponent(user)}`);
    toast(res.success ? `Deleted ${user}.` : (res.error || 'Failed.'));
    loadUsers();
  }
});

// ── IndiaAI GPU Capacity Workload Controls ──────────────────────────────────
const gpuBtnStart = $('gpu-btn-start');
const gpuBtnStop = $('gpu-btn-stop');
const gpuStatusPill = $('gpu-status-pill');
const gpuStatusLabel = $('gpu-status-label');
const gpuOptImages = $('gpu-opt-images');
const gpuOptConcurrency = $('gpu-opt-concurrency');

async function refreshGpuStatus() {
  if (!gpuBtnStart) return;
  try {
    const data = await api('GET', '/api/admin/gpu/status');
    if (!data || !data.success) return;
    const telem = data.telemetry || {};
    const work = data.workload || {};

    const running = !!work.running;
    gpuBtnStart.disabled = running;
    gpuBtnStop.disabled = !running;

    if (gpuStatusPill && gpuStatusLabel) {
      gpuStatusPill.className = 'gpu-pill ' + (running ? 'running' : 'idle');
      gpuStatusLabel.textContent = running ? `RUNNING (${work.concurrency || 2} WORKERS)` : 'IDLE';
    }

    const gpuUtil = Math.round(telem.gpu_util || 0);
    const memPct = Math.round(telem.mem_util || 0);
    const memUsedGb = ((telem.mem_used_mb || 0) / 1024).toFixed(1);
    const memTotalGb = ((telem.mem_total_mb || 81920) / 1024).toFixed(0);

    const setT = (id, val) => { const el = $(id); if (el) el.textContent = val; };
    setT('gpu-val-util', gpuUtil);
    setT('gpu-val-name', telem.name || 'NVIDIA GPU');
    setT('gpu-val-mem-used', memUsedGb);
    setT('gpu-val-mem-total', memTotalGb);
    setT('gpu-val-mem-pct', memPct);
    setT('gpu-val-temp', Math.round(telem.temperature || 0));
    setT('gpu-val-power', Math.round(telem.power_draw_w || 0));
    setT('gpu-val-power-limit', Math.round(telem.power_limit_w || 700));
    setT('gpu-val-tok-s', Math.round(work.tokens_per_second || 0));
    setT('gpu-val-reqs', `${work.total_requests || 0} req`);
    setT('gpu-val-lat', Math.round(work.avg_latency_ms || 0));

    const mUtil = $('gpu-meter-util'); if (mUtil) mUtil.style.width = `${Math.min(100, Math.max(0, gpuUtil))}%`;
    const mMem = $('gpu-meter-mem'); if (mMem) mMem.style.width = `${Math.min(100, Math.max(0, memPct))}%`;
  } catch (_) { /* ignore polling errors */ }
}

if (gpuBtnStart && gpuBtnStop) {
  gpuBtnStart.addEventListener('click', async () => {
    gpuBtnStart.disabled = true;
    toast('Starting synthetic GPU workload…');
    try {
      const concurrency = parseInt(gpuOptConcurrency ? gpuOptConcurrency.value : 2, 10) || 2;
      const includeImages = gpuOptImages ? gpuOptImages.checked : true;
      const res = await api('POST', '/api/admin/gpu/start', { concurrency, include_images: includeImages });
      if (res && res.success) {
        toast('Continuous synthetic workload running against local model server.');
      } else {
        toast((res && res.error) || 'Failed to start.');
      }
    } catch (e) {
      toast('Start error: ' + e.message);
    } finally {
      refreshGpuStatus();
    }
  });

  gpuBtnStop.addEventListener('click', async () => {
    gpuBtnStop.disabled = true;
    toast('Stopping synthetic workload instantly…');
    try {
      const res = await api('POST', '/api/admin/gpu/stop');
      if (res && res.success) {
        toast('Workload stopped. GPU returned to idle state.');
      } else {
        toast((res && res.error) || 'Failed to stop.');
      }
    } catch (e) {
      toast('Stop error: ' + e.message);
    } finally {
      refreshGpuStatus();
    }
  });
}

// ── Boot ──────────────────────────────────────────────────────────────────────
(async function boot() {
  await loadMe();
  await loadUsers();
  refreshGpuStatus();
  setInterval(refreshGpuStatus, 1500);
})();
