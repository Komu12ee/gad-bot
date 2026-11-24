'use strict';

// ── Client for IndiaAI GPU Capacity Demonstration & Workload Generator ──────
const $ = (id) => document.getElementById(id);
const BASE = (typeof window !== 'undefined' && window.__APP_BASE__) || '';

const toast = (msg) => {
  const t = $('toast');
  if (!t) return;
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove('show'), 2600);
};

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function api(method, path, body) {
  const opts = { method, headers: {}, credentials: 'same-origin' };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(BASE + path, opts);
  if (res.status === 401) { window.location.href = `${BASE}/login`; throw new Error('unauth'); }
  if (res.status === 403) { window.location.href = `${BASE}/`; throw new Error('forbidden'); }
  return res.json().catch(() => ({}));
}

// ── State ────────────────────────────────────────────────────────────────────
let isRunning = false;
let pollTimer = null;
let historyData = [];

// ── DOM Elements ─────────────────────────────────────────────────────────────
const btnStart = $('btn-start');
const btnStop = $('btn-stop');
const statusPill = $('status-pill');
const statusText = $('status-text');
const chkImages = $('chk-images');
const selConcurrency = $('sel-concurrency');
const uptimeDisplay = $('uptime-display');

const valGpuUtil = $('val-gpu-util');
const barGpuUtil = $('bar-gpu-util');
const valGpuName = $('val-gpu-name');

const valMemUsed = $('val-mem-used');
const valMemTotal = $('val-mem-total');
const valMemPct = $('val-mem-pct');
const barMemUtil = $('bar-mem-util');
const valDriver = $('val-driver');

const valTemp = $('val-temp');
const valPower = $('val-power');
const valPowerLimit = $('val-power-limit');

const valTokensSec = $('val-tokens-sec');
const valRequestsCount = $('val-requests-count');
const valLatency = $('val-latency');
const targetModelBadge = $('target-model-badge');
const logRows = $('log-rows');
const chartCanvas = $('chart-canvas');

// ── Polling & UI Updates ─────────────────────────────────────────────────────
async function refreshStatus() {
  try {
    const data = await api('GET', '/api/admin/gpu/status');
    if (!data || !data.success) return;

    const telem = data.telemetry || {};
    const work = data.workload || {};
    historyData = data.history || [];

    isRunning = !!work.running;

    // Button states
    btnStart.disabled = isRunning;
    btnStop.disabled = !isRunning;

    // Status pill
    if (isRunning) {
      statusPill.className = 'pill-status running';
      statusText.textContent = `RUNNING — ${work.concurrency || 2} WORKERS ACTIVE`;
    } else {
      statusPill.className = 'pill-status idle';
      statusText.textContent = 'IDLE — READY';
    }

    // Uptime
    if (uptimeDisplay) {
      const u = work.uptime_seconds || 0;
      const mins = Math.floor(u / 60);
      const secs = u % 60;
      uptimeDisplay.textContent = `Uptime: ${mins > 0 ? mins + 'm ' : ''}${secs}s`;
    }

    // Target model
    if (targetModelBadge && work.target_model) {
      targetModelBadge.textContent = `Model: ${work.target_model} on ${work.target_url || ':8000'}`;
    }

    // Telemetry - GPU
    const gpuUtil = Math.round(telem.gpu_util || 0);
    valGpuUtil.textContent = gpuUtil;
    barGpuUtil.style.width = `${Math.min(100, Math.max(0, gpuUtil))}%`;
    valGpuName.textContent = telem.name || 'NVIDIA GPU';

    // Telemetry - VRAM
    const memUsedGb = ((telem.mem_used_mb || 0) / 1024).toFixed(1);
    const memTotalGb = ((telem.mem_total_mb || 81920) / 1024).toFixed(0);
    valMemUsed.textContent = memUsedGb;
    valMemTotal.textContent = memTotalGb;
    valMemPct.textContent = Math.round(telem.mem_util || 0);
    barMemUtil.style.width = `${Math.min(100, Math.max(0, telem.mem_util || 0))}%`;
    valDriver.textContent = `Driver ${telem.driver_version || '550.54'}`;

    // Telemetry - Thermal & Power
    valTemp.textContent = Math.round(telem.temperature || 0);
    valPower.textContent = Math.round(telem.power_draw_w || 0);
    valPowerLimit.textContent = Math.round(telem.power_limit_w || 700);

    // Workload - Throughput
    valTokensSec.textContent = Math.round(work.tokens_per_second || 0);
    valRequestsCount.textContent = `${work.total_requests || 0} req (${work.text_requests || 0} text · ${work.image_requests || 0} img)`;
    valLatency.textContent = Math.round(work.avg_latency_ms || 0);

    // Recent logs table
    const logs = data.recent_logs || [];
    if (logs.length > 0) {
      logRows.innerHTML = logs.map((l) => `
        <tr>
          <td>${esc(l.time)}</td>
          <td><strong>${esc(l.worker)}</strong></td>
          <td><span class="badge-type ${l.type === 'image' ? 'image' : 'text'}">${esc(l.type)}</span></td>
          <td>${esc(l.prompt)}</td>
          <td>${esc(l.tokens)} tok</td>
          <td>${esc(l.latency_ms)} ms</td>
          <td>${l.success ? '<span class="log-ok">200 OK</span>' : `<span class="log-err">${esc(l.error || 'Err')}</span>`}</td>
        </tr>
      `).join('');
    }

    // Redraw Canvas chart
    drawChart();

  } catch (err) {
    console.error('Failed to fetch GPU status:', err);
  }
}

// ── 60-Second Real-Time Canvas Chart ─────────────────────────────────────────
function drawChart() {
  if (!chartCanvas) return;
  const ctx = chartCanvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = chartCanvas.clientWidth;
  const h = chartCanvas.clientHeight;

  if (chartCanvas.width !== w * dpr || chartCanvas.height !== h * dpr) {
    chartCanvas.width = w * dpr;
    chartCanvas.height = h * dpr;
  }
  ctx.resetTransform();
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.strokeStyle = 'rgba(128, 140, 160, 0.12)';
  ctx.lineWidth = 1;
  const gridLines = 4;
  for (let i = 0; i <= gridLines; i++) {
    const y = Math.round((h / gridLines) * i);
    ctx.beginPath();
    ctx.moveTo(40, y);
    ctx.lineTo(w - 10, y);
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '10px sans-serif';
    ctx.fillText(`${100 - i * 25}%`, 8, Math.min(h - 4, Math.max(12, y + 3)));
  }

  if (!historyData || historyData.length < 2) return;

  const maxPoints = 60;
  const points = historyData.slice(-maxPoints);
  const stepX = (w - 60) / (maxPoints - 1);
  const startX = 45;

  function renderCurve(key, strokeColor, fillColor) {
    ctx.beginPath();
    points.forEach((pt, idx) => {
      const val = Math.min(100, Math.max(0, pt[key] || 0));
      const x = startX + idx * stepX;
      const y = h - (val / 100) * (h - 20) - 10;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    if (fillColor) {
      const lastX = startX + (points.length - 1) * stepX;
      ctx.lineTo(lastX, h - 10);
      ctx.lineTo(startX, h - 10);
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();
    }

    ctx.beginPath();
    points.forEach((pt, idx) => {
      const val = Math.min(100, Math.max(0, pt[key] || 0));
      const x = startX + idx * stepX;
      const y = h - (val / 100) * (h - 20) - 10;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2.2;
    ctx.stroke();
  }

  // Draw VRAM curve (Blue)
  renderCurve('mem_util', '#3b82f6', 'rgba(59, 130, 246, 0.08)');

  // Draw GPU compute curve (Green)
  renderCurve('gpu_util', '#10b981', 'rgba(16, 185, 129, 0.12)');

  // Draw Tokens/sec rate scaled 0-500 tok/s to 0-100%
  ctx.beginPath();
  points.forEach((pt, idx) => {
    const rawRate = pt.tokens_per_sec || 0;
    const scaledPct = Math.min(100, Math.max(0, (rawRate / 500) * 100));
    const x = startX + idx * stepX;
    const y = h - (scaledPct / 100) * (h - 20) - 10;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 1.6;
  ctx.setLineDash([4, 3]);
  ctx.stroke();
  ctx.setLineDash([]);
}

// ── Control Handlers (ONE button to start, ONE button to stop) ────────────────
btnStart.addEventListener('click', async () => {
  btnStart.disabled = true;
  toast('Starting synthetic GPU workload…');
  try {
    const concurrency = parseInt(selConcurrency.value, 10) || 2;
    const includeImages = chkImages.checked;
    const res = await api('POST', '/api/admin/gpu/start', { concurrency, include_images: includeImages });
    if (res && res.success) {
      toast('Continuous synthetic workload running against local model server.');
    } else {
      toast((res && res.error) || 'Failed to start workload.');
    }
  } catch (err) {
    toast('Error starting workload: ' + err.message);
  } finally {
    refreshStatus();
  }
});

btnStop.addEventListener('click', async () => {
  btnStop.disabled = true;
  toast('Stopping synthetic workload instantly…');
  try {
    const res = await api('POST', '/api/admin/gpu/stop');
    if (res && res.success) {
      toast('Workload stopped. GPU returned to idle state.');
    } else {
      toast((res && res.error) || 'Failed to stop workload.');
    }
  } catch (err) {
    toast('Error stopping workload: ' + err.message);
  } finally {
    refreshStatus();
  }
});

// ── Boot ─────────────────────────────────────────────────────────────────────
refreshStatus();
pollTimer = setInterval(refreshStatus, 1200);

window.addEventListener('resize', drawChart);
