'use strict';

/**
 * Calls to the bot's Python service (core/05_webui/dept_service.py) on 127.0.0.1:
 * reading + embedding a file, encoding a question, writing an answer. Each call
 * carries the internal key (HMAC of the session secret) that service requires;
 * the browser never reaches these routes (server.js refuses /api/dept).
 */

const crypto = require('crypto');
const http = require('http');

function internalKey() {
  const s = process.env.GAD_SESSION_SECRET || process.env.SESSION_SECRET || process.env.JWT_SECRET || '';
  return s ? crypto.createHmac('sha256', s).update('dept-internal-v1').digest('hex') : '';
}

/**
 * @param {string} baseUrl  e.g. http://127.0.0.1:5001
 * @returns {(path: string, body: object, timeoutMs?: number) => Promise<object>}
 *   resolves the JSON reply when success; rejects with .status and the reply's error
 */
function createClient(baseUrl) {
  const base = new URL(baseUrl);
  return function call(path, body, timeoutMs = 120000) {
    const data = Buffer.from(JSON.stringify(body || {}));
    return new Promise((resolve, reject) => {
      const req = http.request({
        hostname: base.hostname, port: base.port, path, method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': data.length, 'X-Dept-Internal': internalKey() },
      }, (res) => {
        const parts = [];
        res.on('data', (d) => parts.push(d));
        res.on('end', () => {
          let json = null;
          try { json = JSON.parse(Buffer.concat(parts).toString('utf8')); } catch (_) { /* below */ }
          if (res.statusCode === 200 && json && json.success) return resolve(json);
          const err = new Error((json && json.error) || `The answer service replied HTTP ${res.statusCode}.`);
          err.status = res.statusCode === 400 ? 400 : 503;
          reject(err);
        });
      });
      req.setTimeout(timeoutMs, () => req.destroy(Object.assign(new Error('The answer service took too long.'), { status: 504 })));
      req.on('error', (e) => reject(Object.assign(
        new Error(e.status === 504 ? e.message : 'The answer service is not running.'), { status: e.status || 503 })));
      req.end(data);
    });
  };
}

module.exports = { createClient, internalKey };
