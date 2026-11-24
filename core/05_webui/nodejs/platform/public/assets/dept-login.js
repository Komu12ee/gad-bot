'use strict';

// Department sign-in page (/dept/<short name>/login). Requests are relative
// (api/…), so the page works under any proxy prefix.

(function () {
  const $ = (id) => document.getElementById(id);
  const api = window.Platform.makeApi('');

  const show = (text) => { $('msg').textContent = text; $('msg').classList.toggle('hidden', !text); };

  api('GET', 'api/info').then((r) => {
    if (!r.success) { show(r.error || 'This department was not found.'); $('submit').disabled = true; return; }
    $('dept-name').textContent = r.department.name;
    document.title = `${r.department.name} · sign in`;
  });

  $('form').addEventListener('submit', async (e) => {
    e.preventDefault();
    show('');
    const username = $('username').value.trim();
    const password = $('password').value;
    if (!username || !password) return show('Enter your username and password.');
    $('submit').disabled = true;
    try {
      const r = await api('POST', 'api/login', { username, password });
      if (!r.success) return show(r.error || 'Could not sign in.');
      window.location.href = './';
    } finally {
      $('submit').disabled = false;
    }
  });
})();
