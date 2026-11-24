'use strict';

// Relative URLs keep registration inside this department, including when the
// application is served under /pipeline/chips/ by the VM proxy.
(function () {
  const $ = id => document.getElementById(id);
  const api = window.Platform.makeApi('');
  let ready = false, busy = false, created = false;
  const show = text => { $('msg').textContent = text; $('msg').classList.toggle('hidden', !text); };
  $('submit').disabled = true;

  api('GET', 'api/info').then(result => {
    if (!result.success) return show(result.error || 'This department was not found.');
    $('dept-name').textContent = result.department.name;
    document.title = `${result.department.name} · create account`;
    if (result.department.status !== 'active') return show('Registration is unavailable while this department is disabled.');
    ready = true;
    $('submit').disabled = false;
  });

  $('form').addEventListener('submit', async event => {
    event.preventDefault();
    if (!ready || busy || created) return;
    show('');
    const username = $('username').value.trim();
    const password = $('password').value;
    if (password !== $('confirm-password').value) return show('The passwords do not match.');
    busy = true;
    $('submit').disabled = true;
    $('submit').textContent = 'Creating account…';
    try {
      const result = await api('POST', 'api/signup', { username, password });
      if (!result.success) return show(result.error || 'Could not create the account.');
      created = true;
      $('password').value = '';
      $('confirm-password').value = '';
      $('fields').querySelectorAll('input').forEach(input => { input.disabled = true; });
      $('fields').classList.add('hidden');
      $('submit').classList.add('hidden');
      $('signin-hint').classList.add('hidden');
      $('done-msg').textContent = result.message || 'Your account is ready. You can sign in now.';
      $('done').classList.remove('hidden');
    } finally {
      busy = false;
      $('submit').disabled = created;
      $('submit').textContent = 'Create account';
    }
  });
})();
