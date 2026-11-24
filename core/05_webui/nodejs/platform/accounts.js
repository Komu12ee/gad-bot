'use strict';

/**
 * A department's own accounts: admins sign in to the department panel, users will
 * sign in to the department's chatbot. Kept by the CHiPS admin (any account of any
 * department) and by the department's admins (their own department only, never
 * their own account, never the last active admin — see store.guardChange).
 *
 * Passwords are bcrypt-hashed (cost 10, as the bot accounts). Signing in compares
 * against a real hash even for an unknown name, so the time taken does not tell
 * which names exist.
 */

const bcrypt = require('bcryptjs');
const { failure } = require('./http');

const ROLES = ['admin', 'user'];
const DUMMY_HASH = bcrypt.hashSync('no department account has this password', 10);

function username(raw) {
  const u = String(raw || '').trim().toLowerCase();
  if (!/^[a-z0-9._-]{3,32}$/.test(u)) {
    throw failure(400, 'Username: 3 to 32 characters — letters, digits, dot, dash or underscore.');
  }
  return u;
}

function password(raw) {
  if (typeof raw !== 'string' || raw.length < 8) throw failure(400, 'The password must be at least 8 characters.');
  if (raw.length > 128) throw failure(400, 'The password is too long (128 characters at most).');
  return raw;
}

const hash = (pw) => bcrypt.hash(pw, 10);

function createAccounts({ store }) {
  return {
    list: (departmentId) => store.listUsers(departmentId),

    async create(departmentId, body, by) {
      const b = body || {};
      if (!ROLES.includes(b.role)) throw failure(400, 'Choose admin or user.');
      const name = username(b.username);
      return store.createUser(departmentId, { username: name, role: b.role, passwordHash: await hash(password(b.password)), by });
    },

    /** {role?, status?, password?}; `actor` = the department admin making the change (none for CHiPS). */
    async update(departmentId, userId, body, actor) {
      const b = body || {};
      const patch = {};
      if (b.role !== undefined) {
        if (!ROLES.includes(b.role)) throw failure(400, 'Choose admin or user.');
        patch.role = b.role;
      }
      if (b.status !== undefined) {
        if (!['active', 'blocked'].includes(b.status)) throw failure(400, 'Invalid status.');
        patch.status = b.status;
      }
      if (b.password !== undefined) patch.passwordHash = await hash(password(b.password));
      if (!Object.keys(patch).length) throw failure(400, 'Nothing to change.');
      return store.updateUser(departmentId, userId, patch, { actor });
    },

    remove: (departmentId, userId, actor) => store.deleteUser(departmentId, userId, { actor }),

    /** The account for a right name + password (status not checked here), else null. */
    async verify(departmentId, name, pw) {
      const u = String(name || '').trim().toLowerCase();
      const rec = /^[a-z0-9._-]{3,32}$/.test(u) ? await store.getLogin(departmentId, u) : null;
      const ok = await bcrypt.compare(String(pw || ''), rec ? rec.passwordHash : DUMMY_HASH);
      if (!rec || !ok) return null;
      const { passwordHash, ...account } = rec;
      return account;
    },

    /** A signed-in account changes its own password (the current one is asked for). */
    async changeOwn(user, current, next) {
      const me = await this.verify(user.departmentId, user.username, current);
      if (!me || me.id !== user.id) throw failure(400, 'Your current password is not right.');
      return store.updateUser(user.departmentId, user.id, { passwordHash: await hash(password(next)) });
    },
  };
}

/** What a page may see of an account. */
const publicAccount = (a) => ({
  id: a.id, username: a.username, role: a.role, status: a.status,
  createdBy: a.createdBy, createdAt: a.createdAt, lastLoginAt: a.lastLoginAt,
});

module.exports = { createAccounts, publicAccount, ROLES };
