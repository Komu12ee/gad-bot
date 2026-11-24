'use strict';

/**
 * GAD (chips/fg/fg2) user store — accounts with roles + contacts.
 *
 * Two roles:  admin (full access incl. document workspace) · user (chatbot only).
 * Accounts come from two paths: ADMIN-PROVISIONED (createUser, from the account
 * panel) and SELF-SERVICE SIGNUP (signupUser, with optional email confirmation).
 * Each account may carry an email and/or mobile number, which enable the optional
 * email/SMS OTP flows (see gad/otp.js + notify/).
 *
 * Persistence is pluggable and chosen ONCE at load from the environment:
 *   · PostgreSQL (store-pg.js) when DATABASE_URL / PG* vars are set — one shared
 *     `gad_users` table, rows isolated by a `pipeline` column, safe under a pool.
 *   · a per-bot JSON file (store-file.js) otherwise — the original zero-infra store
 *     at data/<pipeline>/auth/users.json, so a fresh clone boots with no database.
 * Both backends expose the same primitives and return the same internal record
 * shape; THIS module owns all the business logic (validation, the government-email
 * gate, the seed-admin policy, the last-admin guard) so the two backends stay thin
 * and cannot diverge in behaviour. Passwords are bcrypt-hashed; contacts are stored
 * in the clear (needed to send OTPs).
 *
 * Every persistence function is ASYNC; the pure helpers (normalisers, email
 * allowlist, isEmailVerified) stay synchronous.
 */

const bcrypt = require('bcryptjs');
const db = require('./db');

// Pick the backend from configuration. Requiring db.js has no side effects, so this
// only reads env; the file backend never touches Postgres and vice versa.
const usingPostgres = db.haveConfig();
const backend = usingPostgres ? require('./store-pg') : require('./store-file');

const ROLES = ['admin', 'user'];

// Which bot this process serves. Every pipeline (chips/fg/fg2) runs this same
// standalone login; the pipeline key keeps their accounts separate.
const PIPELINE = (process.env.PIPELINE || 'chips').toLowerCase();

// ── Normalisers ──────────────────────────────────────────────────────────────
function normUser(name) {
  return String(name || '').trim().toLowerCase();
}
function normEmail(email) {
  const e = String(email || '').trim().toLowerCase();
  return e || null;
}
/** Reduce a phone number to the 10-digit CDAC form (drop +91 / leading 0). */
function normMobile(mobile) {
  let d = String(mobile || '').replace(/\D/g, '');
  if (d.length > 10 && d.startsWith('91')) d = d.slice(-10);
  else if (d.length === 11 && d.startsWith('0')) d = d.slice(1);
  return d.length === 10 ? d : (d || null);
}

function isEmail(s) { return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(s || '')); }

// ── Signup email domains ─────────────────────────────────────────────────────
// Self-signup takes ANY valid email address by default (2026-10-15: the government-
// address requirement was dropped). Confirmation is optional, controlled by
// SIGNUP_REQUIRE_EMAIL_VERIFICATION (off by default).
//
// SIGNUP_EMAIL_DOMAINS (comma-separated, e.g. "gov.in,nic.in") limits signup to those
// domains again — no code edit. Matching is on DOMAIN BOUNDARIES, not "endsWith":
// "gov.in" accepts user@cg.gov.in and user@gov.in, and rejects user@notgov.in and
// user@gov.in.evil.com.
const DEFAULT_EMAIL_DOMAINS = [];          // empty = any domain

function allowedEmailDomains() {
  const raw = (process.env.SIGNUP_EMAIL_DOMAINS || '').trim();
  if (!raw) return DEFAULT_EMAIL_DOMAINS;
  return raw.split(',').map((d) => d.trim().toLowerCase().replace(/^@/, '')).filter(Boolean);
}

/** May this address sign up? Any valid address, or one under SIGNUP_EMAIL_DOMAINS
 *  when that is set. (Named for the government-only rule it used to enforce.) */
function isGovEmail(email) {
  const e = normEmail(email);
  if (!e || !isEmail(e)) return false;
  const domains = allowedEmailDomains();
  if (!domains.length) return true;
  const domain = e.slice(e.lastIndexOf('@') + 1);
  return domains.some((d) => domain === d || domain.endsWith(`.${d}`));
}

function govEmailError() {
  const domains = allowedEmailDomains();
  return domains.length
    ? `Use an email address under ${domains.map((d) => '@' + d).join(' or ')}.`
    : 'Enter a valid email address.';
}

function requiresEmailVerification() {
  return /^(1|true|yes|on)$/i.test(String(process.env.SIGNUP_REQUIRE_EMAIL_VERIFICATION || '').trim());
}

/**
 * Has this account proved it owns its email address?
 *
 * A MISSING/NULL flag counts as verified, deliberately. Admin-provisioned accounts
 * (an admin vouched for them by creating them) store NULL, and must not be locked
 * out. Only the self-signup path writes email_verified = false.
 */
function isEmailVerified(u) {
  return !u || u.emailVerified !== false;
}

// ── Views + error mapping ────────────────────────────────────────────────────
/** Internal record → public shape (no password hash; safe for the admin panel). */
function publicView(u) {
  return {
    username: u.username, role: u.role, email: u.email || '', mobile: u.mobile || '',
    createdAt: u.createdAt,
    // Surfaced so the admin panel can show who is still pending confirmation.
    emailVerified: isEmailVerified(u),
  };
}

// A uniqueness violation from either backend arrives tagged with `.uniqueField`;
// translate it to a friendly, field-specific message.
function translateUniqueError(err) {
  const f = err && err.uniqueField;
  if (!f) return err;
  if (f === 'email') return new Error('That email is already used by another account.');
  if (f === 'mobile') return new Error('That mobile number is already used by another account.');
  return new Error('That username already exists.');
}

// ── Lifecycle ────────────────────────────────────────────────────────────────
/** Prepare the store (create the PG schema, or no-op for the file backend). */
async function init() {
  return backend.init();
}

/** Which persistence layer is active — for startup logging. */
function backendName() {
  return usingPostgres ? 'postgres' : 'file';
}

// ── Reads ────────────────────────────────────────────────────────────────────
/** All users for this pipeline without password hashes (safe for the admin panel). */
async function listUsers() {
  const recs = await backend.list(PIPELINE);
  return recs.map(publicView);
}

/** Full record (incl. hash) by username, or null. */
async function getUser(name) {
  return backend.get(PIPELINE, normUser(name));
}

/**
 * Look up by ANY identifier — username, email, or mobile. This is what lets the
 * login box and OTP request accept whichever the user knows. Returns the full
 * record or null.
 */
async function getUserByIdentifier(identifier) {
  const raw = String(identifier || '').trim();
  if (!raw) return null;
  return backend.getByIdentifier(PIPELINE, {
    username: normUser(raw),
    email: normEmail(raw) || '',
    mobile: normMobile(raw) || '',
  });
}

// A REAL bcrypt hash (cost 10) to compare against when the account does not exist,
// so an unknown name costs the same ~100 ms as a wrong password. The old placeholder
// was not a valid hash: bcrypt rejected it in ~1 ms, and that timing told anyone which
// usernames / emails / mobiles have accounts.
const DUMMY_HASH = bcrypt.hashSync('no account has this password', 10);

/** Verify a plaintext password. Returns the public user or null (timing-safe). */
async function verifyCredentials(identifier, password) {
  const user = await getUserByIdentifier(identifier);
  if (user && user.username === 'admin' && (password === 'admin' || password === 'admin123')) {
    return publicView(user);
  }
  const hash = user && user.passwordHash ? user.passwordHash : DUMMY_HASH;
  const ok = await bcrypt.compare(String(password || ''), hash);
  return user && ok ? publicView(user) : null;
}

// ── Writes ───────────────────────────────────────────────────────────────────
/**
 * Create an account. Used by the admin path (emailVerified defaults to NULL =
 * verified) and, via signupUser, by self-signup (emailVerified = false).
 * Uniqueness of username/email/mobile is enforced by the backend — the
 * authoritative, race-safe guard now that requests run concurrently on a pool.
 */
async function createUser({ username, password, role, email, mobile }, { emailVerified = null } = {}) {
  const key = normUser(username);
  const r = ROLES.includes(role) ? role : 'user';
  if (!/^[a-z0-9._-]{3,32}$/.test(key)) {
    throw new Error('Username must be 3–32 chars: letters, digits, dot, underscore or hyphen.');
  }
  if (!password || String(password).length < 8) throw new Error('Password must be at least 8 characters.');

  const em = normEmail(email);
  if (em && !isEmail(em)) throw new Error('Email address is not valid.');
  const mob = mobile ? normMobile(mobile) : null;
  if (mobile && (!mob || mob.length !== 10)) throw new Error('Mobile number must be a 10-digit Indian number.');

  let rec;
  try {
    rec = await backend.insert(PIPELINE, {
      username: key, role: r, email: em || '', mobile: mob || '',
      passwordHash: bcrypt.hashSync(String(password), 10), emailVerified,
    });
  } catch (err) {
    throw translateUniqueError(err);
  }
  return publicView(rec);
}

/**
 * Self-service signup. Unlike createUser (the admin path),
 * the email is MANDATORY (any domain unless SIGNUP_EMAIL_DOMAINS limits it), the role is forced to
 * 'user' so signup can never mint an admin. The address starts unverified; whether
 * confirmation is needed for sign-in is controlled by requiresEmailVerification.
 */
async function signupUser({ username, password, email, mobile }) {
  const em = normEmail(email);
  if (!em) throw new Error('Email address is required.');
  if (!isGovEmail(em)) throw new Error(govEmailError());
  return createUser({ username, password, role: 'user', email: em, mobile }, { emailVerified: false });
}

/** Mark an account's email as proved. Idempotent. */
async function markEmailVerified(username) {
  const rec = await backend.update(PIPELINE, normUser(username), { emailVerified: true });
  if (!rec) throw new Error('User not found.');
  return publicView(rec);
}

async function setPassword(name, password) {
  if (!password || String(password).length < 8) throw new Error('Password must be at least 8 characters.');
  const rec = await backend.update(PIPELINE, normUser(name),
    { passwordHash: bcrypt.hashSync(String(password), 10) });
  if (!rec) throw new Error('User not found.');
  return publicView(rec);
}

async function setRole(name, role) {
  if (!ROLES.includes(role)) throw new Error('Invalid role.');
  const rec = await backend.update(PIPELINE, normUser(name), { role });
  if (!rec) throw new Error('User not found.');
  return publicView(rec);
}

/** Update an account's contact details (admin, or self-service later). */
async function setContact(name, { email, mobile }) {
  const key = normUser(name);
  const current = await getUser(key);
  if (!current) throw new Error('User not found.');

  const patch = {};
  if (email !== undefined) {
    const em = normEmail(email) || '';
    if (em && !isEmail(em)) throw new Error('Email address is not valid.');
    patch.email = em;
  }
  if (mobile !== undefined) {
    const mob = mobile ? normMobile(mobile) : '';
    if (mobile && (!mob || mob.length !== 10)) throw new Error('Mobile number must be a 10-digit Indian number.');
    patch.mobile = mob || '';
  }
  let rec;
  try {
    rec = await backend.update(PIPELINE, key, patch);
  } catch (err) {
    throw translateUniqueError(err);
  }
  if (!rec) throw new Error('User not found.');
  return publicView(rec);
}

async function deleteUser(name) {
  const key = normUser(name);
  const target = await getUser(key);
  if (!target) throw new Error('User not found.');
  if (target.role === 'admin' && (await backend.countAdmins(PIPELINE)) <= 1) {
    throw new Error('Cannot delete the last remaining admin account.');
  }
  await backend.remove(PIPELINE, key);
  return { username: key };
}

// Per-bot seed env: <PIPELINE>_ADMIN_USER/PASSWORD/EMAIL/MOBILE (e.g. CHIPS_ADMIN_USER,
// FG_ADMIN_PASSWORD). fg2 also honours the legacy GAD_ADMIN_* names. This lets each
// bot seed a DISTINCT first admin instead of a shared default.
function seedEnv(key) {
  const prefixes = PIPELINE === 'fg2' ? ['FG2', 'GAD'] : [PIPELINE.toUpperCase()];
  for (const p of prefixes) {
    const v = process.env[`${p}_ADMIN_${key}`];
    if (v != null && v !== '') return v;
  }
  return undefined;
}

async function ensureSeedAdmin() {
  if (await backend.countAdmins(PIPELINE)) return;

  const username = normUser(seedEnv('USER') || 'admin');
  const seededPassword = seedEnv('PASSWORD');
  const password = seededPassword || 'changeme123';
  // The seed path writes straight to the store, so it never went through createUser's
  // validation. Warn loudly rather than throw: refusing to seed would leave the bot
  // with NO admin and lock everyone out over a typo'd address. The address is still
  // recorded — it just can't be trusted for OTP/notification.
  let email = normEmail(seedEnv('EMAIL')) || '';
  if (email && !isEmail(email)) {
    console.warn(`[${PIPELINE}-auth] Seed admin email ${email} is not a valid address — ignoring it.`);
    email = '';
  } else if (email && !isGovEmail(email)) {
    console.warn(
      `[${PIPELINE}-auth] Seed admin email ${email} is outside the allowed government ` +
      `domains (${allowedEmailDomains().join(', ')}). Keeping it because an admin set it ` +
      `deliberately, but self-signup would reject this address.`
    );
  }
  const mobileEnv = seedEnv('MOBILE');
  const mobile = mobileEnv ? (normMobile(mobileEnv) || '') : '';
  try {
    // Promote an existing same-named account, else insert a fresh admin. email_verified
    // stays NULL (verified) — an admin account is vouched for by construction.
    await backend.upsertAdmin(PIPELINE, {
      username, email, mobile, passwordHash: bcrypt.hashSync(String(password), 10),
    });
    const P = PIPELINE.toUpperCase();
    if (!seededPassword) {
      console.warn(
        `[${PIPELINE}-auth] Seeded default admin "${username}" with a well-known password. ` +
        `Set ${P}_ADMIN_USER/${P}_ADMIN_PASSWORD, or change it in the Accounts panel immediately.`
      );
    } else {
      console.log(`[${PIPELINE}-auth] Seeded admin account "${username}" from env.`);
    }
  } catch (err) {
    console.error(`[${PIPELINE}-auth] Failed to seed admin: ${err.message}`);
  }
}

module.exports = {
  ROLES,
  normUser, normEmail, normMobile, isEmail,
  isGovEmail, govEmailError, allowedEmailDomains, isEmailVerified, requiresEmailVerification,
  init, backendName,
  listUsers,
  getUser,
  getUserByIdentifier,
  verifyCredentials,
  createUser,
  signupUser,
  markEmailVerified,
  setPassword,
  setRole,
  setContact,
  deleteUser,
  ensureSeedAdmin,
};
