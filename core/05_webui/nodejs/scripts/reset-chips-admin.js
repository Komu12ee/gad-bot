#!/usr/bin/env node
'use strict';

// Run on the server with RESET_ADMIN_PASSWORD in the environment.
// Only the CHiPS account named admin is reset; use the existing store so both
// file-backed and PostgreSQL deployments follow the application's password rules.
const path = require('path');
const fs = require('fs/promises');
const repoRoot = path.resolve(__dirname, '..', '..', '..', '..');
require('dotenv').config({ path: path.join(repoRoot, '.env') });
process.env.PIPELINE = 'chips';

const store = require('../gad/store');
const db = require('../gad/db');

async function main() {
    const password = process.env.RESET_ADMIN_PASSWORD;
    delete process.env.RESET_ADMIN_PASSWORD;
    if (!password || password.length < 8) {
        throw new Error('Set RESET_ADMIN_PASSWORD to a password of at least 8 characters.');
    }
    if (store.backendName() === 'file') {
        const file = path.join(process.env.RAG_DATA_ROOT || path.join(repoRoot, 'data'),
            'chips', 'auth', 'users.json');
        let content;
        try {
            content = await fs.readFile(file, 'utf8');
        } catch (err) {
            if (err.code !== 'ENOENT') throw err;
        }
        if (content !== undefined) {
            const parsed = JSON.parse(content);
            if (!Array.isArray(parsed.users)) throw new Error('CHiPS users.json has an invalid users list.');
            const backup = `${file}.backup-${Date.now()}`;
            await fs.writeFile(backup, content, { flag: 'wx', mode: 0o600 });
            console.log(`Account backup saved: ${backup}`);
        }
    }
    await store.init();
    const existing = await store.getUser('admin');
    if (existing) {
        if (existing.role !== 'admin') {
            throw new Error('The CHiPS account named admin has a different role; no password changed.');
        }
        if (!store.isEmailVerified(existing)) {
            throw new Error('CHiPS admin requires email confirmation; a password reset alone will not allow login.');
        }
        await store.setPassword('admin', password);
    } else {
        await store.createUser({ username: 'admin', password, role: 'admin' });
    }
    console.log(`CHiPS admin password saved (${store.backendName()} store).`);
    console.log('Sign in to CHiPS with username: admin');
}

main().catch(err => {
    console.error(`Reset failed: ${err.message}`);
    process.exitCode = 1;
}).finally(async () => {
    try { await db.close(); } catch (_) { process.exitCode = 1; }
});
