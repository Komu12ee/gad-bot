'use strict';

// Approved evidence, never a model-training queue. Each bot/department has a
// separate namespace; PostgreSQL commits the fact, vectors, audit and eval case
// together. Readers query live rows, so withdrawal never relies on cache expiry.
const crypto = require('crypto');
const express = require('express');
const db = require('./gad/db');
const { internalKey } = require('./platform/pyservice');
const fail = (status, message) => Object.assign(new Error(message), { status });
const MODEL = 'BAAI/bge-m3';
const UUID = /^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;
const enabled = () => !/^(0|off|false|no)$/i.test(process.env.VERIFIED_CORRECTIONS || '1');

function indiaDate() {
  const parts = Object.fromEntries(new Intl.DateTimeFormat('en', {
    timeZone: 'Asia/Kolkata', year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(new Date()).map(p => [p.type, p.value]));
  return `${parts.year}-${parts.month}-${parts.day}`;
}
function date(value, name, optional = false) {
  if (optional && !value) return null;
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)
      || !Number.isFinite(Date.parse(value)) || new Date(value).toISOString().slice(0, 10) !== value) {
    throw fail(400, `${name}: use a valid YYYY-MM-DD date.`);
  }
  return value;
}
function text(value, name, max) {
  if (typeof value !== 'string' || !value.trim() || value.length > max || /\u0000/.test(value)) {
    throw fail(400, `${name} is required (${max} characters at most).`);
  }
  return value.trim();
}
function officialUrl(value) {
  let u;
  try { u = new URL(value); } catch (_) { throw fail(400, 'Give the official source URL.'); }
  if (!['http:', 'https:'].includes(u.protocol) || u.username || u.password
      || !/(^|\.)(gov|nic)\.in$/i.test(u.hostname) || u.port) {
    throw fail(400, 'Use an official gov.in or nic.in source URL without credentials or a custom port.');
  }
  return u.href;
}
function validate(body) {
  const today = indiaDate();
  const record = {
    topic: text(body.topic, 'Topic', 160), question: text(body.question, 'Question', 2000),
    answer: text(body.answer, 'Correct answer', 2500),
    sourceTitle: text(body.sourceTitle, 'Source title / order reference', 300),
    sourceUrl: officialUrl(text(body.sourceUrl, 'Source URL', 2000)),
    sourceExcerpt: text(body.sourceExcerpt, 'Supporting source excerpt', 3500),
    effectiveFrom: date(body.effectiveFrom, 'Effective from'),
    effectiveUntil: date(body.effectiveUntil, 'Effective until', true),
    reviewBy: date(body.reviewBy, 'Review by'),
    variants: [],
  };
  if (body.verified !== true) throw fail(400, 'Verify the source and applicability before publishing.');
  if (record.effectiveUntil && record.effectiveUntil < record.effectiveFrom) throw fail(400, 'Effective until must be on/after Effective from.');
  if (record.reviewBy < today) throw fail(400, 'Review by must be today or later.');
  const variants = body.variants ?? [];
  if (!Array.isArray(variants) || variants.length > 8 || variants.some(v=>typeof v !== 'string')) throw fail(400, 'Use up to eight alternative questions.');
  record.variants = [...new Set(variants.filter(v => v.trim())
    .map(v => text(v, 'Alternative question', 2000)))].filter(v => v !== record.question);
  return record;
}
function vector(value) {
  if (!Array.isArray(value) || value.length !== 1024 || value.some(x => !Number.isFinite(x))
      || !value.some(x => Math.abs(x) > 1e-12)) throw fail(400, 'A valid bge-m3 embedding is required.');
  return '[' + value.join(',') + ']';
}
function scope(value) {
  if (!/^[a-z0-9_-]+(?::dept:[1-9][0-9]*)?$/.test(value)) throw fail(400, 'Invalid correction scope.');
  return value;
}
const rowItem = row => ({ ...row.record, id: row.id, revision: row.revision,
  state: row.state, updatedAt: row.updated_at });

function createService({ py }) {
  let ready = false, starting = null, lastTry = 0;
  let error = null;
  async function init() {
    if (!enabled()) return false;
    if (ready) return true;
    if (starting) return starting;
    if (!db.haveConfig()) { error = 'Configure PostgreSQL with pgvector to use verified corrections.'; return false; }
    if (Date.now() - lastTry < 30000) return false;
    lastTry = Date.now();
    starting = (async () => {
      let c;
      try {
        c = await db.pool().connect();
        await c.query('BEGIN');
        await c.query("SELECT pg_advisory_xact_lock(hashtext('verified-corrections-schema-v1'))");
        await c.query('CREATE EXTENSION IF NOT EXISTS vector');
        await c.query(`CREATE TABLE IF NOT EXISTS verified_corrections (
          pipeline TEXT NOT NULL, id UUID NOT NULL, revision INTEGER NOT NULL,
          state TEXT NOT NULL CHECK(state IN ('approved','withdrawn')),
          record JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          PRIMARY KEY(pipeline,id))`);
        await c.query(`CREATE TABLE IF NOT EXISTS verified_correction_questions (
          pipeline TEXT NOT NULL, correction_id UUID NOT NULL, position INTEGER NOT NULL,
          question TEXT NOT NULL, embedding vector(1024) NOT NULL,
          PRIMARY KEY(pipeline,correction_id,position),
          FOREIGN KEY(pipeline,correction_id) REFERENCES verified_corrections(pipeline,id))`);
        await c.query(`CREATE TABLE IF NOT EXISTS verified_correction_history (
          pipeline TEXT NOT NULL, correction_id UUID NOT NULL, revision INTEGER NOT NULL,
          event TEXT NOT NULL, actor TEXT NOT NULL, record JSONB NOT NULL,
          saved_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY(pipeline,correction_id,revision))`);
        await c.query(`CREATE TABLE IF NOT EXISTS verified_correction_evals (
          pipeline TEXT NOT NULL, correction_id UUID NOT NULL, revision INTEGER NOT NULL,
          position INTEGER NOT NULL, question TEXT NOT NULL, expected_answer TEXT NOT NULL, source_url TEXT NOT NULL,
          validity JSONB NOT NULL, PRIMARY KEY(pipeline,correction_id,revision,position))`);
        await c.query('COMMIT'); ready = true; error = null;
      } catch (err) {
        if (c) await c.query('ROLLBACK').catch(() => {});
        console.error('[corrections] schema unavailable:', err.message);
        error = 'Verified corrections are unavailable. Check PostgreSQL and its pgvector extension on the server.';
      } finally { if (c) c.release(); }
      return ready;
    })();
    try { return await starting; } finally { starting = null; }
  }
  async function requireReady() {
    if (!(await init())) throw fail(503, enabled() ? error : 'Verified corrections are disabled by configuration.');
  }
  async function transact(fn) {
    const c = await db.pool().connect();
    try { await c.query('BEGIN'); const out = await fn(c); await c.query('COMMIT'); return out; }
    catch (err) { await c.query('ROLLBACK').catch(() => {}); throw err; }
    finally { c.release(); }
  }
  async function locked(c, key, id) {
    await c.query('SELECT pg_advisory_xact_lock(hashtext($1))', [`correction:${key}:${id}`]);
    return (await c.query('SELECT * FROM verified_corrections WHERE pipeline=$1 AND id=$2 FOR UPDATE', [key,id])).rows[0];
  }
  async function audit(c, key, item, event, actor) {
    await c.query(`INSERT INTO verified_correction_history(pipeline,correction_id,revision,event,actor,record)
      VALUES($1,$2,$3,$4,$5,$6)`, [key,item.id,item.revision,event,actor,JSON.stringify(item)]);
  }
  return {
    init,
    async status() { await init(); return { enabled: enabled(), ready, error, today: indiaDate(), model: MODEL }; },
    async list(key, { page = 1, q = '', state = '' } = {}) {
      await requireReady(); scope(key);
      const today = indiaDate();
      const filter = `pipeline=$1 AND ($2='' OR record->>'topic' ILIKE $2 OR record->>'question' ILIKE $2)
        AND ($3='' OR ($3='withdrawn' AND state='withdrawn')
        OR ($3='active' AND state='approved' AND record->>'reviewBy'>=$4 AND record->>'effectiveFrom'<=$4
          AND (record->>'effectiveUntil' IS NULL OR record->>'effectiveUntil'>=$4))
        OR ($3='expired' AND state='approved' AND record->>'reviewBy'<$4)
        OR ($3='historical' AND state='approved' AND record->>'reviewBy'>=$4 AND record->>'effectiveUntil'<$4)
        OR ($3='future' AND state='approved' AND record->>'reviewBy'>=$4 AND record->>'effectiveFrom'>$4))`;
      const args = [key,q ? `%${String(q).slice(0,160)}%` : '',state,today];
      const total = Number((await db.query(`SELECT count(*) AS n FROM verified_corrections WHERE ${filter}`, args)).rows[0].n);
      page = Math.max(1, Math.min(Number.parseInt(page,10) || 1, Math.max(1,Math.ceil(total/20))));
      const rows = (await db.query(`SELECT * FROM verified_corrections WHERE ${filter} ORDER BY updated_at DESC,id LIMIT 20 OFFSET $5`, [...args,(page-1)*20])).rows;
      return { items: rows.map(rowItem), total, page, pageSize: 20 };
    },
    async publish(key, actor, body) {
      await requireReady(); scope(key);
      if (!UUID.test(body.id || '') || !Number.isSafeInteger(body.revision) || body.revision < 0 || body.revision > 2000000000) throw fail(400, 'Invalid correction ID or revision.');
      const record = validate(body);
      const questions = [record.question,...record.variants];
      const encoded = await py('/api/dept/correction-embeddings', { questions });
      if (encoded.model !== MODEL || encoded.vectors?.length !== questions.length) throw fail(503, 'Correction indexing failed; nothing was published.');
      const vectors = encoded.vectors.map(vector);
      return transact(async c => {
        const old = await locked(c,key,body.id);
        // A lost HTTP response can be retried without creating a second revision.
        if (old?.revision === body.revision+1 && old.state==='approved' && old.record.approvedBy===actor
            && Object.keys(record).every(k=>JSON.stringify(old.record[k])===JSON.stringify(record[k]))) return rowItem(old);
        if ((old?.revision || 0) !== body.revision) throw fail(409, 'This correction changed. Refresh before editing it again.');
        const revision = body.revision + 1;
        Object.assign(record, { approvedBy: actor, approvedAt: new Date().toISOString(), embeddingModel: MODEL });
        const item = { ...record, id: body.id, revision, state: 'approved' };
        await c.query(`INSERT INTO verified_corrections(pipeline,id,revision,state,record) VALUES($1,$2,$3,'approved',$4)
          ON CONFLICT(pipeline,id) DO UPDATE SET revision=$3,state='approved',record=$4,updated_at=now()`, [key,item.id,revision,JSON.stringify(record)]);
        await c.query('DELETE FROM verified_correction_questions WHERE pipeline=$1 AND correction_id=$2', [key,item.id]);
        for (let n=0;n<questions.length;n++) {
          await c.query(`INSERT INTO verified_correction_questions(pipeline,correction_id,position,question,embedding)
            VALUES($1,$2,$3,$4,$5::vector)`,[key,item.id,n,questions[n],vectors[n]]);
          await c.query(`INSERT INTO verified_correction_evals(pipeline,correction_id,revision,position,question,expected_answer,source_url,validity)
            VALUES($1,$2,$3,$4,$5,$6,$7,$8)`,[key,item.id,revision,n,questions[n],record.answer,record.sourceUrl,
            JSON.stringify({effectiveFrom:record.effectiveFrom,effectiveUntil:record.effectiveUntil,reviewBy:record.reviewBy})]);
        }
        await audit(c,key,item,old ? 'revised' : 'approved',actor);
        return item;
      });
    },
    async withdraw(key, actor, id, revision) {
      await requireReady(); scope(key);
      if (!UUID.test(id)) throw fail(400,'Invalid correction ID.');
      return transact(async c => {
        const old = await locked(c,key,id);
        if (!old) throw fail(404,'Correction not found.');
        if (old.revision !== revision) throw fail(409,'This correction changed. Refresh first.');
        if (old.state === 'withdrawn') return rowItem(old);
        const item = {...rowItem(old),revision:old.revision+1,state:'withdrawn',withdrawnBy:actor,withdrawnAt:new Date().toISOString()};
        await c.query(`UPDATE verified_corrections SET revision=$3,state='withdrawn',updated_at=now() WHERE pipeline=$1 AND id=$2`,[key,id,item.revision]);
        await audit(c,key,item,'withdrawn',actor);
        return item;
      });
    },
    async history(key,id) {
      await requireReady(); scope(key);
      if (!UUID.test(id)) throw fail(400,'Invalid correction ID.');
      return (await db.query(`SELECT revision,event,actor,record,saved_at FROM verified_correction_history
        WHERE pipeline=$1 AND correction_id=$2 ORDER BY revision DESC`,[key,id])).rows;
    },
    async dataset(key) {
      await requireReady(); scope(key);
      const today=indiaDate();
      return (await db.query(`SELECT e.* FROM verified_correction_evals e
        JOIN verified_corrections c ON e.pipeline=c.pipeline AND e.correction_id=c.id
        WHERE e.pipeline=$1 AND e.revision=c.revision AND c.state='approved'
        ORDER BY e.correction_id,e.position`,[key])).rows.map(row=>({...row,
          enabled:row.validity.effectiveFrom<=today && (!row.validity.effectiveUntil || row.validity.effectiveUntil>=today)
            && row.validity.reviewBy>=today}));
    },
    async search(key, dense, question, dateQuestion = question) {
      if (!(await init())) return [];
      scope(key);
      const embedding = vector(dense);
      const today = indiaDate();
      const window = dateWindow(dateQuestion,today);
      if (!window) return [];
      const minScore = Number(process.env.CORRECTION_MIN_SCORE || '0.84');
      const rows = (await db.query(`SELECT c.id,c.revision,c.record,MAX(1-(q.embedding <=> $2::vector)) AS score
        FROM verified_corrections c JOIN verified_correction_questions q ON q.pipeline=c.pipeline AND q.correction_id=c.id
        WHERE c.pipeline=$1 AND c.state='approved' AND c.record->>'embeddingModel'=$3 AND c.record->>'reviewBy'>=$4
          AND c.record->>'effectiveFrom'<=$5 AND (c.record->>'effectiveUntil' IS NULL OR c.record->>'effectiveUntil'>=$6)
        GROUP BY c.pipeline,c.id ORDER BY score DESC LIMIT 3`,[key,embedding,MODEL,today,window.from,window.to])).rows;
      return rows.filter(r => Number(r.score) >= (Number.isFinite(minScore) && minScore>=0 && minScore<=1 ? minScore : 0.84)).map(r => ({
        id:r.id, revision:r.revision, score:Number(r.score), topic:r.record.topic, question:r.record.question,
        answer:r.record.answer, sourceTitle:r.record.sourceTitle,sourceUrl:r.record.sourceUrl,sourceExcerpt:r.record.sourceExcerpt,
        effectiveFrom:r.record.effectiveFrom,effectiveUntil:r.record.effectiveUntil,reviewBy:r.record.reviewBy,
        approvedAt:r.record.approvedAt, target:window,
      }));
    },
  };
}

// Only unambiguous relative dates / ISO dates / whole years are resolved here.
// Ambiguous Hindi "कल" and unsupported numeric dates use the document search.
function dateWindow(question,today=indiaDate()) {
  const q = String(question || '').normalize('NFKC').replace(/[०-९]/g,c=>String(c.charCodeAt(0)-0x966));
  const iso = [...q.matchAll(/\b((?:19|20|21)\d{2}-\d{2}-\d{2})\b/g)].map(m=>m[1]);
  try { iso.forEach(d=>date(d,'Question date')); } catch (_) { return null; }
  if (iso.length) return {from:[...iso].sort()[0],to:[...iso].sort().at(-1)};
  if (/\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}/.test(q)
    || /\d/.test(q) && /\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर/i.test(q)) return null;
  if (/(^|[^ऀ-ॿA-Za-z])(?:परसों|कल)($|[^ऀ-ॿA-Za-z])|\b(?:parso[n]?|kal)\b/i.test(q)) return null;
  if (/\b(?:last|next|previous)\s+(?:week|month|year)\b|(?:पिछले|अगले|गत|आगामी)\s*(?:वर्ष|साल|महीने|सप्ताह)/i.test(q)) return null;
  const offsets=[];
  if (/\btomorrow\b/i.test(q)) offsets.push(/day\s+after\s+tomorrow/i.test(q)?2:1);
  if (/\byesterday\b/i.test(q)) offsets.push(/day\s+before\s+yesterday/i.test(q)?-2:-1);
  if (/\btoday\b|\baaj\b|(^|[^ऀ-ॿA-Za-z])आज($|[^ऀ-ॿA-Za-z])/i.test(q)) offsets.push(0);
  if (offsets.length) {
    const dates=offsets.map(offset=>{
      const d=new Date(today+'T12:00:00Z');d.setUTCDate(d.getUTCDate()+offset);return d.toISOString().slice(0,10);
    }).sort();
    return {from:dates[0],to:dates.at(-1)};
  }
  const years = [...q.matchAll(/\b((?:19|20|21)\d{2})\b/g)].map(m=>m[1]).sort();
  if (years.length) return {from:years[0]+'-01-01',to:years.at(-1)+'-12-31'};
  return {from:today,to:today};
}
function passage(item) {
  return `Admin-verified correction: ${item.topic}\nApplicable question: ${item.question}\n`
    + `Effective: ${item.effectiveFrom} to ${item.effectiveUntil || 'until superseded'}; review by ${item.reviewBy}.\n`
    + `Official source: ${item.sourceTitle}\nApproved answer: ${item.answer}\nSupporting source excerpt:\n${item.sourceExcerpt}`;
}

function createRouter(service,{scope: getScope, actor: getActor}) {
  const router = express.Router();
  router.use((req,res,next) => {
    res.setHeader('Cache-Control','no-store');
    const actor = getActor(req);
    if (!actor || actor.role !== 'admin') return res.status(403).json({success:false,error:'Admin access required.'});
    req.correctionActor = actor.username; next();
  });
  router.use(express.json({limit:'128kb'}));
  router.use((req,_res,next) => {
    if (typeof req.body?.b64 === 'string') {
      try { req.body = JSON.parse(Buffer.from(req.body.b64,'base64').toString('utf8')); }
      catch (_) { return next(fail(400,'Invalid correction data.')); }
    }
    next();
  });
  const key = req => scope(typeof getScope === 'function' ? getScope(req) : getScope);
  const handle = fn => (req,res,next) => Promise.resolve(fn(req,res)).catch(next);
  router.get('/status',handle(async (_req,res)=>res.json({success:true,...await service.status()})));
  router.get('/',handle(async (req,res)=>res.json({success:true,...await service.list(key(req),req.query)})));
  router.get('/dataset',handle(async (req,res)=>res.json({success:true,scope:key(req),cases:await service.dataset(key(req))})));
  router.get('/:id/history',handle(async (req,res)=>res.json({success:true,events:await service.history(key(req),req.params.id)})));
  router.post('/',handle(async (req,res)=>res.json({success:true,item:await service.publish(key(req),req.correctionActor,req.body || {})})));
  router.patch('/:id',handle(async (req,res)=>res.json({success:true,item:await service.withdraw(key(req),req.correctionActor,req.params.id,req.body?.revision)})));
  router.use((err,_req,res,_next)=>{
    const status = err.status || 503;
    if (status >= 500) console.error('[corrections]',err.message);
    res.status(status).json({success:false,error:status >= 500 ? 'The correction service is unavailable; nothing was published. Check the server log.' : err.message});
  });
  return router;
}

function internalRouter(service,pipeline) {
  const router = express.Router();
  router.use((req,res,next)=>{
    const expected = Buffer.from(internalKey());
    const given = Buffer.from(req.get('X-Corrections-Internal') || '');
    if (!['127.0.0.1','::1','::ffff:127.0.0.1'].includes(req.socket.remoteAddress)
      || !expected.length || expected.length !== given.length || !crypto.timingSafeEqual(expected,given)) return res.sendStatus(403);
    res.setHeader('Cache-Control','no-store'); next();
  });
  router.use(express.json({limit:'128kb'}));
  router.post('/search',(req,res,next)=>Promise.resolve(service.search(pipeline,req.body?.dense,req.body?.question,req.body?.dateQuestion))
    .then(matches=>res.json({success:true,matches})).catch(next));
  router.use((err,_req,res,_next)=>{ console.error('[corrections] lookup:',err.message); res.status(503).json({success:false,error:'Correction lookup unavailable.'}); });
  return router;
}
module.exports = { createService,createRouter,internalRouter,dateWindow,passage,MODEL };
