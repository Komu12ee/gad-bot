'use strict';

/**
 * A question to a department's chatbot:
 *
 *   question (+ the last turns) ──> Python: standalone question, variations, vectors
 *        ──> PostgreSQL: the department's nearest passages per variation, exact
 *            identifiers (store.searchDense / searchExact) ──> retrieval.rank
 *        ──> Python: rerank, the bots' answer prompt, the answer model, Sources
 *
 * Only the asking department's passages are ever searched. Every question is
 * logged with its answer and sources (department_queries) for the history and
 * review screens.
 */

const retrieval = require('./retrieval');
const { failure } = require('./http');

const MAX_QUESTION = 2000;
const MAX_TURNS = 4;
const MAX_CANDIDATES = 100;
const ANSWER_TIMEOUT_MS = 280 * 1000;          // under nginx's 300 s, so the page gets a clear message

const hindi = (s) => /[ऀ-ॿ]/.test(s);
const NOTHING = {
  hi: 'इस विभाग के दस्तावेज़ों में इस प्रश्न का उत्तर नहीं मिला।',
  en: 'The department’s documents do not answer this question.',
};
const NO_DOCUMENTS = {
  hi: 'इस विभाग के दस्तावेज़ अभी तैयार नहीं हैं — विभाग के व्यवस्थापक को उन्हें प्रोसेस करना होगा।',
  en: 'This department’s documents are not ready yet — the department admin has to process them first.',
};

/** What the page sends as chat history: [{question, answer}], recent turns only, text only. */
function cleanHistory(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.slice(-MAX_TURNS)
    .filter((t) => t && typeof t.question === 'string')
    .map((t) => ({ question: t.question.slice(0, MAX_QUESTION), answer: String(t.answer || '').slice(0, 4000) }));
}

function createChat({ store, py, log = console, corrections, correctionScope }) {
  // One question at a time per account: the answer model is shared by every bot,
  // and a second click (or a script) must not queue more work behind the first.
  const asking = new Set();

  async function record(dept, user, question, out, t0) {
    try {
      await store.logQuery({
        departmentId: dept.id, userId: user.id, username: user.username, question,
        answer: out.answer || null, sources: out.sources || [], status: out.status, error: out.error,
        seconds: (Date.now() - t0) / 1000,
      });
    } catch (err) {
      log.error(`[platform] question not logged: ${err.message}`);
    }
  }

  return async function ask(dept, user, body) {
    const key = `${dept.id}:${user.id}`;
    if (asking.has(key)) throw failure(429, 'Wait for the answer to your previous question.');
    asking.add(key);
    try {
      return await answer(dept, user, body);
    } finally {
      asking.delete(key);
    }
  };

  async function answer(dept, user, body) {
    const t0 = Date.now();
    const question = String((body && body.question) || '').trim();
    if (!question) throw failure(400, 'Type a question.');
    if (question.length > MAX_QUESTION) throw failure(400, `The question is too long (${MAX_QUESTION} characters at most).`);
    if (dept.status !== 'active') throw failure(403, 'CHiPS has switched this department’s assistant off.');
    const history = cleanHistory(body && body.history);
    const lang = hindi(question) ? 'hi' : 'en';

    const haveDocuments = !!(await store.countChunks(dept.id));
    if (!haveDocuments && !corrections) {
      const out = { answer: NO_DOCUMENTS[lang], sources: [], status: 'no_documents' };
      await record(dept, user, question, out, t0);
      return out;
    }

    let enc;
    try {
      enc = await py('/api/dept/encode', { question, history });
    } catch (err) {
      await record(dept, user, question, { status: 'failed', error: err.message }, t0);
      throw failure(err.status === 400 ? 400 : 503, err.message);
    }
    const variations = enc.variations || [];
    const lists = haveDocuments ? await Promise.all(variations.map((v) => store.searchDense(dept.id, v.dense, enc.dense_limit || 60))) : [];
    const exact = haveDocuments ? await store.searchExact(dept.id, retrieval.exactTerms(question), 10) : [];
    const ranked = retrieval.rank(lists, variations[0] && variations[0].sparse, enc.alpha ?? 0.6, exact,
      Math.min(enc.candidates || MAX_CANDIDATES, MAX_CANDIDATES));
    let memory = [];
    if (corrections && variations[0]?.dense) {
      try { memory = await corrections.search(correctionScope(dept.id),variations[0].dense,enc.search_question || question); }
      catch (err) { log.error('[corrections] department lookup unavailable:',err.message); }
    }
    ranked.unshift(...memory.map(item=>({id:`correction:${item.id}:v${item.revision}`,
      fileId:`verified-correction:${item.id}:v${item.revision}`,fileName:item.sourceTitle,seq:0,
      text:require('../verified-corrections').passage(item),verifiedCorrection:item})));
    if (!ranked.length) {
      const out = { answer: haveDocuments ? NOTHING[lang] : NO_DOCUMENTS[lang], sources: [], status: haveDocuments ? 'not_found' : 'no_documents' };
      await record(dept, user, question, out, t0);
      return out;
    }

    let res;
    try {
      res = await py('/api/dept/answer', {
        department: dept.name, question, search_question: enc.search_question || question, history,
        candidates: ranked.map((c) => ({
          id: c.id, file_id: c.fileId, file_name: c.fileName, seq: c.seq, text: c.text,
          context_line: c.contextLine, section: c.section,
          verified_correction: c.verifiedCorrection,
        })),
      }, ANSWER_TIMEOUT_MS);
    } catch (err) {
      await record(dept, user, question, { status: 'failed', error: err.message }, t0);
      throw failure(err.status === 400 ? 400 : 503, err.message);
    }
    const out = {
      answer: res.answer,
      sources: (res.sources || []).map((s) => ({ n: s.n, fileId: s.file_id, name: s.name, url: s.url })),
      status: 'answered',
    };
    await record(dept, user, question, out, t0);
    return out;
  }
}

module.exports = { createChat, cleanHistory };
