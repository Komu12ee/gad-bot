'use strict';

/**
 * Ranking a department's passages for a question — the same steps as the bots'
 * retrieval (rag_pipeline.retrieve_context), over PostgreSQL instead of Qdrant:
 *
 *   1. dense: each query variation's nearest passages (pgvector, cosine); a passage
 *      found by several variations keeps its best score
 *   2. sparse: those candidates re-scored by bge-m3's lexical weights (dot product
 *      with the question's) — sparse re-ranks what dense found, as in the bots
 *   3. hybrid: reciprocal rank fusion, alpha dense / 1 - alpha sparse (k = 60)
 *   4. exact identifiers (an order number such as "F-4-1/2023") put the passages
 *      that contain them first, at most two
 *
 * The result goes to the answer step (dept_service.py), which reranks it with the
 * cross-encoder when that is switched on.
 */

const RRF_K = 60;
const MAX_EXACT = 2;

/** Variations' dense hits → one list, best score per passage, best first. */
function mergeDense(lists) {
  const best = new Map();
  for (const list of lists) {
    for (const c of list || []) {
      const had = best.get(c.id);
      if (!had || c.score > had.score) best.set(c.id, c);
    }
  }
  return [...best.values()].sort((a, b) => b.score - a.score);
}

function sparseScore(query, chunk) {
  let s = 0;
  for (const [token, w] of Object.entries(query || {})) {
    const v = chunk[token];
    if (v) s += w * v;
  }
  return s;
}

/** RRF over the dense order and the sparse order of the same candidates. */
function hybrid(dense, querySparse, alpha) {
  const scores = new Map();
  dense.forEach((c, rank) => scores.set(c.id, alpha / (RRF_K + rank + 1)));
  if (querySparse && Object.keys(querySparse).length) {
    const sparse = dense
      .map((c) => ({ c, s: sparseScore(querySparse, c.sparse || {}) }))
      .filter((x) => x.s > 0)
      .sort((a, b) => b.s - a.s);
    sparse.forEach(({ c }, rank) => scores.set(c.id, (scores.get(c.id) || 0) + (1 - alpha) / (RRF_K + rank + 1)));
  }
  const byId = new Map(dense.map((c) => [c.id, c]));
  return [...scores.entries()].sort((a, b) => b[1] - a[1]).map(([id, s]) => ({ ...byId.get(id), fused: s }));
}

/**
 * Words of the question that identify something exactly: contain a digit and a
 * letter or separator ("F-4-1/2023", "क्र./1234", "GAD-12"). Bare numbers and
 * years are too common to mean one passage.
 */
function exactTerms(question) {
  const out = new Set();
  for (const raw of String(question || '').split(/[\s,;:()"'“”‘’?]+/)) {
    const t = raw.replace(/^[.\-/]+|[.\-/]+$/g, '');
    if (t.length < 4 || t.length > 40) continue;
    if (!/\d/.test(t)) continue;
    if (/^\d+$/.test(t)) continue;
    if (!/[A-Za-zऀ-ॿ]|[-/.]/.test(t)) continue;
    out.add(t);
  }
  return [...out].slice(0, 5);
}

/** Exact-identifier passages first (at most MAX_EXACT), then the hybrid order. */
function withExact(ranked, exact) {
  const first = (exact || []).slice(0, MAX_EXACT);
  const ids = new Set(first.map((c) => c.id));
  return [...first, ...ranked.filter((c) => !ids.has(c.id))];
}

/**
 * The full ranking. `lists`: dense hits per variation; `querySparse`: the
 * question's lexical weights; `exact`: passages holding an exact identifier.
 */
function rank(lists, querySparse, alpha, exact, limit) {
  const ranked = withExact(hybrid(mergeDense(lists), querySparse, alpha), exact);
  return limit ? ranked.slice(0, limit) : ranked;
}

module.exports = { mergeDense, sparseScore, hybrid, exactTerms, withExact, rank };
