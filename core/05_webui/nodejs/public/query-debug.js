// Loaded only by the chat. The API independently checks the admin session and
// trace owner; DOM visibility is not the access control.
function buildQueryDebug(id, base = '', error = '') {
  const panel = document.createElement('details');
  panel.className = 'query-debug';
  const heading = document.createElement('summary');
  heading.textContent = 'Query processing details · Admin';
  panel.appendChild(heading);
  const content = document.createElement('div'); content.className = 'query-debug-content';
  panel.appendChild(content);
  let loaded = false, busy = false;
  const showText = text => { content.textContent = text; };
  panel.addEventListener('toggle', async () => {
    if (!panel.open || loaded || busy) return;
    if (!id) {
      showText(error || 'No processing trace was captured for this answer. Ask the question again after this update.');
      return;
    }
    busy = true;
    showText('Loading processing steps…');
    try {
      const response = await fetch(`${base}/api/admin/query-debug/${encodeURIComponent(id)}`,
        { credentials: 'same-origin', cache: 'no-store' });
      const body = await response.json();
      if (!response.ok || !body.success) throw new Error(body.error || 'Could not load processing details.');
      const trace = body.debug;
      content.textContent = '';
      const top = document.createElement('div'); top.className = 'query-debug-toolbar';
      const note = document.createElement('span');
      const requests = trace.steps.filter(s => s.output?.request_body?.model);
      const model = requests.at(-1)?.output.request_body.model;
      note.textContent = `${model || 'No LLM call'} · ${trace.steps.length} steps · ${trace.elapsed_s}s`;
      const download = document.createElement('button'); download.type = 'button';
      download.textContent = 'Download JSON';
      download.addEventListener('click', () => {
        const url = URL.createObjectURL(new Blob([JSON.stringify(trace, null, 2)], { type: 'application/json' }));
        const link = document.createElement('a'); link.href = url; link.download = `query-${trace.id}.json`;
        link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      });
      top.append(note, download); content.appendChild(top);
      if (trace.limit_reached) {
        const warning = document.createElement('p'); warning.textContent = 'Some outputs exceeded the 32 MiB trace limit. Omitted steps are marked.';
        content.appendChild(warning);
      }
      const stamp = document.createElement('p'); stamp.className = 'query-debug-stamp';
      stamp.textContent = `${trace.pipeline} · ${trace.created_at} · ${trace.id}`;
      content.appendChild(stamp);
      for (const step of trace.steps) {
        const row = document.createElement('details'); row.className = 'query-debug-step';
        const title = document.createElement('summary');
        title.textContent = `${step.number}. ${step.title}  (+${step.elapsed_s}s)`;
        row.appendChild(title);
        // Build large outputs only when that step is expanded.
        let rendered = false;
        row.addEventListener('toggle', () => {
          if (!row.open || rendered) return;
          rendered = true;
          renderDebugOutput(row, step.output);
        });
        content.appendChild(row);
      }
      loaded = true;
    } catch (err) { showText(`${err.message} Close and reopen to retry.`); }
    finally { busy = false; }
  });
  return panel;
}

function renderDebugOutput(parent, output) {
  const chunks = output?.chunks;
  const remaining = Array.isArray(chunks) ? Object.fromEntries(Object.entries(output).filter(([key]) => key !== 'chunks')) : output;
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(remaining, null, 2);
  parent.appendChild(pre);
  if (!Array.isArray(chunks)) return;
  const count = document.createElement('p'); count.textContent = `${chunks.length} chunk(s) — expand one to see its full text and metadata.`;
  parent.appendChild(count);
  for (const chunk of chunks) {
    const row = document.createElement('details'); row.className = 'query-debug-chunk';
    const summary = document.createElement('summary');
    const payload = chunk.payload || {};
    summary.textContent = `#${chunk.rank} · ${payload.source || chunk.id} · chunk ${payload.chunk ?? '?'} · score ${chunk.score ?? 'n/a'}`;
    row.appendChild(summary);
    let rendered = false;
    row.addEventListener('toggle', () => {
      if (!row.open || rendered) return;
      rendered = true;
      const text = document.createElement('pre'); text.className = 'query-debug-chunk-text';
      text.textContent = payload.text || '(No stored text)'; row.appendChild(text);
      const metadata = document.createElement('pre');
      metadata.textContent = JSON.stringify({ ...chunk, payload: { ...payload, text: undefined } }, null, 2);
      row.appendChild(metadata);
    });
    parent.appendChild(row);
  }
}
