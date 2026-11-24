'use strict';

// The same correction editor for a pipeline's Evaluation page and a department
// admin's panel. API scope and administrator identity are enforced server-side.
(function () {
  function node(tag, className = '', value = '') {
    const el = document.createElement(tag); el.className = className; el.textContent = value; return el;
  }
  function uuid() {
    if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = bytes[6] & 15 | 64; bytes[8] = bytes[8] & 63 | 128;
    const s = [...bytes].map(b=>b.toString(16).padStart(2,'0')).join('');
    return `${s.slice(0,8)}-${s.slice(8,12)}-${s.slice(12,16)}-${s.slice(16,20)}-${s.slice(20)}`;
  }
  function mount(root, { api, login, onPublished } = {}) {
    root.innerHTML = `
      <div class="vc-toolbar"><div><h2>Verified corrections</h2><p>Approved facts shared across this assistant's chats. Other assistants keep separate records.</p></div>
        <div class="vc-actions"><button type="button" data-vc="export">Export evaluation cases</button><button type="button" data-vc="new">New correction</button></div></div>
      <div class="vc-search"><label>Search<input type="search" data-vc="search" placeholder="Topic or question…"></label><label>Show<select data-vc="filter"><option value="">All corrections</option><option value="active">Active now</option><option value="expired">Needs re-verification</option><option value="historical">Historical only</option><option value="future">Future effective date</option><option value="withdrawn">Withdrawn</option></select></label><button type="button" data-vc="refresh">Refresh</button></div>
      <p data-vc="status" role="status"></p><div data-vc="records" class="vc-records"></div>
      <div class="vc-pager"><button type="button" data-vc="prev" disabled>Previous</button><span data-vc="page"></span><button type="button" data-vc="next" disabled>Next</button></div>
      <dialog class="vc-dialog"><form class="vc-form">
        <div class="vc-dialog-header"><h2>Publish verified correction</h2><button type="button" data-vc="cancel" aria-label="Close correction editor">Close</button></div>
        <p>Check the official source yourself before publishing. Publication supplies evidence to the chatbot; it does not retrain the model.</p>
        <details data-vc="original" hidden><summary>Original bot answer</summary><pre></pre></details>
        <div class="vc-grid">
          <label>Topic<input name="topic" required maxlength="160" placeholder="e.g. child care leave eligibility"></label>
          <label class="vc-full">Question<textarea name="question" required maxlength="2000" rows="2"></textarea></label>
          <label class="vc-full">Correct answer<textarea name="answer" required maxlength="2500" rows="4"></textarea></label>
          <label class="vc-full">Alternative questions (optional, one per line, up to eight)<textarea name="variants" rows="2" maxlength="16008" placeholder="Include Hindi / English equivalents that ask for the same fact."></textarea></label>
          <label class="vc-full">Official source title / order reference<input name="sourceTitle" required maxlength="300"></label>
          <label class="vc-full">Official source URL (gov.in or nic.in)<input name="sourceUrl" type="url" required maxlength="2000" placeholder="https://…"></label>
          <label class="vc-full">Supporting excerpt from the official source<textarea name="sourceExcerpt" required maxlength="3500" rows="4" placeholder="Paste the source text that supports this correction."></textarea></label>
          <label>Effective from<input name="effectiveFrom" type="date" required></label>
          <label>Effective until (optional)<input name="effectiveUntil" type="date"></label>
          <label>Review by<input name="reviewBy" type="date" required></label>
        </div>
        <p class="vc-hint">The review date limits how long this verification stays active. Recheck changing facts such as officer postings and holidays. A new publication replaces the previous version immediately.</p>
        <label class="vc-confirm"><input name="verified" type="checkbox" required> I checked the official source, the applicable department/position and these dates, and approve this correction for this assistant's users.</label>
        <p data-vc="form-status" role="status"></p><div class="vc-dialog-footer"><button type="submit" data-vc="publish">Approve and publish</button></div>
      </form></dialog>`;
    const $ = name => root.querySelector(`[data-vc="${name}"]`);
    const dialog = root.querySelector('dialog');
    const form = root.querySelector('form');
    const field = name => form.elements.namedItem(name);
    let page = 1, total = 0, today = '', ready = false, busy = false, editing = null, context = null, sequence = 0;
    async function request(method, suffix = '', body) {
      const response = await fetch(api + suffix, { method, credentials:'same-origin', headers:{'Content-Type':'application/json',Accept:'application/json'},
        ...(body === undefined ? {} : {body:JSON.stringify({b64:window.Platform.toBase64(new TextEncoder().encode(JSON.stringify(body)))})}) });
      if (response.status === 401) { location.href = login; throw new Error('Please sign in.'); }
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.error || 'The correction request failed.');
      return data;
    }
    const message = (text,error=false) => { $('status').textContent=text; $('status').className=error?'vc-error':''; };
    function stateLabel(item) {
      return item.state === 'withdrawn' ? 'Withdrawn'
        : item.reviewBy < today ? 'Needs re-verification'
        : item.effectiveUntil && item.effectiveUntil < today ? 'Historical only'
        : item.effectiveFrom > today ? 'Future effective date' : 'Active';
    }
    function button(label, action) {
      const b = node('button','',label); b.type='button'; b.addEventListener('click',action); return b;
    }
    function render(items) {
      $('records').replaceChildren();
      for (const item of items) {
        const article = node('article','vc-record');
        const head = node('div','vc-record-header');
        head.append(node('strong','',item.topic),node('span','vc-badge',stateLabel(item)));
        article.append(head,node('p','vc-question',item.question));
        const detail = node('details'); detail.append(node('summary','','Answer, source and dates'));
        detail.append(node('p','vc-answer',item.answer));
        const source = node('a','',item.sourceTitle); source.href=item.sourceUrl; source.target='_blank'; source.rel='noopener noreferrer';
        detail.append(source,node('p','vc-dates',`Effective ${item.effectiveFrom} to ${item.effectiveUntil || 'until superseded'} · Review by ${item.reviewBy}`),
          node('p','vc-hint',`Version ${item.revision} · Verified by ${item.approvedBy} on ${new Date(item.approvedAt).toLocaleString()}`));
        const excerpt = node('details'); excerpt.append(node('summary','','Supporting excerpt'),node('pre','',item.sourceExcerpt)); detail.append(excerpt);
        article.append(detail);
        const actions = node('div','vc-actions');
        actions.append(button(item.state==='withdrawn'?'Re-verify and publish':'Edit and re-verify',()=>open({item})));
        if (item.state !== 'withdrawn') actions.append(button('Withdraw',async()=>{
          if (!confirm(`Withdraw “${item.topic}”? Future requests will stop using this correction.`)) return;
          try { await request('PATCH',`/${item.id}`,{revision:item.revision}); await load(); }
          catch (err) { message(err.message,true); }
        }));
        actions.append(button('History',async()=>{
          try {
            const data = await request('GET',`/${item.id}/history`);
            let history = article.querySelector('.vc-history');
            if (history) { history.remove(); return; }
            history = node('details','vc-history'); history.open=true; history.append(node('summary','','Revision history'));
            for (const event of data.events) {
              const version = node('details');
              version.append(node('summary','',`v${event.revision} · ${event.event} · ${event.actor} · ${new Date(event.saved_at).toLocaleString()}`),
                node('pre','',JSON.stringify(event.record,null,2))); history.append(version);
            }
            article.append(history);
          } catch (err) { message(err.message,true); }
        }));
        article.append(actions); $('records').append(article);
      }
      $('prev').disabled = page<=1; $('next').disabled=page*20>=total;
      $('page').textContent=`Page ${page} of ${Math.max(1,Math.ceil(total/20))} · ${total} correction${total===1?'':'s'}`;
      message(items.length?'':'No corrections in this view.');
    }
    async function load() {
      const seq=++sequence;
      try {
        const state = await request('GET','/status');
        if (seq!==sequence) return;
        ready=state.enabled && state.ready; today=state.today;
        $('new').disabled=!ready; $('export').disabled=!ready;
        if (!ready) { message(state.error || 'Verified corrections are disabled.',true); return; }
        const params = new URLSearchParams({page,q:$('search').value,state:$('filter').value});
        const data=await request('GET','?'+params);
        if (seq!==sequence) return;
        page=data.page; total=data.total; render(data.items);
      } catch (err) { if(seq===sequence) message(err.message,true); }
    }
    function open({item, question='',originalAnswer='',evaluationId=null} = {}) {
      if (!ready) { message('Verified corrections are not ready on the server.',true); return; }
      form.reset(); editing=item || {id:uuid(),revision:0}; context={evaluationId};
      for (const name of ['topic','question','answer','sourceTitle','sourceUrl','sourceExcerpt','effectiveFrom','effectiveUntil','reviewBy']) field(name).value=item?.[name] || '';
      if (!item) {
        field('question').value=question;
        field('effectiveFrom').value=today;
        const due=new Date(today+'T12:00:00Z'); due.setUTCDate(due.getUTCDate()+30); field('reviewBy').value=due.toISOString().slice(0,10);
      }
      field('variants').value=(item?.variants || []).join('\n');
      $('original').hidden=!originalAnswer; $('original').querySelector('pre').textContent=originalAnswer;
      $('form-status').textContent=''; dialog.showModal();
    }
    form.addEventListener('submit',async event=>{
      event.preventDefault(); if(busy || !form.reportValidity()) return;
      const data={id:editing.id,revision:editing.revision,verified:field('verified').checked};
      for(const name of ['topic','question','answer','sourceTitle','sourceUrl','sourceExcerpt','effectiveFrom','effectiveUntil','reviewBy']) data[name]=field(name).value;
      data.variants=field('variants').value.split('\n').map(s=>s.trim()).filter(Boolean);
      busy=true; $('publish').disabled=true; $('cancel').disabled=true; $('form-status').textContent='Indexing and publishing…';
      try {
        await request('POST','',data); dialog.close(); await load(); message('Verified correction published for this assistant.');
        if(onPublished) { try { await onPublished(context); } catch(err) { message(`Published, but the review queue could not be updated: ${err.message}`,true); } }
      } catch(err) { $('form-status').textContent=err.message; }
      finally { busy=false; $('publish').disabled=false; $('cancel').disabled=false; }
    });
    dialog.addEventListener('cancel',event=>{if(busy)event.preventDefault();});
    $('cancel').addEventListener('click',()=>{if(!busy)dialog.close();});
    $('new').addEventListener('click',()=>open()); $('refresh').addEventListener('click',load);
    $('prev').addEventListener('click',()=>{page--;load();}); $('next').addEventListener('click',()=>{page++;load();});
    $('filter').addEventListener('change',()=>{page=1;load();});
    let debounce; $('search').addEventListener('input',()=>{clearTimeout(debounce);debounce=setTimeout(()=>{page=1;load();},250);});
    $('export').addEventListener('click',async()=>{
      try {
        const data=await request('GET','/dataset');
        const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
        const href=URL.createObjectURL(blob); const a=node('a'); a.href=href;a.download='verified-correction-evaluations.json';a.click();setTimeout(()=>URL.revokeObjectURL(href),1000);
      } catch(err) {message(err.message,true);}
    });
    $('new').disabled=true; $('export').disabled=true;
    const initialized=load();
    return {load,open:async options=>{await initialized;open(options);}};
  }
  window.VerifiedCorrections={mount};
})();
