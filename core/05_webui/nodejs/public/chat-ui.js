'use strict';

// The original chatbot's layout, answer renderer, message rows and recent-chat
// controls. Bot and department controllers supply their own authentication/API.
(function () {
function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Inline formatting: **bold** (the model emits this around headings, filenames
// and figures). Escaping runs first, so the ** markers are the only thing left
// to interpret — no raw HTML can slip through.
function renderInline(escaped) {
  // Input is already HTML-escaped. [label](url) becomes a clickable link — the
  // answer's Sources list uses that form, and without this the raw markdown text
  // would show on screen. Only http(s) URLs match, so the label cannot inject markup.
  return escaped
    .replace(/\[([^\]\n]+?)\]\((https?:\/\/[^)\s]+?)\)/g,
             '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>');
}

// Block renderer for assistant answers. Turns the model's plain text into real
// paragraphs, bold, and bullet lists so "**Total = 4.44**" shows bold and
// "- item" / "* item" / "1. item" lines become a proper list.
function renderMarkdown(raw) {
    const lines = escapeHtml(raw).split('\n');
    let html = '';
    let para = [];
    let list = null;
    let table = [];
    const flushPara = () => { if (para.length) { html += `<p>${renderInline(para.join('<br>'))}</p>`; para = []; } };
    const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
    const flushTable = () => {
      if (!table.length) return;
      const rows = table.filter((r) => r.replace(/[|:\-\s]/g, '') !== '')      // drop |---|---| lines
        .map((r) => r.trim().replace(/^\||\|$/g, '').split('|').map((c) => renderInline(c.trim())));
      if (rows.length) {
        html += '<div class="table-wrap"><table class="answer-table"><thead><tr>'
          + rows[0].map((c) => `<th>${c}</th>`).join('') + '</tr></thead><tbody>'
          + rows.slice(1).map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')
          + '</tbody></table></div>';
      }
      table = [];
    };
    for (const line of lines) {
      const t = line.trim();
      if (t.startsWith('|')) { flushPara(); closeList(); table.push(t); continue; }
      flushTable();
      const bullet = t.match(/^[-*•]\s+(.*)$/);
      const num = t.match(/^\d+[.)]\s+(.*)$/);
      const head = t.match(/^#{1,4}\s+(.*)$/);
      if (head) { flushPara(); closeList(); html += `<p><strong>${renderInline(head[1])}</strong></p>`; }
      else if ((bullet || num) && !t.startsWith('**')) {
        flushPara();
        const kind = bullet ? 'ul' : 'ol';
        if (list !== kind) { closeList(); html += `<${kind} class="answer-list">`; list = kind; }
        html += `<li>${renderInline((bullet || num)[1])}</li>`;
      } else if (t === '') { flushPara(); closeList(); }
      else { closeList(); para.push(line); }
    }
    flushPara(); closeList(); flushTable();
    return html;
  }

// Qwen can include <think>...</think> inside its content. Keep it separate from
// the answer, including when either tag is split across streaming chunks.
function splitThinking(raw, streaming = false) {
  let text = String(raw || '');
  if (streaming) {
    const start = text.lastIndexOf('<');
    if (start !== -1) {
      const suffix = text.slice(start).toLowerCase();
      if (['<think>', '</think>'].some(tag => suffix.length < tag.length && tag.startsWith(suffix))) {
        text = text.slice(0, start);
      }
    }
  }
  const thoughts = [];
  const answer = [];
  const tags = /<\/?think\s*>/gi;
  let inside = false;
  let hasThinking = false;
  let offset = 0;
  let match;
  while ((match = tags.exec(text))) {
    (inside ? thoughts : answer).push(text.slice(offset, match.index));
    inside = !match[0].startsWith('</');
    if (inside) hasThinking = true;
    offset = tags.lastIndex;
  }
  (inside ? thoughts : answer).push(text.slice(offset));
  return { thinking: thoughts.join('\n').trim(), answer: answer.join('').trim(),
    hasThinking, active: streaming && inside };
}

// A previous answer as it goes back with a follow-up: plain text, without the
// model's thinking and without anything tag-shaped. The server ignores both, and a
// network firewall in front of the bot answers HTTP 403 to a request body that
// carries tags like <think> (seen with Sarvam-30B, whose replies hold one).
function historyText(answer) {
  return splitThinking(answer || '').answer
    .replace(/<\/?[a-z][^<>]*>/gi, ' ').replace(/[ \t]{2,}/g, ' ').trim();
}

function renderAnswer(raw, { streaming = false, thinkingOpen = false } = {}) {
  const parts = splitThinking(raw, streaming);
  let html = '';
  // An empty think block (a model told not to think writes one) shows no panel.
  if (parts.hasThinking && (parts.thinking || parts.active)) {
    html += `<details class="model-thinking${parts.active ? ' is-active' : ''}"${thinkingOpen ? ' open' : ''}>` +
      '<summary><span class="thinking-chevron" aria-hidden="true">›</span>' +
      `<span class="thinking-label">${parts.active ? 'Thinking…' : 'Thought process'}</span>` +
      '<span class="thinking-indicator" aria-hidden="true"></span></summary>' +
      `<div class="model-thinking-content">${renderMarkdown(parts.thinking)}</div></details>`;
  }
  html += renderMarkdown(parts.answer);
  if (streaming && parts.answer) html += '<span class="stream-cursor"></span>';
  if (!streaming && parts.hasThinking && !parts.answer) {
    html += '<p>No final answer was returned. Please try again.</p>';
  }
  return html;
}


const CHAT_TEMPLATE = `      <!-- Chat pane -->
      <div id="chat-pane">
        <div id="chat-empty" class="chat-empty">
          <img class="empty-logo" src="img/cg-emblem.png" alt="" data-brand-logo />
          <h1 class="empty-title" data-brand-label>Document Assistant</h1>
          <p>Ask a question about your department's documents and letters.
             Type your question below and press <strong>Send</strong>.</p>
        </div>
        <div id="chat-messages"></div>
      </div>

      <!-- Query bar -->
      <div id="query-bar">
        <div id="query-wrap">
          <textarea id="query-input" rows="1" placeholder="Type your question here…"
            spellcheck="false"></textarea>
          <button id="btn-send" disabled title="Send question">
            <span class="btn-send-label">Send</span>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"
              stroke-linecap="round" stroke-linejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
        <div id="query-controls">
          <button id="btn-new-chat" type="button" class="mode-toggle"
            title="Start a new chat: follow-up questions are read in the context of this chat only">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M12 5v14M5 12h14"></path>
            </svg>
            <span>New chat</span>
          </button>
          <button id="btn-thinking" type="button" class="mode-toggle" aria-pressed="false"
            title="Thinking mode: searches several times, checks what is missing, and shows its reasoning. Slower, for questions that need more than one lookup.">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z"></path>
            </svg>
            <span>Thinking mode</span>
          </button>
          <div id="query-hint">Press Enter to send · Shift + Enter for a new line</div>
        </div>
      </div>
`;

function mount(container, { imageBase = 'img', title = 'Document Assistant', description,
                            thinking = true, maxLength, branding = true } = {}) {
  container.innerHTML = CHAT_TEMPLATE;
  for (const image of container.querySelectorAll('img')) image.src = `${imageBase}/cg-emblem.png`;
  container.querySelector('.empty-title').textContent = title;
  if (description) container.querySelector('.chat-empty p').textContent = description;
  if (!thinking) container.querySelector('#btn-thinking').remove();
  if (maxLength) container.querySelector('#query-input').maxLength = maxLength;
  if (!branding) {
    for (const node of container.querySelectorAll('[data-brand-logo], [data-brand-label]')) {
      node.removeAttribute('data-brand-logo'); node.removeAttribute('data-brand-label');
    }
  }
}

function message(role, text, { avatars, meta = {} }) {
  const msg = document.createElement('div');
  msg.className = `msg msg-${role}`;
  // Avatar
  const avatar = document.createElement('img');
  avatar.className = 'msg-avatar';
  avatar.src = avatars[role] || avatars.assistant;
  avatar.alt = role === 'user' ? 'You' : 'Assistant';

  // Content column (bubble + any attachments)
  const content = document.createElement('div');
  content.className = 'msg-content';

  const body = document.createElement('div');
  body.className = `msg-body ${role}-body`;

  if (role === 'thinking') {
    body.innerHTML =
      '<div class="thinking-dot"></div>' +
      '<div class="thinking-dot"></div>' +
      '<div class="thinking-dot"></div>';
  } else if (role === 'assistant') {
    body.innerHTML = renderAnswer(text);
    if (meta.standalone) {
      const note = document.createElement('div'); note.className = 'search-note';
      note.textContent = `Searched for: ${meta.standalone}`;
      body.prepend(note);
    }
  } else {
    // User's own message — plain text, no markdown interpretation.
    body.innerHTML = escapeHtml(text)
      .split('\n\n').filter(Boolean)
      .map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`)
      .join('');
  }

  content.appendChild(body);
  msg.appendChild(avatar); msg.appendChild(content);
  return { msg, body, content };
}

function renderChatList(list, items, { activeId, disabled = false, onOpen, onAction }) {
  list.replaceChildren();
  const chats = [...items].sort((a, b) => Number(b.pinned) - Number(a.pinned)
    || b.updatedAt.localeCompare(a.updatedAt));
  if (!chats.length) {
    const li = document.createElement('li');
    li.className = 'chats-empty'; li.textContent = 'Your saved chats will appear here.';
    list.appendChild(li);
  }
  for (const chat of chats) {
    const li = document.createElement('li'); li.className = 'chat-row';
    li.classList.toggle('is-active', chat.id === activeId);
    const open = document.createElement('button'); open.type = 'button'; open.className = 'chat-open';
    open.textContent = chat.title; open.title = chat.title;
    open.disabled = disabled;
    if (chat.id === activeId) open.setAttribute('aria-current', 'true');
    open.addEventListener('click', () => onOpen(chat.id));
    li.appendChild(open);
    if (chat.pinned) {
      const pin = document.createElement('span'); pin.className = 'chat-pin'; pin.textContent = 'Pinned';
      li.appendChild(pin);
    }
    const details = document.createElement('details'); details.className = 'chat-options';
    const toggle = document.createElement('summary'); toggle.textContent = '⋯';
    toggle.setAttribute('aria-label', `Options for ${chat.title}`);
    details.appendChild(toggle);
    details.addEventListener('toggle', () => {
      if (details.open) list.querySelectorAll('details[open]').forEach(d => { if (d !== details) d.open = false; });
    });
    const menu = document.createElement('div'); menu.className = 'chat-menu';
    for (const action of ['pin', 'delete']) {
      const btn = document.createElement('button'); btn.type = 'button';
      btn.textContent = action === 'delete' ? 'Delete' : chat.pinned ? 'Unpin' : 'Pin';
      btn.classList.toggle('delete-chat', action === 'delete');
      btn.disabled = disabled;
      btn.addEventListener('click', () => { details.open = false; onAction(chat, action); });
      menu.appendChild(btn);
    }
    details.appendChild(menu); li.appendChild(details); list.appendChild(li);
  }
}

function resizeInput(input) {
  input.style.height = 'auto';
  input.style.height = `${input.scrollHeight}px`;
}

window.ChatUI = { resizeInput, mount, message, renderChatList, escapeHtml, renderInline, renderMarkdown,
                  splitThinking, historyText, renderAnswer };
})();
