'use strict';

// Department API adapter for the shared chatbot UI. Rendering and recent-chat
// controls come from ChatUI; persistence uses the same server store as the bots.
window.DepartmentChat = {
  async create({ container, department, user, api, toast, onSelect }) {
    const C = window.ChatUI;
    const $ = id => document.getElementById(id);
    C.mount(container, { imageBase: '../../img', title: `${department.name} Document Assistant`,
      description: "Ask a question about your department's documents and letters. Type your question below and press Send.",
      thinking: false, maxLength: 2000, branding: false });
    const avatars = { user: '../../img/avatar-user.svg', assistant: '../../img/cg-emblem.png', thinking: '../../img/cg-emblem.png' };
    const key = `dept-chat:${department.id}:${user.id}`;
    let chats = [], messages = [], chatId = null, busy = false, dirty = false;
    const request = async (method, path = '', body) => {
      const result = await api(method, `api/history${path}`, body);
      if (!result.success) throw new Error(result.error || 'Could not save or load this chat.');
      return result;
    };
    const remember = () => { try { chatId ? localStorage.setItem(key, chatId) : localStorage.removeItem(key); } catch (_) { } };
    const update = () => {
      $('btn-send').disabled = busy || department.status !== 'active';
      $('btn-new-chat').disabled = busy;
      renderList();
    };
    const renderList = () => C.renderChatList($('chat-list'), chats, {
      activeId: chatId, disabled: busy, onOpen: open, onAction: action,
    });
    function append(message) {
      $('chat-empty').classList.add('hidden');
      const { msg, body } = C.message(message.role, message.text, { avatars });
      if (message.role === 'assistant' && message.results?.length) {
        const footer = document.createElement('div'); footer.className = 'msg-sources';
        const button = document.createElement('button'); button.type = 'button'; button.className = 'sources-btn';
        button.innerHTML = `<span class="source-count">${message.results.length}</span> View sources`;
        const sources = document.createElement('div'); sources.className = 'department-sources hidden';
        for (const source of message.results) {
          const link = document.createElement('a');
          link.href = typeof source.url === 'string' && /^https?:\/\//i.test(source.url)
            ? source.url : `api/sources/${encodeURIComponent(source.fileId)}`;
          link.target = '_blank'; link.rel = 'noopener';
          link.textContent = `[${source.n}] ${source.name}`;
          sources.appendChild(link);
        }
        button.setAttribute('aria-expanded', 'false');
        button.addEventListener('click', () => {
          const hidden = sources.classList.toggle('hidden');
          button.setAttribute('aria-expanded', String(!hidden));
        });
        footer.append(button, sources); body.appendChild(footer);
      }
      $('chat-messages').appendChild(msg);
      msg.scrollIntoView({ behavior: 'smooth', block: 'end' });
      return msg;
    }
    function display() {
      $('chat-messages').replaceChildren();
      $('chat-empty').classList.toggle('hidden', !!messages.length);
      for (const message of messages) append(message);
      $('query-input').value = ''; resize();
    }
    function upsert(chat) {
      const summary = { id: chat.id, title: chat.title, pinned: chat.pinned, updatedAt: chat.updatedAt };
      chats = [summary, ...chats.filter(row => row.id !== chat.id)];
      chatId = chat.id; remember(); renderList();
    }
    async function save() {
      const result = await request(chatId ? 'PUT' : 'POST', chatId ? `/${chatId}` : '', { messages });
      dirty = false;
      upsert(result.chat);
    }
    async function open(id) {
      if (busy) return;
      busy = true; update();
      try {
        if (dirty) await save();
        const { chat } = await request('GET', `/${encodeURIComponent(id)}`);
        chatId = chat.id; messages = chat.messages; remember(); display(); onSelect?.();
      } catch (err) { toast(err.message, 6000); }
      finally { busy = false; update(); }
    }
    async function action(chat, kind) {
      if (busy) return;
      if (kind === 'delete' && !confirm(`Delete "${chat.title}"?`)) return;
      busy = true; update();
      try {
        if (dirty) await save();
        if (kind === 'delete') {
          await request('DELETE', `/${chat.id}`);
          chats = chats.filter(row => row.id !== chat.id);
          if (chatId === chat.id) { chatId = null; messages = []; remember(); display(); }
        } else {
          const { chat: updated } = await request('PATCH', `/${chat.id}`, { pinned: !chat.pinned });
          chats = chats.map(row => row.id === chat.id ? updated : row);
        }
      } catch (err) { toast(err.message, 6000); }
      finally { busy = false; update(); }
    }
    function resize() { C.resizeInput($('query-input')); }
    async function send() {
      const question = $('query-input').value.trim();
      if (!question || busy || department.status !== 'active') return;
      busy = true; update();
      const turns = [];
      for (let i = 0; i < messages.length - 1; i++) {
        if (messages[i].role === 'user' && messages[i + 1].role === 'assistant' && messages[i + 1].success) {
          turns.push({ question: messages[i].text, answer: C.historyText(messages[i + 1].text) });
        }
      }
      const row = { role: 'user', text: question };
      const wasDirty = dirty;
      messages.push(row);
      dirty = true;
      try { await save(); }
      catch (err) { messages.pop(); dirty = wasDirty; toast(err.message, 6000); busy = false; update(); return; }
      $('query-input').value = ''; resize(); append(row);
      const wait = append({ role: 'thinking', text: '' });
      let reply;
      try {
        const result = await api('POST', 'api/ask', { question, history: turns.slice(-4) });
        reply = { role: 'assistant', success: !!result.success,
          text: result.success ? String(result.answer || 'No answer was returned. Please try again.').replace(/\n+\*\*(?:Sources|स्रोत):\*\*\n[\s\S]*$/, '').trim()
            : result.error || 'The assistant could not answer.', results: result.sources || [] };
      } catch (err) { reply = { role: 'assistant', success: false, text: err.message }; }
      finally { wait.remove(); }
      messages.push(reply); append(reply);
      dirty = true;
      try { await save(); } catch (err) { toast(err.message, 6000); }
      busy = false; update(); $('query-input').focus();
    }
    $('btn-send').addEventListener('click', send);
    $('query-input').addEventListener('input', resize);
    $('query-input').addEventListener('keydown', event => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); send(); }
    });
    $('btn-new-chat').addEventListener('click', async () => {
      if (busy) return;
      busy = true; update();
      try {
        if (dirty) await save();
        chatId = null; messages = []; remember(); display(); onSelect?.(); $('query-input').focus();
      } catch (err) { toast(err.message, 6000); }
      finally { busy = false; update(); }
    });
    try {
      chats = (await request('GET')).chats;
      let previous; try { previous = localStorage.getItem(key); } catch (_) { }
      if (chats.some(chat => chat.id === previous)) await open(previous);
    } catch (err) { toast(err.message, 6000); }
    update();
    return { focus: () => {
      $('chat-pane').scrollTop = $('chat-pane').scrollHeight;
      $('query-input').focus();
    } };
  },
};
