/**
 * DocuMind AI — Frontend Application
 * ─────────────────────────────────────
 * Handles:
 *   - PDF upload (drag & drop + click)
 *   - Document list management
 *   - Chat interface (send query, display answer)
 *   - Source citations rendering
 *   - Toast notifications
 *   - API health monitoring
 */

const API_BASE = window.location.origin;

// ── State ──────────────────────────────────────────────────────────────────────
const state = {
  documents:      [],       // all indexed documents
  selectedDocIds: [],       // filter queries to these docs (empty = all)
  isQuerying:     false,    // prevent double-submit
};

// ── DOM References ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const dom = {
  uploadArea:    $('uploadArea'),
  uploadInput:   $('uploadInput'),
  uploadProgress:$('uploadProgress'),
  progressBar:   $('progressBar'),
  progressLabel: $('progressLabel'),
  docList:       $('docList'),
  chatArea:      $('chatArea'),
  welcome:       $('welcome'),
  queryInput:    $('queryInput'),
  sendBtn:       $('sendBtn'),
  statusDot:     $('statusDot'),
  statusText:    $('statusText'),
  docFilter:     $('docFilter'),
  toastContainer:$('toastContainer'),
};


// ── API Helpers ────────────────────────────────────────────────────────────────

async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    if (err.name === 'TypeError') {
      throw new Error('Cannot connect to API. Is the backend running?');
    }
    throw err;
  }
}


// ── Toast Notifications ────────────────────────────────────────────────────────

function showToast(message, type = 'info', duration = 4000) {
  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type]}</span>
    <span class="toast-msg">${message}</span>
  `;
  dom.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}


// ── Health Check ───────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const data = await apiFetch('/api/health');
    dom.statusDot.className  = 'status-dot online';
    dom.statusText.textContent = 'API Online';

    if (!data.groq_configured) {
      showToast(
        '⚠️ Groq API key not set. Add GROQ_API_KEY to your .env file for AI answers.',
        'info',
        8000
      );
    }
  } catch {
    dom.statusDot.className   = 'status-dot offline';
    dom.statusText.textContent = 'API Offline';
  }
}


// ── Document Management ────────────────────────────────────────────────────────

async function loadDocuments() {
  try {
    const data = await apiFetch('/api/documents');
    state.documents = data.documents;
    renderDocumentList();
  } catch (err) {
    console.error('Failed to load documents:', err);
  }
}

function renderDocumentList() {
  const docs = state.documents;

  if (docs.length === 0) {
    dom.docList.innerHTML = `
      <div class="empty-docs">
        📄 No documents yet.<br>Upload a PDF to get started.
      </div>`;
    renderDocFilter();
    return;
  }

  dom.docList.innerHTML = docs.map(doc => `
    <div class="doc-item ${state.selectedDocIds.includes(doc.doc_id) ? 'selected' : ''}"
         onclick="toggleDocSelection('${doc.doc_id}')"
         title="${doc.filename}">
      <div class="doc-icon">📄</div>
      <div class="doc-info">
        <div class="doc-name">${doc.filename}</div>
        <div class="doc-meta">${doc.total_pages} pages · ${doc.total_chunks} chunks · ${doc.file_size_kb.toFixed(1)} KB</div>
      </div>
      <button class="doc-delete"
              onclick="deleteDocument(event, '${doc.doc_id}', '${doc.filename}')"
              title="Remove document">✕</button>
    </div>
  `).join('');

  renderDocFilter();
}

function toggleDocSelection(docId) {
  const idx = state.selectedDocIds.indexOf(docId);
  if (idx === -1) {
    state.selectedDocIds.push(docId);
  } else {
    state.selectedDocIds.splice(idx, 1);
  }
  renderDocumentList();
}

function renderDocFilter() {
  if (state.selectedDocIds.length === 0 || state.documents.length === 0) {
    dom.docFilter.innerHTML = '';
    return;
  }

  const names = state.selectedDocIds.map(id => {
    const doc = state.documents.find(d => d.doc_id === id);
    return doc ? `<span class="doc-filter-chip">📄 ${doc.filename}</span>` : '';
  }).join('');

  dom.docFilter.innerHTML = `
    <span style="color:var(--text-muted);font-size:12px;">Searching in:</span>
    ${names}
    <button onclick="clearDocFilter()" style="border:none;background:none;color:var(--text-muted);cursor:pointer;font-size:12px;margin-left:4px;">✕ Clear</button>
  `;
}

function clearDocFilter() {
  state.selectedDocIds = [];
  renderDocumentList();
}

async function deleteDocument(event, docId, filename) {
  event.stopPropagation(); // don't trigger toggleDocSelection
  if (!confirm(`Remove "${filename}" from the knowledge base?`)) return;

  try {
    await apiFetch(`/api/documents/${docId}`, { method: 'DELETE' });
    state.documents = state.documents.filter(d => d.doc_id !== docId);
    state.selectedDocIds = state.selectedDocIds.filter(id => id !== docId);
    renderDocumentList();
    showToast(`"${filename}" removed successfully.`, 'success');
  } catch (err) {
    showToast(`Failed to remove document: ${err.message}`, 'error');
  }
}


// ── PDF Upload ─────────────────────────────────────────────────────────────────

function setupUpload() {
  // Click to upload
  dom.uploadInput.addEventListener('change', e => {
    if (e.target.files[0]) handleUpload(e.target.files[0]);
  });

  // Drag & drop
  dom.uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    dom.uploadArea.classList.add('drag-over');
  });
  dom.uploadArea.addEventListener('dragleave', () => {
    dom.uploadArea.classList.remove('drag-over');
  });
  dom.uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    dom.uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  });
}

async function handleUpload(file) {
  // Validate
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showToast('Please upload a PDF file only.', 'error');
    return;
  }
  if (file.size > 50 * 1024 * 1024) {
    showToast('File too large. Maximum size is 50MB.', 'error');
    return;
  }

  // Show progress UI
  dom.uploadProgress.style.display = 'block';
  dom.progressBar.style.width = '15%';
  dom.progressLabel.textContent = 'Uploading...';

  const formData = new FormData();
  formData.append('file', file);

  try {
    // Animate progress (fake progress for UX — real progress needs streaming)
    let progress = 15;
    const progressInterval = setInterval(() => {
      progress = Math.min(progress + Math.random() * 8, 85);
      dom.progressBar.style.width = progress + '%';
      if (progress > 30)  dom.progressLabel.textContent = 'Extracting text...';
      if (progress > 55)  dom.progressLabel.textContent = 'Generating embeddings...';
      if (progress > 75)  dom.progressLabel.textContent = 'Indexing chunks...';
    }, 400);

    const result = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    clearInterval(progressInterval);

    if (!result.ok) {
      const err = await result.json();
      throw new Error(err.detail || 'Upload failed');
    }

    const data = await result.json();

    dom.progressBar.style.width = '100%';
    dom.progressLabel.textContent = 'Complete!';

    setTimeout(() => {
      dom.uploadProgress.style.display = 'none';
      dom.progressBar.style.width = '0%';
    }, 1500);

    await loadDocuments();
    showToast(data.message, 'success', 5000);

    // Reset file input
    dom.uploadInput.value = '';

  } catch (err) {
    dom.uploadProgress.style.display = 'none';
    dom.progressBar.style.width = '0%';
    showToast(`Upload failed: ${err.message}`, 'error', 6000);
  }
}


// ── Chat Interface ─────────────────────────────────────────────────────────────

function setupChat() {
  // Send on button click
  dom.sendBtn.addEventListener('click', sendQuery);

  // Send on Enter (Shift+Enter for newline)
  dom.queryInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  });

  // Auto-resize textarea
  dom.queryInput.addEventListener('input', () => {
    dom.queryInput.style.height = 'auto';
    dom.queryInput.style.height = Math.min(dom.queryInput.scrollHeight, 120) + 'px';
  });
}

async function sendQuery() {
  const question = dom.queryInput.value.trim();
  if (!question || state.isQuerying) return;

  if (state.documents.length === 0) {
    showToast('Please upload a PDF document first.', 'info');
    return;
  }

  state.isQuerying = true;
  dom.sendBtn.disabled = true;
  dom.queryInput.value = '';
  dom.queryInput.style.height = 'auto';

  // Hide welcome screen on first message
  if (dom.welcome) dom.welcome.style.display = 'none';

  // Render user message
  appendMessage('user', question);

  // Show typing indicator
  const typingEl = appendTypingIndicator();

  try {
    const payload = {
      question,
      doc_ids: state.selectedDocIds.length > 0 ? state.selectedDocIds : null,
    };

    const data = await apiFetch('/api/query', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    typingEl.remove();
    appendAssistantMessage(data);

  } catch (err) {
    typingEl.remove();
    appendMessage('assistant', `⚠️ Error: ${err.message}`);
    showToast(err.message, 'error');
  } finally {
    state.isQuerying = false;
    dom.sendBtn.disabled = false;
    dom.queryInput.focus();
  }
}

function appendMessage(role, text) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const label = role === 'user' ? 'You' : 'DocuMind AI';
  wrap.innerHTML = `
    <div class="msg-label">${label}</div>
    <div class="msg-bubble">${escapeHtml(text)}</div>
  `;

  dom.chatArea.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function appendAssistantMessage(data) {
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';

  // Confidence badge
  const confMap = {
    HIGH:   ['badge-green',  '● High confidence'],
    MEDIUM: ['badge-amber',  '● Medium confidence'],
    LOW:    ['badge-red',    '● Low confidence'],
  };
  const [confClass, confLabel] = confMap[data.confidence] || ['badge-gray', '● Unknown'];

  // Grounded badge
  const groundedBadge = data.is_grounded
    ? '<span class="meta-badge badge-green">✓ Grounded</span>'
    : '<span class="meta-badge badge-amber">⚠ May not be fully grounded</span>';

  // Sources HTML
  const sourcesHtml = data.sources && data.sources.length > 0
    ? buildSourcesHtml(data.sources)
    : '';

  wrap.innerHTML = `
    <div class="msg-label">DocuMind AI</div>
    <div class="msg-bubble">
      <div style="white-space:pre-wrap;word-wrap:break-word">${escapeHtml(data.answer)}</div>
      <div class="answer-meta">
        <span class="meta-badge ${confClass}">${confLabel}</span>
        ${groundedBadge}
        <span class="meta-badge badge-blue">🔍 ${data.chunks_retrieved} sources</span>
        <span class="meta-badge badge-purple">⚡ ${data.model_used}</span>
        <span class="meta-badge badge-gray">⏱ ${data.latency_ms}ms</span>
      </div>
    </div>
    ${sourcesHtml}
  `;

  dom.chatArea.appendChild(wrap);
  scrollToBottom();

  // Attach source toggle handlers
  const header = wrap.querySelector('.sources-header');
  if (header) {
    header.addEventListener('click', () => {
      const list   = wrap.querySelector('.sources-list');
      const toggle = wrap.querySelector('.sources-toggle');
      list.classList.toggle('open');
      toggle.classList.toggle('open');
    });
  }
}

function buildSourcesHtml(sources) {
  const items = sources.map((src, i) => `
    <div class="source-item">
      <div class="source-meta">
        <span class="source-file">📄 ${escapeHtml(src.doc_name)}</span>
        <span class="source-page">Page ${src.page_number}</span>
        <span class="source-sim">${(src.similarity * 100).toFixed(1)}% match</span>
      </div>
      <div class="source-text">"${escapeHtml(src.text)}"</div>
    </div>
  `).join('');

  return `
    <div class="sources-panel">
      <div class="sources-header">
        <span class="sources-title">📚 ${sources.length} Source${sources.length > 1 ? 's' : ''} used</span>
        <span class="sources-toggle">▼</span>
      </div>
      <div class="sources-list">
        ${items}
      </div>
    </div>
  `;
}

function appendTypingIndicator() {
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';
  wrap.innerHTML = `
    <div class="msg-label">DocuMind AI</div>
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  dom.chatArea.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function scrollToBottom() {
  dom.chatArea.scrollTop = dom.chatArea.scrollHeight;
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}


// ── Sample Questions ────────────────────────────────────────────────────────────

function askSample(question) {
  dom.queryInput.value = question;
  sendQuery();
}


// ── Init ───────────────────────────────────────────────────────────────────────

async function init() {
  setupUpload();
  setupChat();
  await checkHealth();
  await loadDocuments();

  // Poll health every 30 seconds
  setInterval(checkHealth, 30000);
}

document.addEventListener('DOMContentLoaded', init);
