/**
 * GAD Temporal Knowledge Graph & Retriever Explorer Client Logic
 * Hierarchical Clustered & Radial Hub-and-Spoke Architecture
 */

let network = null;
let graphData = { nodes: [], edges: [] };
let selectedNodeId = null;
let currentYearFilter = null;
let currentStatusFilter = 'all';
let currentLimit = 120;
let viewMode = 'hierarchical'; // 'hierarchical' | 'network'
let expandedCategories = new Set(['leave', 'compassionate']);
let categoriesList = [];
let currentDocumentMarkdown = '';
let currentTargetDoc = null;
let drawerMdMode = 'render'; // 'render' | 'raw'
let modalMdMode = 'render';

document.addEventListener('DOMContentLoaded', () => {
  initGraph();
  loadStats();
  loadCategories();
  loadHierarchicalGraph();
});

function switchView(viewName) {
  const graphBtn = document.getElementById('tab-graph-btn');
  const expBtn = document.getElementById('tab-exp-btn');
  const expPanel = document.getElementById('experiment-panel');
  const controlsBar = document.getElementById('controls-bar');

  if (viewName === 'graph') {
    graphBtn.classList.add('active');
    expBtn.classList.remove('active');
    expPanel.classList.remove('active');
    controlsBar.style.display = 'flex';
  } else {
    expBtn.classList.add('active');
    graphBtn.classList.remove('active');
    expPanel.classList.add('active');
    controlsBar.style.display = 'none';
  }
}

function initGraph() {
  const container = document.getElementById('graph-container');
  const data = {
    nodes: new vis.DataSet([]),
    edges: new vis.DataSet([])
  };

  // Turn off continuous physics to eliminate CPU usage and keep nodes static
  const options = {
    nodes: {
      font: { color: '#f8fafc', size: 12, face: 'sans-serif' },
      borderWidth: 2,
      shadow: false
    },
    edges: {
      width: 1.5,
      font: { color: '#94a3b8', size: 10, align: 'middle' },
      smooth: { type: 'continuous' }
    },
    physics: {
      enabled: false // Static layout: zero CPU movement, mathematically positioned
    },
    interaction: {
      hover: true,
      tooltipDelay: 100,
      zoomView: true,
      dragNodes: true,
      dragView: true
    }
  };

  network = new vis.Network(container, data, options);

  network.on('click', (params) => {
    if (params.nodes.length > 0) {
      const clickedId = params.nodes[0];
      const nodeObj = (graphData.nodes || []).find(n => n.id === clickedId);
      if (nodeObj && nodeObj.is_hub) {
        // Toggle expansion of the category hub
        toggleCategory(nodeObj.category_id);
      } else {
        // Open document full OCR and history drawer
        openDrawer(clickedId);
      }
    }
  });
}

// ─── HIERARCHICAL CATEGORY HUB LOGIC ─────────────────────────────

async function loadCategories() {
  try {
    const res = await fetch('/api/graph/categories');
    const data = await res.json();
    if (data.success && data.categories) {
      categoriesList = data.categories;
      renderCategoryPills();
    }
  } catch (err) {
    console.error('Failed to load categories:', err);
  }
}

function renderCategoryPills() {
  const container = document.getElementById('category-pills-list');
  if (!container) return;
  container.innerHTML = '';

  const shortNames = {
    'compassionate': '🤝 अनुकंपा',
    'leave': '🌴 अवकाश',
    'vigilance': '⚖️ सतर्कता व जांच',
    'pension': '💰 वेतन व पेंशन',
    'secretariat': '🏛️ सचिवालय',
    'rti': 'ℹ️ सूचना का अधिकार',
    'rules': '📜 सेवा नियम',
    'promotion': '🎖️ पदोन्नति व भर्ती',
    'transfer': '🔄 स्थानांतरण',
    'general': '📑 सामान्य परिपत्र'
  };

  categoriesList.forEach(cat => {
    const isExpanded = expandedCategories.has(cat.id);
    const pill = document.createElement('div');
    pill.className = `pill ${isExpanded ? 'active' : ''}`;
    pill.style.fontSize = '11px';
    pill.style.padding = '4px 10px';
    pill.innerHTML = `
      <span>${shortNames[cat.id] || cat.name}</span>
      <span class="badge" style="background:${isExpanded ? 'rgba(56,189,248,0.3)' : 'rgba(255,255,255,0.1)'};color:${isExpanded ? '#38bdf8' : '#94a3b8'};padding:1px 5px;font-size:10px">${cat.total}</span>
    `;
    pill.onclick = () => toggleCategory(cat.id);
    container.appendChild(pill);
  });
}

function toggleCategory(categoryId) {
  if (expandedCategories.has(categoryId)) {
    expandedCategories.delete(categoryId);
  } else {
    expandedCategories.add(categoryId);
  }
  renderCategoryPills();
  if (viewMode === 'hierarchical') {
    loadHierarchicalGraph();
  }
}

function expandAllCategories() {
  expandedCategories = new Set(categoriesList.map(c => c.id));
  renderCategoryPills();
  if (viewMode === 'hierarchical') {
    loadHierarchicalGraph();
  }
}

function collapseAllCategories() {
  expandedCategories.clear();
  renderCategoryPills();
  if (viewMode === 'hierarchical') {
    loadHierarchicalGraph();
  }
}

function switchViewMode(mode) {
  viewMode = mode;
  const btnH = document.getElementById('btn-mode-hierarchical');
  const btnN = document.getElementById('btn-mode-network');
  if (btnH && btnN) {
    btnH.classList.toggle('active', mode === 'hierarchical');
    btnN.classList.toggle('active', mode === 'network');
  }

  if (mode === 'hierarchical') {
    loadHierarchicalGraph();
  } else {
    loadSubgraph();
  }
}

async function loadHierarchicalGraph() {
  try {
    document.getElementById('stats-badge').innerText = 'Loading category hubs…';
    const expandedParam = Array.from(expandedCategories).join(',');
    const url = `/api/graph/hierarchical?expanded=${encodeURIComponent(expandedParam)}&limit=${currentLimit}`;

    const res = await fetch(url);
    const data = await res.json();
    if (data.success) {
      graphData = data;
      renderVisNetwork(data.nodes, data.edges);
      const hubCount = data.nodes.filter(n => n.is_hub).length;
      const childCount = data.nodes.length - hubCount;
      document.getElementById('stats-badge').innerText =
        `${hubCount} Hubs · ${childCount} Documents visible (Total: ${data.total_in_graph})`;
    }
  } catch (err) {
    console.error('Failed to load hierarchical graph:', err);
    document.getElementById('stats-badge').innerText = 'Load error';
  }
}

// ─── LEGACY SUBGRAPH / SEARCH LOGIC ─────────────────────────────

async function loadStats() {
  try {
    const res = await fetch('/api/graph/stats');
    const data = await res.json();
    if (data.success && data.stats) {
      const s = data.stats;
      document.getElementById('stats-badge').innerText =
        `${s.total_nodes} Orders · ${s.total_edges} Edges`;
    }
  } catch (err) {
    console.error('Failed to load stats:', err);
  }
}

async function loadSubgraph(params = {}) {
  try {
    document.getElementById('stats-badge').innerText = 'Loading graph…';
    const limit = params.limit || currentLimit;
    let url = `/api/graph/subgraph?limit=${limit}`;
    if (params.center) url += `&center=${encodeURIComponent(params.center)}`;
    if (params.query) url += `&query=${encodeURIComponent(params.query)}`;
    if (params.year) url += `&year=${params.year}`;

    const res = await fetch(url);
    const data = await res.json();
    if (data.success) {
      graphData = data;
      renderVisNetwork(data.nodes, data.edges);
      document.getElementById('stats-badge').innerText =
        `${data.count} Nodes visible (Total: ${data.total_in_graph})`;
    }
  } catch (err) {
    console.error('Failed to load subgraph:', err);
    document.getElementById('stats-badge').innerText = 'Load error';
  }
}

function changeNodeLimit(val) {
  currentLimit = parseInt(val, 10) || 120;
  if (viewMode === 'hierarchical') {
    loadHierarchicalGraph();
  } else {
    loadSubgraph();
  }
}

function renderVisNetwork(rawNodes, rawEdges) {
  let filteredNodes = rawNodes;
  if (currentStatusFilter !== 'all') {
    // Keep hubs visible, filter only leaf nodes
    filteredNodes = rawNodes.filter(n => n.is_hub || n.status === currentStatusFilter);
  }

  const validIds = new Set(filteredNodes.map(n => n.id));
  const filteredEdges = rawEdges.filter(e => validIds.has(e.from) && validIds.has(e.to));

  const nodesDataSet = new vis.DataSet(filteredNodes);
  const edgesDataSet = new vis.DataSet(filteredEdges);

  network.setData({ nodes: nodesDataSet, edges: edgesDataSet });
}

// ─── SEMANTIC & KEYWORD SEARCH + CANVAS HIGHLIGHTING ─────────────

let searchDebounceTimer = null;
let isHighlightActive = false;

function onSearchInputDebounced(val) {
  const query = (val || '').trim();
  const clearBtn = document.getElementById('search-clear-btn');
  if (clearBtn) {
    clearBtn.style.display = query ? 'block' : 'none';
  }

  if (searchDebounceTimer) clearTimeout(searchDebounceTimer);

  if (!query) {
    clearCanvasHighlight();
    return;
  }

  searchDebounceTimer = setTimeout(() => {
    executeSemanticSearch(false);
  }, 220);
}

function searchGraph() {
  executeSemanticSearch(true);
}

async function executeSemanticSearch(isEnterPress = false) {
  const input = document.getElementById('graph-search-input');
  const query = input ? input.value.trim() : '';
  if (!query) {
    clearCanvasHighlight();
    return;
  }

  try {
    const res = await fetch(`/api/graph/semantic_search?q=${encodeURIComponent(query)}&limit=40`);
    const data = await res.json();
    if (data.success) {
      renderSearchSuggestions(data);
      if (isEnterPress && data.officer_match) {
        displayOfficerDossier(data.officer_match.id);
      } else if (isEnterPress || data.matched_category || (data.matched_node_ids && data.matched_node_ids.length > 0)) {
        applyCanvasHighlight(data.matched_node_ids || [], data.matched_hub_ids || [], query, data.matched_category);
      }
    }
  } catch (err) {
    console.error('Semantic search failed:', err);
  }
}

function renderSearchSuggestions(data) {
  const container = document.getElementById('search-suggestions');
  if (!container) return;

  const officer = data.officer_match;
  const cat = data.matched_category;
  const results = data.results || [];

  if (!officer && !cat && results.length === 0) {
    container.innerHTML = `
      <div style="padding:14px;color:#94a3b8;font-size:12px;text-align:center">
        No circulars, officers, or categories found matching "<strong>${escapeHtml(data.query)}</strong>".
      </div>
    `;
    container.style.display = 'block';
    return;
  }

  let html = '';

  // 0. Matched Officer Profile (Instant Officer Dossier)
  if (officer) {
    const notifCount = officer.notifications ? officer.notifications.length : (officer.notification_count || 0);
    html += `
      <div class="sugg-category-banner" style="background:linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(6, 78, 59, 0.35));border:1px solid #10b981;cursor:pointer;margin-bottom:8px" onclick="displayOfficerDossier('${officer.id}')">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:11px;color:#34d399;text-transform:uppercase;font-weight:700;letter-spacing:0.5px">👤 Officer Profile Found · ${escapeHtml(officer.category_label || 'Officer')}</span>
          <span class="badge" style="background:#059669;color:#fff">${notifCount} Linked Orders</span>
        </div>
        <div style="font-size:14.5px;font-weight:700;color:#fff;margin-top:3px">${escapeHtml(officer.name_en)} (${escapeHtml(officer.name_hi)})</div>
        <div style="font-size:12px;color:#cbd5e1;margin-top:2px"><strong>${escapeHtml(officer.designation_hi || officer.designation_en)}</strong> · ${escapeHtml(officer.department_hi || officer.department_en)}</div>
        ${officer.additional_charges && officer.additional_charges.length > 0 ? `<div style="font-size:11px;color:#fcd34d;margin-top:3px">⭐ अतिरिक्त प्रभार: ${escapeHtml(officer.additional_charges.join(', '))}</div>` : ''}
        <div style="font-size:11px;color:#38bdf8;margin-top:5px;font-weight:600">👉 Click to open Officer Dossier and view notification orders on canvas</div>
      </div>
    `;
  }

  // 1. Matched Governance Category Hub (Instant Category Routing)
  if (cat) {
    html += `
      <div class="sugg-category-banner" onclick="applySemanticCategory('${cat.id}')">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:11px;color:#38bdf8;text-transform:uppercase;font-weight:700;letter-spacing:0.5px">🎯 Matched Category Hub</span>
          <span class="badge" style="background:#0284c7;color:#fff">${cat.count} Circulars</span>
        </div>
        <div style="font-size:13.5px;font-weight:600;color:#fff;margin-top:3px">${cat.name}</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px">Click to expand spoke circulars & zoom directly to this Hub</div>
      </div>
    `;
  }

  // 2. Matching Documents list
  if (results.length > 0) {
    html += `<div class="sugg-section-title">Matching Documents (${data.total_matches} found)</div>`;
    results.slice(0, 8).forEach(r => {
      const st = r.status || 'current';
      html += `
        <div class="sugg-item" onclick="focusAndHighlightNode('${r.id}')">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <strong style="color:#38bdf8;font-size:12.5px">${escapeHtml(r.order_no)}</strong>
            <span class="badge badge-${st}">${st.toUpperCase()}</span>
          </div>
          <div style="font-size:11px;color:#94a3b8;margin-top:2px;display:flex;gap:8px">
            <span>📅 ${r.doc_date || 'Date N/A'}</span>
            <span>📂 ${escapeHtml(r.category || '')}</span>
          </div>
          <div style="font-size:12px;color:#cbd5e1;margin-top:3px;line-height:1.4">
            ${escapeHtml(r.title_hi || r.title_en || r.summary || '')}
          </div>
        </div>
      `;
    });
  }

  container.innerHTML = html;
  container.style.display = 'block';
}

function applySemanticCategory(categoryId) {
  const suggestions = document.getElementById('search-suggestions');
  if (suggestions) suggestions.style.display = 'none';

  // Ensure this category is expanded
  if (!expandedCategories.has(categoryId)) {
    expandedCategories.add(categoryId);
    renderCategoryPills();
  }

  loadHierarchicalGraph().then(() => {
    const hubId = `hub_${categoryId}`;
    const matchedDocs = [];
    if (nodesDataSet) {
      nodesDataSet.forEach(node => {
        if (node.category_id === categoryId && !node.is_hub) {
          matchedDocs.push(node.id);
        }
      });
    }

    executeHighlight(new Set(matchedDocs), new Set([hubId]));

    if (network && nodesDataSet.get(hubId)) {
      network.focus(hubId, {
        scale: 1.15,
        animation: { duration: 900, easingFunction: 'easeInOutQuad' }
      });
    }

    showHighlightStatusPill(`${matchedDocs.length} circulars in ${categoryId.toUpperCase()} highlighted (others dimmed)`);
  });
}

function focusAndHighlightNode(nodeId) {
  const suggestions = document.getElementById('search-suggestions');
  if (suggestions) suggestions.style.display = 'none';

  const node = nodesDataSet ? nodesDataSet.get(nodeId) : null;
  if (!node) {
    // If not currently visible, determine its category, auto-expand, then highlight
    fetch(`/api/graph/history/${encodeURIComponent(nodeId)}`)
      .then(res => res.json())
      .then(data => {
        if (data.success && data.target) {
          const catId = data.target.category_id;
          if (catId && !expandedCategories.has(catId)) {
            expandedCategories.add(catId);
            renderCategoryPills();
            loadHierarchicalGraph().then(() => {
              executeSingleNodeHighlight(nodeId);
            });
            return;
          }
        }
        executeSingleNodeHighlight(nodeId);
      });
    return;
  }

  executeSingleNodeHighlight(nodeId);
}

function executeSingleNodeHighlight(nodeId) {
  const node = nodesDataSet ? nodesDataSet.get(nodeId) : null;
  const hubSet = new Set(node && node.parent_hub ? [node.parent_hub] : []);
  executeHighlight(new Set([nodeId]), hubSet);

  if (network && nodesDataSet.get(nodeId)) {
    network.focus(nodeId, {
      scale: 1.45,
      animation: { duration: 800, easingFunction: 'easeInOutQuad' }
    });
  }

  showHighlightStatusPill(`Focused: ${(node && node.order_no) || nodeId}`);
  openDrawer(nodeId);
}

function applyCanvasHighlight(matchedNodeIds, matchedHubIds, query, matchedCategory) {
  const matchedSet = new Set(matchedNodeIds);
  const hubSet = new Set(matchedHubIds || []);

  let needsReload = false;
  hubSet.forEach(hubId => {
    const cid = hubId.replace('hub_', '');
    if (!expandedCategories.has(cid)) {
      expandedCategories.add(cid);
      needsReload = true;
    }
  });

  if (needsReload) {
    renderCategoryPills();
    loadHierarchicalGraph().then(() => {
      executeHighlight(matchedSet, hubSet);
      showHighlightStatusPill(`${matchedNodeIds.length} matches for "${query}" highlighted`);
    });
    return;
  }

  executeHighlight(matchedSet, hubSet);
  showHighlightStatusPill(`${matchedNodeIds.length} matches for "${query}" highlighted`);
}

function executeHighlight(matchedSet, hubSet) {
  if (!nodesDataSet) return;
  isHighlightActive = true;

  const updates = [];
  const edgeUpdates = [];

  nodesDataSet.forEach(node => {
    const isMatchedDoc = matchedSet.has(node.id);
    const isMatchedHub = hubSet.has(node.id) || (node.category_id && hubSet.has(`hub_${node.category_id}`));

    if (isMatchedDoc || isMatchedHub) {
      // Highlight: vibrant color, thick border, increased size
      updates.push({
        id: node.id,
        size: node.is_hub ? 46 : 22,
        borderWidth: node.is_hub ? 4 : 3,
        color: {
          background: isMatchedDoc ? '#38bdf8' : (node.color.background || '#10b981'),
          border: '#ffffff',
          highlight: { background: '#60a5fa', border: '#ffffff' }
        },
        font: {
          color: '#ffffff',
          size: node.is_hub ? 14 : 11,
          bold: true
        },
        opacity: 1.0
      });
    } else {
      // Dim: faint gray background and transparent border
      updates.push({
        id: node.id,
        size: node.is_hub ? 24 : 7,
        borderWidth: 1,
        color: {
          background: 'rgba(30, 41, 59, 0.35)',
          border: 'rgba(51, 65, 85, 0.2)'
        },
        font: {
          color: 'rgba(100, 116, 139, 0.3)',
          size: 8
        },
        opacity: 0.15
      });
    }
  });

  if (edgesDataSet) {
    edgesDataSet.forEach(edge => {
      const fromMatch = matchedSet.has(edge.from) || hubSet.has(edge.from);
      const toMatch = matchedSet.has(edge.to) || hubSet.has(edge.to);
      if (fromMatch && toMatch) {
        edgeUpdates.push({
          id: edge.id,
          color: { color: '#38bdf8', opacity: 0.9 },
          width: 2.5
        });
      } else {
        edgeUpdates.push({
          id: edge.id,
          color: { color: 'rgba(51, 65, 85, 0.08)', opacity: 0.08 },
          width: 0.5
        });
      }
    });
  }

  nodesDataSet.update(updates);
  if (edgesDataSet && edgeUpdates.length > 0) {
    edgesDataSet.update(edgeUpdates);
  }

  // Camera glide to matched targets
  const allTargets = Array.from(matchedSet).concat(Array.from(hubSet));
  const validTargets = allTargets.filter(id => nodesDataSet.get(id));

  if (validTargets.length === 1 && network) {
    network.focus(validTargets[0], {
      scale: 1.35,
      animation: { duration: 800, easingFunction: 'easeInOutQuad' }
    });
  } else if (validTargets.length > 1 && network) {
    network.fit({
      nodes: validTargets,
      animation: { duration: 900, easingFunction: 'easeInOutQuad' }
    });
  }
}

function showHighlightStatusPill(text) {
  const pill = document.getElementById('highlight-status-pill');
  const label = document.getElementById('highlight-status-text');
  if (pill && label) {
    label.innerText = `🔍 ${text}`;
    pill.style.display = 'flex';
  }
}

function clearCanvasHighlight() {
  isHighlightActive = false;
  const input = document.getElementById('graph-search-input');
  if (input) input.value = '';

  const clearBtn = document.getElementById('search-clear-btn');
  if (clearBtn) clearBtn.style.display = 'none';

  const suggestions = document.getElementById('search-suggestions');
  if (suggestions) suggestions.style.display = 'none';

  const pill = document.getElementById('highlight-status-pill');
  if (pill) pill.style.display = 'none';

  // Restore normal rendering
  renderVisNetwork(graphData.nodes, graphData.edges);
  if (network) {
    network.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
  }
}

// Dismiss suggestions when clicking outside
document.addEventListener('click', (e) => {
  const searchBox = document.querySelector('.search-box');
  const suggestions = document.getElementById('search-suggestions');
  if (suggestions && searchBox && !searchBox.contains(e.target)) {
    suggestions.style.display = 'none';
  }
});

function filterStatus(status, elem) {
  currentStatusFilter = status;
  document.querySelectorAll('.filter-pills .pill').forEach(p => p.classList.remove('active'));
  elem.classList.add('active');
  renderVisNetwork(graphData.nodes, graphData.edges);
}

function onYearSlider(val) {
  currentYearFilter = parseInt(val, 10);
  document.getElementById('year-label').innerText = val;
  switchViewMode('network');
  loadSubgraph({ year: currentYearFilter });
}

function resetYearFilter() {
  currentYearFilter = null;
  document.getElementById('year-label').innerText = 'All';
  switchViewMode('hierarchical');
}

// ─── DRAWER & FULL OCR DOCUMENT VIEWER ───────────────────────────

async function openDrawer(nodeId) {
  selectedNodeId = nodeId;
  const drawer = document.getElementById('history-drawer');
  drawer.classList.add('open');

  try {
    const res = await fetch(`/api/graph/history/${encodeURIComponent(nodeId)}`);
    const data = await res.json();
    if (data.success) {
      populateDrawer(data);
    }
  } catch (err) {
    console.error('Failed to load history:', err);
  }
}

function closeDrawer() {
  document.getElementById('history-drawer').classList.remove('open');
}

function populateDrawer(data) {
  const target = data.target || {};
  const standing = data.standing_order || {};
  const status = target.status || 'unknown';

  document.getElementById('d-order-no').innerText = target.order_no || 'Order No N/A';
  document.getElementById('d-date').innerText = target.doc_date || 'Date N/A';
  document.getElementById('d-title').innerText = target.title_hi || target.title_en || 'Document Title';
  document.getElementById('d-dept').innerText = target.department || 'सामान्य प्रशासन विभाग';

  // Badge styling
  const badge = document.getElementById('d-status-badge');
  badge.className = 'badge';
  if (status === 'current') {
    badge.classList.add('badge-current');
    badge.innerText = 'ACTIVE STANDING';
  } else if (status === 'amended') {
    badge.classList.add('badge-amended');
    badge.innerText = 'AMENDED';
  } else if (status === 'superseded') {
    badge.classList.add('badge-superseded');
    badge.innerText = 'SUPERSEDED';
  } else {
    badge.innerText = status.toUpperCase();
  }

  // Populate Chronological History Timeline
  const list = document.getElementById('history-timeline-list');
  list.innerHTML = '';

  const timeline = data.timeline_chain || [];
  if (timeline.length === 0) {
    list.innerHTML = '<div style="font-size:12px;color:#94a3b8">No historical references linked yet.</div>';
  } else {
    timeline.forEach(item => {
      const isSelected = item.is_selected;
      const isStanding = (item.node_id === standing.node_id && item.status === 'current');
      const itemStatus = item.status || 'unknown';

      const div = document.createElement('div');
      div.className = `timeline-node ${itemStatus} ${isSelected ? 'active' : ''}`;
      div.innerHTML = `
        <div class="timeline-card" onclick="openDrawer('${item.node_id}')">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
            <strong style="color:#fff">${item.order_no || item.doc_date || 'Order'}</strong>
            <span class="badge badge-${itemStatus}">${itemStatus.toUpperCase()}</span>
          </div>
          <div style="color:#94a3b8;font-size:12px">${item.doc_date || ''}</div>
          <div style="color:#cbd5e1;font-size:12px;margin-top:4px">${item.title_hi || item.title_en || ''}</div>
          ${isStanding ? '<div style="margin-top:6px;font-size:11px;color:#34d399;font-weight:600">⭐ Current Standing Rule in Series</div>' : ''}
        </div>
      `;
      list.appendChild(div);
    });
  }

  // Restore OCR Document viewer visibility if hidden by officer dossier
  const ocrWrap = document.getElementById('d-full-ocr-wrapper');
  if (ocrWrap) ocrWrap.style.display = 'block';
  const toggleBtn = document.getElementById('btn-toggle-ocr');
  if (toggleBtn) toggleBtn.style.display = 'inline-block';
  const expandBtn = document.getElementById('btn-expand-modal-ocr');
  if (expandBtn) expandBtn.style.display = 'inline-block';

  // Populate Full OCR Document (.md)
  currentTargetDoc = target;
  let ocrText = target.full_ocr_text || '';
  if (ocrText) {
    updateOcrDocumentDisplay(ocrText, target);
  } else {
    fetchFullOcrAsync(target.doc_id || target.id);
  }

  // Populate Chunks Preview
  const chunksBox = document.getElementById('d-chunks-container');
  chunksBox.innerHTML = `
    <div class="chunk-preview">
      <strong>Summary:</strong> ${target.summary || 'Summary unavailable'}<br/>
      <div style="margin-top:6px;font-size:12px;color:#64748b">
        Total indexed chunks for this document: ${target.chunk_count || 1}
      </div>
    </div>
  `;
}

// ─── MARKDOWN RENDERING & OCR VIEWER ENGINE ──────────────────────

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function renderMarkdownToHtml(rawMarkdown) {
  if (!rawMarkdown) {
    return '<div style="color:#94a3b8;font-style:italic;padding:12px">No document text available.</div>';
  }

  // Transform page markers (<!-- Page 1 -->) into clean styled visual chips
  let processed = rawMarkdown.replace(/<!--\s*Page\s*(\d+)\s*-->/gi, (match, p1) => {
    return `\n\n<div class="md-page-break"><span class="md-page-chip">📄 Page ${p1}</span></div>\n\n`;
  });

  if (typeof marked !== 'undefined' && marked.parse) {
    try {
      const html = marked.parse(processed, { gfm: true, breaks: true, async: false });
      if (typeof DOMPurify !== 'undefined' && DOMPurify.sanitize) {
        return DOMPurify.sanitize(html, {
          ADD_TAGS: ['div', 'span'],
          ADD_ATTR: ['class', 'style', 'id']
        });
      }
      return html;
    } catch (e) {
      console.warn('Marked parsing error:', e);
    }
  }

  return `<pre style="white-space:pre-wrap;font-family:monospace;margin:0">${escapeHtml(rawMarkdown)}</pre>`;
}

function updateOcrDocumentDisplay(text, docInfo) {
  currentDocumentMarkdown = text || '';
  const lines = text ? text.split('\n').length : 0;
  const kb = text ? (text.length / 1024).toFixed(1) : 0;
  const metaText = `${lines} lines · ${kb} KB`;

  const metaChip = document.getElementById('d-ocr-meta');
  if (metaChip) metaChip.innerText = metaText;

  const renderedBox = document.getElementById('d-full-ocr-rendered');
  const rawBox = document.getElementById('d-full-ocr-raw');

  if (renderedBox) {
    renderedBox.innerHTML = renderMarkdownToHtml(text);
  }
  if (rawBox) {
    rawBox.textContent = text;
  }

  setDrawerMdMode(drawerMdMode || 'render');

  const wrapper = document.getElementById('d-full-ocr-wrapper');
  if (wrapper) wrapper.style.display = 'flex';
  const btnToggle = document.getElementById('btn-toggle-ocr');
  if (btnToggle) btnToggle.innerText = 'Hide .md ▴';
}

async function fetchFullOcrAsync(docId) {
  if (!docId) return;
  const renderedBox = document.getElementById('d-full-ocr-rendered');
  if (renderedBox) {
    renderedBox.innerHTML = '<div style="color:#94a3b8;font-size:12px;padding:12px">Loading full OCR markdown text...</div>';
  }
  try {
    const cleanId = String(docId).replace(/^doc_/, '');
    const res = await fetch(`/api/graph/document/${encodeURIComponent(cleanId)}`);
    const data = await res.json();
    if (data.success && data.full_ocr_text) {
      updateOcrDocumentDisplay(data.full_ocr_text, currentTargetDoc || {});
    } else {
      if (renderedBox) {
        renderedBox.innerHTML = '<div style="color:#64748b;font-size:12px;font-style:italic;padding:12px">(Full OCR document text not found for this circular)</div>';
      }
    }
  } catch (err) {
    if (renderedBox) {
      renderedBox.innerHTML = '<div style="color:#ef4444;font-size:12px;padding:12px">Failed to load OCR text from server.</div>';
    }
  }
}

function setDrawerMdMode(mode) {
  drawerMdMode = mode;
  const renderedBox = document.getElementById('d-full-ocr-rendered');
  const rawBox = document.getElementById('d-full-ocr-raw');
  const btnRender = document.getElementById('btn-md-mode-render');
  const btnRaw = document.getElementById('btn-md-mode-raw');

  if (mode === 'raw') {
    if (renderedBox) renderedBox.style.display = 'none';
    if (rawBox) rawBox.style.display = 'block';
    if (btnRender) btnRender.classList.remove('active');
    if (btnRaw) btnRaw.classList.add('active');
  } else {
    if (renderedBox) renderedBox.style.display = 'block';
    if (rawBox) rawBox.style.display = 'none';
    if (btnRender) btnRender.classList.add('active');
    if (btnRaw) btnRaw.classList.remove('active');
  }
}

function toggleFullOcrView() {
  const wrapper = document.getElementById('d-full-ocr-wrapper');
  const btnToggle = document.getElementById('btn-toggle-ocr');
  if (!wrapper) return;
  if (wrapper.style.display === 'none') {
    wrapper.style.display = 'flex';
    btnToggle.innerText = 'Hide .md ▴';
  } else {
    wrapper.style.display = 'none';
    btnToggle.innerText = 'Show .md ▾';
  }
}

// ─── FULLSCREEN MODAL DOCUMENT READER ────────────────────────────

function openFullscreenMdModal() {
  const backdrop = document.getElementById('md-modal-backdrop');
  if (!backdrop) return;

  const target = currentTargetDoc || {};
  document.getElementById('modal-order-no').innerText = target.order_no || 'Document View';
  document.getElementById('modal-doc-date').innerText = target.doc_date || '';
  document.getElementById('modal-doc-dept').innerText = target.department || 'सामान्य प्रशासन विभाग';

  const badge = document.getElementById('modal-status-badge');
  const status = target.status || 'current';
  badge.className = `badge badge-${status}`;
  badge.innerText = status.toUpperCase();

  const lines = currentDocumentMarkdown ? currentDocumentMarkdown.split('\n').length : 0;
  const kb = currentDocumentMarkdown ? (currentDocumentMarkdown.length / 1024).toFixed(1) : 0;
  const metaBadge = document.getElementById('modal-doc-meta');
  if (metaBadge) metaBadge.innerText = `${lines} lines · ${kb} KB`;

  const modalRendered = document.getElementById('modal-rendered-content');
  const modalRaw = document.getElementById('modal-raw-content');

  if (modalRendered) {
    modalRendered.innerHTML = renderMarkdownToHtml(currentDocumentMarkdown);
  }
  if (modalRaw) {
    modalRaw.textContent = currentDocumentMarkdown;
  }

  setModalMdMode(modalMdMode || 'render');
  backdrop.classList.add('open');
}

function closeFullscreenMdModal() {
  const backdrop = document.getElementById('md-modal-backdrop');
  if (backdrop) backdrop.classList.remove('open');
}

function onModalBackdropClick(event) {
  if (event.target.id === 'md-modal-backdrop') {
    closeFullscreenMdModal();
  }
}

function setModalMdMode(mode) {
  modalMdMode = mode;
  const modalRendered = document.getElementById('modal-rendered-content');
  const modalRaw = document.getElementById('modal-raw-content');
  const btnRender = document.getElementById('modal-btn-render');
  const btnRaw = document.getElementById('modal-btn-raw');

  if (mode === 'raw') {
    if (modalRendered) modalRendered.style.display = 'none';
    if (modalRaw) modalRaw.style.display = 'block';
    if (btnRender) btnRender.classList.remove('active');
    if (btnRaw) btnRaw.classList.add('active');
  } else {
    if (modalRendered) modalRendered.style.display = 'block';
    if (modalRaw) modalRaw.style.display = 'none';
    if (btnRender) btnRender.classList.add('active');
    if (btnRaw) btnRaw.classList.remove('active');
  }
}

function copyDocumentMarkdown() {
  if (!currentDocumentMarkdown) return;
  navigator.clipboard.writeText(currentDocumentMarkdown).then(() => {
    const copyBtns = [
      document.getElementById('btn-copy-ocr'),
      document.querySelector('#md-modal-backdrop button[onclick="copyDocumentMarkdown()"]')
    ];
    copyBtns.forEach(btn => {
      if (btn) {
        const orig = btn.innerText;
        btn.innerText = '✓ Copied!';
        btn.style.color = '#34d399';
        setTimeout(() => {
          btn.innerText = orig;
          btn.style.color = '';
        }, 1800);
      }
    });
  }).catch(err => {
    console.error('Clipboard copy failed:', err);
  });
}

function printModalDocument() {
  const content = modalMdMode === 'raw'
    ? document.getElementById('modal-raw-content').innerHTML
    : document.getElementById('modal-rendered-content').innerHTML;
  const title = (currentTargetDoc && currentTargetDoc.order_no) || 'Government Circular';

  const printWindow = window.open('', '_blank');
  printWindow.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>${title}</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans Devanagari", sans-serif; padding: 30px; line-height: 1.6; color: #111; }
        h1, h2, h3 { color: #0f172a; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left; }
        th { background: #f1f5f9; }
        .md-page-break { border-top: 2px dashed #94a3b8; margin: 30px 0; text-align: center; }
        .md-page-chip { background: #e2e8f0; padding: 2px 8px; font-size: 11px; font-weight: bold; }
        @media print { .no-print { display: none; } }
      </style>
    </head>
    <body>
      <h2>${title}</h2>
      <div>${content}</div>
    </body>
    </html>
  `);
  printWindow.document.close();
  printWindow.focus();
  setTimeout(() => { printWindow.print(); }, 250);
}

// Close modals on Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeFullscreenMdModal();
    closeOfficersDirectoryModal();
  }
});

function focusLineage() {
  if (selectedNodeId) {
    switchViewMode('network');
    loadSubgraph({ center: selectedNodeId });
  }
}

// ─── RETRIEVER EXPERIMENT LOGIC ─────────────────────────────────

function setPreset(queryText) {
  document.getElementById('exp-query-input').value = queryText;
  runRetrieverExperiment();
}

async function runRetrieverExperiment() {
  const query = document.getElementById('exp-query-input').value.trim();
  if (!query) return;

  const ftsBox = document.getElementById('fts-results-container');
  const graphBox = document.getElementById('graph-results-container');
  const answerBox = document.getElementById('llm-answer-text');

  ftsBox.innerHTML = '<div style="color:#94a3b8;font-size:13px">Retrieving lexical results…</div>';
  graphBox.innerHTML = '<div style="color:#94a3b8;font-size:13px">Traversing temporal knowledge graph…</div>';
  answerBox.innerHTML = '<em>Generating grounded answer with remote LLM (ocr-32b)…</em>';

  try {
    const res = await fetch('/api/retriever/experiment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 5, generate_answer: true })
    });

    const data = await res.json();
    if (!data.success) {
      ftsBox.innerHTML = `<div style="color:#ef4444">Error: ${data.error}</div>`;
      graphBox.innerHTML = '';
      answerBox.innerHTML = '';
      return;
    }

    const exp = data.experiment;
    renderExperimentResults(exp);
  } catch (err) {
    console.error('Experiment failed:', err);
    ftsBox.innerHTML = `<div style="color:#ef4444">Request failed: ${err.message}</div>`;
  }
}

function renderExperimentResults(exp) {
  const fts = exp.lexical_fts;
  const tg = exp.temporal_graph;

  // Latency & counts
  document.getElementById('fts-latency').innerText = `${fts.latency_ms} ms (${fts.total_candidates} matches)`;
  document.getElementById('graph-latency').innerText = `${tg.latency_ms} ms (${tg.total_candidates} enriched)`;

  // Render FTS Chunks
  const ftsBox = document.getElementById('fts-results-container');
  ftsBox.innerHTML = '';
  if (fts.results.length === 0) {
    ftsBox.innerHTML = '<div style="color:#94a3b8;font-size:13px">No lexical matches found.</div>';
  } else {
    fts.results.forEach((r, idx) => {
      const card = document.createElement('div');
      card.className = 'chunk-preview';
      card.innerHTML = `
        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
          <strong>#${idx + 1} ${r.order_no || 'Order'}</strong>
          <span style="font-size:11px;color:#94a3b8">Score: ${r.score.toFixed(3)}</span>
        </div>
        <div style="font-size:12px;color:#94a3b8;margin-bottom:6px">${r.doc_date || 'Date N/A'}</div>
        <div style="font-size:12px;color:#cbd5e1;line-height:1.4">${r.content.slice(0, 200)}…</div>
      `;
      ftsBox.appendChild(card);
    });
  }

  // Render Temporal Graph Results (with supersession alerts!)
  const graphBox = document.getElementById('graph-results-container');
  graphBox.innerHTML = '';

  // Render Temporal Warnings
  if (tg.warnings && tg.warnings.length > 0) {
    const wBox = document.createElement('div');
    wBox.style.background = 'rgba(239, 68, 68, 0.15)';
    wBox.style.border = '1px solid #ef4444';
    wBox.style.borderRadius = '6px';
    wBox.style.padding = '8px 12px';
    wBox.style.marginBottom = '12px';
    wBox.style.fontSize = '12px';
    wBox.style.color = '#fca5a5';
    wBox.innerHTML = '<strong>⚠️ Temporal Lineage Warnings:</strong><br/>' + tg.warnings.map(w => `• ${w}`).join('<br/>');
    graphBox.appendChild(wBox);
  }

  if (tg.results.length === 0) {
    graphBox.innerHTML += '<div style="color:#94a3b8;font-size:13px">No graph-enriched results found.</div>';
  } else {
    tg.results.forEach((r, idx) => {
      const card = document.createElement('div');
      card.className = 'chunk-preview';
      const isSuper = (r.effective_status === 'superseded');
      card.style.borderLeftColor = isSuper ? '#ef4444' : (r.effective_status === 'amended' ? '#f59e0b' : '#10b981');

      card.innerHTML = `
        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
          <strong>#${idx + 1} ${r.order_no || 'Order'}</strong>
          <span class="badge badge-${r.effective_status}">${r.effective_status.toUpperCase()}</span>
        </div>
        <div style="font-size:12px;color:#94a3b8;margin-bottom:4px">
          Date: ${r.doc_date || 'N/A'} | Source: ${r.retrieval_source}
        </div>
        ${r.superseded_by ? `<div style="font-size:11px;color:#f87171;font-weight:600;margin-bottom:4px">⚠️ Superseded by: ${r.superseded_by}</div>` : ''}
        ${r.lineage_summary ? `<div style="font-size:11px;color:#60a5fa;margin-bottom:4px">${r.lineage_summary}</div>` : ''}
        <div style="font-size:12px;color:#cbd5e1;line-height:1.4">${r.content.slice(0, 220)}…</div>
      `;
      graphBox.appendChild(card);
    });
  }

  // Render AI Answer
  const answerBox = document.getElementById('llm-answer-text');
  if (exp.llm_answer) {
    answerBox.innerHTML = exp.llm_answer.replace(/\n/g, '<br/>');
  } else {
    answerBox.innerHTML = '<span style="color:#94a3b8">No answer generated.</span>';
  }
}

// ─── OFFICER DOSSIER & LEADERSHIP DIRECTORY ───────────────────────

let cachedOfficersList = null;
let currentOfficerModalCategory = 'all';

async function displayOfficerDossier(officerId) {
  const suggestions = document.getElementById('search-suggestions');
  if (suggestions) suggestions.style.display = 'none';
  closeOfficersDirectoryModal();

  try {
    const res = await fetch(`/api/officer/${encodeURIComponent(officerId)}`);
    const data = await res.json();
    if (!data.success || !data.officer) {
      console.error('Officer not found:', data.error);
      return;
    }
    const off = data.officer;
    populateOfficerDrawer(off);
    highlightOfficerOrders(off);
  } catch (err) {
    console.error('Failed to load officer dossier:', err);
  }
}

function populateOfficerDrawer(off) {
  const drawer = document.getElementById('history-drawer');
  drawer.classList.add('open');

  // Badge
  const badge = document.getElementById('d-status-badge');
  badge.className = 'badge';
  badge.style.background = '#059669';
  badge.style.color = '#fff';
  badge.innerText = `OFFICER DOSSIER · ${(off.category_label || off.category || 'ADMIN').toUpperCase()}`;

  // Name & Header
  document.getElementById('d-order-no').innerText = `${off.name_en} (${off.name_hi || ''})`;
  document.getElementById('d-date').innerText = off.category_label || 'Active Civil Service';
  document.getElementById('d-title').innerText = `${off.designation_hi || off.designation_en || 'Designation'} · ${off.department_hi || off.department_en || ''}`;
  document.getElementById('d-dept').innerText = 'Government of Chhattisgarh';

  // Hide circular full OCR preview for officer dossier
  const ocrWrap = document.getElementById('d-full-ocr-wrapper');
  if (ocrWrap) ocrWrap.style.display = 'none';
  const toggleBtn = document.getElementById('btn-toggle-ocr');
  if (toggleBtn) toggleBtn.style.display = 'none';
  const expandBtn = document.getElementById('btn-expand-modal-ocr');
  if (expandBtn) expandBtn.style.display = 'none';

  // Details box in Chunks container
  const chunksBox = document.getElementById('d-chunks-container');
  let chargesHtml = '';
  if (off.additional_charges && off.additional_charges.length > 0) {
    chargesHtml = `
      <div class="officer-meta-row" style="align-items:flex-start">
        <span class="officer-meta-label">अतिरिक्त प्रभार:</span>
        <span style="color:#fcd34d;font-weight:600">${escapeHtml(off.additional_charges.join(', '))}</span>
      </div>
    `;
  }

  chunksBox.innerHTML = `
    <div class="officer-dossier-box">
      <div style="font-size:11px;font-weight:700;color:#34d399;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px">
        🏛️ Administrative Contact & Profile
      </div>
      <div class="officer-meta-row">
        <span class="officer-meta-label">Designation:</span>
        <strong style="color:#fff">${escapeHtml(off.designation_hi || off.designation_en || 'N/A')}</strong>
      </div>
      <div class="officer-meta-row">
        <span class="officer-meta-label">Department:</span>
        <span>${escapeHtml(off.department_hi || off.department_en || 'N/A')}</span>
      </div>
      ${chargesHtml}
      <div class="officer-meta-row">
        <span class="officer-meta-label">Office Phone:</span>
        <span>${escapeHtml(off.office_phone || 'N/A')}</span>
      </div>
      <div class="officer-meta-row">
        <span class="officer-meta-label">Res. Phone:</span>
        <span>${escapeHtml(off.res_phone || 'N/A')}</span>
      </div>
      <div class="officer-meta-row">
        <span class="officer-meta-label">Email:</span>
        <span>${off.email ? `<a href="mailto:${escapeHtml(off.email)}" style="color:#38bdf8;text-decoration:none">${escapeHtml(off.email)}</a>` : 'N/A'}</span>
      </div>
      <div class="officer-meta-row" style="margin-top:10px;padding-top:8px;border-top:1px dashed #1e3a5f">
        <span class="officer-meta-label">Source Record:</span>
        <a href="${off.source_url || 'https://gad.cg.gov.in/'}" target="_blank" style="color:#38bdf8;text-decoration:none;font-weight:600">
          🌐 ${escapeHtml(off.source_title || off.source_name || 'Civil List Directory')} ↗
        </a>
      </div>
      <div class="officer-meta-row">
        <span class="officer-meta-label">File / Doc ID:</span>
        <span style="font-family:monospace;font-size:11px;color:#94a3b8">
          ${escapeHtml(off.filename || off.source_doc_id || 'Display_civil_list_Email.aspx')} (${escapeHtml(off.source_doc_id || '')})
        </span>
      </div>
    </div>
  `;

  // Linked Notifications Timeline
  const list = document.getElementById('history-timeline-list');
  list.innerHTML = '';
  const notifs = off.notifications || [];

  if (notifs.length === 0) {
    list.innerHTML = '<div style="font-size:12px;color:#94a3b8;padding:10px">No linked GAD circulars or notifications found in repository yet.</div>';
  } else {
    notifs.forEach(notif => {
      const typeBadge = notif.notification_type ? notif.notification_type.toUpperCase() : 'NOTIFICATION';
      const div = document.createElement('div');
      div.className = 'officer-notif-card';
      div.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
          <strong style="color:#38bdf8;font-size:13px">${escapeHtml(notif.order_no || 'Order')}</strong>
          <span class="badge" style="background:#1e3a5f;color:#60a5fa;font-size:10px">${typeBadge}</span>
        </div>
        <div style="font-size:11.5px;color:#94a3b8;margin-bottom:4px">
          📅 ${escapeHtml(notif.order_date || notif.doc_date || 'Date N/A')}
        </div>
        <div style="font-size:12.5px;color:#f1f5f9;font-weight:600;margin-bottom:4px">
          📄 ${escapeHtml(notif.doc_title || notif.subject || 'Government Order')}
        </div>
        <div style="font-size:11px;color:#64748b;margin-bottom:6px;display:flex;gap:10px;flex-wrap:wrap">
          <span>📁 ${escapeHtml(notif.filename || notif.doc_id + '.pdf')}</span>
          <span>🆔 <code style="font-size:10.5px">${escapeHtml(notif.doc_id)}</code></span>
        </div>
        <div style="font-size:12px;color:#cbd5e1;line-height:1.4;margin-bottom:8px">
          ${escapeHtml(notif.excerpt || 'Government order relating to this officer.')}
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          <button class="pill" style="font-size:11px;padding:3px 8px;background:#0369a1;color:#fff;border-color:#0284c7" onclick="openDrawer('${notif.doc_id}')">
            📖 View Circular & Full OCR (.md)
          </button>
          <button class="pill" style="font-size:11px;padding:3px 8px;background:#1e293b;border-color:#334155;color:#94a3b8" onclick="focusAndHighlightNode('${notif.doc_id}')">
            🎯 Focus in Graph
          </button>
          ${notif.source_url ? `<a href="${escapeHtml(notif.source_url)}" target="_blank" class="pill" style="font-size:11px;padding:3px 8px;background:#064e3b;border-color:#10b981;color:#6ee7b7;text-decoration:none" title="Open official live document link on Government of Chhattisgarh portal">🌐 Official Link ↗</a>` : ''}
        </div>
      `;
      list.appendChild(div);
    });
  }
}

function highlightOfficerOrders(off) {
  const notifs = off.notifications || [];
  const matchedDocIds = notifs.map(n => n.doc_id).filter(Boolean);
  if (matchedDocIds.length === 0) {
    showHighlightStatusPill(`Officer: ${off.name_en} (0 linked circulars in active graph)`);
    return;
  }

  // If in hierarchical view, auto-expand categories that contain these docs
  if (viewMode === 'hierarchical' && rawGraphData && rawGraphData.nodes) {
    const requiredCategories = new Set();
    const docToCat = {};
    rawGraphData.nodes.forEach(n => {
      if (n.category_id) docToCat[n.id] = n.category_id;
    });

    matchedDocIds.forEach(id => {
      const cat = docToCat[id];
      if (cat && !expandedCategories.has(cat)) {
        requiredCategories.add(cat);
      }
    });

    if (requiredCategories.size > 0) {
      requiredCategories.forEach(c => expandedCategories.add(c));
      renderCategoryPills();
      loadHierarchicalGraph().then(() => {
        applyOfficerCanvasHighlight(matchedDocIds, off.name_en);
      });
      return;
    }
  }

  applyOfficerCanvasHighlight(matchedDocIds, off.name_en);
}

function applyOfficerCanvasHighlight(matchedDocIds, officerName) {
  const nodeSet = new Set(matchedDocIds);
  const hubSet = new Set();

  if (nodesDataSet) {
    nodesDataSet.forEach(node => {
      if (nodeSet.has(node.id) && node.parent_hub) {
        hubSet.add(node.parent_hub);
      }
    });
  }

  executeHighlight(nodeSet, hubSet);

  // Focus camera on first available matched node
  if (network && nodesDataSet) {
    for (const id of matchedDocIds) {
      if (nodesDataSet.get(id)) {
        network.focus(id, {
          scale: 1.25,
          animation: { duration: 800, easingFunction: 'easeInOutQuad' }
        });
        break;
      }
    }
  }

  showHighlightStatusPill(`👤 ${officerName}: ${matchedDocIds.length} linked circulars highlighted`);
}

function openOfficersDirectoryModal() {
  const modal = document.getElementById('officers-modal-backdrop');
  if (!modal) return;
  modal.classList.add('open');

  const searchInput = document.getElementById('officer-modal-search');
  if (searchInput) {
    searchInput.value = '';
    searchInput.focus();
  }

  if (!cachedOfficersList) {
    fetch('/api/officers?limit=350')
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          cachedOfficersList = data.officers || [];
          renderOfficersGrid();
        }
      })
      .catch(err => console.error('Failed to load officers:', err));
  } else {
    renderOfficersGrid();
  }
}

function closeOfficersDirectoryModal() {
  const modal = document.getElementById('officers-modal-backdrop');
  if (modal) modal.classList.remove('open');
}

function onOfficersBackdropClick(event) {
  if (event.target.id === 'officers-modal-backdrop') {
    closeOfficersDirectoryModal();
  }
}

function filterOfficerDirectory(cat, btn) {
  currentOfficerModalCategory = cat;
  const tabs = document.querySelectorAll('#officer-category-tabs .pill');
  tabs.forEach(t => t.classList.remove('active'));
  if (btn) btn.classList.add('active');
  renderOfficersGrid();
}

function onOfficerModalSearch(val) {
  renderOfficersGrid();
}

function renderOfficersGrid() {
  const container = document.getElementById('officers-grid-container');
  if (!container || !cachedOfficersList) return;

  const searchVal = (document.getElementById('officer-modal-search')?.value || '').toLowerCase().trim();
  const cat = currentOfficerModalCategory;

  const filtered = cachedOfficersList.filter(off => {
    // Category match
    if (cat !== 'all') {
      if (cat === 'secretaries' && !['secretaries', 'chief_secretary'].includes(off.category)) return false;
      else if (cat !== 'secretaries' && off.category !== cat) return false;
    }

    // Text match
    if (searchVal) {
      let extra = '';
      if (Array.isArray(off.additional_charges)) {
        extra = off.additional_charges.join(' ');
      } else if (typeof off.additional_charges === 'string') {
        extra = off.additional_charges;
      }
      const haystack = `${off.name_en || ''} ${off.name_hi || ''} ${off.designation_en || ''} ${off.designation_hi || ''} ${off.department_en || ''} ${off.department_hi || ''} ${extra}`.toLowerCase();
      if (!haystack.includes(searchVal)) return false;
    }

    return true;
  });

  if (filtered.length === 0) {
    container.innerHTML = `
      <div style="grid-column: 1 / -1; padding: 40px; text-align: center; color: #94a3b8">
        No officers found matching your criteria.
      </div>
    `;
    return;
  }

  let html = '';
  filtered.forEach(off => {
    const notifCount = off.notification_count || (off.notifications ? off.notifications.length : 0);
    const notifBadge = notifCount > 0 
      ? `<span class="badge" style="background:#059669;color:#fff">${notifCount} Orders</span>`
      : '';

    html += `
      <div class="officer-item-card" onclick="displayOfficerDossier('${off.id}')">
        <div style="display:flex;justify-content:space-between;align-items:flex-start">
          <span class="officer-card-badge">${escapeHtml(off.category_label || off.category || 'Officer')}</span>
          ${notifBadge}
        </div>
        <div style="font-size:14px;font-weight:700;color:#fff;margin-top:2px">
          ${escapeHtml(off.name_en)} ${off.name_hi ? `<span style="font-weight:normal;color:#94a3b8;font-size:12.5px">(${escapeHtml(off.name_hi)})</span>` : ''}
        </div>
        <div style="font-size:12px;color:#38bdf8;font-weight:600">
          ${escapeHtml(off.designation_hi || off.designation_en || 'Designation N/A')}
        </div>
        <div style="font-size:11.5px;color:#cbd5e1">
          ${escapeHtml(off.department_hi || off.department_en || '')}
        </div>
        ${off.email ? `<div style="font-size:11px;color:#94a3b8;margin-top:2px">✉️ ${escapeHtml(off.email)}</div>` : ''}
      </div>
    `;
  });

  container.innerHTML = html;
}
