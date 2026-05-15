// dashboard.js — Stats panel + documents view

// ── Dashboard ────────────────────────────────────────
async function loadDashboard() {
  try {
    const res  = await fetch(`${window.API}/stats`);
    const data = await res.json();

    animateCount("statDocs",    data.doc_count);
    animateCount("statChunks",  data.index_vectors);
    animateCount("statQueries", data.query_count);
    animateCount("statToday",   data.today_queries);

    const recentList = document.getElementById("recentList");
    if (recentList) {
      if (data.recent_queries && data.recent_queries.length > 0) {
        recentList.innerHTML = "";
        data.recent_queries.forEach(q => {
          const li = document.createElement("li");
          li.textContent = q.question.length > 70
            ? q.question.slice(0, 70) + "…"
            : q.question;
          recentList.appendChild(li);
        });
      } else {
        recentList.innerHTML = `<li class="recent-empty">No queries yet.</li>`;
      }
    }
  } catch {
    // API offline — silently ignore
  }
}

function animateCount(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  const duration = 800;
  const start = performance.now();
  const from = parseInt(el.textContent) || 0;
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(from + (target - from) * ease);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

window.loadDashboard = loadDashboard;

// ── Documents View ───────────────────────────────────
async function loadDocuments() {
  const grid = document.getElementById("docsGrid");
  if (!grid) return;
  grid.innerHTML = `<div class="docs-loading">Loading documents…</div>`;

  try {
    const res  = await fetch(`${window.API}/documents`);
    const data = await res.json();
    renderDocuments(data.documents);
  } catch {
    grid.innerHTML = `<div class="docs-empty">Could not load documents. Is the API running?</div>`;
  }
}

let allDocs = [];

function renderDocuments(docs) {
  allDocs = docs;
  filterDocs("");
}

function filterDocs(query) {
  const grid = document.getElementById("docsGrid");
  const filtered = allDocs.filter(d =>
    d.filename.toLowerCase().includes(query.toLowerCase())
  );

  if (filtered.length === 0) {
    grid.innerHTML = `<div class="docs-empty">No documents found. Upload some docs first.</div>`;
    return;
  }

  grid.innerHTML = "";
  filtered.forEach(doc => {
    const ext = doc.filename.split(".").pop().toLowerCase();
    const card = document.createElement("div");
    card.className = "doc-card";
    card.innerHTML = `
      <div class="doc-card-header">
        <span class="doc-type-badge ${ext}">${ext.toUpperCase()}</span>
        <span class="doc-name">${escapeHtml(doc.filename)}</span>
      </div>
      <div class="doc-meta">
        <span>${doc.chunk_count} chunks</span>
        <span>${doc.page_count} pages</span>
        <span>${formatDate(doc.created_at)}</span>
      </div>
      <div class="doc-actions">
        <button class="btn-secondary small btn-summarize" data-id="${doc.id}" data-name="${escapeHtml(doc.filename)}">
          Summarize
        </button>
        <button class="btn-delete" data-id="${doc.id}" data-name="${escapeHtml(doc.filename)}">
          Delete
        </button>
      </div>
    `;
    grid.appendChild(card);
  });

  grid.querySelectorAll(".btn-delete").forEach(btn => {
    btn.addEventListener("click", () => confirmDelete(btn.dataset.id, btn.dataset.name));
  });

  grid.querySelectorAll(".btn-summarize").forEach(btn => {
    btn.addEventListener("click", () => summarizeDoc(btn.dataset.id, btn.dataset.name));
  });
}

async function summarizeDoc(id, name) {
  window.showToast(`Summarizing "${name}"...`, "info");
  try {
    const res = await fetch(`${window.API}/summarize/${id}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Summarization failed");
    
    // Show summary in a modal or alert
    alert(`Summary for ${name}:\n\n${data.summary}`);
  } catch (err) {
    window.showToast("Could not summarize: " + err.message, "error");
  }
}

async function confirmDelete(id, name) {
  if (!confirm(`Delete "${name}" from the knowledge base? This cannot be undone.`)) return;
  try {
    const res = await fetch(`${window.API}/documents/${id}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Delete failed");
    window.showToast(`"${name}" removed.`, "info");
    loadDocuments();
    loadDashboard();
  } catch (err) {
    window.showToast("Could not delete: " + err.message, "error");
  }
}

// Filter input
document.getElementById("docSearch")?.addEventListener("input", e => {
  filterDocs(e.target.value);
});
document.getElementById("refreshDocs")?.addEventListener("click", loadDocuments);

window.loadDocuments = loadDocuments;

// ── Helpers ──────────────────────────────────────────
function formatDate(str) {
  if (!str) return "";
  try {
    return new Date(str + "Z").toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch { return str; }
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
