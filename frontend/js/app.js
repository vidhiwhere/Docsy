// app.js — Core routing, sidebar, toasts, and health check

const API = "http://127.0.0.1:5000/api";

// ── View Router ──────────────────────────────────────
const views   = document.querySelectorAll(".view");
const navItems = document.querySelectorAll(".nav-item");
const topbarTitle = document.getElementById("topbarTitle");

const VIEW_TITLES = {
  query: "Ask Docsy",
  upload: "Upload Documents",
  documents: "Knowledge Base",
  dashboard: "Dashboard",
};

function showView(name) {
  views.forEach(v => v.classList.remove("active"));
  navItems.forEach(n => n.classList.remove("active"));

  const target = document.getElementById(`view-${name}`);
  const navEl   = document.getElementById(`nav-${name}`);
  if (target) target.classList.add("active");
  if (navEl)  navEl.classList.add("active");
  if (topbarTitle) topbarTitle.textContent = VIEW_TITLES[name] || name;

  // Trigger view-specific refresh
  if (name === "dashboard") window.loadDashboard?.();
  if (name === "documents") window.loadDocuments?.();

  closeSidebar();
}

navItems.forEach(item => {
  item.addEventListener("click", e => {
    e.preventDefault();
    const view = item.dataset.view;
    if (view) showView(view);
  });
});

// ── Sidebar (mobile) ─────────────────────────────────
const sidebar = document.getElementById("sidebar");
const hamburger = document.getElementById("hamburger");
const sidebarClose = document.getElementById("sidebarClose");

function openSidebar()  { sidebar.classList.add("open"); }
function closeSidebar() { sidebar.classList.remove("open"); }

hamburger?.addEventListener("click", openSidebar);
sidebarClose?.addEventListener("click", closeSidebar);

// Close on backdrop click
document.addEventListener("click", e => {
  if (sidebar.classList.contains("open") &&
      !sidebar.contains(e.target) &&
      e.target !== hamburger) {
    closeSidebar();
  }
});

// ── Toast Notifications ──────────────────────────────
function showToast(msg, type = "info", duration = 3500) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = "opacity 0.3s";
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

window.showToast = showToast;
window.API = API;

// ── Health Check ─────────────────────────────────────
async function checkHealth() {
  const statusDot   = document.querySelector(".status-dot");
  const statusEl    = document.querySelector(".status-indicator span:last-child");
  const modeLabel   = document.getElementById("modeLabel");

  try {
    const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      const data = await res.json();
      statusDot?.classList.add("online");
      if (statusEl) statusEl.textContent = "API Online";
      if (modeLabel) modeLabel.textContent = data.mode === "openai" ? "OpenAI Mode" : "Local Mode";
    } else {
      throw new Error("not ok");
    }
  } catch {
    statusDot?.classList.remove("online");
    if (statusEl) statusEl.textContent = "API Offline";
    if (modeLabel) modeLabel.textContent = "Disconnected";
  }
}

checkHealth();
setInterval(checkHealth, 30000);
