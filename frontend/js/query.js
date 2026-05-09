// query.js — Q&A form: submit, render answer + sources

const queryForm      = document.getElementById("queryForm");
const queryInput     = document.getElementById("queryInput");
const queryBtn       = document.getElementById("queryBtn");
const answerSection  = document.getElementById("answerSection");
const answerText     = document.getElementById("answerText");
const sourcesGrid    = document.getElementById("sourcesGrid");
const loadingState   = document.getElementById("loadingState");
const queryEmptyState = document.getElementById("queryEmptyState");
const copyBtn        = document.getElementById("copyBtn");

// Suggestion chips
document.querySelectorAll(".suggestion-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    queryInput.value = chip.dataset.q;
    queryInput.focus();
  });
});

// Form submit
queryForm?.addEventListener("submit", async e => {
  e.preventDefault();
  const question = queryInput.value.trim();
  if (!question) return;
  await runQuery(question);
});

async function runQuery(question) {
  // Show loading
  queryEmptyState.style.display = "none";
  answerSection.style.display   = "none";
  loadingState.style.display    = "block";
  queryBtn.disabled = true;

  try {
    const res = await fetch(`${window.API}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Query failed");
    }

    const data = await res.json();
    renderAnswer(data);

  } catch (err) {
    loadingState.style.display = "none";
    queryEmptyState.style.display = "block";
    window.showToast(`Error: ${err.message}`, "error");
  } finally {
    queryBtn.disabled = false;
  }
}

function renderAnswer(data) {
  loadingState.style.display = "none";
  answerSection.style.display = "block";

  // Typewriter effect
  answerText.textContent = "";
  typewriterEffect(answerText, data.answer);

  // Sources
  sourcesGrid.innerHTML = "";
  if (data.sources && data.sources.length > 0) {
    data.sources.forEach(src => {
      const ext = src.source_file.split(".").pop().toLowerCase();
      const card = document.createElement("div");
      card.className = "source-card";
      card.innerHTML = `
        <div class="source-card-header">
          <svg class="source-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          <span class="source-name">${escapeHtml(src.source_file)}</span>
          <span class="source-page">Page ${src.page}</span>
        </div>
        <div class="source-excerpt">${escapeHtml(src.excerpt)}</div>
      `;
      sourcesGrid.appendChild(card);
    });
  } else {
    sourcesGrid.innerHTML = `<p style="color:var(--text-muted);font-size:0.82rem;">No source references available.</p>`;
  }
}

function typewriterEffect(el, text, speed = 12) {
  let i = 0;
  el.textContent = "";
  const interval = setInterval(() => {
    el.textContent += text[i];
    i++;
    if (i >= text.length) clearInterval(interval);
  }, speed);
}

// Copy answer
copyBtn?.addEventListener("click", () => {
  const text = answerText.textContent;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    window.showToast("Answer copied to clipboard!", "success");
  });
});

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
