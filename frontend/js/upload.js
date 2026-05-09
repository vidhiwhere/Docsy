// upload.js — Drag & drop file upload with progress feedback

const uploadZone    = document.getElementById("uploadZone");
const fileInput     = document.getElementById("fileInput");
const browseBtn     = document.getElementById("browseBtn");
const uploadList    = document.getElementById("uploadList");
const uploadItems   = document.getElementById("uploadItems");
const startUploadBtn = document.getElementById("startUploadBtn");
const uploadResults = document.getElementById("uploadResults");

let stagedFiles = [];

// ── Browse ──────────────────────────────────────────
browseBtn?.addEventListener("click", () => fileInput.click());
uploadZone?.addEventListener("click", e => {
  if (e.target === uploadZone || e.target.closest(".upload-zone-inner")) {
    fileInput.click();
  }
});

fileInput?.addEventListener("change", () => {
  addFiles([...fileInput.files]);
  fileInput.value = "";
});

// ── Drag & Drop ─────────────────────────────────────
uploadZone?.addEventListener("dragover", e => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone?.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone?.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  addFiles([...e.dataTransfer.files]);
});

// ── Stage files ──────────────────────────────────────
function addFiles(files) {
  const allowed = [".pdf", ".docx", ".md", ".markdown", ".txt"];
  files.forEach(file => {
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!allowed.includes(ext)) {
      window.showToast(`"${file.name}" is not a supported file type.`, "error");
      return;
    }
    if (!stagedFiles.find(f => f.name === file.name)) {
      stagedFiles.push(file);
    }
  });
  renderStagedFiles();
}

function renderStagedFiles() {
  if (stagedFiles.length === 0) {
    uploadList.style.display = "none";
    return;
  }
  uploadList.style.display = "block";
  uploadItems.innerHTML = "";

  stagedFiles.forEach((file, idx) => {
    const item = document.createElement("div");
    item.className = "upload-item";
    item.innerHTML = `
      <svg class="upload-item-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
      </svg>
      <span class="upload-item-name">${file.name}</span>
      <span class="upload-item-size">${formatSize(file.size)}</span>
      <button class="upload-item-remove" data-idx="${idx}" title="Remove">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    `;
    uploadItems.appendChild(item);
  });

  uploadItems.querySelectorAll(".upload-item-remove").forEach(btn => {
    btn.addEventListener("click", () => {
      stagedFiles.splice(parseInt(btn.dataset.idx), 1);
      renderStagedFiles();
    });
  });
}

// ── Upload ───────────────────────────────────────────
startUploadBtn?.addEventListener("click", async () => {
  if (stagedFiles.length === 0) return;
  startUploadBtn.disabled = true;
  startUploadBtn.textContent = "Uploading…";

  const formData = new FormData();
  stagedFiles.forEach(f => formData.append("files", f));

  try {
    const res = await fetch(`${window.API}/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    renderResults(data.results);
    stagedFiles = [];
    renderStagedFiles();
    window.showToast("Upload complete!", "success");
  } catch (err) {
    window.showToast("Upload failed: " + err.message, "error");
  } finally {
    startUploadBtn.disabled = false;
    startUploadBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>
        <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
      </svg>
      Upload & Index`;
  }
});

function renderResults(results) {
  uploadResults.style.display = "block";
  uploadResults.innerHTML = `<h3 class="upload-list-title" style="margin-bottom:12px">Results</h3>`;
  results.forEach(r => {
    const div = document.createElement("div");
    div.className = `upload-result-item ${r.status}`;
    let msg = r.filename;
    if (r.status === "success") msg += ` — ${r.chunks} chunks indexed (${r.pages} pages)`;
    else if (r.message) msg += ` — ${r.message}`;
    div.textContent = msg;
    uploadResults.appendChild(div);
  });
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}
