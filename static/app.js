// Runbook Copilot Web Client Controller
document.addEventListener("DOMContentLoaded", () => {
  // State
  const state = {
    mode: "agent", // "agent" | "rag"
    topK: 4,
    sourceFilter: "All Sources",
    hostname: "cluster-node-01",
    isLoading: false,
    historyCount: 0,
    history: [],
  };

  // DOM Elements
  const modeAgentBtn = document.getElementById("mode-agent-btn");
  const modeRagBtn = document.getElementById("mode-rag-btn");
  const activeModeLabel = document.getElementById("active-mode-label");
  const connectionStatus = document.getElementById("connection-status");
  const statusText = document.getElementById("status-text");

  const btnToggleSettings = document.getElementById("btn-toggle-settings");
  const btnCloseSettings = document.getElementById("btn-close-settings");
  const settingsPanel = document.getElementById("settings-panel");
  const settingSource = document.getElementById("setting-source");
  const settingTopK = document.getElementById("setting-topk");
  const topKVal = document.getElementById("topk-val");
  const settingHost = document.getElementById("setting-host");

  const statQdrant = document.getElementById("stat-qdrant");
  const statChunks = document.getElementById("stat-chunks");
  const statOllama = document.getElementById("stat-ollama");

  const chatViewport = document.getElementById("chat-viewport");
  const welcomeHero = document.getElementById("welcome-hero");
  const chatStream = document.getElementById("chat-stream");
  const queryForm = document.getElementById("query-form");
  const queryInput = document.getElementById("query-input");
  const btnSubmit = document.getElementById("btn-submit");
  const btnClearChat = document.getElementById("btn-clear-chat");
  const quickChips = document.querySelectorAll(".quick-chip");

  // Mode switching
  function setMode(mode) {
    state.mode = mode;
    if (mode === "agent") {
      modeAgentBtn.classList.add("active");
      modeRagBtn.classList.remove("active");
      activeModeLabel.textContent = "Agent Mode (Safety-Gated + Telemetry)";
    } else {
      modeRagBtn.classList.add("active");
      modeAgentBtn.classList.remove("active");
      activeModeLabel.textContent = "Direct RAG Mode (Semantic Search Only)";
    }
  }

  modeAgentBtn.addEventListener("click", () => setMode("agent"));
  modeRagBtn.addEventListener("click", () => setMode("rag"));

  // Settings
  btnToggleSettings.addEventListener("click", () => {
    settingsPanel.classList.toggle("hidden");
  });

  btnCloseSettings.addEventListener("click", () => {
    settingsPanel.classList.add("hidden");
  });

  settingTopK.addEventListener("input", (e) => {
    state.topK = parseInt(e.target.value, 10);
    topKVal.textContent = state.topK;
  });

  settingSource.addEventListener("change", (e) => {
    state.sourceFilter = e.target.value;
  });

  settingHost.addEventListener("input", (e) => {
    state.hostname = e.target.value.trim() || "cluster-node-01";
  });

  // Fetch backend status
  async function checkBackendStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) throw new Error("Status endpoint error");
      const data = await res.json();

      if (data.status === "ready") {
        connectionStatus.className = "status-pill status-connected";
        statusText.textContent = `Online • ${data.qdrant.indexed_chunks} Chunks`;
      } else {
        connectionStatus.className = "status-pill status-connecting";
        statusText.textContent = "Services Degraded";
      }

      if (data.qdrant) {
        statQdrant.textContent = `Connected (${data.qdrant.host})`;
        statChunks.textContent = data.qdrant.indexed_chunks || "0";
      }
      if (data.ollama) {
        statOllama.textContent = data.ollama.active_model;
      }
    } catch (err) {
      connectionStatus.className = "status-pill status-error";
      statusText.textContent = "Offline / Connection Error";
    }
  }

  checkBackendStatus();
  setInterval(checkBackendStatus, 15000);

  // Auto-resize textarea
  queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + "px";
  });

  // Enter to submit, Shift+Enter for new line
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!state.isLoading && queryInput.value.trim()) {
        queryForm.requestSubmit();
      }
    }
  });

  // Clear chat
  btnClearChat.addEventListener("click", () => {
    chatStream.innerHTML = "";
    welcomeHero.classList.remove("hidden");
    state.historyCount = 0;
    state.history = [];
  });

  // Quick Chips
  quickChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      if (query) {
        queryInput.value = query;
        queryInput.style.height = "auto";
        queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + "px";
        queryForm.requestSubmit();
      }
    });
  });

  // Scroll to bottom
  function scrollToBottom() {
    chatViewport.scrollTo({
      top: chatViewport.scrollHeight,
      behavior: "smooth",
    });
  }

  // Handle Form Submit
  queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query || state.isLoading) return;

    state.isLoading = true;
    welcomeHero.classList.add("hidden");

    // UI Loading state
    const btnText = btnSubmit.querySelector(".btn-text");
    const btnIcon = btnSubmit.querySelector(".btn-icon");
    const spinner = btnSubmit.querySelector(".loader-spinner");
    btnSubmit.disabled = true;
    btnText.textContent = "Analyzing...";
    btnIcon.classList.add("hidden");
    spinner.classList.remove("hidden");

    // Clear input
    queryInput.value = "";
    queryInput.style.height = "48px";

    // 1. Append user turn
    const turnId = `turn-${Date.now()}`;
    const userTurnHtml = `
      <div class="chat-turn" id="${turnId}">
        <div class="user-query-card">
          <div class="user-query-meta">
            <span>Query #${++state.historyCount}</span>
            <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
          </div>
          <div class="user-query-text">${escapeHtml(query)}</div>
        </div>

        <!-- Assistant Pending Placeholder -->
        <div class="assistant-response-card" id="pending-${turnId}">
          <div class="response-header">
            <div class="response-identity">
              <div class="agent-avatar">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
              </div>
              <span class="agent-title-text">${state.mode === "agent" ? "Runbook Copilot Agent" : "Direct RAG Pipeline"}</span>
            </div>
            <div class="response-meta-tags">
              <span class="meta-tag">Retrieving & Reasoning...</span>
            </div>
          </div>
          <div class="markdown-body">
            <p style="color: var(--text-muted); font-style: italic;">
              Evaluating incident against deterministic safety rules, retrieving Qdrant runbook chunks, and generating verified remediation...
            </p>
          </div>
        </div>
      </div>
    `;

    chatStream.insertAdjacentHTML("beforeend", userTurnHtml);
    scrollToBottom();

    // 2. Call API
    try {
      const payload = {
        query: query,
        mode: state.mode,
        top_k: state.topK,
        source_filter: state.sourceFilter,
        hostname: state.hostname,
        history: state.history,
      };

      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with ${res.status}`);
      }

      const result = await res.json();
      renderAssistantResponse(turnId, result);

      if (!result.is_escalated) {
        state.history.push({ role: "user", content: query });
        state.history.push({ role: "assistant", content: result.response });
        if (state.history.length > 8) {
          state.history = state.history.slice(-8);
        }
      }
    } catch (err) {
      renderErrorResponse(turnId, err.message);
    } finally {
      state.isLoading = false;
      btnSubmit.disabled = false;
      btnText.textContent = "Run Copilot";
      btnIcon.classList.remove("hidden");
      spinner.classList.add("hidden");
      scrollToBottom();
    }
  });

  // Render Assistant Result
  function renderAssistantResponse(turnId, data) {
    const turnEl = document.getElementById(turnId);
    if (!turnEl) return;

    const pendingEl = document.getElementById(`pending-${turnId}`);
    if (pendingEl) pendingEl.remove();

    const isEscalated = data.is_escalated;
    const latencySec = (data.latency_ms / 1000).toFixed(2);
    const modeBadge = data.mode === "agent" ? "LangGraph Agent" : "Direct RAG";

    let escalationHtml = "";
    if (isEscalated) {
      escalationHtml = `
        <div class="escalation-banner">
          <div class="escalation-header">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
            <span>INCIDENT ESCALATED TO HUMAN ON-CALL ENGINEER</span>
          </div>
          <div class="escalation-body">
            <strong>Escalation Trigger:</strong> 
            <span class="escalation-reason-highlight">${escapeHtml(data.escalation_reason || "Safety Policy / Confidence Threshold")}</span>
            <p style="margin-top: 6px;">Automated remediation has been safely aborted. Please follow team emergency protocol.</p>
          </div>
          <div class="escalation-footer">
            Audit Record: Logged to <code>eval/escalations.log</code>
          </div>
        </div>
      `;
    }

    let telemetryHtml = "";
    if (data.system_status) {
      const s = data.system_status;
      const statusColor = s.service_status === "running" ? "#10b981" : "#f59e0b";
      telemetryHtml = `
        <div class="telemetry-card">
          <div class="telemetry-header">
            <span class="telemetry-title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>
              Live System Telemetry (${escapeHtml(s.hostname || "cluster-node-01")})
            </span>
            <span class="font-mono" style="font-size: 0.72rem; color: ${statusColor}; font-weight: 600; text-transform: uppercase;">
              Service: ${escapeHtml(s.service_status || "unknown")}
            </span>
          </div>
          <div class="telemetry-grid">
            <div class="telemetry-stat-item">
              <span class="telemetry-stat-label">CPU Utilization</span>
              <span class="telemetry-stat-val">${s.cpu_percent}%</span>
            </div>
            <div class="telemetry-stat-item">
              <span class="telemetry-stat-label">Memory Used</span>
              <span class="telemetry-stat-val">${s.memory_used_percent}%</span>
            </div>
            <div class="telemetry-stat-item">
              <span class="telemetry-stat-label">Free Storage</span>
              <span class="telemetry-stat-val">${s.disk_free_gb} GB</span>
            </div>
            <div class="telemetry-stat-item">
              <span class="telemetry-stat-label">15m Load Avg</span>
              <span class="telemetry-stat-val">${s.load_average_15m}</span>
            </div>
          </div>
        </div>
      `;
    }

    // Markdown parse response
    const parsedMarkdown = marked.parse(data.response || "No response generated.");

    // Citations
    let citationsHtml = "";
    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
      const chunkItems = data.retrieved_chunks.map((chunk, idx) => {
        const sourceClass = getSourceBadgeClass(chunk.source);
        const scorePct = Math.round(chunk.score * 100);
        return `
          <div class="citation-card">
            <div class="citation-top">
              <div>
                <span class="chip-badge ${sourceClass}">${escapeHtml(chunk.source)}</span>
                <span class="citation-title-text" style="margin-left: 8px;">${escapeHtml(chunk.title)}</span>
              </div>
              <span class="citation-score-badge" title="Cosine Similarity">${chunk.score.toFixed(4)} (${scorePct}%)</span>
            </div>
            ${chunk.header_path ? `<div class="citation-header-path">${escapeHtml(chunk.header_path)}</div>` : ''}
            <div class="citation-snippet">${escapeHtml(chunk.content.substring(0, 320))}${chunk.content.length > 320 ? '...' : ''}</div>
          </div>
        `;
      }).join("");

      citationsHtml = `
        <div class="citations-section">
          <div class="citations-header" onclick="this.nextElementSibling.classList.toggle('hidden')">
            <span class="citations-title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
              Grounded Runbook Citations (${data.retrieved_chunks.length} chunks)
            </span>
            <span style="font-size: 0.72rem; color: var(--accent-cyan);">Toggle Sources ▾</span>
          </div>
          <div class="citation-list">
            ${chunkItems}
          </div>
        </div>
      `;
    }

    const cardHtml = `
      <div class="assistant-response-card">
        <div class="response-header">
          <div class="response-identity">
            <div class="agent-avatar" style="${isEscalated ? 'background: var(--accent-rose);' : ''}">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            </div>
            <span class="agent-title-text">${isEscalated ? "Escalation Safety Trigger" : "Remediation Plan"}</span>
          </div>
          <div class="response-meta-tags">
            <span class="meta-tag">${modeBadge}</span>
            <span class="meta-tag">${latencySec}s</span>
          </div>
        </div>

        ${escalationHtml}
        ${telemetryHtml}

        <div class="markdown-body">
          ${parsedMarkdown}
        </div>

        ${citationsHtml}
      </div>
    `;

    turnEl.insertAdjacentHTML("beforeend", cardHtml);
    addCodeCopyButtons(turnEl);
  }

  // Render Error
  function renderErrorResponse(turnId, message) {
    const turnEl = document.getElementById(turnId);
    if (!turnEl) return;

    const pendingEl = document.getElementById(`pending-${turnId}`);
    if (pendingEl) pendingEl.remove();

    const errHtml = `
      <div class="assistant-response-card" style="border-color: rgba(239, 68, 68, 0.4);">
        <div class="response-header">
          <div class="response-identity">
            <div class="agent-avatar" style="background: var(--accent-rose);">!</div>
            <span class="agent-title-text" style="color: #fca5a5;">Pipeline Execution Error</span>
          </div>
        </div>
        <div class="markdown-body" style="color: #fca5a5;">
          <p>Failed to execute copilot query: ${escapeHtml(message)}</p>
          <p style="font-size: 0.8rem; color: var(--text-muted);">Ensure Ollama (llama3.2) and Qdrant are running on their configured ports.</p>
        </div>
      </div>
    `;
    turnEl.insertAdjacentHTML("beforeend", errHtml);
  }

  // Helper for source badge CSS
  function getSourceBadgeClass(source) {
    const s = (source || "").toLowerCase();
    if (s.includes("proxmox")) return "proxmox";
    if (s.includes("nakivo")) return "nakivo";
    if (s.includes("microsoft") || s.includes("learn")) return "mslearn";
    if (s.includes("serverfault")) return "serverfault";
    return "proxmox";
  }

  // Helper: Escape HTML
  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Helper: Add copy buttons to code blocks
  function addCodeCopyButtons(container) {
    const pres = container.querySelectorAll("pre");
    pres.forEach((pre) => {
      if (pre.querySelector(".code-copy-btn")) return;
      const copyBtn = document.createElement("button");
      copyBtn.className = "code-copy-btn";
      copyBtn.textContent = "Copy";
      copyBtn.addEventListener("click", () => {
        const code = pre.querySelector("code");
        const text = code ? code.innerText : pre.innerText;
        navigator.clipboard.writeText(text).then(() => {
          copyBtn.textContent = "Copied!";
          setTimeout(() => (copyBtn.textContent = "Copy"), 2000);
        });
      });
      pre.appendChild(copyBtn);
    });
  }
});
