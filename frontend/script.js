/**
 * QueryMate AI - Frontend Logic
 * This file manages the user interface, real-time AI streaming, document management,
 * and collaborative features like sharing and exporting.
 */

// --- GLOBAL CONFIGURATION ---
const API_BASE = "http://127.0.0.1:8000";

// --- APPLICATION STATE ---
let currentConversationId = null;
let currentVectorStoreId = null;
let activeFilterTag = null; // Track if we are filtering the sidebar by a specific tag
let currentSummaryData = null;
let currentConversationData = null;
let currentSearchQuery = "";
let searchMode = "titles"; // 'titles' or 'content'
let selectedConversations = new Set(); // For batch operations
let hasGlobalDocs = false; // Tracks if user has previous document history

// Persistence for User Preferences
let darkMode = localStorage.getItem("darkMode") === "true";
const token = localStorage.getItem("accessToken");

// --- CORE DOM ELEMENTS ---
const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const popover = document.getElementById("citationPopover");
const popoverContent = document.getElementById("popoverContent");
const popoverTitle = document.getElementById("popoverTitle");

// --- UI MODALS & BUTTONS ---
const summaryBtn = document.getElementById("summaryBtn");
const summaryModal = document.getElementById("summaryModal");
const exportBtn = document.getElementById("exportBtn");
const exportModal = document.getElementById("exportModal");
const summaryContent = document.getElementById("summaryContent");

/**
 * === UTILITIES ===
 */

// Displays brief floating notifications
function showToast(message, duration = 3000) {
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.classList.add("show"), 10);
    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Returns a glowing loader component
function showLoading(text = "Processing...") {
    const loader = document.createElement("div");
    loader.className = "loading-inline";
    loader.innerHTML = `
        <span class="spinner-mini"></span>
        <span class="loading-text">${text}</span>
    `;
    return loader;
}

/**
 * === AESTHETICS (Dark Mode) ===
 */
function toggleDarkMode() {
    darkMode = !darkMode;
    localStorage.setItem("darkMode", darkMode);
    applyDarkMode();
    showToast(darkMode ? "🌙 Night mode active" : "☀️ Day mode active");
}

function applyDarkMode() {
    if (darkMode) {
        document.documentElement.classList.add("dark-mode");
    } else {
        document.documentElement.classList.remove("dark-mode");
    }
}

/**
 * === KEYBOARD SHORTCUTS ===
 */
document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter" && !userInput.disabled) {
        e.preventDefault();
        handleSend();
    }
    if (e.ctrlKey && e.key === "k") {
        e.preventDefault();
        document.getElementById("searchInput")?.focus();
    }
    if (e.ctrlKey && e.key === "n") {
        e.preventDefault();
        handleNewChat();
    }
    if (e.key === "Escape") {
        summaryModal.style.display = "none";
        exportModal.style.display = "none";
        popover.style.display = "none";
    }
});

/**
 * === INITIALIZATION ===
 */
document.addEventListener("DOMContentLoaded", () => {
    // 1. Session & Theme Guard
    if (!token && !window.location.pathname.includes("login.html")) {
        window.location.href = "login.html";
        return;
    }
    applyDarkMode();

    // 2. Data Hydration
    loadUserInfo();
    loadConversations();

    // 3. Global click delegates
    document.addEventListener("click", (e) => {
        if (e.target === summaryModal) summaryModal.style.display = "none";
        if (e.target === exportModal) exportModal.style.display = "none";
        const statsModal = document.getElementById("statsModal");
        if (e.target === statsModal) statsModal.style.display = "none";
        if (popover && !popover.contains(e.target) && !e.target.classList.contains("citation-link")) {
            popover.style.display = "none";
        }
    });

    // 4. Link UI events
    const actions = {
        "logoutBtn": () => { localStorage.removeItem("accessToken"); window.location.href = "login.html"; },
        "newChatBtn": handleNewChat,
        "sendBtn": handleSend,
        "summaryBtn": handleSummaryClick,
        "downloadSummaryBtn": handleDownloadSummary,
        "exportPdfBtn": () => exportConversation("pdf"),
        "exportHtmlBtn": () => exportConversation("html"),
        "exportJsonBtn": () => exportConversation("json"),
        "exportBtn": () => { document.getElementById("exportModal").style.display = "flex"; },
        "closePopover": () => popover.style.display = "none",
        "closeModalBtn": () => summaryModal.style.display = "none",
        "closeExportModalBtn": () => exportModal.style.display = "none",
        "sidebarStatsBtn": () => loadUserStats(true),
        "sessionStatsBtn": () => showConversationStats(currentConversationId),
        "knowledgeMapBtn": () => loadKnowledgeMap(),
        "themeToggle": () => toggleDarkMode(),
        "modeTitles": () => setSearchMode('titles'),
        "modeContent": () => setSearchMode('content')
    };

    const searchInp = document.getElementById("searchInput");
    if (searchInp) {
        searchInp.onkeyup = (e) => {
            currentSearchQuery = e.target.value;
            if (searchMode === 'titles') searchConversations(currentSearchQuery);
            else handleDeepSearch(currentSearchQuery);
        };
    }

    Object.entries(actions).forEach(([id, fn]) => {
        const el = document.getElementById(id);
        if (el) el.onclick = fn;
    });
});

async function loadUserInfo() {
    try {
        const res = await fetch(`${API_BASE}/users/me`, { headers: { "Authorization": `Bearer ${token}` } });
        if (res.ok) {
            const user = await res.json();
            document.getElementById("usernameDisplay").textContent = user.username;
        }
    } catch (e) { console.error("User info error:", e); }
    loadUserStats(); // Trigger global state check
}

async function loadUserStats(showModal = false) {
    try {
        const res = await fetch(`${API_BASE}/users/stats/dashboard`, { headers: { "Authorization": `Bearer ${token}` } });
        if (res.ok) {
            const stats = await res.json();
            hasGlobalDocs = stats.total_pdfs > 0;
            if (showModal) showStatsModal(stats);
            // Handle logical workspace setup based on history
            if (!currentConversationId) handleNewChat();
            return stats;
        }
    } catch (e) { console.error("Dashboard stats error:", e); }
}

function toggleDarkMode() {
    document.body.classList.toggle("dark-mode");
    const isDark = document.body.classList.contains("dark-mode");
    localStorage.setItem("theme", isDark ? "dark" : "light");
    showToast(isDark ? "Dark Mode Active 🌙" : "Light Mode Active ☀️");
}

function showStatsModal(stats) {
    document.getElementById("stat-convs").textContent = stats.total_conversations;
    document.getElementById("stat-questions").textContent = stats.total_questions;
    document.getElementById("stat-pdfs").textContent = stats.total_pdfs;
    document.getElementById("stat-msgs").textContent = stats.total_messages;
    document.getElementById("statsModal").style.display = "flex";
}

async function loadKnowledgeMap() {
    document.getElementById("knowledgeMapModal").style.display = "flex";
    const content = document.getElementById("knowledgeMapContent");
    const linksDiv = document.getElementById("knowledgeLinks");
    content.innerHTML = `<div class="spinner-mini"></div>`;
    linksDiv.innerHTML = "";

    try {
        const res = await fetch(`${API_BASE}/users/library/concepts`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        const data = await res.json();
        content.innerHTML = "";

        if (!data.themes || !data.themes.length) {
            content.innerHTML = "Upload more documents to generate a knowledge map!";
            return;
        }

        data.themes.forEach(t => {
            const node = document.createElement("div");
            node.className = "theme-node";
            node.style.cursor = "pointer";
            node.textContent = t;
            node.onclick = () => {
                document.getElementById("knowledgeMapModal").style.display = "none";
                quickAddTag(t); // Add as tag to current chat
                setSearchMode('content');
                document.getElementById("searchInput").value = t;
                handleDeepSearch(t);
            };
            content.appendChild(node);
        });

        if (data.links && data.links.length) {
            linksDiv.innerHTML = "<strong>Key Connections Identified:</strong><br>";
            data.links.forEach(l => {
                linksDiv.innerHTML += `• ${l.source} ↔ ${l.target}<br>`;
            });
        }
    } catch (e) {
        content.innerHTML = "Error generating Map.";
    }
}

function setSearchMode(mode) {
    searchMode = mode;
    document.getElementById("modeTitles").classList.toggle("active", mode === "titles");
    document.getElementById("modeContent").classList.toggle("active", mode === "content");
    document.getElementById("searchInput").placeholder = mode === "titles" ? "Search Titles..." : "Deep Content Search...";
    if (currentSearchQuery) {
        if (mode === 'titles') searchConversations(currentSearchQuery);
        else handleDeepSearch(currentSearchQuery);
    }
}

/**
 * === SEARCH ENGINE ===
 */
async function handleDeepSearch(query) {
    if (!query.trim() || query.length < 3) { if (!query.trim()) loadConversations(); return; }
    try {
        const res = await fetch(`${API_BASE}/conversations/search/deep?q=${encodeURIComponent(query)}`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        if (!res.ok) return;
        const data = await res.json();
        const list = document.getElementById("conversationList");
        list.innerHTML = data.results.length ? "" : `<div class="empty-state">🔍 No deep matches</div>`;

        data.results.forEach(r => {
            const div = document.createElement("div");
            div.className = "chat-item result-card";
            div.style.padding = "14px";
            div.style.marginBottom = "10px";
            div.style.borderLeft = "4px solid var(--primary)";
            div.style.cursor = "pointer";

            div.innerHTML = `
                <div class="chat-item-title" style="color: var(--primary); font-family: 'Outfit'; font-size: 0.9rem;">📄 ${r.source}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 6px;">Found on Page ${r.page}</div>
                <div class="chat-item-preview" style="font-style: italic; font-size: 0.8rem; line-height: 1.4; opacity: 0.9;">
                    "...${r.content.substring(0, 100)}..."
                </div>
            `;

            // On click, open the citation preview for that specific page
            div.onclick = () => showCitationPreview(r.raw_source || r.source, r.page);
            list.appendChild(div);
        });
    } catch (e) { console.error("Deep search error:", e); }
}
async function searchConversations(query) {
    if (!query.trim()) { loadConversations(); return; }
    try {
        const res = await fetch(`${API_BASE}/conversations/search/${encodeURIComponent(query)}`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        if (!res.ok) throw new Error("Search service unavailable");
        const data = await res.json();
        const list = document.getElementById("conversationList");
        list.innerHTML = data.results.length ? "" : `<div class="empty-state">🔍 No matches found</div>`;
        data.results.forEach(c => {
            const div = document.createElement("div");
            div.className = `chat-item ${currentConversationId === c.conversation_id ? 'active' : ''}`;
            div.innerHTML = `
                <div class="chat-item-title">${c.title}</div>
                <div class="chat-item-preview">${c.preview?.substring(0, 50) + '...' || ''}</div>
            `;
            div.onclick = () => loadChat(c.conversation_id);
            list.appendChild(div);
        });
        showToast(`Found ${data.results.length} matches`);
    } catch (e) { showToast("Search error: " + e.message); }
}

/**
 * === TAGGING SYSTEM ===
 */
async function addTag(conversationId, tagName) {
    try {
        const res = await fetch(`${API_BASE}/conversations/${conversationId}/tags`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
            body: JSON.stringify({ tag: tagName })
        });
        if (res.ok) { showToast(`✅ Tag applied`); loadConversationTags(conversationId); }
    } catch (e) { showToast("Tag error: " + e.message); }
}

async function removeTag(conversationId, tagName) {
    try {
        const res = await fetch(`${API_BASE}/conversations/${conversationId}/tags/${tagName}`, {
            method: "DELETE",
            headers: { "Authorization": `Bearer ${token}` }
        });
        if (res.ok) { showToast(`Tag removed`); loadConversationTags(conversationId); }
    } catch (e) { showToast("Tag error: " + e.message); }
}

async function loadConversationTags(conversationId) {
    try {
        const res = await fetch(`${API_BASE}/conversations/${conversationId}/tags`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        if (res.ok) {
            const data = await res.json();
            const container = document.getElementById("conversationTags");
            if (container) {
                let tagsHtml = data.tags.map(t => `
                    <span class="tag">${t}<button class="tag-remove" onclick="removeTag(${conversationId}, '${t}')">×</button></span>
                `).join("");

                // Add a '+' button for manual tagging
                tagsHtml += `<button class="tag-add-btn" onclick="addTagPrompt(${conversationId})" title="Add Tag">+</button>`;

                container.innerHTML = tagsHtml;
            }
        }
    } catch (e) { console.error("Tags load error:", e); }
}

/**
 * Helper to quickly apply a tag to the current session
 */
async function quickAddTag(tagName) {
    if (!currentConversationId) {
        showToast("Open a chat first to apply tags");
        return;
    }
    await addTag(currentConversationId, tagName);
}

/**
 * === FEEDBACK & REACTIONS ===
 */
async function addMessageFeedback(messageId, rating) {
    try {
        const res = await fetch(`${API_BASE}/messages/${messageId}/feedback`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
            body: JSON.stringify({ rating: rating })
        });
        if (res.ok) showToast("✅ Feedback saved");
    } catch (e) { showToast("Feedback error"); }
}

/**
 * === SHARING UTILITIES ===
 */
async function createShareLink(id = null) {
    const targetId = id || currentConversationId;
    if (!targetId) return;
    const days = prompt("How many days should this link last? (0=never)", "7");
    if (days === null) return;
    try {
        const res = await fetch(`${API_BASE}/conversations/${targetId}/share`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
            body: JSON.stringify({ expires_in_days: days ? parseInt(days) : null })
        });
        if (res.ok) {
            const data = await res.json();
            navigator.clipboard.writeText(data.share_url);
            showToast("✅ Link copied!");
            alert(`Link generated: ${data.share_url}`);
        }
    } catch (e) { showToast("Share error"); }
}

/**
 * === BATCH ACTIONS ===
 */
function toggleConversationSelection(id) {
    if (selectedConversations.has(id)) selectedConversations.delete(id); else selectedConversations.add(id);
    if (selectedConversations.size) showToast(`${selectedConversations.size} selected`);
}

async function batchDeleteSelected() {
    if (!selectedConversations.size) { showToast("Select items first"); return; }
    if (!confirm(`Relly purge ${selectedConversations.size} chats?`)) return;
    try {
        const res = await fetch(`${API_BASE}/conversations/batch/delete`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
            body: JSON.stringify({ conversation_ids: Array.from(selectedConversations) })
        });
        if (res.ok) { showToast("Purged successfully"); selectedConversations.clear(); loadConversations(); }
    } catch (e) { showToast("Purge failed"); }
}

async function batchAddTags() {
    if (!selectedConversations.size) return;
    const tag = prompt("Tag to apply to selection:");
    if (!tag) return;
    try {
        const res = await fetch(`${API_BASE}/conversations/batch/tags`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
            body: JSON.stringify({ conversation_ids: Array.from(selectedConversations), tag: tag })
        });
        if (res.ok) { showToast(`Tagged ${selectedConversations.size} items`); selectedConversations.clear(); loadConversations(); }
    } catch (e) { showToast("Batch tag error"); }
}

/**
 * === EXPORT ENGINE ===
 */
async function exportConversation(format) {
    if (!currentConversationId) return;
    try {
        const res = await fetch(`${API_BASE}/conversations/${currentConversationId}/export`, { headers: { "Authorization": `Bearer ${token}` } });
        if (!res.ok) throw new Error("Data retrieval failed");
        const data = await res.json();
        if (format === "pdf") exportToPdf(data);
        else if (format === "html") exportToHtml(data);
        else if (format === "json") exportToJson(data);
        exportModal.style.display = "none";
        showToast("✅ Export successful");
    } catch (e) { showToast("Export error"); }
}

function exportToPdf(data) {
    const html = generateConversationHtml(data);
    const element = document.createElement("div");
    element.innerHTML = html;
    const opt = { margin: 10, filename: `QueryMate_Export_${data.id}.pdf`, jsPDF: { unit: 'mm', format: 'a4' } };
    html2pdf().set(opt).from(element).save();
}

function exportToHtml(data) {
    const html = generateConversationHtml(data);
    const blob = new Blob([html], { type: "text/html" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `QueryMate_Export_${data.id}.html`;
    link.click();
}

function exportToJson(data) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `QueryMate_Data_${data.id}.json`;
    link.click();
}

function generateConversationHtml(data) {
    return `<h1>${data.title}</h1><p>Owner: ${data.user}</p><hr/>` +
        data.messages.map(m => `<div><p><b>${m.role.toUpperCase()}:</b></p><div>${m.content}</div></div>`).join("");
}

/**
 * === CHAT LOGIC (Real-time Streaming) ===
 */
async function handleSend() {
    const text = userInput.value.trim();
    if (!text) return;

    // 1. If we are in a fresh session (no convo ID yet), create one first
    if (!currentConversationId) {
        try {
            const createRes = await fetch(`${API_BASE}/conversations`, {
                method: "POST",
                headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
                body: JSON.stringify({ title: "Global Research Chat" })
            });
            if (!createRes.ok) throw new Error("Failed to initialize session");
            const newConvo = await createRes.json();
            currentConversationId = newConvo.id;
            loadConversations(); // Update list in sidebar

            // Clean welcome message
            chatBox.innerHTML = "";
            document.getElementById("chatTitle").textContent = "Global Research Chat";
            ["summaryBtn", "exportBtn", "shareBtn", "chatStatsBtn", "deleteBtn"].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.style.display = "block";
            });
        } catch (e) {
            showToast("Initialization error: " + e.message);
            return;
        }
    }

    appendMessage('user', text);
    userInput.value = "";
    const botContentDiv = appendMessage('bot', "");
    botContentDiv.innerHTML = showLoading("AI is thinking...").innerHTML;

    try {
        const response = await fetch(`${API_BASE}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
            body: JSON.stringify({ question: text, conversation_id: currentConversationId })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = "";
        let isFirst = true;
        let fuRendered = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            if (isFirst) { botContentDiv.innerHTML = ""; isFirst = false; }
            const chunk = decoder.decode(value);
            fullText += chunk;

            // Display logic: show text but hide metadata tags
            let display = fullText;

            // Handle Follow-ups
            const fuMatch = fullText.match(/\[FOLLOW_UPS\]([\s\S]*?)(\[MSID:\d+\]|$)/);
            if (fuMatch && !fuRendered) {
                const jsonStr = fuMatch[1].trim();
                // Check if JSON looks complete (ends with ])
                if (jsonStr.endsWith("]")) {
                    renderFollowUps(jsonStr);
                    fuRendered = true;
                }
            }

            // Clean display text (strip tags)
            display = display.replace(/\[FOLLOW_UPS\][\s\S]*?(\[MSID:\d+\]|$)/, "$1");

            // Handle MSID (for feedback buttons)
            const idMatch = display.match(/\[MSID:(\d+)\]/);
            if (idMatch) {
                const msgDiv = botContentDiv.closest(".message");
                if (msgDiv) {
                    msgDiv.dataset.messageId = idMatch[1];
                    msgDiv.querySelectorAll('.msg-rating-btn').forEach(b => b.disabled = false);
                }
                display = display.replace(/\[MSID:\d+\]/, "");
            }

            renderTextWithCitations(botContentDiv, display.trim());
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    } catch (e) { botContentDiv.textContent = "AI error: " + e.message; }
}

/**
 * === RESEARCH ANALYSIS & SUMMARIES ===
 */
async function handleSummaryClick() {
    summaryModal.style.display = "flex";
    summaryContent.innerHTML = showLoading("Drafting executive summary...").innerHTML;
    try {
        const response = await fetch(`${API_BASE}/conversations/${currentConversationId}/summarize-stream`, {
            method: "POST", headers: { "Authorization": `Bearer ${token}` }
        });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        currentSummaryData = "";
        let isFirst = true;
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            if (isFirst) { summaryContent.innerHTML = ""; isFirst = false; }
            currentSummaryData += decoder.decode(value);
            renderTextWithCitations(summaryContent, currentSummaryData);
        }
    } catch (e) { summaryContent.textContent = "Summary failed"; }
}

function handleDownloadSummary() {
    if (!currentSummaryData) return;
    const element = document.createElement("div");
    element.innerHTML = `<h1>Research Analysis</h1><hr/>` + marked.parse(currentSummaryData);
    html2pdf().from(element).save("QueryMate_Summary.pdf");
}

/**
 * === CITATION ENGINE ===
 */
function renderTextWithCitations(element, text) {
    if (!text) return;
    let html = marked.parse(text);
    html = html.replace(/\[Source: (.*?) - Page (\d+)\]/g, (m, f, p) =>
        `<span class="citation-link" onclick="showCitationPreview(event, ${p}, '${f.replace(/'/g, "\\'")}')">📄 ${f} (Pg ${p})</span>`);
    html = html.replace(/\[Page (\d+)\]/g, (m, p) =>
        `<span class="citation-link" onclick="showCitationPreview(event, ${p})">📄 Page ${p}</span>`);
    html = html.replace(/\[(\d+)\]/g, (m, p) =>
        `<span class="citation-link" onclick="showCitationPreview(event, ${p})">📄 Page ${p}</span>`);
    element.innerHTML = html;
}

async function showCitationPreview(event, page, filename = null) {
    event.stopPropagation();
    if (!currentVectorStoreId) return;
    const rect = event.target.getBoundingClientRect();
    popover.style.display = "block";
    popover.style.left = `${Math.min(rect.left, window.innerWidth - 370)}px`;
    popover.style.top = `${rect.bottom + 10}px`;
    popoverTitle.textContent = filename ? `${filename} (Pg ${page})` : `Page ${page}`;
    popoverContent.innerHTML = showLoading().innerHTML;
    try {
        const res = await fetch(`${API_BASE}/preview/${currentVectorStoreId}/${encodeURIComponent(filename || "document")}/${page}`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        const data = await res.json();
        popoverContent.innerHTML = `<div class="preview-text">${data.text}</div>`;
    } catch (e) { popoverContent.innerHTML = "Preview unavailable"; }
}

/**
 * === SIDEBAR MANAGEMENT ===
 */
async function loadChat(id) {
    currentConversationId = id;
    try {
        const res = await fetch(`${API_BASE}/conversations/${id}`, { headers: { "Authorization": `Bearer ${token}` } });
        const data = await res.json();
        currentVectorStoreId = data.vector_store_id;
        document.getElementById("chatTitle").textContent = data.title;
        chatBox.innerHTML = "";
        data.messages.forEach((m, index) => {
            const isLast = index === data.messages.length - 1;
            let contentText = m.content;

            // Check for Follow-ups in the last message
            if (isLast && m.role === 'assistant') {
                const fuMatch = contentText.match(/\[FOLLOW_UPS\]([\s\S]*?)(\[MSID:\d+\]|$)/);
                if (fuMatch) {
                    const jsonStr = fuMatch[1].trim();
                    if (jsonStr.endsWith("]")) {
                        // Use a timeout to render at the bottom after DOM update
                        setTimeout(() => renderFollowUps(jsonStr), 50);
                    }
                }
            }

            // Always strip the tag for display
            contentText = contentText.replace(/\[FOLLOW_UPS\][\s\S]*?(\[MSID:\d+\]|$)/, "$1");
            // IDs are not needed in display text (handled via dataset)
            contentText = contentText.replace(/\[MSID:\d+\]/, "");

            const contentEl = appendMessage(m.role, "");
            const div = contentEl.closest(".message");
            if (div) {
                div.dataset.messageId = m.id;
                div.querySelectorAll('.msg-rating-btn').forEach(b => b.disabled = false);
            }
            renderTextWithCitations(contentEl, contentText.trim());
        });
        userInput.disabled = false;
        document.getElementById("sendBtn").disabled = false;
        userInput.placeholder = "Ask about these documents...";
        ["summaryBtn", "exportBtn", "shareBtn", "sessionStatsBtn", "deleteBtn"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = "block";
        });
        loadConversationTags(id);
    } catch (e) { showToast("Load error"); }
}

async function loadConversations() {
    try {
        const res = await fetch(`${API_BASE}/conversations`, { headers: { "Authorization": `Bearer ${token}` } });
        let chats = await res.json();
        const list = document.getElementById("conversationList");

        list.innerHTML = chats.length ? "" : `<div class="empty-state">No chats yet</div>`;

        chats.forEach(c => {
            const div = document.createElement("div");
            div.className = `chat-item ${currentConversationId === c.id ? 'active' : ''}`;

            const tagHtml = (c.tags || []).map(t => `
                <span class="sidebar-tag ${t === activeFilterTag ? 'active' : ''}" 
                      onclick="event.stopPropagation(); handleSidebarTagClick('${t}')">${t}</span>
            `).join("");

            div.innerHTML = `
                <div style="flex: 1;">
                    <div class="chat-item-title">${c.title}</div>
                    <div class="sidebar-tags-container">${tagHtml}</div>
                    <div class="chat-item-date">${new Date(c.created_at).toLocaleDateString()}</div>
                </div>
            `;
            div.addEventListener("contextmenu", (e) => { e.preventDefault(); showContextMenu(e, c.id); });
            div.onclick = () => loadChat(c.id);
            list.appendChild(div);
        });
    } catch (e) { console.error("History load error"); }
}

/**
 * Handles sidebar tag interaction:
 * 1. Highlights the tag for quick visual cross-referencing.
 * 2. Does NOT modify chat metadata (fixes propagation bug).
 * 3. Keeps your full conversation list visible.
 */
async function handleSidebarTagClick(tagName) {
    if (activeFilterTag === tagName) {
        activeFilterTag = null;
    } else {
        activeFilterTag = tagName;
    }
    loadConversations();
}

function showContextMenu(e, id) {
    const menu = document.createElement("div");
    menu.className = "context-menu";
    menu.style.left = `${e.clientX}px`;
    menu.style.top = `${e.clientY}px`;
    menu.innerHTML = `<button onclick="showConversationStats(${id})">📊 Stats</button>
                      <button onclick="addTagPrompt(${id})">🏷️ Tag</button>
                      <button onclick="createShareLink(${id})">🔗 Share</button>
                      <button onclick="deleteConversation(${id})">🗑️ Delete</button>`;
    document.body.appendChild(menu);
    const close = () => { menu.remove(); document.removeEventListener('click', close); };
    setTimeout(() => document.addEventListener('click', close), 10);
}

/**
 * === DELETION & UPLOAD ===
 */
async function deleteConversation(id) {
    if (!confirm("Delete permanently?")) return;
    try {
        const res = await fetch(`${API_BASE}/conversations/${id}`, { method: "DELETE", headers: { "Authorization": `Bearer ${token}` } });
        if (res.ok) { showToast("Purged"); if (currentConversationId === id) handleNewChat(); loadConversations(); }
    } catch (e) { showToast("Delete failed"); }
}

function handleNewChat() {
    currentConversationId = null;
    currentVectorStoreId = null;

    // UI Setup based on user's global knowledge state
    if (hasGlobalDocs) {
        chatBox.innerHTML = `
            <div class="message bot">
                <div class="msg-content">
                    👋 **Global Research Mode Active**<br><br>
                    You have previously uploaded documents. You can start typing below to query your **entire knowledge base** across all past research!<br><br>
                    QueryMate will automatically cross-reference all your PDF and image history.
                </div>
            </div>`;
        document.getElementById("chatTitle").textContent = "Global Research";
        userInput.disabled = false;
        userInput.placeholder = "Search across all your documents...";
        document.getElementById("sendBtn").disabled = false;
    } else {
        chatBox.innerHTML = `
            <div class="message bot">
                <div class="msg-content">
                    <b>Welcome to QueryMate!</b><br><br>
                    To begin, please upload a PDF or image. Once you've added your first document, my reasoning engine will be unlocked for this and all future chats!
                </div>
            </div>`;
        document.getElementById("chatTitle").textContent = "Fresh Session";
        userInput.disabled = true;
        userInput.placeholder = "Upload document to unlock chat...";
        document.getElementById("sendBtn").disabled = true;
    }

    ["summaryBtn", "exportBtn", "shareBtn", "sessionStatsBtn", "deleteBtn"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = "none";
    });
}

function addTagPrompt(id) {
    const t = prompt("Tag name:");
    if (t) addTag(id, t);
}

// Global Help / Core rendering
function appendMessage(role, text) {
    const displayRole = (role === "assistant" || role === "bot") ? "bot" : "user";
    const div = document.createElement("div");
    div.className = `message ${displayRole}`;
    const content = document.createElement("div");
    content.className = "msg-content";
    content.textContent = text;

    if (displayRole === "bot") {
        const wrapper = document.createElement("div");
        wrapper.className = "msg-wrapper";
        const btns = document.createElement("div");
        btns.className = "msg-buttons-container";
        const copy = document.createElement("button");
        copy.className = "msg-copy-btn";
        copy.innerHTML = "📋 Copy";
        copy.onclick = (e) => {
            e.stopPropagation();
            navigator.clipboard.writeText(content.innerText);
            showToast("Copied");

            // Button Feedback Animation
            const originalText = copy.innerHTML;
            copy.innerHTML = "✅ Copied!";
            copy.classList.add("copied-active");
            setTimeout(() => {
                copy.innerHTML = originalText;
                copy.classList.remove("copied-active");
            }, 2000);
        };
        const rates = document.createElement("div");
        rates.className = "msg-ratings";
        const up = document.createElement("button");
        up.className = "msg-rating-btn thumbs-up";
        up.innerHTML = "👍";
        up.disabled = true;
        up.onclick = (e) => { e.stopPropagation(); addMessageFeedback(div.dataset.messageId, 1); up.classList.add("active"); };
        const down = document.createElement("button");
        down.className = "msg-rating-btn thumbs-down";
        down.innerHTML = "👎";
        down.disabled = true;
        down.onclick = (e) => { e.stopPropagation(); addMessageFeedback(div.dataset.messageId, -1); down.classList.add("active"); };
        rates.append(up, down);
        btns.append(copy, rates);
        wrapper.append(content, btns);
        div.appendChild(wrapper);
    } else {
        div.appendChild(content);
    }
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return content;
}

function renderFollowUps(jsonStr) {
    try {
        const questions = JSON.parse(jsonStr);
        const container = document.createElement("div");
        container.className = "follow-ups-container";
        container.innerHTML = `<div class="fu-label">💡 Suggested Follow-ups:</div>`;
        questions.forEach(q => {
            const chip = document.createElement("button");
            chip.className = "fu-chip";
            chip.textContent = q;
            chip.onclick = () => {
                userInput.value = q;
                handleSend();
            };
            container.appendChild(chip);
        });
        chatBox.appendChild(container);
        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (e) { console.error("Follow-up error:", e); }
}

// File Upload Listener
document.getElementById("pdfInput")?.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // 1. Interactive UI: Show QueryMate is working inside the chat
    const statusMsg = appendMessage('bot', `🔍 **Analysis in progress...** I'm reading **${file.name}** and indexing its content for our research.`);
    const spinner = showLoading("Ingesting pages...");
    statusMsg.appendChild(spinner);

    // Also show a subtle overlay in case they are looking elsewhere
    const overlay = document.getElementById("uploadOverlay");
    if (overlay) overlay.style.display = "flex";

    const form = new FormData();
    form.append("file", file);
    if (currentConversationId) form.append("conversation_id", currentConversationId);

    try {
        const res = await fetch(`${API_BASE}/upload_pdf`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${token}` },
            body: form
        });

        if (overlay) overlay.style.display = "none";

        if (res.ok) {
            const data = await res.json();
            statusMsg.innerHTML = `✅ **Analysis Complete!** I have indexed **${file.name}**. What would you like to know about it?`;
            showToast("Document Ready");
            loadConversations();
            loadChat(data.conversation_id);
        } else {
            statusMsg.innerHTML = `❌ **Upload Failed.** I couldn't process the file. Please ensure it's a valid PDF or Image.`;
            showToast("Upload Error");
        }
    } catch (e) {
        if (overlay) overlay.style.display = "none";
        statusMsg.textContent = "Error: Service unreachable.";
        showToast("Service Error");
    }
    e.target.value = "";
});

// Final Keyboard Helper
userInput?.addEventListener("keypress", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } });

async function showConversationStats(id) {
    try {
        const res = await fetch(`${API_BASE}/conversations/${id}/stats`, { headers: { "Authorization": `Bearer ${token}` } });
        const s = await res.json();
        alert(`📊 Stats\nMessages: ${s.total_messages}\nQuestions: ${s.total_questions}\nResponses: ${s.total_responses}`);
    } catch (e) { showToast("Stats error"); }
}
