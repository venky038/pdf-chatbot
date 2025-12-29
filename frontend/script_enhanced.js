const API_BASE = "http://127.0.0.1:8000";
let currentConversationId = null;
let currentVectorStoreId = null; 
let currentSummaryData = null;
let currentConversationData = null;
let currentSearchQuery = "";
let selectedConversations = new Set();
let darkMode = localStorage.getItem("darkMode") === "true";

const token = localStorage.getItem("accessToken");
const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const popover = document.getElementById("citationPopover");
const popoverContent = document.getElementById("popoverContent");
const popoverTitle = document.getElementById("popoverTitle");

// UI Elements
const summaryBtn = document.getElementById("summaryBtn");
const summaryModal = document.getElementById("summaryModal");
const exportBtn = document.getElementById("exportBtn");
const exportModal = document.getElementById("exportModal");
const summaryContent = document.getElementById("summaryContent");
const closeModalBtn = document.getElementById("closeModalBtn");
const downloadSummaryBtn = document.getElementById("downloadSummaryBtn");

// === UTILITIES ===
function showToast(message, duration = 3000) {
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = message;
    document.body.appendChild(toast);
    
    // Trigger animation
    setTimeout(() => toast.classList.add("show"), 10);
    
    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

function showLoading(text = "Loading...") {
    const loader = document.createElement("div");
    loader.className = "loading-inline";
    loader.innerHTML = `
        <span class="spinner-mini"></span>
        <span class="loading-text">${text}</span>
    `;
    return loader;
}

// === DARK MODE ===
function toggleDarkMode() {
    darkMode = !darkMode;
    localStorage.setItem("darkMode", darkMode);
    applyDarkMode();
    showToast(darkMode ? "🌙 Dark mode enabled" : "☀️ Light mode enabled");
}

function applyDarkMode() {
    if (darkMode) {
        document.documentElement.classList.add("dark-mode");
    } else {
        document.documentElement.classList.remove("dark-mode");
    }
}

// === KEYBOARD SHORTCUTS ===
document.addEventListener("keydown", (e) => {
    // Ctrl+Enter: Send message
    if (e.ctrlKey && e.key === "Enter") {
        e.preventDefault();
        if (!userInput.disabled) handleSend();
    }
    
    // Ctrl+K: Focus search
    if (e.ctrlKey && e.key === "k") {
        e.preventDefault();
        const searchInput = document.getElementById("searchInput");
        if (searchInput) searchInput.focus();
    }
    
    // Ctrl+N: New conversation
    if (e.ctrlKey && e.key === "n") {
        e.preventDefault();
        document.getElementById("newChatBtn").click();
    }
    
    // Ctrl+E: Export current chat
    if (e.ctrlKey && e.key === "e") {
        e.preventDefault();
        if (currentConversationId) exportBtn.click();
    }
    
    // Esc: Close modals
    if (e.key === "Escape") {
        summaryModal.style.display = "none";
        exportModal.style.display = "none";
        popover.style.display = "none";
    }
});

// === INITIALIZATION ===
document.addEventListener("DOMContentLoaded", () => {
    if (!token) window.location.href = "login.html";
    
    applyDarkMode();
    loadUserInfo();
    loadConversations();
    
    // Close modals on outside click
    document.addEventListener("click", (e) => {
        if (!popover.contains(e.target) && !e.target.classList.contains("citation-link")) {
            popover.style.display = "none";
        }
        if (e.target === summaryModal) {
            summaryModal.style.display = "none";
        }
        if (e.target === exportModal) {
            exportModal.style.display = "none";
        }
    });
});

// === USER INFO & PROFILE ===
async function loadUserInfo() {
    try {
        const res = await fetch(`${API_BASE}/users/me`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        if (res.ok) {
            const user = await res.json();
            document.getElementById("usernameDisplay").textContent = user.username;
        }
    } catch (e) {
        console.error("Failed to load user info:", e);
    }
}

async function loadUserStats() {
    try {
        const res = await fetch(`${API_BASE}/users/stats/dashboard`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        if (res.ok) {
            const stats = await res.json();
            // Display stats in a modal or dashboard
            console.log("User Stats:", stats);
            showStatsModal(stats);
        }
    } catch (e) {
        console.error("Failed to load stats:", e);
    }
}

function showStatsModal(stats) {
    alert(`📊 Your Statistics:\n\n` +
        `Total Conversations: ${stats.total_conversations}\n` +
        `Questions Asked: ${stats.total_questions_asked}\n` +
        `PDFs Uploaded: ${stats.total_pdfs_uploaded}\n` +
        `Avg Messages/Chat: ${stats.avg_messages_per_conversation}`);
}

// === MODAL SETUP ===
document.getElementById("closePopover").onclick = () => popover.style.display = "none";
closeModalBtn.onclick = () => summaryModal.style.display = "none";
document.getElementById("closeExportModalBtn").onclick = () => exportModal.style.display = "none";

// === SEARCH CONVERSATIONS ===
async function searchConversations(query) {
    if (!query.trim()) {
        loadConversations();
        return;
    }
    
    try {
        const res = await fetch(`${API_BASE}/conversations/search/${encodeURIComponent(query)}`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        
        if (!res.ok) throw new Error("Search failed");
        const data = await res.json();
        
        const list = document.getElementById("conversationList");
        list.innerHTML = "";
        
        if (data.results.length === 0) {
            list.innerHTML = `<div class="empty-state">🔍 No results found</div>`;
            return;
        }
        
        data.results.forEach(c => {
            const div = document.createElement("div");
            div.className = `chat-item ${currentConversationId === c.conversation_id ? 'active' : ''}`;
            div.innerHTML = `
                <div class="chat-item-title">${c.title}</div>
                <div class="chat-item-preview">${c.preview ? c.preview.substring(0, 50) + '...' : ''}</div>
            `;
            div.onclick = () => loadChat(c.conversation_id);
            list.appendChild(div);
        });
        
        showToast(`Found ${data.results.length} result(s)`);
    } catch (e) {
        showToast("Search failed: " + e.message);
    }
}

// === TAGS ===
async function addTag(conversationId, tagName) {
    try {
        const res = await fetch(`${API_BASE}/conversations/${conversationId}/tags`, {
            method: "POST",
            headers: { 
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ tag: tagName })
        });
        
        if (res.ok) {
            showToast(`✅ Tag "${tagName}" added`);
            loadConversationTags(conversationId);
        }
    } catch (e) {
        showToast("Failed to add tag: " + e.message);
    }
}

async function removeTag(conversationId, tagName) {
    try {
        const res = await fetch(`${API_BASE}/conversations/${conversationId}/tags/${tagName}`, {
            method: "DELETE",
            headers: { "Authorization": `Bearer ${token}` }
        });
        
        if (res.ok) {
            showToast(`Removed tag "${tagName}"`);
            loadConversationTags(conversationId);
        }
    } catch (e) {
        showToast("Failed to remove tag: " + e.message);
    }
}

async function loadConversationTags(conversationId) {
    try {
        const res = await fetch(`${API_BASE}/conversations/${conversationId}/tags`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        
        if (res.ok) {
            const data = await res.json();
            const tagsContainer = document.getElementById("conversationTags");
            if (tagsContainer) {
                tagsContainer.innerHTML = data.tags.map(tag => `
                    <span class="tag">
                        ${tag}
                        <button class="tag-remove" onclick="removeTag(${conversationId}, '${tag}')">×</button>
                    </span>
                `).join("");
            }
        }
    } catch (e) {
        console.error("Failed to load tags:", e);
    }
}

// === MESSAGE REACTIONS ===
async function addMessageFeedback(messageId, rating) {
    try {
        const res = await fetch(`${API_BASE}/messages/${messageId}/feedback`, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ rating: rating })
        });
        
        if (res.ok) {
            showToast(rating > 0 ? "👍 Thanks for the feedback!" : "👎 Thanks for the feedback!");
        }
    } catch (e) {
        console.error("Failed to add feedback:", e);
        showToast("❌ Failed to record feedback");
    }
}

// === CONVERSATION SHARING ===
async function createShareLink() {
    if (!currentConversationId) return;
    
    const days = prompt("Share for how many days? (0 for never expires)", "7");
    if (days === null) return;
    
    try {
        const res = await fetch(`${API_BASE}/conversations/${currentConversationId}/share`, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ expires_in_days: days ? parseInt(days) : null })
        });
        
        if (res.ok) {
            const data = await res.json();
            const shareUrl = data.share_url;
            
            // Copy to clipboard
            navigator.clipboard.writeText(shareUrl);
            showToast("✅ Share link copied to clipboard!");
            
            alert(`Share Link Created!\n\n${shareUrl}\n\nLink copied to clipboard!`);
        }
    } catch (e) {
        showToast("Failed to create share link: " + e.message);
    }
}

// === BATCH OPERATIONS ===
function toggleConversationSelection(conversationId) {
    if (selectedConversations.has(conversationId)) {
        selectedConversations.delete(conversationId);
    } else {
        selectedConversations.add(conversationId);
    }
    
    // Update UI to show selected count
    const count = selectedConversations.size;
    if (count > 0) {
        showToast(`${count} conversation(s) selected`);
    }
}

async function batchDeleteSelected() {
    if (selectedConversations.size === 0) {
        showToast("No conversations selected");
        return;
    }
    
    if (!confirm(`Delete ${selectedConversations.size} conversation(s)?`)) return;
    
    try {
        const res = await fetch(`${API_BASE}/conversations/batch/delete`, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ conversation_ids: Array.from(selectedConversations) })
        });
        
        if (res.ok) {
            const data = await res.json();
            showToast(`✅ Deleted ${data.deleted_count} conversation(s)`);
            selectedConversations.clear();
            loadConversations();
        }
    } catch (e) {
        showToast("Failed to delete conversations: " + e.message);
    }
}

async function batchAddTags() {
    if (selectedConversations.size === 0) {
        showToast("No conversations selected");
        return;
    }
    
    const tag = prompt("Enter tag name:");
    if (!tag) return;
    
    try {
        const res = await fetch(`${API_BASE}/conversations/batch/tags`, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ 
                conversation_ids: Array.from(selectedConversations),
                tag: tag
            })
        });
        
        if (res.ok) {
            const data = await res.json();
            showToast(`✅ Tag "${tag}" added to ${data.updated_count} conversation(s)`);
            selectedConversations.clear();
            loadConversations();
        }
    } catch (e) {
        showToast("Failed to add tags: " + e.message);
    }
}

// === EXPORT LOGIC ===
exportBtn.onclick = () => {
    exportModal.style.display = "flex";
};

document.getElementById("exportPdfBtn").onclick = () => exportConversation("pdf");
document.getElementById("exportHtmlBtn").onclick = () => exportConversation("html");
document.getElementById("exportJsonBtn").onclick = () => exportConversation("json");

async function exportConversation(format) {
    if (!currentConversationId) return;
    
    try {
        const res = await fetch(`${API_BASE}/conversations/${currentConversationId}/export`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        
        if (!res.ok) throw new Error("Export failed");
        const data = await res.json();
        currentConversationData = data;
        
        if (format === "pdf") {
            exportToPdf(data);
        } else if (format === "html") {
            exportToHtml(data);
        } else if (format === "json") {
            exportToJson(data);
        }
        
        exportModal.style.display = "none";
        showToast("✅ Export successful!");
    } catch (e) {
        showToast("Export failed: " + e.message);
    }
}

function exportToPdf(data) {
    const html = generateConversationHtml(data);
    const element = document.createElement("div");
    element.innerHTML = html;
    
    const opt = {
        margin: 10,
        filename: `conversation_${data.title.replace(/\s+/g, "_")}.pdf`,
        image: { type: "jpeg", quality: 0.98 },
        html2canvas: { scale: 2 },
        jsPDF: { orientation: "portrait", unit: "mm", format: "a4" }
    };
    
    html2pdf().set(opt).from(element).save();
}

function exportToHtml(data) {
    const html = generateConversationHtml(data);
    const fullHtml = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${data.title} - Conversation Export</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8fafc; padding: 2rem; }
        .container { max-width: 900px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 2rem; }
        .header { border-bottom: 2px solid #e2e8f0; padding-bottom: 1rem; margin-bottom: 2rem; }
        .header h1 { color: #0f172a; font-size: 1.8rem; }
        .header p { color: #64748b; margin-top: 0.5rem; font-size: 0.95rem; }
        .messages { display: flex; flex-direction: column; gap: 1rem; }
        .message { display: flex; flex-direction: column; }
        .message.user { align-items: flex-end; }
        .message.bot { align-items: flex-start; }
        .msg-content { padding: 12px 18px; border-radius: 12px; line-height: 1.6; max-width: 70%; word-wrap: break-word; }
        .message.user .msg-content { background: #3b82f6; color: white; border-radius: 12px 12px 0 12px; }
        .message.bot .msg-content { background: #e2e8f0; color: #1e293b; border-radius: 12px 12px 12px 0; }
        .timestamp { font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem; }
        @media (max-width: 600px) {
            .msg-content { max-width: 90%; }
            .container { padding: 1rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📄 ${data.title}</h1>
            <p><strong>User:</strong> ${data.user}</p>
            <p><strong>Date:</strong> ${new Date(data.created_at).toLocaleString()}</p>
        </div>
        <div class="messages">
            ${data.messages.map(m => `
                <div class="message ${m.role}">
                    <div class="msg-content">${m.content.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>
                    <div class="timestamp">${new Date(m.timestamp).toLocaleTimeString()}</div>
                </div>
            `).join("")}
        </div>
    </div>
</body>
</html>
    `;
    
    const blob = new Blob([fullHtml], { type: "text/html;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `conversation_${data.title.replace(/\s+/g, "_")}.html`;
    link.click();
}

function exportToJson(data) {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: "application/json;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `conversation_${data.title.replace(/\s+/g, "_")}.json`;
    link.click();
}

function generateConversationHtml(data) {
    return `
        <h1>${data.title}</h1>
        <p><strong>User:</strong> ${data.user}</p>
        <p><strong>Date:</strong> ${new Date(data.created_at).toLocaleString()}</p>
        <hr/>
        ${data.messages.map(m => `
            <div style="margin: 1.5rem 0; padding: 1rem; background: ${m.role === "user" ? "#eff6ff" : "#f1f5f9"}; border-radius: 8px;">
                <strong>${m.role === "user" ? "👤 You" : "🤖 Assistant"}:</strong>
                <p style="margin-top: 0.5rem;">${m.content}</p>
                <small style="color: #64748b;">${new Date(m.timestamp).toLocaleTimeString()}</small>
            </div>
        `).join("")}
    `;
}

// === COPY MESSAGE TO CLIPBOARD ===
function copyMessageToClipboard(content) {
    if (!content || content.trim() === "") {
        showToast("⚠️ Nothing to copy");
        return;
    }
    
    navigator.clipboard.writeText(content).then(() => {
        showToast("✅ Copied to clipboard!");
    }).catch(err => {
        console.error("Copy failed:", err);
        // Fallback for older browsers
        const textArea = document.createElement("textarea");
        textArea.value = content;
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand("copy");
            showToast("✅ Copied to clipboard!");
        } catch (e) {
            showToast("❌ Copy failed");
        }
        document.body.removeChild(textArea);
    });
}

// === CHAT LOGIC ===
async function handleSend() {
    const text = userInput.value.trim();
    if (!text || !currentConversationId) return;
    
    appendMessage('user', text);
    userInput.value = "";
    
    const botContentDiv = appendMessage('bot', "");
    botContentDiv.innerHTML = showLoading("Thinking...").innerHTML;
    
    try {
        const response = await fetch(`${API_BASE}/ask`, {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}` 
            },
            body: JSON.stringify({ question: text, conversation_id: currentConversationId })
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = "";
        let isFirstChunk = true;
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            if (isFirstChunk) {
                botContentDiv.innerHTML = "";
                isFirstChunk = false;
            }
            
            fullText += decoder.decode(value);
            renderTextWithCitations(botContentDiv, fullText);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    } catch (e) {
        botContentDiv.textContent = "Error: " + e.message;
    }
}

// === SUMMARY LOGIC ===
summaryBtn.onclick = async () => {
    summaryModal.style.display = "flex";
    summaryContent.innerHTML = showLoading("Generating summary...").innerHTML;
    
    try {
        const response = await fetch(`${API_BASE}/conversations/${currentConversationId}/summarize-stream`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${token}` } 
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        currentSummaryData = "";
        let isFirstChunk = true;
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            if (isFirstChunk) {
                summaryContent.innerHTML = "";
                isFirstChunk = false;
            }
            
            currentSummaryData += decoder.decode(value);
            renderTextWithCitations(summaryContent, currentSummaryData);
        }
    } catch (e) {
        summaryContent.textContent = "Failed to generate summary: " + e.message;
    }
};

downloadSummaryBtn.onclick = () => {
    if (!currentSummaryData) return;
    
    const html = `
        <div style="font-family: 'Inter', Arial, sans-serif; padding: 20px; line-height: 1.6; color: #1e293b;">
            <h1 style="color: #0f172a; font-size: 1.8em; margin-bottom: 1rem;">Document Summary</h1>
            <div style="border-top: 2px solid #e2e8f0; padding-top: 1rem;">
                ${marked.parse(currentSummaryData)}
            </div>
            <div style="border-top: 1px solid #e2e8f0; margin-top: 2rem; padding-top: 1rem; font-size: 0.9em; color: #64748b;">
                <p>Generated on: ${new Date().toLocaleString()}</p>
            </div>
        </div>
    `;
    
    const opt = {
        margin: 10,
        filename: 'summary_report.pdf',
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2 },
        jsPDF: { orientation: 'portrait', unit: 'mm', format: 'a4' }
    };
    
    html2pdf().set(opt).from(html).save();
    showToast("✅ Summary downloaded!");
};

// === RENDER CITATIONS ===
function renderTextWithCitations(element, text) {
    let html = marked.parse(text);
    
    // Handle full format: [Source: filename - Page X]
    html = html.replace(/\[Source: (.*?) - Page (\d+)\]/g, (match, filename, page) => {
        return `<span class="citation-link" onclick="showCitationPreview(event, ${page})">📄 ${filename} (Pg ${page})</span>`;
    });

    // Handle standard format: [Page X]
    html = html.replace(/\[Page (\d+)\]/g, (match, page) => {
        return `<span class="citation-link" onclick="showCitationPreview(event, ${page})">📄 Page ${page}</span>`;
    });

    // Handle lazy format: [4] -> Convert to Page 4 pill
    html = html.replace(/\[(\d+)\]/g, (match, page) => {
        return `<span class="citation-link" onclick="showCitationPreview(event, ${page})">📄 Page ${page}</span>`;
    });
    
    element.innerHTML = html;
}

// === POPOVER PREVIEW LOGIC ===
async function showCitationPreview(event, page) {
    event.stopPropagation();
    
    const rect = event.target.getBoundingClientRect();
    popover.style.left = `${rect.left}px`;
    popover.style.top = `${rect.bottom + 10}px`; 
    popover.style.display = "block";
    
    popoverTitle.textContent = `Source: Page ${page}`;
    popoverContent.innerHTML = showLoading().innerHTML;
    
    try {
        const res = await fetch(`${API_BASE}/preview/${currentVectorStoreId}/${page}`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        
        if (!res.ok) throw new Error("Failed to load");
        const data = await res.json();
        popoverContent.textContent = data.text; 
        
    } catch (e) {
        popoverContent.textContent = "Error loading preview.";
    }
}

// === CONVERSATION STATS ===
async function showConversationStats() {
    if (!currentConversationId) return;
    
    try {
        const res = await fetch(`${API_BASE}/conversations/${currentConversationId}/stats`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        
        if (res.ok) {
            const stats = await res.json();
            alert(`📊 Chat Statistics\n\n` +
                `Total Messages: ${stats.total_messages}\n` +
                `Questions: ${stats.total_questions}\n` +
                `Responses: ${stats.total_responses}\n` +
                `Avg Question Length: ${stats.avg_question_length} chars\n` +
                `Avg Response Length: ${stats.avg_response_length} chars\n` +
                `Session Duration: ${Math.round(stats.session_duration_seconds / 60)} minutes`);
        }
    } catch (e) {
        showToast("Failed to load stats: " + e.message);
    }
}

// === HELPERS ===
function appendMessage(role, text) {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    const content = document.createElement("div");
    content.className = "msg-content";
    
    // Add buttons for bot messages
    if (role === "assistant") {
        const wrapper = document.createElement("div");
        wrapper.className = "msg-wrapper";
        
        content.textContent = text;
        
        // Create button container
        const btnContainer = document.createElement("div");
        btnContainer.className = "msg-buttons-container";
        
        // Copy button
        const copyBtn = document.createElement("button");
        copyBtn.className = "msg-copy-btn";
        copyBtn.textContent = "📋 Copy";
        copyBtn.title = "Copy message to clipboard";
        copyBtn.onclick = (e) => {
            e.stopPropagation();
            copyMessageToClipboard(text || "");
        };
        
        // Ratings container
        const ratingsContainer = document.createElement("div");
        ratingsContainer.className = "msg-ratings";
        
        // Get message ID from the database (stored in data attribute)
        // We'll set this when the message is rendered from history
        const thumbsUp = document.createElement("button");
        thumbsUp.className = "msg-rating-btn thumbs-up";
        thumbsUp.textContent = "👍";
        thumbsUp.title = "Helpful";
        thumbsUp.onclick = (e) => {
            e.stopPropagation();
            const msgId = div.dataset.messageId;
            if (msgId) {
                addMessageFeedback(msgId, 1);
                thumbsUp.classList.add("active");
                thumbsDown.classList.remove("active");
            }
        };
        
        const thumbsDown = document.createElement("button");
        thumbsDown.className = "msg-rating-btn thumbs-down";
        thumbsDown.textContent = "👎";
        thumbsDown.title = "Not helpful";
        thumbsDown.onclick = (e) => {
            e.stopPropagation();
            const msgId = div.dataset.messageId;
            if (msgId) {
                addMessageFeedback(msgId, -1);
                thumbsUp.classList.remove("active");
                thumbsDown.classList.add("active");
            }
        };
        
        ratingsContainer.appendChild(thumbsUp);
        ratingsContainer.appendChild(thumbsDown);
        
        btnContainer.appendChild(copyBtn);
        btnContainer.appendChild(ratingsContainer);
        
        wrapper.appendChild(content);
        wrapper.appendChild(btnContainer);
        div.appendChild(wrapper);
    } else {
        content.textContent = text;
        div.appendChild(content);
    }
    
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return content;
}

async function loadChat(id) {
    currentConversationId = id;
    
    try {
        const res = await fetch(`${API_BASE}/conversations/${id}`, { 
            headers: { "Authorization": `Bearer ${token}` } 
        });
        
        if (!res.ok) throw new Error("Failed to load chat");
        const data = await res.json();
        
        currentVectorStoreId = data.vector_store_id;
        document.getElementById("chatTitle").textContent = data.title;
        
        chatBox.innerHTML = "";
        data.messages.forEach(m => {
            const content = appendMessage(m.role, "");
            // Find the message div to attach the ID
            const msgDiv = content.closest(".message");
            if (msgDiv && m.id) {
                msgDiv.dataset.messageId = m.id;
            }
            renderTextWithCitations(content, m.content);
        });
        
        userInput.disabled = false;
        document.getElementById("sendBtn").disabled = false;
        summaryBtn.style.display = "block";
        exportBtn.style.display = "block";
        
        // Load tags
        loadConversationTags(id);
        
    } catch (e) {
        showToast("Failed to load chat: " + e.message);
    }
}

async function loadConversations(page = 0) {
    try {
        const res = await fetch(`${API_BASE}/conversations`, { 
            headers: { "Authorization": `Bearer ${token}` } 
        });
        
        if (!res.ok) throw new Error("Failed to load conversations");
        const chats = await res.json();
        
        const list = document.getElementById("conversationList");
        list.innerHTML = "";
        
        if (chats.length === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <div>📚 No conversations yet</div>
                    <small>Upload a PDF or create a new chat to start</small>
                </div>
            `;
            return;
        }
        
        // Pagination: Show 10 items per page
        const itemsPerPage = 10;
        const start = page * itemsPerPage;
        const end = start + itemsPerPage;
        const pageChats = chats.slice(start, end);
        
        pageChats.forEach(c => {
            const div = document.createElement("div");
            div.className = `chat-item ${currentConversationId === c.id ? 'active' : ''}`;
            div.innerHTML = `
                <div style="flex: 1;">
                    <div class="chat-item-title">${c.title}</div>
                    <div class="chat-item-date">${new Date(c.created_at).toLocaleDateString()}</div>
                </div>
            `;
            
            // Add context menu (right-click)
            div.addEventListener("contextmenu", (e) => {
                e.preventDefault();
                showContextMenu(e, c.id);
            });
            
            div.onclick = () => loadChat(c.id);
            list.appendChild(div);
        });
        
        // Add pagination if needed
        if (chats.length > itemsPerPage) {
            const paginationDiv = document.createElement("div");
            paginationDiv.className = "pagination";
            paginationDiv.innerHTML = `
                ${page > 0 ? `<button onclick="loadConversations(${page - 1})">← Prev</button>` : ""}
                <span>${page + 1} / ${Math.ceil(chats.length / itemsPerPage)}</span>
                ${end < chats.length ? `<button onclick="loadConversations(${page + 1})">Next →</button>` : ""}
            `;
            list.appendChild(paginationDiv);
        }
        
    } catch (e) {
        showToast("Failed to load conversations: " + e.message);
    }
}

function showContextMenu(event, conversationId) {
    const menu = document.createElement("div");
    menu.className = "context-menu";
    menu.style.left = event.clientX + "px";
    menu.style.top = event.clientY + "px";
    menu.innerHTML = `
        <button onclick="showConversationStats()">📊 Stats</button>
        <button onclick="addTagPrompt(${conversationId})">🏷️ Add Tag</button>
        <button onclick="createShareLink()">🔗 Share</button>
        <button onclick="deleteConversation(${conversationId})">🗑️ Delete</button>
    `;
    
    document.body.appendChild(menu);
    document.addEventListener("click", () => menu.remove(), { once: true });
}

async function deleteConversation(id) {
    if (!confirm("Delete this conversation?")) return;
    
    try {
        const res = await fetch(`${API_BASE}/conversations/${id}`, {
            method: "DELETE",
            headers: { "Authorization": `Bearer ${token}` }
        });
        
        if (res.ok) {
            showToast("✅ Conversation deleted");
            if (currentConversationId === id) currentConversationId = null;
            loadConversations();
        }
    } catch (e) {
        showToast("Failed to delete: " + e.message);
    }
}

function addTagPrompt(conversationId) {
    const tag = prompt("Enter tag name:");
    if (tag) addTag(conversationId, tag);
}

// Upload handler with improved UX
document.getElementById("pdfInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (!confirm("Upload this file?")) return;
    
    const formData = new FormData();
    formData.append("file", file);
    if (currentConversationId) formData.append("conversation_id", currentConversationId);
    
    showToast("📤 Uploading...");
    
    try {
        const res = await fetch(`${API_BASE}/upload_pdf`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${token}` },
            body: formData
        });
        
        if (!res.ok) {
            const error = await res.json();
            showToast("Upload failed: " + error.detail);
            return;
        }
        
        const data = await res.json();
        
        if (data.status === "duplicate") {
            showToast("⚠️ File already uploaded to this chat");
        } else {
            showToast(`✅ Upload successful! (${data.pdf_count} PDF(s))`);
            loadConversations();
            if (data.conversation_id) loadChat(data.conversation_id);
        }
    } catch (e) {
        showToast("Upload error: " + e.message);
    }
    
    e.target.value = "";
});

// New chat button
document.getElementById("newChatBtn").onclick = () => {
    currentConversationId = null;
    chatBox.innerHTML = `<div class="message bot"><div class="msg-content">Select or upload a PDF to start chatting!</div></div>`;
    document.getElementById("chatTitle").textContent = "New Chat";
    userInput.disabled = true;
    document.getElementById("sendBtn").disabled = true;
    summaryBtn.style.display = "none";
    exportBtn.style.display = "none";
};

// Send button handler
document.getElementById("sendBtn").onclick = handleSend;

// Allow Enter to send
userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

// Logout
document.getElementById("logoutBtn").onclick = () => {
    localStorage.removeItem("accessToken");
    window.location.href = "login.html";
};
