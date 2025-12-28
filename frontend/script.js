const API_BASE = "http://127.0.0.1:8000";
let currentConversationId = null;
let currentVectorStoreId = null; 
let currentSummaryData = null; // Store summary for PDF download
let currentConversationData = null; // Store conversation for export
const token = localStorage.getItem("accessToken");

const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const popover = document.getElementById("citationPopover");
const popoverContent = document.getElementById("popoverContent");
const popoverTitle = document.getElementById("popoverTitle");

// Summary Elements
const summaryBtn = document.getElementById("summaryBtn");
const summaryModal = document.getElementById("summaryModal");
const exportBtn = document.getElementById("exportBtn");
const exportModal = document.getElementById("exportModal");
const summaryContent = document.getElementById("summaryContent");
const closeModalBtn = document.getElementById("closeModalBtn");
const downloadSummaryBtn = document.getElementById("downloadSummaryBtn");

// --- INIT ---
document.addEventListener("DOMContentLoaded", () => {
    if(!token) window.location.href = "login.html";
    loadUserInfo();  // Load and display current user
    loadConversations();
    
    // Close popover on outside click
    document.addEventListener("click", (e) => {
        if (!popover.contains(e.target) && !e.target.classList.contains("citation-link")) {
            popover.style.display = "none";
        }
        if (e.target === summaryModal) {
            summaryModal.style.display = "none";
        }
    });
});

// --- LOAD USER INFO ---
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

document.getElementById("closePopover").onclick = () => popover.style.display = "none";
closeModalBtn.onclick = () => summaryModal.style.display = "none";
document.getElementById("closeExportModalBtn").onclick = () => exportModal.style.display = "none";

// --- EXPORT LOGIC ---
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
    } catch (e) {
        alert("Export failed: " + e.message);
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
        .citation-link { color: #3b82f6; font-weight: 600; cursor: pointer; }
        h2, h3 { color: #0f172a; margin-top: 1rem; margin-bottom: 0.5rem; }
        p { margin-bottom: 0.5rem; }
        code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }
        pre { background: #1e293b; color: #e2e8f0; padding: 1rem; border-radius: 8px; overflow-x: auto; }
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

// --- CHAT LOGIC ---
async function handleSend() {
    const text = userInput.value.trim();
    if(!text || !currentConversationId) return;
    
    appendMessage('user', text);
    userInput.value = "";
    
    // 1. ADD "THINKING" INDICATOR
    const botContentDiv = appendMessage('bot', "");
    botContentDiv.innerHTML = `
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>`;
    
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
        
        // 2. Clear indicator on first chunk
        let isFirstChunk = true;
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            if (isFirstChunk) {
                botContentDiv.innerHTML = ""; // Remove dots
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

// --- SUMMARY LOGIC ---
summaryBtn.onclick = async () => {
    summaryModal.style.display = "flex";
    summaryContent.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
    
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
                summaryContent.innerHTML = ""; // Remove loading indicator
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
    if(!currentSummaryData) return;
    
    // Properly render markdown in PDF using html2pdf
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
};


// --- RENDER CITATIONS ---
function renderTextWithCitations(element, text) {
    let html = marked.parse(text);
    
    // 1. Handle full format: [Source: filename - Page X]
    html = html.replace(/\[Source: (.*?) - Page (\d+)\]/g, (match, filename, page) => {
        return `<span class="citation-link" onclick="showCitationPreview(event, ${page})">📄 ${filename} (Pg ${page})</span>`;
    });

    // 2. Handle standard format: [Page X]
    html = html.replace(/\[Page (\d+)\]/g, (match, page) => {
        return `<span class="citation-link" onclick="showCitationPreview(event, ${page})">📄 Page ${page}</span>`;
    });

    // 3. Handle lazy format: [4] -> Convert to Page 4 pill
    // This fixes the issue where the AI just returns a number
    html = html.replace(/\[(\d+)\]/g, (match, page) => {
        return `<span class="citation-link" onclick="showCitationPreview(event, ${page})">📄 Page ${page}</span>`;
    });
    
    element.innerHTML = html;
}
// --- POPOVER PREVIEW LOGIC ---
async function showCitationPreview(event, page) {
    event.stopPropagation();
    
    const rect = event.target.getBoundingClientRect();
    popover.style.left = `${rect.left}px`;
    popover.style.top = `${rect.bottom + 10}px`; 
    popover.style.display = "block";
    
    popoverTitle.textContent = `Source: Page ${page}`;
    popoverContent.textContent = "Loading preview...";
    
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

// --- HELPERS ---
function appendMessage(role, text) {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    const content = document.createElement("div");
    content.className = "msg-content";
    content.textContent = text;
    div.appendChild(content);
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return content;
}

async function loadChat(id) {
    currentConversationId = id;
    const res = await fetch(`${API_BASE}/conversations/${id}`, { headers: { "Authorization": `Bearer ${token}` } });
    const data = await res.json();
    currentVectorStoreId = data.vector_store_id;
    document.getElementById("chatTitle").textContent = data.title;
    
    chatBox.innerHTML = "";
    data.messages.forEach(m => {
        const div = appendMessage(m.role, "");
        renderTextWithCitations(div, m.content);
    });
    
    userInput.disabled = false;
    document.getElementById("sendBtn").disabled = false;
    summaryBtn.style.display = "block"; // Show button when chat loads
    exportBtn.style.display = "block"; // Show export button when chat loads
}

async function loadConversations() {
    const res = await fetch(`${API_BASE}/conversations`, { headers: { "Authorization": `Bearer ${token}` } });
    const chats = await res.json();
    const list = document.getElementById("conversationList");
    list.innerHTML = "";
    chats.forEach(c => {
        const div = document.createElement("div");
        div.className = `chat-item ${currentConversationId === c.id ? 'active' : ''}`;
        div.textContent = c.title;
        div.onclick = () => loadChat(c.id);
        list.appendChild(div);
    });
}

// Upload
document.getElementById("pdfInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if(!file) return;
    if(!confirm("Upload this file?")) return;
    
    const formData = new FormData();
    formData.append("file", file);
    if (currentConversationId) formData.append("conversation_id", currentConversationId);
    
    try {
        const res = await fetch(`${API_BASE}/upload_pdf`, {
            method: "POST", headers: { "Authorization": `Bearer ${token}` }, body: formData
        });
        const data = await res.json();
        if(!currentConversationId) { loadConversations(); loadChat(data.conversation_id); }
        else { alert("File added!"); }
    } catch(e) { alert(e.message); }
});

document.getElementById("sendBtn").onclick = handleSend;
userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault(); 
        if (!userInput.disabled) handleSend();
    }
});

document.getElementById("newChatBtn").onclick = () => {
    currentConversationId = null;
    chatBox.innerHTML = "";
    document.getElementById("chatTitle").textContent = "New Conversation";
    summaryBtn.style.display = "none";
};
document.getElementById("logoutBtn").onclick = () => {
    localStorage.removeItem("accessToken");
    window.location.href = "login.html";
};