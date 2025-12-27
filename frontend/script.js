// --- DOM ELEMENTS ---
const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const pdfInput = document.getElementById("pdfInput");
const conversationList = document.getElementById("conversationList");
const chatTitle = document.getElementById("chatTitle");
const logoutBtn = document.getElementById("logoutBtn");
const usernameDisplay = document.getElementById("usernameDisplay");
const searchInput = document.getElementById("searchChats");

// Summary Elements
const summaryBtn = document.getElementById("summaryBtn");
const summaryModal = document.getElementById("summaryModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const summaryContent = document.getElementById("summaryContent");
const downloadSummaryPdfBtn = document.getElementById("downloadSummaryPdfBtn");

// --- STATE ---
const API_BASE = "http://127.0.0.1:8000";
let currentConversationId = null;
let currentVectorStoreId = null; 
let currentSummaryData = null;
let conversationsCache = [];
const token = localStorage.getItem("accessToken");

// --- INITIALIZATION ---
document.addEventListener("DOMContentLoaded", async () => {
    if (!token) {
        window.location.href = "login.html";
        return;
    }
    try {
        const user = await fetchWithAuth("/users/me");
        usernameDisplay.textContent = user.username;
        await loadConversations();
    } catch (e) {
        console.error("Init Error:", e);
    }
});

// --- API HELPER ---
async function fetchWithAuth(url, options = {}) {
    const headers = { ...options.headers, 'Authorization': `Bearer ${token}` };
    if (options.body && typeof options.body === 'string' && !headers['Content-Type']) {
        headers['Content-Type'] = 'application/json';
    }
    const res = await fetch(`${API_BASE}${url}`, { ...options, headers });
    if (res.status === 401) {
        localStorage.removeItem("accessToken");
        window.location.href = "login.html";
        throw new Error("Unauthorized");
    }
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `API Error ${res.status}`);
    }
    return res.json();
}

// --- CONVERSATION LOGIC ---
async function loadConversations() {
    try {
        conversationsCache = await fetchWithAuth("/conversations");
        renderConversationList();
    } catch (e) { console.error(e); }
}

function renderConversationList(filter = "") {
    conversationList.innerHTML = "";
    const filtered = conversationsCache.filter(c => c.title.toLowerCase().includes(filter.toLowerCase()));
    
    if (filtered.length === 0) {
        conversationList.innerHTML = `<div style="text-align:center; padding:1rem; opacity:0.6;">No chats found</div>`;
        return;
    }

    filtered.forEach(convo => {
        const div = document.createElement("div");
        div.className = `chat-item ${currentConversationId === convo.id ? 'active' : ''}`;
        
        div.innerHTML = `
            <span>${convo.title}</span>
            <div class="chat-item-actions">
                <button class="icon-btn" onclick="event.stopPropagation(); deleteChat(${convo.id})">🗑️</button>
            </div>
        `;
        div.onclick = () => loadChatHistory(convo.id);
        conversationList.appendChild(div);
    });
}

async function deleteChat(id) {
    if (!confirm("Delete this conversation?")) return;
    try {
        await fetchWithAuth(`/conversations/${id}`, { method: "DELETE" });
        conversationsCache = conversationsCache.filter(c => c.id !== id);
        if (currentConversationId === id) {
            currentConversationId = null;
            currentVectorStoreId = null;
            chatTitle.textContent = "Select a Conversation";
            chatBox.innerHTML = "";
            setChatInputEnabled(false);
        }
        renderConversationList(searchInput.value);
    } catch (e) { alert(e.message); }
}

// --- CHAT HISTORY ---
async function loadChatHistory(convoId) {
    try {
        currentConversationId = convoId;
        chatBox.innerHTML = '<div style="text-align:center; padding:2rem;">Loading...</div>';
        
        const data = await fetchWithAuth(`/conversations/${convoId}`);
        currentVectorStoreId = data.vector_store_id; 
        chatTitle.textContent = data.title;
        
        chatBox.innerHTML = ""; 
        data.messages.forEach(msg => appendMessage(msg.role, msg.content));
        
        setChatInputEnabled(true);
        renderConversationList(searchInput.value);
        
    } catch (e) {
        chatBox.innerHTML = `<div style="color:red; text-align:center;">Error: ${e.message}</div>`;
    }
}

// --- MESSAGE RENDERING ---
function appendMessage(role, text) {
    const isUser = role === 'user';
    const div = document.createElement("div");
    div.className = `message ${isUser ? 'user' : 'bot'}`;
    
    const contentDiv = document.createElement("div");
    contentDiv.className = "msg-content";
    
    if (isUser) {
        contentDiv.textContent = text;
    } else {
        // Parse Markdown & Citations
        let html = marked.parse(text || "");
        if (currentVectorStoreId) {
            html = html.replace(/\[(Source:\s?)?Page\s?(\d+)\]/gi, (match, prefix, pageNum) => {
                const url = `${API_BASE}/uploads/${currentVectorStoreId}.pdf#page=${pageNum}`;
                return `<a href="${url}" target="_blank" class="citation-link" title="Open PDF Page ${pageNum}">📄 Page ${pageNum}</a>`;
            });
        }
        contentDiv.innerHTML = html;
    }
    
    div.appendChild(contentDiv);
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return contentDiv;
}

async function handleSend() {
    const text = userInput.value.trim();
    if (!text || !currentConversationId) return;
    
    appendMessage('user', text);
    userInput.value = "";
    
    const botContent = appendMessage('bot', "");
    botContent.innerHTML = `<div class="loading-dots"><span></span><span></span><span></span></div>`;
    
    try {
        const res = await fetchWithAuth("/ask", {
            method: "POST",
            body: JSON.stringify({ question: text, conversation_id: currentConversationId })
        });
        
        const parent = botContent.parentElement;
        parent.remove(); 
        appendMessage('bot', res.answer);
        
    } catch (e) {
        botContent.innerHTML = `<span style="color:red">Error: ${e.message}</span>`;
    }
}

// --- UPLOAD ---
pdfInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    if (confirm(`Upload ${file.name}?`)) {
        const originalText = chatTitle.textContent;
        chatTitle.textContent = "⏳ Uploading & Indexing...";
        
        const formData = new FormData();
        formData.append("file", file);
        
        try {
            const res = await fetch(`${API_BASE}/upload_pdf`, {
                method: "POST",
                headers: { "Authorization": `Bearer ${token}` },
                body: formData
            });
            
            if (!res.ok) throw new Error("Upload failed");
            const data = await res.json();
            
            await loadConversations();
            await loadChatHistory(data.conversation_id);
            
        } catch (e) {
            alert("Upload Error: " + e.message);
            chatTitle.textContent = originalText;
        } finally {
            pdfInput.value = "";
        }
    } else {
        pdfInput.value = "";
    }
});

// --- SUMMARY LOGIC ---
summaryBtn.addEventListener("click", async () => {
    if (!currentConversationId) return;
    
    summaryModal.classList.add("show");
    summaryContent.innerHTML = `<div class="loading-dots"><span></span><span></span><span></span></div>`;
    
    try {
        const data = await fetchWithAuth(`/conversations/${currentConversationId}/summarize`);
        currentSummaryData = data;
        
        let html = marked.parse(data.generated_summary);
        if (currentVectorStoreId) {
             html = html.replace(/\[(Source:\s?)?Page\s?(\d+)\]/gi, (match, prefix, pageNum) => {
                const url = `${API_BASE}/uploads/${currentVectorStoreId}.pdf#page=${pageNum}`;
                return `<a href="${url}" target="_blank" class="citation-link">📄 Page ${pageNum}</a>`;
            });
        }
        summaryContent.innerHTML = html;
        
    } catch (e) {
        summaryContent.innerHTML = `<span style="color:red">Error: ${e.message}</span>`;
    }
});

closeModalBtn.addEventListener("click", () => summaryModal.classList.remove("show"));
window.onclick = (e) => { if (e.target === summaryModal) summaryModal.classList.remove("show"); };

// --- PDF REPORT GENERATION (FIXED) ---
function cleanMarkdown(text) {
    if (!text) return "";
    return text
        .replace(/\*\*(.*?)\*\*/g, '$1') // Bold
        .replace(/###\s?/g, '')           // Headers
        .replace(/##\s?/g, '')
        .replace(/#\s?/g, '')
        .replace(/\[Source: Page \d+\]/g, '') // Remove citations for cleaner PDF report
        .replace(/\n\n/g, '\n');          // Fix double spacing
}

downloadSummaryPdfBtn.addEventListener("click", () => {
    if (!currentSummaryData) return;
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    doc.setFontSize(16);
    doc.text("Conversation Summary", 10, 10);
    
    doc.setFontSize(12);
    // Use CLEANED text, not raw Markdown
    const cleanText = cleanMarkdown(currentSummaryData.generated_summary);
    
    const splitText = doc.splitTextToSize(cleanText, 180);
    doc.text(splitText, 10, 20);
    
    doc.save("summary.pdf");
});

function setChatInputEnabled(enabled) {
    userInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
    summaryBtn.style.display = enabled ? "flex" : "none"; 
    userInput.placeholder = enabled ? "Ask a question..." : "Select a chat to start...";
    if (!enabled) chatTitle.textContent = "Select a Conversation";
}

sendBtn.addEventListener("click", handleSend);
userInput.addEventListener("keypress", (e) => { if (e.key === "Enter" && !userInput.disabled) handleSend(); });
searchInput.addEventListener("input", (e) => renderConversationList(e.target.value));
logoutBtn.addEventListener("click", () => {
    localStorage.removeItem("accessToken");
    window.location.href = "login.html";
});