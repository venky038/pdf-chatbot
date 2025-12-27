const API_BASE = "http://127.0.0.1:8000";
let currentConversationId = null;
let currentVectorStoreId = null; 
let currentSummaryData = null; // Store summary for PDF download
const token = localStorage.getItem("accessToken");

const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const popover = document.getElementById("citationPopover");
const popoverContent = document.getElementById("popoverContent");
const popoverTitle = document.getElementById("popoverTitle");

// Summary Elements
const summaryBtn = document.getElementById("summaryBtn");
const summaryModal = document.getElementById("summaryModal");
const summaryContent = document.getElementById("summaryContent");
const closeModalBtn = document.getElementById("closeModalBtn");
const downloadSummaryBtn = document.getElementById("downloadSummaryBtn");

// --- INIT ---
document.addEventListener("DOMContentLoaded", () => {
    if(!token) window.location.href = "login.html";
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

document.getElementById("closePopover").onclick = () => popover.style.display = "none";
closeModalBtn.onclick = () => summaryModal.style.display = "none";

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
        const res = await fetch(`${API_BASE}/conversations/${currentConversationId}/summarize`, {
             headers: { "Authorization": `Bearer ${token}` } 
        });
        const data = await res.json();
        currentSummaryData = data.generated_summary;
        
        renderTextWithCitations(summaryContent, currentSummaryData);
    } catch (e) {
        summaryContent.textContent = "Failed to generate summary.";
    }
};

downloadSummaryBtn.onclick = () => {
    if(!currentSummaryData) return;
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    // Strip markdown for clean PDF
    const cleanText = currentSummaryData.replace(/\*\*/g, "").replace(/###/g, ""); 
    
    doc.setFontSize(12);
    const splitText = doc.splitTextToSize(cleanText, 180);
    doc.text(splitText, 10, 10);
    doc.save("Summary.pdf");
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