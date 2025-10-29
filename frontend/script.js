// --- DOM Elements ---
const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const pdfInput = document.getElementById("pdfInput");
const uploadBtn = document.getElementById("uploadBtn");
const pdfLabel = document.getElementById("pdfLabel");
const conversationList = document.getElementById("conversationList");
const chatTitle = document.getElementById("chatTitle");
const logoutBtn = document.getElementById("logoutBtn");
const usernameDisplay = document.getElementById("usernameDisplay");
const summaryBtn = document.getElementById("summaryBtn");
const summaryModal = document.getElementById("summaryModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const summaryContent = document.getElementById("summaryContent");
const downloadPdfBtn = document.getElementById("downloadPdfBtn");

// --- API & State ---
const API_BASE = "http://127.0.0.1:8000";
let currentConversationId = null;
let currentSummaryData = null; // Store summary data for PDF download
const token = localStorage.getItem("accessToken");

// --- AUTH & INITIALIZATION ---
document.addEventListener("DOMContentLoaded", async () => {
    if (!token) {
        window.location.href = "login.html"; return;
    }
    try { // Fetch username
        const userInfo = await fetchWithAuth("/users/me");
        usernameDisplay.textContent = userInfo.username;
    } catch (error) {
        console.error("Failed to fetch username:", error); usernameDisplay.textContent = "Error";
    }
    loadConversations(); // Load chat list
});

logoutBtn.addEventListener("click", () => {
    localStorage.removeItem("accessToken"); window.location.href = "login.html";
});

// --- API HELPER ---
async function fetchWithAuth(url, options = {}) {
    const headers = { ...options.headers, 'Authorization': `Bearer ${token}` };
    const res = await fetch(`${API_BASE}${url}`, { ...options, headers });
    if (res.status === 401) { localStorage.removeItem("accessToken"); window.location.href = "login.html"; throw new Error("Unauthorized"); }
    if (!res.ok) { const errData = await res.json(); throw new Error(errData.detail || "API error"); }
    return res.json();
}

// --- UI HELPERS ---
function setChatBoxMessage(htmlContent) {
    chatBox.innerHTML = '';
    const msg = document.createElement("div"); msg.className = "message bot"; msg.innerHTML = htmlContent; chatBox.appendChild(msg);
}
function appendMessage(sender, message) {
    const msg = document.createElement("div"); msg.className = sender === "user" ? "message user" : "message bot";
    // Sanitize user input before adding to prevent XSS
    if (sender === "user") { msg.textContent = message; } 
    else { msg.innerHTML = marked.parse(message || ""); } // Handle potential null/undefined message
    chatBox.appendChild(msg); chatBox.scrollTop = chatBox.scrollHeight;
}
function replaceLastBotMessage(newText) {
    const lastBotMessage = chatBox.querySelector(".message.bot:last-child");
    if (lastBotMessage && lastBotMessage.textContent === "⏳ Thinking...") {
        lastBotMessage.innerHTML = marked.parse(newText || ""); // Handle potential null/undefined
    } else { appendMessage("bot", newText); }
}
function setChatInputEnabled(enabled) {
    userInput.disabled = !enabled; sendBtn.disabled = !enabled;
    summaryBtn.style.display = enabled ? "block" : "none"; // Toggle summary button visibility
    userInput.placeholder = enabled ? "Ask a question..." : "Start a new chat or select one.";
}

// --- CONVERSATION LOGIC ---
async function loadConversations() {
    try {
        const convos = await fetchWithAuth("/conversations");
        conversationList.innerHTML = "";
        convos.forEach(convo => {
            const item = document.createElement("div"); item.className = "convo-item"; item.textContent = convo.title; item.dataset.id = convo.id;
            item.addEventListener("click", () => loadChatHistory(convo.id));
            conversationList.appendChild(item);
        });
        // If a chat was active before reload, try to keep it active
        if (currentConversationId) {
             const activeItem = conversationList.querySelector(`.convo-item[data-id='${currentConversationId}']`);
             if (activeItem) activeItem.classList.add('active');
        }
    } catch (error) { console.error("Failed load convos:", error); setChatBoxMessage("Failed load history."); }
}
async function loadChatHistory(convoId) {
    try {
        setChatBoxMessage("⏳ Loading chat history...");
        const history = await fetchWithAuth(`/conversations/${convoId}`);
        chatBox.innerHTML = "";
        history.messages.forEach(msg => appendMessage(msg.role === 'assistant' ? 'bot' : 'user', msg.content)); // Use correct roles
        currentConversationId = convoId; chatTitle.textContent = history.title; setChatInputEnabled(true);
        document.querySelectorAll('.convo-item').forEach(item => item.classList.toggle('active', item.dataset.id == convoId));
    } catch (error) { console.error("Failed load chat:", error); setChatBoxMessage("Failed load chat."); setChatInputEnabled(false); }
}

// --- EVENT LISTENERS ---
pdfInput.addEventListener("change", () => { pdfLabel.textContent = pdfInput.files.length > 0 ? pdfInput.files[0].name : "Choose PDF"; });

uploadBtn.addEventListener("click", async () => {
    const file = pdfInput.files[0]; if (!file) { alert("Select PDF first!"); return; }
    currentConversationId = null; setChatInputEnabled(false);
    chatTitle.textContent = "Processing PDF..."; setChatBoxMessage("⏳ Processing PDF...");
    const formData = new FormData(); formData.append("file", file);
    try {
        const data = await fetchWithAuth("/upload_pdf", { method: "POST", body: formData });
        await loadConversations(); // Refresh sidebar FIRST
        await loadChatHistory(data.conversation_id); // THEN load the new chat
    } catch (error) { console.error("Upload failed:", error); setChatBoxMessage(`❌ Upload Error: ${error.message}`); chatTitle.textContent = "PDF Chat Assistant"; }
    finally { pdfInput.value = ""; pdfLabel.textContent = "Choose PDF"; }
});

async function handleSendQuestion() {
    const question = userInput.value.trim(); if (!question || !currentConversationId) return;
    appendMessage("user", question); userInput.value = ""; appendMessage("bot", "⏳ Thinking...");
    try {
        const data = await fetchWithAuth("/ask", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question, conversation_id: currentConversationId }),
        });
        replaceLastBotMessage(`🤖 ${data.answer}`);
    } catch (error) { replaceLastBotMessage(`❌ Error: ${error.message}`); console.error(error); }
}
sendBtn.addEventListener("click", handleSendQuestion);
userInput.addEventListener("keypress", (e) => { if (e.key === "Enter" && !userInput.disabled) { handleSendQuestion(); } });

// --- SUMMARY MODAL ---
summaryBtn.addEventListener("click", async () => {
    if (!currentConversationId) return;
    summaryContent.innerHTML = "<p>⏳ Generating summary...</p>"; summaryModal.classList.add("show");
    currentSummaryData = null; downloadPdfBtn.disabled = true;
    try {
        const data = await fetchWithAuth(`/conversations/${currentConversationId}/summarize`);
        currentSummaryData = data; // Store raw data for PDF
        let html = "<h4>✨ AI Generated Summary</h4>";
        html += marked.parse(data.generated_summary || "");
        html += "<hr><h4>📜 Full Transcript</h4>";
        data.messages.forEach(msg => {
            const role = msg.role === 'user' ? 'User' : 'Assistant';
            html += `<div class="transcript-item ${msg.role === 'user' ? 'user' : 'assistant'}"><strong>${role}:</strong>`;
            const tempDiv = document.createElement('div'); tempDiv.textContent = msg.content || ""; // Ensure content exists
            html += `<div>${tempDiv.innerHTML.replace(/\n/g, '<br>')}</div></div>`;
        });
        summaryContent.innerHTML = html;
        downloadPdfBtn.disabled = false;
    } catch (error) { summaryContent.innerHTML = `<p>❌ Error: ${error.message}</p>`; console.error("Summary failed:", error); }
});

closeModalBtn.addEventListener("click", () => { summaryModal.classList.remove("show"); });
summaryModal.addEventListener("click", (e) => { if (e.target === summaryModal) { summaryModal.classList.remove("show"); } });

// --- PDF DOWNLOAD ---
downloadPdfBtn.addEventListener("click", () => {
    if (!currentSummaryData) { alert("Summary data missing."); return; }
    try {
        const { jsPDF } = window.jspdf; const doc = new jsPDF();
        const margin = 15; let currentY = margin; const pageWidth = doc.internal.pageSize.width; const pageHeight = doc.internal.pageSize.height;
        const addText = (text, x, y, options) => {
            const lines = doc.splitTextToSize(text || " ", pageWidth - margin * 2 - (options?.indent || 0)); // Ensure text is not empty
            if (y + (lines.length * 5) > pageHeight - margin) { doc.addPage(); y = margin; } // Check for page break
            doc.text(lines, x, y); return y + (lines.length * 5); // Return new Y position
        };
        // Title
        doc.setFontSize(16); currentY = addText(`Summary: ${chatTitle.textContent}`, margin, currentY, {}) + 5;
        // AI Summary
        doc.setFontSize(12); doc.setFont("helvetica", "bold"); currentY = addText("AI Generated Summary:", margin, currentY, {}) + 2;
        doc.setFont("helvetica", "normal"); currentY = addText(currentSummaryData.generated_summary, margin, currentY, {}) + 7;
        // Transcript
        doc.setFont("helvetica", "bold"); currentY = addText("Full Transcript:", margin, currentY, {}) + 2;
        currentSummaryData.messages.forEach(msg => {
            const role = msg.role === 'user' ? 'User' : 'Assistant';
            doc.setFont("helvetica", "bold");
            if (currentY + 10 > pageHeight - margin) { doc.addPage(); currentY = margin; } // Check before adding role
            currentY = addText(`${role}:`, margin, currentY, {}) + 1;
            doc.setFont("helvetica", "normal");
            currentY = addText(msg.content, margin + 5, currentY, { indent: 5 }) + 3; // Indent message slightly
        });
        doc.save(`summary-${currentConversationId}.pdf`);
    } catch (error) {
        console.error("Failed to generate PDF:", error);
        alert("Failed to generate PDF. Check console for details.");
    }
});