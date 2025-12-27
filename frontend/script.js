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
const sidebar = document.getElementById("sidebar");
const resizer = document.getElementById("resizer");
const chatContainer = document.getElementById("chatContainer");
const searchInput = document.getElementById("searchChats");

// --- API & State ---
const API_BASE = "http://127.0.0.1:8000";
let currentConversationId = null;
let currentSummaryData = null;
const token = localStorage.getItem("accessToken");
let conversationsCache = [];

// --- AUTH & INITIALIZATION ---
document.addEventListener("DOMContentLoaded", async () => {
    if (!token) {
        window.location.href = "login.html";
        return;
    }
    try {
        const userInfo = await fetchWithAuth("/users/me");
        usernameDisplay.textContent = userInfo.username;
    } catch (error) {
        console.error("Failed fetch username:", error);
        usernameDisplay.textContent = "Error";
    }

    const savedWidth = localStorage.getItem("sidebarWidth");
    if (savedWidth) {
        sidebar.style.width = savedWidth;
    }
    await loadConversations();
});

logoutBtn.addEventListener("click", () => {
    localStorage.removeItem("accessToken");
    window.location.href = "login.html";
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

    // Check for errors
    if (!res.ok) {
        let errorDetail = `API Error (${res.status})`;
        try {
            const errData = await res.json();
            errorDetail = errData.detail || errorDetail;
        } catch (e) { /* Ignore if error response not JSON */ }
        throw new Error(errorDetail);
    }

    // Return parsed JSON
    return res.json();
}


// --- UI HELPERS ---
function setChatBoxMessage(htmlContent) {
    chatBox.innerHTML = '';
    const msg = document.createElement("div");
    msg.className = "message bot";
    msg.innerHTML = htmlContent;
    chatBox.appendChild(msg);
}

function appendMessage(sender, message) {
    const msg = document.createElement("div");
    msg.className = sender === "user" ? "message user" : "message bot";

    if (sender === "user") {
        msg.textContent = message;
    } else {
        // Use marked if available, otherwise plain text
        if (typeof marked !== 'undefined') {
            msg.innerHTML = marked.parse(message || "");
        } else {
            msg.textContent = message;
        }
    }

    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msg; // Return element so we can modify it later if needed
}

function setChatInputEnabled(enabled) {
    userInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
    summaryBtn.style.display = enabled ? "block" : "none";
    userInput.placeholder = enabled ? "Ask..." : "Start/select chat.";
}

function renderConversationList(filter = "") {
    conversationList.innerHTML = "";
    const filteredConvos = conversationsCache.filter(convo => convo.title.toLowerCase().includes(filter.toLowerCase()));

    if (filteredConvos.length === 0 && filter) {
        conversationList.innerHTML = `<div class="convo-item-empty">No chats found.</div>`;
    } else if (conversationsCache.length === 0) {
        conversationList.innerHTML = `<div class="convo-item-empty">No chats yet.</div>`;
    }

    filteredConvos.forEach(convo => {
        const item = document.createElement("div");
        item.className = "convo-item";
        item.dataset.id = convo.id;

        const titleSpan = document.createElement("span");
        titleSpan.textContent = convo.title;
        titleSpan.title = convo.title;
        item.appendChild(titleSpan);

        const actionsDiv = document.createElement("div");
        actionsDiv.className = "convo-actions";

        const editBtn = document.createElement("button");
        editBtn.textContent = "✏️";
        editBtn.title = "Rename";
        editBtn.onclick = (e) => {
            e.stopPropagation();
            handleRename(convo.id, convo.title);
        };
        actionsDiv.appendChild(editBtn);

        const deleteBtn = document.createElement("button");
        deleteBtn.textContent = "🗑️";
        deleteBtn.title = "Delete";
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            handleDelete(convo.id, convo.title);
        };
        actionsDiv.appendChild(deleteBtn);

        item.appendChild(actionsDiv);
        item.addEventListener("click", () => loadChatHistory(convo.id));
        conversationList.appendChild(item);
    });

    if (currentConversationId) {
        const activeItem = conversationList.querySelector(`.convo-item[data-id='${currentConversationId}']`);
        if (activeItem) activeItem.classList.add('active');
    }
}


// --- CORE LOGIC ---
async function loadConversations() {
    try {
        conversationsCache = await fetchWithAuth("/conversations");
        renderConversationList(searchInput.value);
    } catch (error) {
        console.error("Failed load convos:", error);
        setChatBoxMessage("Failed load history.");
    }
}

async function loadChatHistory(convoId) {
    try {
        setChatBoxMessage("⏳ Loading chat history...");
        const history = await fetchWithAuth(`/conversations/${convoId}`);
        
        chatBox.innerHTML = "";
        history.messages.forEach(msg => appendMessage(msg.role === 'assistant' ? 'bot' : 'user', msg.content));
        
        currentConversationId = convoId;
        chatTitle.textContent = history.title;
        setChatInputEnabled(true);
        renderConversationList(searchInput.value);
    } catch (error) {
        console.error("Failed load chat:", error);
        setChatBoxMessage("Failed load chat.");
        setChatInputEnabled(false);
    }
}

async function handleRename(convoId, oldTitle) {
    const newTitle = prompt("Enter new title:", oldTitle);
    if (newTitle && newTitle.trim() && newTitle !== oldTitle) {
        try {
            await fetchWithAuth(`/conversations/${convoId}`, {
                method: "PUT",
                body: JSON.stringify({ title: newTitle.trim() })
            });
            const convoIndex = conversationsCache.findIndex(c => c.id === convoId);
            if (convoIndex > -1) conversationsCache[convoIndex].title = newTitle.trim();
            renderConversationList(searchInput.value);
            if (currentConversationId === convoId) chatTitle.textContent = newTitle.trim();
        } catch (error) {
            console.error("Rename fail:", error);
            alert(`Error: ${error.message}`);
        }
    }
}

async function handleDelete(convoId, title) {
    if (confirm(`Delete "${title}"?`)) {
        try {
            await fetchWithAuth(`/conversations/${convoId}`, { method: "DELETE" });
            conversationsCache = conversationsCache.filter(c => c.id !== convoId);
            renderConversationList(searchInput.value);
            if (currentConversationId === convoId) {
                currentConversationId = null;
                chatTitle.textContent = "PDF Chat Assistant";
                setChatBoxMessage('<div class="message bot" id="welcomeMessage">Chat deleted.</div>');
                setChatInputEnabled(false);
            }
        } catch (error) {
            console.error("Delete fail:", error);
            alert(`Error: ${error.message}`);
        }
    }
}

// --- EVENT LISTENERS ---
pdfInput.addEventListener("change", () => {
    pdfLabel.textContent = pdfInput.files.length > 0 ? pdfInput.files[0].name : "Choose PDF";
});

uploadBtn.addEventListener("click", async () => {
    const file = pdfInput.files[0];
    if (!file) {
        alert("Select PDF first!");
        return;
    }
    currentConversationId = null;
    setChatInputEnabled(false);
    chatTitle.textContent = "Processing PDF...";
    setChatBoxMessage("⏳ Processing PDF...");
    
    const formData = new FormData();
    formData.append("file", file);

    try {
        const data = await fetchWithAuth("/upload_pdf", {
            method: "POST",
            body: formData
        });
        await loadConversations();
        await loadChatHistory(data.conversation_id);
    } catch (error) {
        console.error("Upload failed:", error);
        setChatBoxMessage(`❌ Upload Error: ${error.message}`);
        chatTitle.textContent = "PDF Chat Assistant";
    } finally {
        pdfInput.value = "";
        pdfLabel.textContent = "Choose PDF";
    }
});

// ===============================================
// --- FIXED NON-STREAMING: handleSendQuestion ---
// ===============================================
async function handleSendQuestion() {
    const question = userInput.value.trim();
    if (!question || !currentConversationId) return;

    // 1. Add User Message
    appendMessage("user", question);
    userInput.value = "";

    // 2. Add "Thinking..." Placeholder
    const thinkingMsg = appendMessage("bot", ""); // Empty container
    thinkingMsg.innerHTML = '<div class="message-content">⏳ Thinking...</div>';

    try {
        // 3. Send Request (Expect standard JSON response)
        const data = await fetchWithAuth("/ask", {
            method: "POST",
            body: JSON.stringify({
                question: question,
                conversation_id: currentConversationId,
            }),
        });

        // 4. Update UI with Full Answer
        // data.answer corresponds to return {"answer": full_response} in app.py
        if (data.answer) {
             thinkingMsg.innerHTML = marked.parse(data.answer);
        } else {
             thinkingMsg.innerHTML = "⚠️ Empty response received.";
        }

    } catch (error) {
        console.error("Chat error:", error);
        thinkingMsg.innerHTML = `<span style="color:red">❌ Error: ${error.message}</span>`;
    } finally {
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}


// --- Chat Input Listeners ---
sendBtn.addEventListener("click", handleSendQuestion);
userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !userInput.disabled) {
        handleSendQuestion();
    }
});

// --- Search Input ---
searchInput.addEventListener("input", (e) => {
    renderConversationList(e.target.value);
});

// --- Resizer Logic ---
let isResizing = false;
resizer.addEventListener('mousedown', (e) => {
    isResizing = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
});
document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;
    const newWidth = e.clientX;
    const minWidth = parseInt(window.getComputedStyle(sidebar).minWidth, 10);
    const maxWidth = window.innerWidth * 0.5;
    if (newWidth >= minWidth && newWidth <= maxWidth) {
        sidebar.style.width = `${newWidth}px`;
    }
});
document.addEventListener('mouseup', () => {
    if (isResizing) {
        isResizing = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        localStorage.setItem('sidebarWidth', sidebar.style.width);
    }
});


// --- Summary Modal & PDF Download ---
summaryBtn.addEventListener("click", async () => {
    if (!currentConversationId) return;
    summaryContent.innerHTML = "<p>⏳ Generating summary...</p>";
    summaryModal.classList.add("show");
    currentSummaryData = null;
    downloadPdfBtn.disabled = true;
    try {
        const data = await fetchWithAuth(`/conversations/${currentConversationId}/summarize`);
        currentSummaryData = data;
        let html = "<h4>✨ AI Summary</h4>";
        html += marked.parse(data.generated_summary || "");
        html += "<hr><h4>📜 Transcript</h4>";
        data.messages.forEach(msg => {
            const role = msg.role === 'user' ? 'User' : 'Assistant';
            html += `<div class="transcript-item ${msg.role === 'user' ? 'user' : 'assistant'}"><strong>${role}:</strong>`;
            const tempDiv = document.createElement('div');
            tempDiv.textContent = msg.content || "";
            html += `<div>${tempDiv.innerHTML.replace(/\n/g, '<br>')}</div></div>`;
        });
        summaryContent.innerHTML = html;
        downloadPdfBtn.disabled = false;
    } catch (error) {
        summaryContent.innerHTML = `<p>❌ Error: ${error.message}</p>`;
        console.error("Summary failed:", error);
    }
});

closeModalBtn.addEventListener("click", () => {
    summaryModal.classList.remove("show");
});
summaryModal.addEventListener("click", (e) => {
    if (e.target === summaryModal) {
        summaryModal.classList.remove("show");
    }
});

downloadPdfBtn.addEventListener("click", () => {
    if (!currentSummaryData) {
        alert("Summary data missing.");
        return;
    }
    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        const margin = 15;
        let currentY = margin;
        const pageWidth = doc.internal.pageSize.width;
        const pageHeight = doc.internal.pageSize.height;
        const addText = (text, x, y, options = {}) => {
            const lines = doc.splitTextToSize(text || " ", pageWidth - margin * 2 - (options.indent || 0));
            const requiredHeight = lines.length * 5 + (options.spacing || 0);
            if (y + requiredHeight > pageHeight - margin) {
                doc.addPage();
                y = margin;
            }
            doc.text(lines, x + (options.indent || 0), y);
            return y + requiredHeight;
        };
        doc.setFontSize(16);
        currentY = addText(`Summary: ${chatTitle.textContent}`, margin, currentY, { spacing: 5 });
        doc.setFontSize(12);
        doc.setFont("helvetica", "bold");
        currentY = addText("AI Summary:", margin, currentY, { spacing: 2 });
        doc.setFont("helvetica", "normal");
        currentY = addText(currentSummaryData.generated_summary, margin, currentY, { spacing: 7 });
        doc.setFont("helvetica", "bold");
        currentY = addText("Transcript:", margin, currentY, { spacing: 2 });
        currentSummaryData.messages.forEach(msg => {
            const role = msg.role === 'user' ? 'User' : 'Assistant';
            doc.setFont("helvetica", "bold");
            if (currentY + 10 > pageHeight - margin) {
                doc.addPage();
                currentY = margin;
            }
            currentY = addText(`${role}:`, margin, currentY, { spacing: 1 });
            doc.setFont("helvetica", "normal");
            currentY = addText(msg.content, margin + 5, currentY, { indent: 5, spacing: 3 });
        });
        doc.save(`summary-${currentConversationId}.pdf`);
    } catch (error) {
        console.error("PDF gen failed:", error);
        alert("Failed PDF gen.");
    }
});