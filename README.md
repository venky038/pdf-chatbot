# 📄 QueryMate - Universal Document, Office & Image AI Assistant

QueryMate is a **premium, feature-rich RAG (Retrieval-Augmented Generation)** application designed for high-precision document research. It transforms static files into interactive knowledge bases using state-of-the-art AI, a stunning glassmorphic UI, and robust analytical tools.

---

## 🚀 Key Features

- **Universal Document Support**: Chat with **PDFs, Word (.docx), Excel (.xlsx), PowerPoint (.pptx)**, MarkDown, HTML, and TXT files.
- **Vision Intelligence**: Understand diagrams, charts, and text in **Images (.png, .jpg, .webp)** using advanced vision models.
- **Deep Search Engineering**: Implements **Hybrid Retrieval** (FAISS Vector + BM25 Keyword) with **Cross-Encoder Re-ranking** for extreme precision.
- **Multi-Query Expansion**: Automatically generates search variations to maximize retrieval recall.
- **Smart Follow-ups**: AI suggests the next 3 logical research questions based on the current context.

### 🕸️ Knowledge & Visualization
- **Knowledge Map**: Generates a conceptual graph of your entire document library, identifying core themes and connections.
- **Global Research Mode**: Query your **entire database** of past uploads across all conversations simultaneously.
- **Interactive Tagging System**: 
  - **Tag Library**: Effortlessly categorize research sessions.
  - **Visual Cross-Reference**: Click a tag in the sidebar to highlight it across all related conversations.
  - **Quick Add**: Apply tags directly from the Knowledge Map or Header with a single click.

### 📊 Analytics & Management
- **Statistics Dashboard**: Track your usage with global and per-session metrics (message counts, token indicators, document stats).
- **Proactive Summarization**: Generates academic-style session titles and comprehensive analytical reports.
- **Export Capabilities**: Save your research in **Professional PDF, HTML, or JSON** formats.

### 💎 Premium User Experience
- **Glassmorphic UI**: High-fidelity design with dynamic light/dark modes and fluid transitions.
- **Real-time Citations**: Clean, UUID-free citations (e.g., `[Source: manual.pdf - Page 12]`) were implemented recently for better readability.
- **Interactive Feedback**: Copy-to-clipboard animations, thumbs up/down reactions, and real-time streaming responses.
- **Lazy Loading**: Optimized backend that loads heavy AI models only when needed, ensuring lightning-fast startup.

---

## 🛠️ Tech Stack

### Backend (Python/FastAPI)
- **Vector Engine**: FAISS (Meta)
- **Keyword Engine**: BM25
- **Re-ranking**: Cross-Encoders (`sentence-transformers`)
- **Document Processing**: PyMuPDF, MarkItDown (Microsoft), RapidOCR (Local OCR fallback)
- **AI Integration**: Perplexity AI (`sonar-pro` vision-enabled)
- **Security**: JWT Authentication + Argon2 Password Hashing

### Frontend (Modern Web)
- **Structure**: Vanilla Javascript (ES6+)
- **Styling**: Premium CSS3 (Custom Design System, Glassmorphism)
- **Components**: Marked.js (Markdown), html2pdf (Professional Exports)

---

## 🔧 Installation & Setup

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/venky038/pdf-chatbot.git
   cd pdf-chatbot
   ```

2. **Backend Infrastructure**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file in the `backend/` folder:
   ```env
   PERPLEXITY_API_KEY=your_key
   JWT_SECRET_KEY=your_secret
   ```

4. **Launch Application**:
   ```bash
   # In backend/
   uvicorn app:app --reload
   ```
   *Then open `frontend/login.html` in your browser.*

---

## 📖 Usage Workflow

1. **Analyze**: Upload any document (PDF, Word, Excel, PPT) or technical diagram.
2. **Explore**: Use the **Knowledge Map** to see how concepts connect.
3. **Query**: Ask complex questions. Use **Deep Search** for needle-in-a-haystack data.
4. **Organize**: Tag your chats and use the sidebar highlights to track research themes.
5. **Export**: Generate a final summary report for your records.

---

## 📄 License
MIT License - Developed as a high-performance research companion.
