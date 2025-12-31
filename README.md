# QueryMate - Intelligent Document & Image Research Assistant

QueryMate is a premium, feature-rich RAG (Retrieval-Augmented Generation) application that allows users to chat with their documents (PDFs and Images) using advanced AI. It features a stunning glassmorphic UI, real-time citation rendering, and multi-modal support.

## 🚀 Key Features

- **Multi-modal RAG**: Chat with both PDFs and Images (.png, .jpg, .webp).
- **Vision Integration**: Uses Perplexity's `sonar-pro` vision capabilities to understand diagrams, charts, and text in images.
- **Local OCR Fallback**: Integrated `RapidOCR` for local text extraction when vision APIs are unavailable.
- **Hybrid Retrieval**: Combines FAISS vector search with BM25 keyword matching for superior context retrieval.
- **Automatic Summarization**: Automatically generates concise one-line conversation titles based on document content.
- **Real-time Citations**: Displays clickable page-pills that show exactly where the information was found.
- **Premium UI**: Modern glassmorphic design with intuitive light/dark mode support.
- **Smart Conversations**: Message feedback (thumbs up/down), copy to clipboard, and session management.

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Vector Store**: FAISS
- **Retrieval**: BM25 (Rank-BM25)
- **AI Models**: 
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  - Vision/Chat: Perplexity AI (`sonar-pro`)
  - OCR: `RapidOCR` (ONNX Runtime)
- **Database**: SQLAlchemy (SQLite)
- **File Handling**: PyMuPDF (Fitz), Aiofiles

### Frontend
- **Logic**: Vanilla JavaScript (ES6+)
- **Styling**: Modern CSS3 (Variables, Glassmorphism, Responsive Design)
- **Rendering**: Marked.js (Markdown), html2pdf (Export)

## 📋 Prerequisites

- Python 3.9+
- [Perplexity API Key](https://docs.perplexity.ai/)

## 🔧 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/querymate.git
   cd querymate
   ```

2. **Backend Setup**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Create a `.env` file in the `backend` directory:
   ```env
   PERPLEXITY_API_KEY=your_api_key_here
   SECRET_KEY=your_jwt_secret_key
   ```

4. **Run the Application**:
   ```bash
   # From the backend directory
   uvicorn app:app --reload
   ```

5. **Access the Frontend**:
   Open `frontend/index.html` in your browser (or use a Live Server).

## 📖 Usage

1. **Sign Up / Login**: Create an account to save your conversations.
2. **Upload**: Drag and drop or click the "Upload File" button to add a PDF or Image.
3. **Chat**: Ask questions like "What are the key takeaways from this chart?" or "Summarize the section on Docker volumes."
4. **Citations**: Click on the `Page 1` pills to see the source text used for the answer.
5. **Manage**: Use the sidebar to search conversations or delete them.

## 📄 License

MIT License - feel free to use this project for your own learning or applications!
