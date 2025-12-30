import os
import logging
import json
import pickle
import io
import base64
import asyncio
import httpx
import numpy as np
import fitz  # PyMuPDF
import shutil
from typing import AsyncGenerator, List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# Set up a dedicated logger for RAG operations to keep logs organized
logger = logging.getLogger("QueryMate.RAG")

# --- OCR ENGINE INITIALIZATION ---
# Scanned PDFs contain images, not text. We use RapidOCR to 'read' these pixels.
try:
    from rapidocr_onnxruntime import RapidOCR
    ocr_engine = RapidOCR()
    OCR_AVAILABLE = True
    logger.info("RapidOCR engine initialized. Scanned PDF support enabled.")
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("RapidOCR not found. System will only process 'selectable' text PDFs.")

class RAGService:
    """
    RAGService handles the entire 'Retrieval-Augmented Generation' lifecycle:
    1. Parsing documents (PDF/Images).
    2. Chunking and indexing into Vector/Keyword stores.
    3. Hybrid Search (Vector + BM25) with Re-ranking.
    4. AI-driven response generation with citations.
    """

    def __init__(self, api_key: str, vector_store_dir: str, upload_dir: str):
        # Sanitize API key: handles accidental 'Bearer' prefix or extra whitespace
        self.api_key = api_key.replace("Bearer ", "").strip() if api_key else ""
        self.vector_store_dir = vector_store_dir
        self.upload_dir = upload_dir
        
        logger.info(f"RAG Service Ready | Vector Storage: {vector_store_dir}")
        
        # 384-dimensional embeddings (compact and fast)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Cross-Encoder for high-quality re-ranking (decides which snippet is BEST)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Intelligent splitter to keep related sentences together
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        
        if not self.api_key:
            logger.error("PERPLEXITY_API_KEY is missing! Reasoning features will be disabled.")

    async def generate_one_liner_summary(self, text: str) -> str:
        """Asks the AI to generate a ultra-concise title for a chat session based on document content."""
        if not text: return "Document Chat"
        prompt = f"Summarize this text into a concise 5-7 word title. No quotes. No asterisks: {text[:2000]}"
        return await self._call_perplexity_simple(prompt)

    async def _call_perplexity_simple(self, prompt: str) -> str:
        """Helper for standard non-streaming completions (like title or summary generation)."""
        url = "https://api.perplexity.ai/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": "sonar", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100}
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                res.raise_for_status()
                content = res.json()["choices"][0]["message"]["content"]
                return content.strip().replace('"', '').replace('*', '')
        except Exception as e:
            logger.error(f"Perplexity simple call failed: {e}")
            return "Document Chat"

    async def process_file(self, file_path: str, vs_id: str, is_append: bool = False):
        """
        Processes a single file using batch-optimized extraction and indexing.
        """
        logger.info(f"Batch Ingestion Initiated: {os.path.basename(file_path)}")
        ext = os.path.splitext(file_path)[1].lower()
        docs = []

        if ext == ".pdf":
            docs = await self._load_pdf_batch(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
            docs = await self._load_image(file_path)
        
        if not docs:
            logger.warning(f"File yielded no indexable content: {file_path}")
            return None

        # Bulk Chunking
        logger.info(f"Batch Processing: {len(docs)} pages -> Generating chunks...")
        chunks = self.splitter.split_documents(docs)
        
        # Bulk Embedding & Indexing
        await self._save_chunks(chunks, vs_id, is_append)
        
        preview = " ".join([d.page_content for d in docs[:2]])
        return {"preview": preview[:2000]}

    async def _load_pdf_batch(self, path: str):
        """Extracts text from PDF in a batch-optimized manner with high-speed OCR fallback."""
        documents = []
        loop = asyncio.get_event_loop()
        
        def extract():
            doc = fitz.open(path)
            fname = os.path.basename(path)
            pages_data = []
            for i, page in enumerate(doc):
                text = page.get_text()
                # Store text and page object for potential OCR
                pages_data.append({"index": i, "text": text, "page": page})
            doc.close()
            return pages_data, fname

        pages_data, fname = await loop.run_in_executor(None, extract)
        
        # Optimization: Process pages with text immediately, batch the OCR for others
        for item in pages_data:
            text = item["text"]
            if len(text.strip()) < 50 and OCR_AVAILABLE:
                # Run OCR in executor for this specific page
                def ocr_page(p):
                    pix = p.get_pixmap(matrix=fitz.Matrix(2,2))
                    res, _ = ocr_engine(pix.tobytes("png"))
                    return "\n".join([line[1] for line in res]) if res else ""
                
                text = await loop.run_in_executor(None, ocr_page, item["page"])
            
            if text.strip():
                documents.append(Document(
                    page_content=text, 
                    metadata={"page": item["index"]+1, "source": fname}
                ))
        return documents

    async def _load_image(self, path: str):
        """Uses Perplexity's vision features to analyze and transcribe images."""
        fname = os.path.basename(path)
        logger.info(f"Analyzing image via AI: {fname}")
        
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": "sonar",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this image in detail. Extract any text or data visible exactly."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}]
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 200:
                    text = res.json()["choices"][0]["message"]["content"]
                    return [Document(page_content=text, metadata={"page": 1, "source": fname})]
        except Exception as e:
            logger.error(f"AI image analysis failed: {e}")
        return []

    async def _save_chunks(self, chunks, vs_id: str, is_append: bool):
        """Saves documents to both a FAISS vector index and a BM25 keyword index."""
        store_path = os.path.join(self.vector_store_dir, vs_id)
        bm25_path = os.path.join(self.vector_store_dir, f"{vs_id}_bm25.pkl")
        loop = asyncio.get_event_loop()

        def sync_save():
            # 1. Update/Create FAISS Vector Store
            if is_append and os.path.exists(store_path):
                logger.info(f"Appending {len(chunks)} chunks to existing FAISS index: {vs_id}")
                kb = FAISS.load_local(store_path, self.embeddings, allow_dangerous_deserialization=True)
                kb.add_documents(chunks)
                kb.save_local(store_path)
            else:
                logger.info(f"Creating new FAISS index: {vs_id}")
                FAISS.from_documents(chunks, self.embeddings).save_local(store_path)
            
            # 2. Update/Create BM25 Keyword Store
            all_chunks = chunks
            if is_append and os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f: 
                    data = pickle.load(f)
                    all_chunks = data["chunks"] + chunks
            
            logger.info(f"Updating BM25 Keyword Index (Total Chunks: {len(all_chunks)})")
            bm25 = BM25Okapi([c.page_content.lower().split() for c in all_chunks])
            with open(bm25_path, "wb") as f: 
                pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)

        await loop.run_in_executor(None, sync_save)
        logger.info(f"Physical indices finalized for vs_id: {vs_id}")

    async def answer_question_stream(self, question: str, vs_id: str, history: List[Dict]) -> AsyncGenerator[str, None]:
        """
        The Main Reasoning Loop:
        1. Context Search (Hybrid).
        2. Prompt Engineering (Injection of facts).
        3. Streaming Output from LLM.
        """
        loop = asyncio.get_event_loop()
        # Retrieve the most relevant paragraphs from your documents
        context = await loop.run_in_executor(None, self._hybrid_search, question, vs_id)
        
        if not context or len(context.strip()) < 10:
            logger.info("Search returned no relevant snippets. Falling back to general knowledge.")
            context = "No relevant document excerpts found."

        sys_prompt = (
            "You are a professional research assistant. "
            "Use the provided context to answer the user request. "
            "Include citations like [Source: filename - Page X] whenever you reference a fact. "
            "If the context doesn't have the answer, state that clearly."
        )
        
        # Prepare structured message list for the AI
        messages = [{"role": "system", "content": sys_prompt}]
        
        # Clean up conversation history to avoid 'user-user' consecutive roles
        clean_history = []
        for i, m in enumerate(history):
            if i == len(history) - 1 and m["role"] == "user": continue
            clean_history.append(m)
        
        messages.extend(clean_history)
        messages.append({"role": "user", "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION: {question}"})

        logger.info("Streaming response from AI...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": "sonar", "messages": messages, "stream": True}
            ) as res:
                if res.status_code != 200:
                    logger.error(f"AI Stream Error: {res.status_code}")
                    yield f"Error: The AI service returned {res.status_code}"
                    return

                async for line in res.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            token = chunk["choices"][0]["delta"].get("content", "")
                            if token: yield token
                        except: pass

    def _hybrid_search(self, query: str, vs_id: str):
        """Combines Vector (Semantic) and BM25 (Keyword) search, then reranks with a Cross-Encoder."""
        logger.info(f"Retrieval Research: '{query}'")
        store_path = os.path.join(self.vector_store_dir, vs_id)
        bm25_path = os.path.join(self.vector_store_dir, f"{vs_id}_bm25.pkl")
        
        if not os.path.exists(store_path) or not os.path.exists(bm25_path):
            logger.warning("Files for this vector store are missing from disk.")
            return ""

        # Step 1: Semantic Retrieval (finds meaning)
        kb = FAISS.load_local(store_path, self.embeddings, allow_dangerous_deserialization=True)
        v_docs = kb.similarity_search(query, k=15)
        
        # Step 2: Keyword Retrieval (finds exact words)
        with open(bm25_path, "rb") as f: 
            data = pickle.load(f)
        b_docs = data["bm25"].get_top_n(query.lower().split(), data["chunks"], n=15)
        
        # Merge and deduplicate candidates
        unique = {d.page_content: d for d in (v_docs + b_docs)}
        candidates = list(unique.values())
        logger.info(f"Candidates gathered: {len(v_docs)} semantic + {len(b_docs)} keyword. Unique: {len(candidates)}")
        
        if not candidates: return ""

        # Step 3: Re-Ranking (The Judge)
        # We send the question and all 30 snippets to a model that decides which ones ARE ACTUALLY USEFUL.
        scores = self.reranker.predict([[query, d.page_content] for d in candidates])
        best_score = max(scores) if len(scores) > 0 else 0
        logger.info(f"Top Rerank Confidence: {best_score:.4f}")
        
        # Pick the top 6 most relevant snippets to feed the AI
        top = [c for c, s in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:6]]
        
        context_blocks = []
        for d in top:
            fname = os.path.basename(d.metadata.get("source", "Unknown PDF"))
            page = d.metadata.get("page", "?")
            context_blocks.append(f"[Source: {fname} - Page {page}] {d.page_content}")
            
        return "\n\n".join(context_blocks)

    async def get_page_preview(self, vs_id: str, filename: str, page_num: int):
        """Locates a specific page and returns raw text. Now handles 'document' alias for better UX."""
        logger.info(f"Preview Request: {filename} (Pg {page_num}) for VS: {vs_id}")
        
        # Check if requested filename is the generic 'document' alias
        is_generic = filename.lower() == "document"
        target = None

        if not is_generic:
            # Try finding the specific file
            possible_path = os.path.join(self.upload_dir, f"{vs_id}_{filename}")
            if os.path.exists(possible_path):
                target = possible_path

        if not target:
            # Broad search fallback: find ANY file associated with this VS ID
            for f in os.listdir(self.upload_dir):
                if f.startswith(vs_id):
                    # If is_generic (from [Page X]), we take the first PDF match
                    # If specific filename given but path was weird, we fuzzy match
                    if is_generic or filename in f:
                        target = os.path.join(self.upload_dir, f)
                        break
        
        if not target or not os.path.exists(target): 
            logger.warning(f"Preview Target NOT FOUND: {filename}")
            return None
        
        try:
            doc = fitz.open(target)
            if 0 <= page_num - 1 < len(doc):
                text = doc[page_num - 1].get_text()
                doc.close()
                return text
            doc.close()
        except Exception as e:
            logger.error(f"Preview extraction error: {e}")
        return None

    async def generate_summary_stream(self, transcript: str) -> AsyncGenerator[str, None]:
        """Provides a total overview with citations mandated for credibility."""
        prompt = (
            "Summarize the following research session. "
            "IMPORTANT: Every major finding MUST be cited using [Page X] markers from the conversation history. "
            "Group by logical sections (Insights, Data points, Conclusions).\n\n"
            f"CHAT HISTORY:\n{transcript[:18000]}"
        )
        messages = [
            {"role": "system", "content": "You are a research assistant who prioritizes data provenance and citations."}, 
            {"role": "user", "content": prompt}
        ]
        
        logger.info("Generating analytical summary with citations...")
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": "sonar", "messages": messages, "stream": True}
            ) as res:
                async for line in res.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            token = chunk["choices"][0]["delta"].get("content", "")
                            if token: yield token
                        except: pass
