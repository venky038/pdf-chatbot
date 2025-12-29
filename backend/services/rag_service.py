import os
import logging
import json
import pickle
import io
import base64
import asyncio
import httpx
import numpy as np
import fitz # PyMuPDF
import shutil
from typing import AsyncGenerator, List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

logger = logging.getLogger("DocuMind.RAG")

# --- RAPID OCR ---
try:
    from rapidocr_onnxruntime import RapidOCR
    ocr_engine = RapidOCR()
    OCR_AVAILABLE = True
    logger.info("RapidOCR engine initialized.")
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("RapidOCR not found. OCR features disabled.")

class RAGService:
    def __init__(self, api_key: str, vector_store_dir: str, upload_dir: str):
        # Sanitize API key: remove 'Bearer ' if present and strip whitespace
        self.api_key = api_key.replace("Bearer ", "").strip() if api_key else ""
        self.vector_store_dir = vector_store_dir
        self.upload_dir = upload_dir
        
        logger.info(f"Initializing RAG Service | Store: {vector_store_dir}")
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        
        if not self.api_key:
            logger.warning("PERPLEXITY_API_KEY is not set. Chat features will error out.")

    async def generate_one_liner_summary(self, text: str) -> str:
        if not text: return "Chat Session"
        prompt = f"Summarize this text into a concise 5-7 word title. No quotes. No asterisks: {text[:2000]}"
        return await self._call_perplexity_simple(prompt)

    async def _call_perplexity_simple(self, prompt: str) -> str:
        url = "https://api.perplexity.ai/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": "sonar", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100}
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code != 200:
                    logger.error(f"Perplexity API Error: {res.status_code} - {res.text}")
                res.raise_for_status()
                content = res.json()["choices"][0]["message"]["content"]
                return content.strip().replace('"', '').replace('*', '')
        except Exception as e:
            logger.error(f"Failed to call Perplexity: {e}")
            return "Document Chat"

    async def process_file(self, file_path: str, vs_id: str, is_append: bool = False):
        logger.info(f"Processing file: {os.path.basename(file_path)} (vs_id: {vs_id})")
        ext = os.path.splitext(file_path)[1].lower()
        docs = []

        if ext == ".pdf":
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, self._load_pdf_local, file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
            docs = await self._load_image(file_path)
        
        if not docs:
            logger.warning(f"No content extracted from {file_path}")
            return None

        logger.info(f"Extracted {len(docs)} pages/segments. Splitting into chunks...")
        chunks = self.splitter.split_documents(docs)
        await self._save_chunks(chunks, vs_id, is_append)
        
        preview = " ".join([d.page_content for d in docs[:2]])
        return {"preview": preview[:2000]}

    def _load_pdf_local(self, path: str):
        documents = []
        try:
            doc = fitz.open(path)
            fname = os.path.basename(path)
            for i, page in enumerate(doc):
                text = page.get_text()
                if len(text.strip()) < 50 and OCR_AVAILABLE:
                    # Page is likely an image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
                    res, _ = ocr_engine(pix.tobytes("png"))
                    if res:
                        # RapidOCR output format: [ [[box], "text", score], ... ]
                        text = "\n".join([line[1] for line in res])
                
                if text.strip():
                    documents.append(Document(page_content=text, metadata={"page": i+1, "source": fname, "vs_id": os.path.dirname(path)}))
            doc.close()
        except Exception as e:
            logger.error(f"PDF Extraction error on {path}: {e}")
        return documents

    async def _load_image(self, path: str):
        fname = os.path.basename(path)
        logger.info(f"Analyzing image via Perplexity: {fname}")
        
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": "sonar",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this image in detail. Extract any text or data visible."},
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
                else:
                    logger.error(f"Image analysis failed: {res.status_code} - {res.text}")
        except Exception as e:
            logger.error(f"Image analysis exception: {e}")
        return []

    async def _save_chunks(self, chunks, vs_id: str, is_append: bool):
        store_path = os.path.join(self.vector_store_dir, vs_id)
        bm25_path = os.path.join(self.vector_store_dir, f"{vs_id}_bm25.pkl")
        loop = asyncio.get_event_loop()

        def sync_save():
            # FAISS
            if is_append and os.path.exists(store_path):
                logger.debug(f"Appending to existing FAISS index at {vs_id}")
                kb = FAISS.load_local(store_path, self.embeddings, allow_dangerous_deserialization=True)
                kb.add_documents(chunks) # Using add_documents instead of merge for simplicity
                kb.save_local(store_path)
            else:
                logger.debug(f"Creating new FAISS index at {vs_id}")
                FAISS.from_documents(chunks, self.embeddings).save_local(store_path)
            
            # BM25
            all_chunks = chunks
            if is_append and os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f: 
                    data = pickle.load(f)
                    all_chunks = data["chunks"] + chunks
            
            logger.debug(f"Updating BM25 index with {len(all_chunks)} total chunks")
            bm25 = BM25Okapi([c.page_content.lower().split() for c in all_chunks])
            with open(bm25_path, "wb") as f: 
                pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)

        await loop.run_in_executor(None, sync_save)
        logger.info(f"Indices saved for {vs_id}")

    async def answer_question_stream(self, question: str, vs_id: str, history: List[Dict]) -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        context = await loop.run_in_executor(None, self._hybrid_search, question, vs_id)
        
        if not context or len(context.strip()) < 10:
            logger.info("No relevant context found in documents.")
            context = "No specific relevant excerpts found in the provided documents."

        sys_prompt = "You are a professional assistant. Answer ONLY based on the provided context. Cite pages as [Page X]."
        
        # Prepare messages: avoid consecutive same roles
        messages = [{"role": "system", "content": sys_prompt}]
        
        clean_history = []
        for i, m in enumerate(history):
            if i == len(history) - 1 and m["role"] == "user":
                continue
            clean_history.append(m)
        
        messages.extend(clean_history)
        messages.append({"role": "user", "content": f"CONTEXT FROM DOCUMENTS:\n{context}\n\nUSER QUESTION: {question}"})

        logger.info(f"Streaming answer from Perplexity (Model: sonar)")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": "sonar", "messages": messages, "stream": True}
            ) as res:
                if res.status_code != 200:
                    logger.error(f"Perplexity Stream Error: {res.status_code}")
                    yield f"Error from AI Service: {res.status_code}"
                    return

                async for line in res.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            token = chunk["choices"][0]["delta"].get("content", "")
                            if token: yield token
                        except: pass

    def _hybrid_search(self, query: str, vs_id: str):
        logger.info(f"Hybrid Search: '{query}'")
        store_path = os.path.join(self.vector_store_dir, vs_id)
        bm25_path = os.path.join(self.vector_store_dir, f"{vs_id}_bm25.pkl")
        
        if not os.path.exists(store_path) or not os.path.exists(bm25_path):
            logger.warning(f"Vector store {vs_id} not found on disk.")
            return ""

        # Vector retrieval
        kb = FAISS.load_local(store_path, self.embeddings, allow_dangerous_deserialization=True)
        v_docs = kb.similarity_search(query, k=15)
        
        # BM25 retrieval
        with open(bm25_path, "rb") as f: 
            data = pickle.load(f)
        
        b_docs = data["bm25"].get_top_n(query.lower().split(), data["chunks"], n=15)
        
        # Re-ranking
        unique = {d.page_content: d for d in (v_docs + b_docs)}
        candidates = list(unique.values())
        logger.info(f"  Retrieval: {len(v_docs)} vector, {len(b_docs)} BM25. Unique: {len(candidates)}")
        
        if not candidates: return ""

        scores = self.reranker.predict([[query, d.page_content] for d in candidates])
        best_score = max(scores) if len(scores) > 0 else 0
        logger.info(f"  Top Re-Rank Score: {best_score:.4f}")
        
        top = [c for c, s in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:6]]
        
        context_blocks = []
        for d in top:
            fname = os.path.basename(d.metadata.get("source", "Unknown PDF"))
            page = d.metadata.get("page", "?")
            context_blocks.append(f"[Source: {fname} - Page {page}] {d.page_content}")
            
        return "\n\n".join(context_blocks)

    async def get_page_preview(self, vs_id: str, filename: str, page_num: int):
        logger.info(f"Fetching preview for vs_id: {vs_id}, file: {filename}, page: {page_num}")
        
        # Files are stored as {vs_id}_{original_filename}
        target = os.path.join(self.upload_dir, f"{vs_id}_{filename}")
        
        if not os.path.exists(target):
            # Fallback check if filename format is different
            logger.warning(f"Preview: {target} not found. Checking alternatives...")
            for f in os.listdir(self.upload_dir):
                if f.startswith(vs_id) and filename in f:
                    target = os.path.join(self.upload_dir, f)
                    break
        
        if not os.path.exists(target):
            logger.error(f"Preview failed: File {filename} not found for {vs_id}")
            return None
        
        try:
            doc = fitz.open(target)
            if 0 <= page_num - 1 < len(doc):
                text = doc[page_num - 1].get_text()
                doc.close()
                return text
            doc.close()
        except Exception as e:
            logger.error(f"Error reading preview for {target}: {e}")
        return None

    async def generate_summary_stream(self, transcript: str) -> AsyncGenerator[str, None]:
        prompt = f"Provide a high-level professional summary of this document interaction:\n\n{transcript[:10000]}"
        messages = [{"role": "system", "content": "You are an expert summarizer."}, {"role": "user", "content": prompt}]
        
        logger.info("Generating streaming summary...")
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
