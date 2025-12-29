import os
import logging
import fitz  # PyMuPDF
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import httpx
import asyncio
import json
import pickle
import numpy as np
import io
import base64
from typing import AsyncGenerator

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    handlers=[logging.FileHandler("server.log", encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
dotenv_path = find_dotenv()
load_dotenv(dotenv_path=dotenv_path)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not PERPLEXITY_API_KEY: logger.critical("PERPLEXITY_API_KEY is missing!")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_stores")

# --- RAPID OCR SETUP ---
try:
    from rapidocr_onnxruntime import RapidOCR
    ocr_engine = RapidOCR()
    OCR_AVAILABLE = True
    logger.info("✅ RapidOCR loaded.")
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("⚠️ RapidOCR not found.")

# --- MODELS ---
try:
    logger.info("Loading AI Models...")
    EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    RERANKER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    logger.info("✅ Models loaded.")
except Exception as e:
    logger.critical(f"❌ Failed to load models: {e}")
    raise e

class NoSearchableTextError(Exception): pass

# --- HELPER FUNCTIONS ---
def _normalize_messages(system_prompt, history, new_user_content):
    final_messages = [{"role": "system", "content": system_prompt}]
    processed_history = [m for m in history if m.get("role") in ["user", "assistant"]]
    for msg in processed_history:
        role = msg["role"]
        content = msg.get("content", "").strip()
        if not content: continue 
        if final_messages[-1]["role"] == role:
            final_messages[-1]["content"] += f"\n\n{content}"
        else:
            final_messages.append({"role": role, "content": content})
    if final_messages[-1]["role"] == "user":
        final_messages[-1]["content"] += f"\n\n---\n{new_user_content}"
    else:
        final_messages.append({"role": "user", "content": new_user_content})
    return final_messages

async def classify_intent(question: str) -> str:
    clean_q = question.strip().lower()
    if any(t in clean_q for t in ["summarize", "summary", "overview", "what is this document"]): return "summary"
    if len(clean_q) < 20 and any(t in clean_q for t in ["hi", "hello", "thanks", "bye"]): return "chat"
    return "rag"

async def generate_one_liner_summary(text_content: str) -> str:
    """Generates a one-line summary of the document content."""
    if not text_content: return "New Chat"
    
    prompt = (
        "Based on the following document excerpt, generate a concise one-line title (max 5-7 words). "
        "The title should be descriptive of the core subject. "
        "Do not use introductory phrases. Just the title text:\n\n"
        f"{text_content[:3000]}"
    )
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            summary = data["choices"][0]["message"]["content"].strip().replace('"', '').replace('.', '').replace('*', '')
            return summary if summary else "New Document"
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return "Processed Document"

# --- PDF PROCESSING (LOCAL HYBRID ONLY) ---
def extract_text_with_rapidocr(page):
    if not OCR_AVAILABLE: return ""
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        result, _ = ocr_engine(img_bytes)
        if not result: return ""
        return "\n".join([line[1] for line in result])
    except Exception: return ""

def load_pdf_hybrid_local(pdf_path: str):
    """Fallback method: Uses PyMuPDF + RapidOCR"""
    documents = []
    try:
        doc = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path)
        logger.info(f"📖 [Hybrid] Processing {filename} locally...")
        
        for i, page in enumerate(doc):
            page_num = i + 1
            text = page.get_text()
            
            if len(text.strip()) < 50 and len(page.get_images()) > 0:
                ocr_text = extract_text_with_rapidocr(page)
                if len(ocr_text) > len(text): text = ocr_text
            
            if len(text.strip()) > 10:
                documents.append(Document(
                    page_content=f"[Page {page_num}] {text}", 
                    metadata={"page": page_num, "source": filename}
                ))
        doc.close()
    except Exception as e:
        logger.error(f"❌ Hybrid Load Error: {e}")
    return documents

def encode_image(image_path: str) -> str | None:
    """Encodes an image file to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

async def extract_text_with_perplexity_vision(image_path: str):
    """Uses Perplexity Vision API to extract text & describe images."""
    try:
        base64_image = encode_image(image_path)
        if not base64_image: return None

        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this image. If it contains charts or diagrams, describe them in detail. Provide a technical summary of the content."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 2048
        }
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"Perplexity Vision API Error: {response.status_code}")
                return None
    except Exception as e:
        logger.error(f"Perplexity Vision Exception: {e}")
        return None

def extract_text_from_image_local(image_path: str):
    """OCR fallback using RapidOCR"""
    if not OCR_AVAILABLE: return ""
    try:
        result, _ = ocr_engine(image_path)
        if not result: return ""
        return "\n".join([line[1] for line in result])
    except Exception as e:
        logger.error(f"Local OCR Error: {e}")
        return ""

async def load_image_for_rag(image_path: str):
    """Process image using Perplexity Vision (primary) or local OCR (fallback)"""
    documents = []
    filename = os.path.basename(image_path)
    logger.info(f"🖼️ Processing image {filename}...")
    
    # Try Perplexity Vision first (now async)
    text = await extract_text_with_perplexity_vision(image_path)
    
    # Fallback to local OCR if Perplexity fails
    if not text:
        logger.info(f"  ⏭️ Perplexity Vision failed. Falling back to local OCR...")
        text = extract_text_from_image_local(image_path)
    
    if text and len(text.strip()) > 10:
        documents.append(Document(
            page_content=f"[Image: {filename}]\n{text}",
            metadata={"page": 1, "source": filename}
        ))
    return documents

async def _process_and_save_async(file_path, vector_store_id, is_append=False):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        # PyMuPDF processing is CPU bound, run in thread
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(None, load_pdf_hybrid_local, file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        documents = await load_image_for_rag(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
        
    if not documents: raise NoSearchableTextError("No text found in document.")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    store_path = os.path.join(VECTOR_STORE_DIR, vector_store_id)
    bm25_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_id}_bm25.pkl")
    
    # Run heavy computation in executor
    loop = asyncio.get_event_loop()
    def compute_and_save():
        # Save FAISS
        if is_append and os.path.exists(store_path):
            existing = FAISS.load_local(store_path, EMBEDDINGS_MODEL, allow_dangerous_deserialization=True)
            existing.merge_from(FAISS.from_documents(chunks, EMBEDDINGS_MODEL))
            existing.save_local(store_path)
        else:
            FAISS.from_documents(chunks, EMBEDDINGS_MODEL).save_local(store_path)
        
        # Save BM25
        all_chunks = chunks
        if is_append and os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f: all_chunks = pickle.load(f)["chunks"] + chunks
                
        bm25 = BM25Okapi([c.page_content.lower().split() for c in all_chunks])
        with open(bm25_path, "wb") as f: pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)
        
        preview = " ".join([d.page_content for d in documents[:2]])
        return {"status": "success", "preview": preview[:5000]}

    return await loop.run_in_executor(None, compute_and_save)

async def process_file_for_rag(file_path, vector_store_id, is_append=False):
    return await _process_and_save_async(file_path, vector_store_id, is_append)

# --- RETRIEVAL & SEARCH (Same as before) ---
def format_context(docs):
    formatted = []
    for doc in docs:
        p = doc.metadata.get("page", "?")
        full_src = doc.metadata.get("source", "doc")
        src = full_src.split("_", 1)[1] if "_" in full_src else full_src
        formatted.append(f"[Source: {src} - Page {p}]\n{doc.page_content.replace(chr(10), ' ')}")
    return "\n\n".join(formatted)

def _get_global_context_sync(vector_store_id):
    bm25_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_id}_bm25.pkl")
    if not os.path.exists(bm25_path): return ""
    with open(bm25_path, "rb") as f: chunks = pickle.load(f)["chunks"]
    indices = np.linspace(0, len(chunks)-1, num=min(40, len(chunks)), dtype=int)
    return format_context([chunks[i] for i in indices])

def _hybrid_search_sync(question, vector_store_id):
    store_path = os.path.join(VECTOR_STORE_DIR, vector_store_id)
    bm25_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_id}_bm25.pkl")
    if not os.path.exists(bm25_path): return ""
    vector_store = FAISS.load_local(store_path, EMBEDDINGS_MODEL, allow_dangerous_deserialization=True)
    with open(bm25_path, "rb") as f: data = pickle.load(f)
    unique_docs = {d.page_content: d for d in vector_store.similarity_search(question, k=20)}
    for d in data["bm25"].get_top_n(question.lower().split(), data["chunks"], n=20): unique_docs[d.page_content] = d
    candidates = list(unique_docs.values())
    if not candidates: return ""
    scores = RERANKER.predict([[question, d.page_content] for d in candidates])
    top_docs = [doc for doc, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:8]]
    return format_context(top_docs)

# --- ANSWERING & SUMMARY (Same as before) ---
async def answer_question_stream(question: str, vector_store_id: str, history: list[dict]) -> AsyncGenerator[str, None]:
    intent = await classify_intent(question)
    # PRIVATE MODE: Do not answer general questions - only answer from uploaded PDFs
    if intent == "chat":
        yield "I can only answer questions related to your uploaded documents (PDFs or Images). Please ask about the content in your files."
        return
    loop = asyncio.get_event_loop()
    context = ""
    try:
        if intent == "summary":
            context = await loop.run_in_executor(None, _get_global_context_sync, vector_store_id)
            system_prompt = "You are an expert analyst. Summarize the document content. Always cite pages using the format [Page X]. IMPORTANT: Only answer based on the provided document context."
        else:
            context = await loop.run_in_executor(None, _hybrid_search_sync, question, vector_store_id)
            system_prompt = (
                "You are a helpful assistant. Answer the user's question based ONLY on the Context below.\n"
                "1. CRITICAL: You MUST cite pages using the exact format: [Page X].\n"
                "2. If the answer is missing, say 'I cannot find this information'.\n"
                "3. IMPORTANT: Do NOT answer questions about topics not covered in the provided context."
            )
        if not context:
            yield "I cannot find relevant information in the uploaded document. Please check if the document contains information related to your question."
            return
        messages = _normalize_messages(system_prompt, history, f"CONTEXT:\n{context}\n\nUSER QUESTION: {question}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", "https://api.perplexity.ai/chat/completions", 
                                     headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}, 
                                     json={"model": "sonar-pro", "messages": messages, "stream": True}) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content: yield content
                        except: pass
    except Exception as e:
        logger.error(f"Stream Error: {e}")
        yield "⚠️ Error processing request."

async def generate_summary(transcript: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Provide a concise, professional summary of the document content and conversation."},
        {"role": "user", "content": f"Summarize the following conversation and document content:\n\n{transcript[:15000]}"}
    ]
    url = "https://api.perplexity.ai/chat/completions"
    payload = {"model": "sonar-pro", "messages": messages, "stream": False}
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else: return "Failed to generate summary."
    except Exception: return "An error occurred during summary."

async def generate_summary_stream(transcript: str) -> AsyncGenerator[str, None]:
    """Stream summary response for full conversation"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Provide a concise, professional summary of the document content and entire conversation."},
        {"role": "user", "content": f"Summarize the following entire conversation and document content:\n\n{transcript[:15000]}"}
    ]
    url = "https://api.perplexity.ai/chat/completions"
    payload = {"model": "sonar-pro", "messages": messages, "stream": True}
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content: yield content
                        except: pass
    except Exception as e:
        logger.error(f"Summary Stream Error: {e}")
        yield f"⚠️ Error generating summary: {str(e)}"