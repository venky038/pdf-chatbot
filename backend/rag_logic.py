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
from typing import AsyncGenerator

# --- NEW IMPORTS FOR LLAMAPARSE ---
try:
    from llama_parse import LlamaParse
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

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
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not PERPLEXITY_API_KEY: logger.critical("PERPLEXITY_API_KEY is missing!")
if not LLAMA_CLOUD_API_KEY: logger.warning("LLAMA_CLOUD_API_KEY is missing. LlamaParse will be skipped.")

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

# --- PDF PROCESSING STRATEGIES ---

# 1. SMART HYBRID LOADER (Local Fallback)
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
            
            # Smart OCR Trigger: Low text density + images present
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

# 2. LLAMAPARSE LOADER (Cloud Primary)
def load_pdf_llamaparse(pdf_path: str):
    """Primary method: Uses LlamaCloud API"""
    if not LLAMA_AVAILABLE or not LLAMA_CLOUD_API_KEY:
        raise ImportError("LlamaParse not configured.")
    
    logger.info(f"☁️ [LlamaParse] Uploading {os.path.basename(pdf_path)} to LlamaCloud...")
    
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",  # Best for preserving tables
        verbose=True
    )
    
    # LlamaParse returns a list of objects
    llama_docs = parser.load_data(pdf_path)
    
    final_docs = []
    filename = os.path.basename(pdf_path)
    
    for i, doc in enumerate(llama_docs):
        # We assume LlamaParse returns pages roughly in order. 
        # Metadata handling depends on response structure, but simple indexing works for most PDFs.
        page_num = i + 1
        content = doc.text
        
        if len(content.strip()) > 0:
            final_docs.append(Document(
                page_content=f"[Page {page_num}] {content}",
                metadata={"page": page_num, "source": filename}
            ))
            
    logger.info(f"☁️ [LlamaParse] Success! Extracted {len(final_docs)} segments.")
    return final_docs


def _process_and_save_sync(pdf_path, vector_store_id, is_append=False):
    # STRATEGY SELECTION
    documents = []
    
    # Try LlamaParse First
    try:
        documents = load_pdf_llamaparse(pdf_path)
    except Exception as e:
        logger.warning(f"⚠️ LlamaParse failed or quota exceeded ({e}). Switching to Hybrid Local.")
        documents = [] # Reset
    
    # Fallback to Local Hybrid
    if not documents:
        documents = load_pdf_hybrid_local(pdf_path)
        
    if not documents: raise NoSearchableTextError("No text found in document.")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    store_path = os.path.join(VECTOR_STORE_DIR, vector_store_id)
    bm25_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_id}_bm25.pkl")
    
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
        
    return f"Processed {len(documents)} pages."

async def process_pdf_for_rag(pdf_path, vector_store_id, is_append=False):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _process_and_save_sync, pdf_path, vector_store_id, is_append)

# --- RETRIEVAL & SEARCH ---
def format_context(docs):
    formatted = []
    for doc in docs:
        p = doc.metadata.get("page", "?")
        # Clean up filename if it has UUID prefix
        full_src = doc.metadata.get("source", "doc")
        src = full_src.split("_", 1)[1] if "_" in full_src else full_src
        
        formatted.append(f"[Source: {src} - Page {p}]\n{doc.page_content.replace(chr(10), ' ')}")
    return "\n\n".join(formatted)

def _get_global_context_sync(vector_store_id):
    bm25_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_id}_bm25.pkl")
    if not os.path.exists(bm25_path): return ""
    with open(bm25_path, "rb") as f: chunks = pickle.load(f)["chunks"]
    # Sample chunks for summary
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

# --- ANSWERING & SUMMARY ---

async def answer_question_stream(question: str, vector_store_id: str, history: list[dict]) -> AsyncGenerator[str, None]:
    intent = await classify_intent(question)
    
    if intent == "chat":
        yield "Hello! How can I help you with your documents today?"
        return

    loop = asyncio.get_event_loop()
    context = ""
    
    try:
        if intent == "summary":
            context = await loop.run_in_executor(None, _get_global_context_sync, vector_store_id)
            system_prompt = "You are an expert analyst. Summarize the document content. Always cite pages using the format [Page X]."
        else:
            context = await loop.run_in_executor(None, _hybrid_search_sync, question, vector_store_id)
            system_prompt = (
                "You are a helpful assistant. Answer the user's question based ONLY on the Context below.\n"
                "1. CRITICAL: You MUST cite pages using the exact format: [Page X] (e.g., [Page 4], [Page 12]).\n"
                "2. Do not use simple numbers like [1]. Use [Page 1].\n"
                "3. If the answer is missing, say 'I cannot find this information'."
            )

        if not context:
            yield "I cannot find relevant information in the document."
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

# --- FIXED SUMMARY FUNCTION ---
async def generate_summary(transcript: str):
    """
    Generates a real summary using Perplexity API based on the chat transcript/context.
    """
    logger.info("Generating real summary via API...")
    
    # We use a simplified prompt to summarize the discussion or the content found so far
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Provide a concise, professional summary of the following conversation/document content."},
        {"role": "user", "content": f"Summarize this interaction and key information found:\n\n{transcript[:15000]}"} # Limit char count
    ]
    
    url = "https://api.perplexity.ai/chat/completions"
    payload = {"model": "sonar-pro", "messages": messages, "stream": False}
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"Summary API Error: {response.text}")
                return "Failed to generate summary from AI provider."
    except Exception as e:
        logger.error(f"Summary Exception: {e}")
        return "An error occurred while generating the summary."