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

# --- GLOBAL VECTOR STORE MANAGEMENT ---
def get_global_vector_store_path(user_id: int) -> str:
    """Get path to user's global vector store"""
    return os.path.join(VECTOR_STORE_DIR, f"user_{user_id}_global")

def check_global_store_exists(user_id: int) -> bool:
    """Check if user has a global vector store"""
    global_path = get_global_vector_store_path(user_id)
    bm25_path = os.path.join(VECTOR_STORE_DIR, f"user_{user_id}_global_bm25.pkl")
    
    # Check if both FAISS and BM25 files exist
    faiss_exists = os.path.exists(os.path.join(global_path, "index.faiss"))
    bm25_exists = os.path.exists(bm25_path)
    
    result = faiss_exists and bm25_exists
    logger.debug(f"🔍 Global store check for user {user_id}: FAISS={faiss_exists}, BM25={bm25_exists}, Result={result}")
    return result

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

def _process_and_save_sync(pdf_path, vector_store_id, is_append=False, user_id=None):
    # DIRECT LOCAL PROCESSING - LLAMAPARSE REMOVED
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
    
    # ALSO SAVE TO GLOBAL STORE (if user_id provided)
    if user_id:
        _save_to_global_store_sync(chunks, user_id)
        
    return f"Processed {len(documents)} pages."

def _save_to_global_store_sync(chunks, user_id: int):
    """Save chunks to user's global vector store with error handling"""
    if not chunks:
        logger.warning(f"No chunks to save for user {user_id}")
        return
        
    global_path = get_global_vector_store_path(user_id)
    global_bm25_path = os.path.join(VECTOR_STORE_DIR, f"user_{user_id}_global_bm25.pkl")
    
    try:
        # Ensure directory exists
        os.makedirs(global_path, exist_ok=True)
        
        # Save FAISS to global
        faiss_index_path = os.path.join(global_path, "index.faiss")
        try:
            if os.path.exists(faiss_index_path):
                # Merge with existing
                existing = FAISS.load_local(global_path, EMBEDDINGS_MODEL, allow_dangerous_deserialization=True)
                existing.merge_from(FAISS.from_documents(chunks, EMBEDDINGS_MODEL))
                existing.save_local(global_path)
                logger.info(f"✅ Merged PDFs to user {user_id}'s global vector store")
            else:
                # Create new
                FAISS.from_documents(chunks, EMBEDDINGS_MODEL).save_local(global_path)
                logger.info(f"✅ Created user {user_id}'s global vector store")
        except Exception as e:
            logger.error(f"❌ Error saving FAISS for user {user_id}: {e}")
            raise
        
        # Save BM25 to global
        try:
            all_chunks = chunks
            if os.path.exists(global_bm25_path):
                with open(global_bm25_path, "rb") as f:
                    all_chunks = pickle.load(f)["chunks"] + chunks
                
            bm25 = BM25Okapi([c.page_content.lower().split() for c in all_chunks])
            with open(global_bm25_path, "wb") as f:
                pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)
            logger.info(f"✅ Saved BM25 index for user {user_id} ({len(all_chunks)} total chunks)")
        except Exception as e:
            logger.error(f"❌ Error saving BM25 for user {user_id}: {e}")
            raise
            
    except Exception as e:
        logger.error(f"❌ Failed to save global store for user {user_id}: {e}")
        # Don't re-raise - allow chat to continue even if global store fails

async def process_pdf_for_rag(pdf_path, vector_store_id, is_append=False, user_id=None):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _process_and_save_sync, pdf_path, vector_store_id, is_append, user_id)

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

def _hybrid_search_global_sync(question, user_id: int):
    """Search user's global vector store"""
    global_path = get_global_vector_store_path(user_id)
    global_bm25_path = os.path.join(VECTOR_STORE_DIR, f"user_{user_id}_global_bm25.pkl")
    
    if not os.path.exists(global_bm25_path): return ""
    
    vector_store = FAISS.load_local(global_path, EMBEDDINGS_MODEL, allow_dangerous_deserialization=True)
    with open(global_bm25_path, "rb") as f: data = pickle.load(f)
    
    unique_docs = {d.page_content: d for d in vector_store.similarity_search(question, k=20)}
    for d in data["bm25"].get_top_n(question.lower().split(), data["chunks"], n=20): unique_docs[d.page_content] = d
    
    candidates = list(unique_docs.values())
    if not candidates: return ""
    
    scores = RERANKER.predict([[question, d.page_content] for d in candidates])
    top_docs = [doc for doc, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:8]]
    return format_context(top_docs)

def _get_global_context_from_store_sync(user_id: int):
    """Get global context for summary from user's global store"""
    global_bm25_path = os.path.join(VECTOR_STORE_DIR, f"user_{user_id}_global_bm25.pkl")
    if not os.path.exists(global_bm25_path): return ""
    
    with open(global_bm25_path, "rb") as f: chunks = pickle.load(f)["chunks"]
    indices = np.linspace(0, len(chunks)-1, num=min(40, len(chunks)), dtype=int)
    return format_context([chunks[i] for i in indices])

# --- ANSWERING & SUMMARY ---
async def answer_question_stream(question: str, vector_store_id: str, history: list[dict], user_id: int = None) -> AsyncGenerator[str, None]:
    loop = asyncio.get_event_loop()
    
    # GENERAL CHAT MODE - No PDF loaded in this specific chat
    if not vector_store_id or vector_store_id == "null":
        # OPTION 1: Check if user has global vector store (previous PDFs)
        if user_id and check_global_store_exists(user_id):
            logger.info(f"📚 Using global vector store for user {user_id}")
            context = await loop.run_in_executor(None, _hybrid_search_global_sync, question, user_id)
            
            # STRICT RAG: Only answer if context is found
            if context:
                system_prompt = (
                    "You are a helpful assistant. Answer questions based on the Context provided below from the user's uploaded documents.\n"
                    "1. Cite pages using the format [Page X] if the answer is in the context.\n"
                    "2. If the answer is not in the context, answer generally using your own knowledge."
                )
                messages = _normalize_messages(system_prompt, history, f"CONTEXT:\n{context}\n\nUSER QUESTION: {question}")
            else:
                # Fallback: User has docs, but this query isn't about them
                # Since you requested "not pure general llm", we indicate no info found in docs.
                logger.info(f"🌐 Global search empty - No matching info in docs")
                yield "❌ I couldn't find any relevant information in your uploaded documents regarding this query."
                return
        
        # OPTION 2: Pure LLM Mode (No PDFs ever uploaded)
        else:
            logger.info(f"🌐 General Chat Mode (No Context) for user {user_id}")
            system_prompt = "You are a helpful assistant. Answer the user's questions clearly and concisely."
            messages = _normalize_messages(system_prompt, history, question)
        
        # --- SEND REQUEST ---
        try:
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
        return
    
    # RAG MODE - Specific PDF loaded in this conversation
    intent = await classify_intent(question)
    
    # If user just says "hi", don't do RAG
    if intent == "chat":
        system_prompt = "You are a helpful assistant. Be polite and concise."
        messages = _normalize_messages(system_prompt, history, question)
    else:
        context = ""
        try:
            if intent == "summary":
                context = await loop.run_in_executor(None, _get_global_context_sync, vector_store_id)
                system_prompt = "You are an expert analyst. Summarize the document content. Always cite pages using the format [Page X]."
            else:
                context = await loop.run_in_executor(None, _hybrid_search_sync, question, vector_store_id)
                system_prompt = (
                    "You are a helpful assistant. Answer the user's question based ONLY on the Context below.\n"
                    "1. CRITICAL: You MUST cite pages using the exact format: [Page X].\n"
                    "2. If the answer is missing, say 'I cannot find this information'.\n"
                )
        except Exception as e:
             logger.error(f"Context retrieval failed: {e}")
             context = ""

        if not context and intent != "chat":
             # Fallback if context is empty but intent was RAG
             yield "I cannot find relevant information in the uploaded document."
             return
             
        messages = _normalize_messages(system_prompt, history, f"CONTEXT:\n{context}\n\nUSER QUESTION: {question}")

    # Stream Response
    try:
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
    """Generate a simple, concise bullet-point summary"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Generate a simple, concise summary of the conversation and document content. Return 5-7 bullet points covering main topics discussed. Be brief - maximum 1-2 lines per bullet point."},
        {"role": "user", "content": f"Summarize this conversation in 5-7 bullet points:\n\n{transcript[:15000]}"}
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