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
import re

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

dotenv_path = find_dotenv()
load_dotenv(dotenv_path=dotenv_path)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    logger.error("PERPLEXITY_API_KEY not found!")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_stores")

# --- MODEL INITIALIZATION ---
logger.info("Loading Embedding Model...")
EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

logger.info("Loading Re-Ranker Model...")
RERANKER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class NoSearchableTextError(Exception):
    pass

# --- Helper Functions ---

def _normalize_messages(system_prompt, history, new_user_content):
    """STRICT NORMALIZATION to prevent API 400 errors."""
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

# --- 1. INTELLIGENT ROUTER ---

async def classify_intent(question: str) -> str:
    """Decides: 'chat', 'summary', or 'rag'."""
    clean_q = question.strip().lower()

    # A. Summary Intent
    summary_triggers = ["summarize", "summary", "overview", "what is this document", "what is this pdf"]
    if any(t in clean_q for t in summary_triggers) and any(x in clean_q for x in ["document", "pdf", "file", "paper", "it", "content"]):
        return "summary"
        
    # B. Chat Intent
    chat_triggers = ["hi", "hello", "thanks", "thank you", "bye", "good morning", "who are you"]
    if len(clean_q) < 30 and any(t in clean_q for t in chat_triggers):
        return "chat"
    
    # C. Default RAG
    return "rag"

# --- 2. PDF PROCESSING WITH METADATA ---

def load_pdf_documents(pdf_path: str):
    """Reads PDF page-by-page."""
    documents = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            if len(text.strip()) > 10:
                page_num = i + 1
                # Inject Page Number for better Searchability
                content_with_page = f"[Page {page_num}]\n{text}"
                
                documents.append(Document(
                    page_content=content_with_page, 
                    metadata={"page": page_num, "source": os.path.basename(pdf_path)}
                ))
        doc.close()
    except Exception as e:
        logger.exception(f"Error reading PDF {pdf_path}")
        return []
    return documents

def split_documents_preserving_metadata(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def _process_and_save_sync(pdf_path, vector_store_id):
    """Saves Indices with Metadata."""
    documents = load_pdf_documents(pdf_path)
    if not documents: raise NoSearchableTextError("No text found.")
    
    chunks = split_documents_preserving_metadata(documents)
    
    # FAISS
    vector_store = FAISS.from_documents(documents=chunks, embedding=EMBEDDINGS_MODEL)
    store_path = os.path.join(VECTOR_STORE_DIR, vector_store_id)
    vector_store.save_local(store_path)
    
    # BM25
    tokenized_corpus = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_id}_bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
        
    logger.info(f"Saved Hybrid Indices for: {vector_store_id}")
    return "\n".join([d.page_content for d in documents])

async def process_pdf_for_rag(pdf_path, vector_store_id):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _process_and_save_sync, pdf_path, vector_store_id)

# --- 3. RETRIEVAL WITH CITATIONS ---

def format_context_with_citations(docs):
    formatted_context = []
    for doc in docs:
        page_num = doc.metadata.get("page", "?")
        content = doc.page_content.replace("\n", " ") 
        formatted_context.append(f"[Source: Page {page_num}]\n{content}")
    return "\n\n".join(formatted_context)

def _get_global_context_sync(vector_store_id):
    """Global Sampler for Summaries."""
    bm25_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_id}_bm25.pkl")
    if not os.path.exists(bm25_path): return ""
    
    with open(bm25_path, "rb") as f:
        chunks = pickle.load(f)["chunks"]
    
    if len(chunks) < 50:
        return format_context_with_citations(chunks)
    
    indices = np.linspace(0, len(chunks)-1, num=50, dtype=int)
    selected_chunks = [chunks[i] for i in indices]
    
    return format_context_with_citations(selected_chunks)

def _hybrid_search_sync(queries, vector_store_id):
    """
    Hybrid Search (Vector + BM25 + Re-Ranking).
    Accepts a list of queries, but we are just passing [original_question] now.
    """
    store_path = os.path.join(VECTOR_STORE_DIR, vector_store_id)
    bm25_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_id}_bm25.pkl")
    
    if not os.path.exists(bm25_path): return ""

    vector_store = FAISS.load_local(store_path, EMBEDDINGS_MODEL, allow_dangerous_deserialization=True)
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
        bm25 = data["bm25"]
        all_chunks = data["chunks"] 

    unique_contents = set()
    combined_candidates = []

    for q in queries:
        # A. Vector
        docs = vector_store.similarity_search(q, k=15) # Increased slightly
        for d in docs:
            if d.page_content not in unique_contents:
                unique_contents.add(d.page_content)
                combined_candidates.append(d)
        
        # B. BM25
        keywords = q.lower().split()
        top_docs_bm25 = bm25.get_top_n(keywords, all_chunks, n=15) # Increased slightly
        
        for d in top_docs_bm25:
            if d.page_content not in unique_contents:
                unique_contents.add(d.page_content)
                combined_candidates.append(d)

    if not combined_candidates: return ""

    # C. Re-Rank
    original_q = queries[-1]
    pairs = [[original_q, d.page_content] for d in combined_candidates]
    scores = RERANKER.predict(pairs)
    
    scored_docs = sorted(zip(combined_candidates, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:6]]
    
    return format_context_with_citations(top_docs)

# --- 4. GENERATION ---

async def get_perplexity_response(context: str, question: str, history: list[dict], mode="rag"):
    url = "https://api.perplexity.ai/chat/completions"
    
    if mode == "chat":
        system_prompt = "You are a friendly AI assistant."
        final_content = question
    elif mode == "summary":
        system_prompt = (
            "You are an expert analyst. Provide a comprehensive summary of the document. "
            "Use the provided context chunks which are marked with [Source: Page X]. "
            "Cite the page numbers in your summary where appropriate."
        )
        final_content = f"DOCUMENT SAMPLES:\n{context}\n\nREQUEST: {question}"
    else:
        system_prompt = (
            "You are a strict RAG assistant. Answer the user's question "
            "ONLY using the provided CONTEXT.\n"
            "Rules:\n"
            "1. Each fact you state MUST be followed by a citation like [Page X].\n"
            "2. If the answer is not in the Context, say 'I cannot find this information'.\n"
            "3. Do not use external knowledge."
        )
        final_content = f"CONTEXT:\n{context}\n\nUSER QUESTION: {question}"

    messages = _normalize_messages(system_prompt, history, final_content)

    payload = {"model": "sonar-pro", "messages": messages, "stream": False}
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200: return f"⚠️ API Error ({response.status_code})."
            return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return "⚠️ Connection error."

# --- MAIN ENTRY POINT ---

async def answer_question(question: str, vector_store_id: str, history: list[dict]):
    intent = await classify_intent(question)
    context = ""
    
    if intent == "summary":
        loop = asyncio.get_event_loop()
        try:
            context = await loop.run_in_executor(None, _get_global_context_sync, vector_store_id)
        except Exception:
            context = "Error retrieving global context."

    elif intent == "rag":
        loop = asyncio.get_event_loop()
        try:
            # DIRECT CALL (No Expansion)
            # We pass the single question as a list because _hybrid_search_sync expects a list
            queries = [question] 
            context = await loop.run_in_executor(None, _hybrid_search_sync, queries, vector_store_id)
            if not context: context = "No relevant context found."
        except Exception:
            return "⚠️ Search failed."
            
    return await get_perplexity_response(context, question, history, mode=intent)

# --- Extra Helpers ---
async def generate_conversation_title(text: str):
    url = "https://api.perplexity.ai/chat/completions"
    prompt = f"Generate a single-line title (max 5 words) for this text. Do NOT use markdown. Text:\n{text[:1000]}"
    messages = [{"role": "user", "content": prompt}]
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}"}, json={"model":"sonar-pro","messages":messages})
        return resp.json()["choices"][0]["message"]["content"].strip('"').split("\n")[0]

async def generate_summary(transcript: str):
    url = "https://api.perplexity.ai/chat/completions"
    prompt = f"Summarize this chat:\n\n{transcript}"
    messages = [{"role": "user", "content": prompt}]
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}"}, json={"model":"sonar-pro","messages":messages})
        return resp.json()["choices"][0]["message"]["content"]