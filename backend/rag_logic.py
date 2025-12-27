import os
import logging
import fitz  # PyMuPDF
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import httpx
import asyncio
import json
import pickle
import numpy as np

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
    summary_triggers = ["summarize", "summary", "overview", "what is this document"]
    if any(t in clean_q for t in summary_triggers) and any(x in clean_q for x in ["document", "pdf", "file", "paper", "it", "content"]):
        logger.info(f"Router: Classified as 'summary' (Global Context)")
        return "summary"
        
    # B. Chat Intent
    chat_triggers = ["hi", "hello", "thanks", "thank you", "bye", "good morning", "who are you"]
    if len(clean_q) < 30 and any(t in clean_q for t in chat_triggers):
        logger.info(f"Router: Classified as 'chat'")
        return "chat"
    
    # C. Default RAG
    logger.info(f"Router: Classified as 'rag'")
    return "rag"

async def generate_query_variations(question: str):
    """
    MULTI-QUERY RETRIEVAL: Generates 3 search variations.
    UPDATED PROMPT: Explicitly asks for technical formatting (IDs, Numbers).
    """
    logger.info("Generating query expansions...")
    url = "https://api.perplexity.ai/chat/completions"
    
    # Smarter prompt to catch "Question 4" -> "Question Number : 4"
    system_prompt = (
        "You are a technical search expert. Break down the user's question into 3 distinct search queries "
        "to find exact matches in a PDF document.\n"
        "Guidelines:\n"
        "1. If the user asks for 'Question 4', generate queries like 'Question Number : 4', 'Question Id', 'Q.No. 4'.\n"
        "2. If the user asks multiple questions (e.g. 'A and B'), split them into separate queries for A and B.\n"
        "3. Return ONLY a JSON list of strings. Example: [\"query1\", \"query2\"]"
    )
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]

    payload = {"model": "sonar-pro", "messages": messages, "temperature": 0.1}
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                # Clean markdown
                content = content.replace("```json", "").replace("```", "").strip()
                variations = json.loads(content)
                if isinstance(variations, list):
                    # Ensure original question is included
                    variations.append(question)
                    logger.info(f"Generated {len(variations)} variations: {variations}")
                    return variations[:5] # Allow up to 5 variations
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
    
    return [question]

# --- PDF Processing ---

def get_pdf_text(pdf_path: str) -> str:
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception:
        return ""
    return text if len(text.strip()) >= 50 else ""

def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def _process_and_save_sync(pdf_path, vector_store_id):
    """Saves FAISS and BM25 indices."""
    text = get_pdf_text(pdf_path)
    if not text: raise NoSearchableTextError("No text found.")
    
    chunks = get_text_chunks(text)
    
    # 1. FAISS
    vector_store = FAISS.from_texts(texts=chunks, embedding=EMBEDDINGS_MODEL)
    store_path = os.path.join(VECTOR_STORE_DIR, vector_store_id)
    vector_store.save_local(store_path)
    
    # 2. BM25
    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_id}_bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
        
    logger.info(f"Saved Hybrid Indices for: {vector_store_id}")
    return text

async def process_pdf_for_rag(pdf_path, vector_store_id):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _process_and_save_sync, pdf_path, vector_store_id)

# --- 2. RETRIEVAL STRATEGIES ---

def _get_global_context_sync(vector_store_id):
    """GLOBAL SAMPLER for Summaries."""
    bm25_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_id}_bm25.pkl")
    if not os.path.exists(bm25_path): return ""
    
    with open(bm25_path, "rb") as f:
        chunks = pickle.load(f)["chunks"]
    
    if len(chunks) < 50: return "\n\n".join(chunks)
    
    indices = np.linspace(0, len(chunks)-1, num=50, dtype=int)
    selected_chunks = [chunks[i] for i in indices]
    
    return "\n\n".join(selected_chunks)

def _hybrid_search_sync(queries, vector_store_id):
    """
    Multi-Query Hybrid Search.
    Retrieved candidates = (Vector_k10 + BM25_k10) * Num_Queries
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

    # Iterate through ALL variations
    for q in queries:
        # A. Vector
        docs = vector_store.similarity_search(q, k=10)
        for d in docs:
            if d.page_content not in unique_contents:
                unique_contents.add(d.page_content)
                combined_candidates.append(d.page_content)
        
        # B. BM25
        keywords = q.lower().split()
        chunks = bm25.get_top_n(keywords, all_chunks, n=10)
        for c in chunks:
            if c not in unique_contents:
                unique_contents.add(c)
                combined_candidates.append(c)

    if not combined_candidates: return ""

    logger.info(f"Hybrid Pool: {len(combined_candidates)} candidates from {len(queries)} queries.")

    # C. Re-Rank
    # We re-rank against the ORIGINAL user question (last in the list)
    # This ensures that even if a sub-query found the chunk, the chunk is scored 
    # based on how well it answers the USER'S actual need.
    original_q = queries[-1]
    pairs = [[original_q, c] for c in combined_candidates]
    scores = RERANKER.predict(pairs)
    
    scored_docs = sorted(zip(combined_candidates, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:6]] # Increase to Top 6
    
    return "\n\n".join(top_docs)

# --- 3. GENERATION ---

async def get_perplexity_response(context: str, question: str, history: list[dict], mode="rag"):
    url = "https://api.perplexity.ai/chat/completions"
    
    if mode == "chat":
        system_prompt = "You are a friendly AI assistant."
        final_content = question
    elif mode == "summary":
        system_prompt = (
            "You are an expert analyst. Provide a comprehensive summary of the document. "
            "Cover all major sections."
        )
        final_content = f"DOCUMENT SAMPLES:\n{context}\n\nREQUEST: {question}"
    else:
        # STRICT ANTI-HALLUCINATION PROMPT
        system_prompt = (
            "You are a strict RAG assistant. You must answer the user's question "
            "ONLY using the information provided in the CONTEXT below.\n"
            "Rules:\n"
            "1. If the answer is not in the Context, explicitly say 'I cannot find this information in the document'.\n"
            "2. DO NOT use external knowledge (e.g., do not mention World Cup winners unless in the PDF).\n"
            "3. If the user asks multiple questions, answer ALL of them based on the context."
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
            # 1. Generate Variations
            queries = await generate_query_variations(question)
            # 2. Hybrid Search for ALL variations
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