import os
import logging
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import asyncio
import pytesseract
from PIL import Image
import io

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)
from dotenv import load_dotenv, find_dotenv

# Find the .env file (searches parent directories)
dotenv_path = find_dotenv() 

# Load the .env file from the specified path
load_dotenv(dotenv_path=dotenv_path)
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY: logger.error("PERPLEXITY_API_KEY not found!")
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_stores")
EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# --- End Setup ---

async def generate_summary(transcript: str):
    """Asks Perplexity to generate a summary of a conversation."""
    logger.info("Querying Perplexity API to generate a summary...")
    url = "https://api.perplexity.ai/chat/completions"
    prompt = f"""Based on the following conversation transcript, please provide a concise summary.
    The summary should capture the main questions asked and the key answers or findings.
    Format the summary as a short paragraph or a few bullet points.

    Transcript:
    {transcript}"""
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, lambda: requests.post(url, headers=headers, json=payload, timeout=60))
        response.raise_for_status()
        data = response.json()
        summary = data["choices"][0]["message"]["content"]
        logger.info("Generated summary successfully.")
        return summary
    except Exception as e:
        logger.exception("Error generating summary with Perplexity")
        return "⚠️ Sorry, something went wrong while generating the summary."

async def generate_conversation_title(text: str):
    """Asks Perplexity to generate a short title for a text."""
    logger.info("Querying Perplexity API to generate a title...")
    url = "https://api.perplexity.ai/chat/completions"
    snippet = text[:3000]
    prompt = f"""Based on the following text, please generate a very short, concise title (5 words or less) for this document.
    Do not include quotes or labels like "Title:". Just return the title itself.
    Text:\n{snippet}"""
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "You are a title generator."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, lambda: requests.post(url, headers=headers, json=payload, timeout=60))
        response.raise_for_status()
        data = response.json()
        title = data["choices"][0]["message"]["content"].strip().strip('"')
        logger.info(f"Generated title: {title}")
        return title
    except Exception as e:
        logger.exception("Error generating title with Perplexity")
        return "New Chat" # Fallback title

def get_pdf_text(pdf_path: str) -> str:
    """Reads text from PDF, with OCR fallback."""
    text = ""
    logger.info(f"Reading PDF file with PyMuPDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            text += page.get_text()
        if len(text.strip()) < 100:
            logger.warning(f"PyMuPDF found little text ({len(text)} chars). Attempting OCR...")
            ocr_text = ""
            for page_num, page in enumerate(doc):
                logger.info(f"Running OCR on page {page_num + 1}...")
                pix = page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                ocr_text_from_page = pytesseract.image_to_string(img)
                ocr_text += ocr_text_from_page + "\n"
            text = ocr_text
        doc.close()
    except Exception as e:
        logger.exception(f"Error while reading PDF {pdf_path}")
        return ""
    logger.info(f"Extracted {len(text)} characters from PDF.")
    return text

def get_text_chunks(text: str):
    """Splits text into overlapping chunks."""
    logger.info("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    logger.info(f"Text split into {len(chunks)} chunks.")
    return chunks

def get_vector_store(chunks):
    """Embeds chunks and stores them in FAISS."""
    logger.info("Creating FAISS vector store...")
    try:
        vector_store = FAISS.from_texts(texts=chunks, embedding=EMBEDDINGS_MODEL)
        logger.info("FAISS vector store created successfully.")
        return vector_store
    except Exception as e:
        logger.exception("Error creating FAISS vector store")
        return None

async def get_perplexity_response(context: str, question: str, history: list[dict]):
    """Queries Perplexity API with context, question, AND chat history."""
    logger.info("Querying Perplexity API with chat history...")
    url = "https://api.perplexity.ai/chat/completions"
    system_prompt = "You are a helpful assistant. Answer questions based on the provided context and our previous conversation. Be clear and concise."
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"""Context:\n{context}\n---\nQuestion: {question}\nBased only on the context above and our previous conversation, answer the question."""
    })
    payload = {"model": "sonar-pro", "messages": messages}
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, lambda: requests.post(url, headers=headers, json=payload, timeout=60))
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        logger.info("Got response from Perplexity API.")
        return answer
    except Exception as e:
        logger.exception("Error querying Perplexity API")
        return "⚠️ Sorry, something went wrong while fetching the response."

async def process_pdf_for_rag(pdf_path, vector_store_id):
    """Loads a PDF, splits, embeds, saves, and returns the extracted text."""
    logger.info(f"Starting RAG processing for vector store: {vector_store_id}")
    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(None, _process_and_save_sync, pdf_path, vector_store_id)
        return text
    except Exception as e:
        logger.exception(f"Failed to process and save for {vector_store_id}")
        return None

def _process_and_save_sync(pdf_path, vector_store_id):
    """Synchronous helper for PDF processing. Returns extracted text."""
    text = get_pdf_text(pdf_path)
    if not text or len(text.strip()) == 0:
        logger.error("Failed to extract text.")
        raise ValueError("Failed to extract text from PDF.")
    chunks = get_text_chunks(text)
    if not chunks:
        logger.error("Failed to create text chunks.")
        raise ValueError("Failed to create text chunks from PDF.")
    vector_store = get_vector_store(chunks)
    if not vector_store:
        logger.error("Failed to create vector store.")
        raise ValueError("Failed to create vector store.")
    store_path = os.path.join(VECTOR_STORE_DIR, vector_store_id)
    vector_store.save_local(store_path)
    logger.info(f"Vector store saved for {vector_store_id} at {store_path}")
    return text

async def answer_question(question: str, vector_store_id: str, history: list[dict]):
    """LOADS the vector store and answers the question."""
    store_path = os.path.join(VECTOR_STORE_DIR, vector_store_id)
    if not os.path.exists(store_path):
        logger.warning(f"No vector store found for: {vector_store_id}")
        return "⚠️ Vector store not found. Please re-upload the PDF."
    logger.info(f"Answering question for: {vector_store_id}")
    loop = asyncio.get_event_loop()
    try:
        context = await loop.run_in_executor(None, _load_and_search_sync, question, store_path)
        return await get_perplexity_response(context, question, history)
    except Exception as e:
        logger.exception(f"Error during load/search for {vector_store_id}")
        return "⚠️ Error retrieving context from your document."

def _load_and_search_sync(question, store_path):
    """Synchronous helper for loading FAISS and searching."""
    logger.info(f"Loading FAISS index from {store_path}")
    vector_store = FAISS.load_local(store_path, EMBEDDINGS_MODEL, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context