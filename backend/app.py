import os
import uuid
import hashlib
import logging
import asyncio
import aiofiles
import secrets
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

import models
import schemas
import auth
from database import SessionLocal, engine
from services.auth_service import AuthService
from services.conversation_service import ConversationService
from services.message_service import MessageService
from services.rag_service import RAGService

# --- LOGGING CONFIGURATION ---
# We use a dual-logging system: 
# 1. A physical 'server.log' file for long-term auditing.
# 2. A 'StreamHandler' for real-time monitoring in your terminal.
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log")

# Setup formatting and UTF-8 encoding (crucial for Windows terminal compatibility)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger("QueryMate.Main")
logger.info(f"System Initialized. Logs stored at: {LOG_FILE}")

# --- APPLICATION SETUP ---
app = FastAPI(
    title="QueryMate AI API",
    description="The core engine for intelligent document research, high-speed vector retrieval, and AI streaming."
)

# Physically create database tables based on models.py if they don't exist
models.Base.metadata.create_all(bind=engine)

# --- DIRECTORY CONFIGURATION ---
# We unify storage in the project root to keep the backend folder clean.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_stores")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# Serve the 'uploads' folder as static files so the UI can link directly to PDFs
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# --- DEPENDENCY INJECTION ---
def get_db():
    """Generates a new database session for every request and ensures it is closed after."""
    db = SessionLocal()
    try: 
        yield db
    finally: 
        db.close()

async def get_current_user(token: str = Depends(auth.oauth2_scheme), db: Session = Depends(get_db)):
    """Auth Guard: Validates the JWT token and returns the current user object."""
    return await auth.get_current_user(token, db)

# --- SERVICE INITIALIZATION ---
# Services are stateful objects that handle specific logic (Database, File Cleanup, RAG)
conv_service = ConversationService(UPLOAD_DIR, VECTOR_DIR)
msg_service = MessageService()
rag_service = RAGService(os.getenv("PERPLEXITY_API_KEY"), VECTOR_DIR, UPLOAD_DIR)

# --- CORS MIDDLEWARE ---
# Security configuration to allow the frontend (even when opened via file://) to talk to the API.
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "null",  # Essential for local testing without a web server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AUTHENTICATION ENDPOINTS ---

@app.post("/register", status_code=201, tags=["Auth"])
async def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Registers a new user account with hashed passwords."""
    logger.info(f"Registering new user: {user.username}")
    return AuthService.register_user(db, user.username, user.password)

@app.post("/login", response_model=schemas.Token, tags=["Auth"])
async def login(form_data: auth.OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticates user and returns a JSON Web Token (JWT)."""
    logger.info(f"Login attempt: {form_data.username}")
    res = AuthService.login_for_access_token(db, form_data.username, form_data.password)
    logger.info(f"Login success: {form_data.username}")
    return res

@app.get("/users/me", response_model=schemas.UserInfo, tags=["Auth"])
async def me(current_user: models.User = Depends(get_current_user)):
    """Returns profile info for whichever user is currently logged in."""
    return {"username": current_user.username}

# --- CONVERSATION MANAGEMENT ---

@app.get("/users/stats/dashboard", tags=["Stats"])
async def dashboard_stats(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Calculates all-time usage statistics for the user dashboard."""
    return conv_service.get_dashboard_stats(db, current_user.id)

@app.get("/users/library/concepts", tags=["Stats"])
async def get_concepts(current_user: models.User = Depends(get_current_user)):
    """Generates a conceptual relationship map for the user's document history."""
    return await rag_service.get_library_themes(current_user.id)

@app.get("/conversations", response_model=List[schemas.ConversationInfo], tags=["Chat"])
async def list_conversations(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Retrieves all chat sessions for the logged-in user."""
    return conv_service.get_user_conversations(db, current_user.id)

@app.get("/conversations/search/deep", tags=["Chat"])
async def deep_search(q: str, current_user: models.User = Depends(get_current_user)):
    """Searches within the actual content of all uploaded documents."""
    results = await rag_service.deep_search(q, current_user.id)
    return {"results": results}

@app.post("/conversations", response_model=schemas.ConversationInfo, tags=["Chat"])
async def create_convo(data: schemas.ConversationUpdate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Creates a fresh chat session without an immediate document upload."""
    new_convo = models.Conversation(
        title=data.title,
        user_id=current_user.id,
        vector_store_id=None # No specific document index yet
    )
    db.add(new_convo)
    db.commit()
    db.refresh(new_convo)
    return new_convo

@app.get("/conversations/{conversation_id}", response_model=schemas.ConversationHistory, tags=["Chat"])
async def get_history(conversation_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Fetches the full message history and metadata for a specific chat session."""
    conv, messages = conv_service.get_conversation_history(db, conversation_id, current_user.id)
    if not conv: 
        logger.warning(f"History request for non-existent chat: {conversation_id}")
        raise HTTPException(404, detail="Conversation not found")
    return {
        "id": conv.id,
        "title": conv.title,
        "vector_store_id": conv.vector_store_id,
        "messages": messages
    }

@app.put("/conversations/{conversation_id}", tags=["Chat"])
async def rename_convo(conversation_id: int, update: schemas.ConversationUpdate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Allows manual renaming of a chat session."""
    logger.info(f"Renaming chat {conversation_id} to: {update.title}")
    c = conv_service.rename_conversation(db, conversation_id, current_user.id, update.title)
    if not c: raise HTTPException(404)
    return c

@app.get("/users/me/tags", tags=["Chat"])
async def get_all_user_tags(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Retrieves all unique tags used across all chats for the current user."""
    tags = conv_service.get_user_tags(db, current_user.id)
    return {"tags": tags}

@app.get("/conversations/{conversation_id}/tags", tags=["Chat"])
async def get_tags(conversation_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Retrieves all tags belonging to a specific conversation."""
    tags = conv_service.get_tags(db, conversation_id, current_user.id)
    return {"tags": tags}

@app.post("/conversations/{conversation_id}/tags", tags=["Chat"])
async def add_tag(conversation_id: int, data: dict, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Manually adds a new tag to a conversation."""
    tag_name = data.get("tag")
    if not tag_name: raise HTTPException(400, detail="Missing tag name")
    t = conv_service.add_tag(db, conversation_id, current_user.id, tag_name)
    if not t: raise HTTPException(404)
    return {"status": "success", "tag": t.tag}

@app.delete("/conversations/{conversation_id}/tags/{tag_name}", tags=["Chat"])
async def remove_tag(conversation_id: int, tag_name: str, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Removes a specific tag from a conversation."""
    if conv_service.remove_tag(db, conversation_id, current_user.id, tag_name):
        return {"status": "success"}
    raise HTTPException(404)

@app.delete("/conversations/{conversation_id}", status_code=204, tags=["Chat"])
async def delete_conversation(conversation_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Deletes the chat record from DB AND cleans up physical vector/pdf files from the server."""
    if conv_service.delete_conversation(db, conversation_id, current_user.id):
        return Response(status_code=204)
    raise HTTPException(404)

# --- DOCUMENT PROCESSING & RAG ---

@app.post("/upload_pdf", tags=["Ingestion"])
async def upload_pdf(file: UploadFile = File(...), conversation_id: Optional[int] = Form(None), db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """
    Main ingestion route:
    1. Checks for duplicates using file hashing.
    2. Stores the file physically.
    3. Triggers RAG Service to create Vector/BM25 indices.
    4. Auto-generates a smart title for the chat.
    """
    logger.info(f"Processing upload: {file.filename}")
    content = await file.read()
    f_hash = hashlib.sha256(content).hexdigest()
    
    vs_id = str(uuid.uuid4())
    is_append = False
    conv = None
    
    # Check if we are adding this file to an existing chat room
    if conversation_id:
        conv = db.query(models.Conversation).filter(models.Conversation.id == conversation_id, models.Conversation.user_id == current_user.id).first()
        if conv:
            # If the convo was 'empty' (e.g. created as Global Research), assign it a real ID now
            if not conv.vector_store_id:
                conv.vector_store_id = vs_id
            else:
                vs_id = conv.vector_store_id
            
            hashes = conv.get_file_hashes()
            if f_hash in hashes.values():
                logger.info(f"Duplicate upload blocked for: {file.filename}")
                return {"status": "duplicate", "message": "File already recognized in this chat", "conversation_id": conv.id}
            is_append = True

    # Save the file with a unique prefix to prevent collisions
    path = os.path.join(UPLOAD_DIR, f"{vs_id}_{file.filename}")
    logger.info(f"Physical Save: {path}")
    async with aiofiles.open(path, "wb") as f: 
        await f.write(content)
    
    # Hand off to RAG service for expensive AI operations (Embedding + Indexing)
    logger.info(f"Handoff to RAG Engine (vs_id: {vs_id})")
    res = await rag_service.process_file(path, vs_id, is_append)
    if not res: 
        logger.error(f"RAG Engine failed for {file.filename}")
        raise HTTPException(500, detail="Document indexing failed")
    
    # Mirror to Global Store
    await rag_service.save_to_global_store(res["chunks"], current_user.id)
    
    # Ask the AI to read the document and give it a name
    title = await rag_service.generate_one_liner_summary(res["preview"])
    tags = await rag_service.generate_tags(res["preview"])
    logger.info(f"AI suggests title: {title} and tags: {tags}")
    
    if not conv:
        # Create a new conversation record
        labels = {file.filename: f_hash}
        conv = models.Conversation(title=title, user_id=current_user.id, vector_store_id=vs_id)
        conv.set_file_hashes(labels)
        db.add(conv)
        db.flush()
        
        # Add Auto-Tags
        for t in tags:
            conv_service.add_tag(db, conv.id, current_user.id, t)
    else:
        # Update existing record with the new file hash
        hashes = conv.get_file_hashes()
        hashes[file.filename] = f_hash
        conv.set_file_hashes(hashes)
        if len(hashes) > 1: conv.title = f"Inter-Doc: {title}"
        
        # Also add new auto-tags for the appended file
        for t in tags:
            conv_service.add_tag(db, conv.id, current_user.id, t)
    
    db.commit()
    db.refresh(conv)
    return {"status": "success", "conversation_id": conv.id, "title": conv.title}

@app.get("/preview/{vector_store_id}/{filename}/{page_num}", tags=["Ingestion"])
async def preview_page(vector_store_id: str, filename: str, page_num: int, current_user: models.User = Depends(get_current_user)):
    """Fetches the raw text of a specific page from a specific file for the 'Citation Preview' feature."""
    text = await rag_service.get_page_preview(vector_store_id, filename, page_num)
    if text is None: raise HTTPException(404, detail="Page content not found")
    return {"text": text}

# --- CHAT & RAG ENDPOINTS ---

@app.post("/ask", tags=["Chat"])
async def ask(data: dict, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """
    The Thinking Engine:
    1. Look up the specific chat and history.
    2. Perform Hybrid Search (Vector + Keyword) across the specific documents.
    3. Stream the AI's reasoned answer word-by-word back to the user.
    """
    q, cid = data.get("question"), data.get("conversation_id")
    if not q or not cid: raise HTTPException(400, detail="Missing question or chat ID")
    
    # --- BLOCKING LOGIC ---
    # Check if this user has ever uploaded a document. If not, they are restricted to upload first.
    if not rag_service.has_documents(current_user.id):
        logger.warning(f"🚫 Blocked Chat: User {current_user.id} has no uploaded documents.")
        async def block_generator():
            yield "⚠️ **Access Restricted:** To start chatting with QueryMate, please upload at least one PDF or Image document first. \n\nOur system is document-centric; once you upload your first file, I can analyze it for you. In the future, every new chat you start will automatically remember all your previously uploaded documents!"
        return StreamingResponse(block_generator(), media_type="text/plain")

    logger.info(f"Query Processed: '{q[:40]}...' (Conv: {cid})")
    
    conv = db.query(models.Conversation).filter(models.Conversation.id == cid, models.Conversation.user_id == current_user.id).first()
    if not conv: raise HTTPException(404)
    
    # Save the user message to DB immediately
    msg_service.add_message(db, cid, "user", q)
    
    # Get conversation history to provide context to the LLM
    msgs = db.query(models.Message).filter(models.Message.conversation_id == cid).all()
    history = [{"role": m.role, "content": m.content} for m in msgs]
    
    async def stream_output():
        """Generator that streams response chunks and saves the final output to DB."""
        full_response = ""
        
        async for chunk in rag_service.answer_question_stream(q, conv.vector_store_id, history, current_user.id):
            full_response += chunk
            yield chunk
            
        if full_response:
            # We save the ENTIRE message including [FOLLOW_UPS] JSON.
            # The RAG Service handles stripping this tag when reading history.
            # The Frontend handles parsing/hiding this tag when rendering.
            assistant_msg = msg_service.add_message(db, cid, "assistant", full_response.strip())
            yield f"\n\n[MSID:{assistant_msg.id}]"

    return StreamingResponse(stream_output(), media_type="text/plain")

@app.post("/conversations/{conversation_id}/summarize-stream", tags=["Analysis"])
async def summarize(conversation_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Generates a high-level summary of the entire conversation transcript."""
    msgs = db.query(models.Message).filter(models.Message.conversation_id == conversation_id).all()
    transcript = "\n".join([f"{m.role}: {m.content}" for m in msgs])
    return StreamingResponse(rag_service.generate_summary_stream(transcript), media_type="text/plain")

# --- FEEDBACK & UTILS ---

@app.post("/messages/{message_id}/feedback", tags=["Analysis"])
async def feedback(message_id: int, req: schemas.MessageFeedbackCreate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Stores user ratings (thumbs up/down) for specific AI messages."""
    msg_service.add_feedback(db, message_id, current_user.id, req.rating)
    return {"status": "success"}

@app.get("/conversations/{conversation_id}/stats", tags=["Stats"])
async def convo_stats(conversation_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Fetches real-time performance and usage metrics for a specific chat."""
    stats = conv_service.get_stats(db, conversation_id, current_user.id)
    if not stats: raise HTTPException(404)
    return stats

@app.get("/messages/{message_id}/feedback", tags=["Analysis"])
async def get_feedback(message_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Retrieves aggregated feedback data for a message."""
    return msg_service.get_feedback_stats(db, message_id)

@app.post("/conversations/{conversation_id}/share", tags=["Chat"])
async def share(conversation_id: int, req: schemas.ShareCreateRequest, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Creates a unique, time-limited token for sharing chat history publicly."""
    c = db.query(models.Conversation).filter(models.Conversation.id == conversation_id, models.Conversation.user_id == current_user.id).first()
    if not c: raise HTTPException(404)
    token = secrets.token_urlsafe(32)
    exp = datetime.utcnow() + timedelta(days=req.expires_in_days) if req.expires_in_days else None
    s = models.PublicShare(conversation_id=conversation_id, share_token=token, expires_at=exp)
    db.add(s)
    db.commit()
    return {"share_token": token, "share_url": f"http://localhost:8000/share/{token}"}

@app.get("/share/{token}", tags=["Chat"])
async def get_share(token: str, db: Session = Depends(get_db)):
    """Endpoint for anonymous users to view shared conversations."""
    s = db.query(models.PublicShare).filter(models.PublicShare.share_token == token, models.PublicShare.is_active == 1).first()
    if not s or (s.expires_at and datetime.utcnow() > s.expires_at): 
        raise HTTPException(404, detail="Share link expired or invalid")
    c = s.conversation
    msgs = db.query(models.Message).filter(models.Message.conversation_id == c.id).order_by(models.Message.timestamp.asc()).all()
    return {"title": c.title, "messages": [{"role": m.role, "content": m.content} for m in msgs]}

@app.get("/conversations/{conversation_id}/export", tags=["Chat"])
async def export(conversation_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Exports the entire chat as a clean JSON structure for external backup."""
    c = db.query(models.Conversation).filter(models.Conversation.id == conversation_id, models.Conversation.user_id == current_user.id).first()
    if not c: raise HTTPException(404)
    msgs = db.query(models.Message).filter(models.Message.conversation_id == c.id).order_by(models.Message.timestamp.asc()).all()
    return {
        "title": c.title, 
        "user": current_user.username,
        "created_at": c.created_at,
        "messages": [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in msgs]
    }

if __name__ == "__main__":
    # Standard entry point for running the API locally
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)