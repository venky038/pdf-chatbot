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

# --- LOGGING ---
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log")

# On Windows, some consoles don't support UTF-8/Emojis by default. 
# We'll configure handlers to use UTF-8 explicitly.
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger("DocuMind")
logger.info(f"Server logs are being saved to: {LOG_FILE} (Encoding: UTF-8)")

# --- APP SETUP ---
app = FastAPI(title="DocuMind AI API")
models.Base.metadata.create_all(bind=engine)

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_stores")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# --- DEPENDENCIES ---
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

async def get_current_user(token: str = Depends(auth.oauth2_scheme), db: Session = Depends(get_db)):
    return await auth.get_current_user(token, db)

# --- SERVICES ---
conv_service = ConversationService(UPLOAD_DIR, VECTOR_DIR)
msg_service = MessageService()
rag_service = RAGService(os.getenv("PERPLEXITY_API_KEY"), VECTOR_DIR, UPLOAD_DIR)

# --- MIDDLEWARE ---
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "null",  # Supports opening index.html via file://
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AUTH ENDPOINTS ---
@app.post("/register", status_code=201)
async def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    return AuthService.register_user(db, user.username, user.password)

@app.post("/login", response_model=schemas.Token)
async def login(form_data: auth.OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    logger.info(f"Login attempt for user: {form_data.username}")
    res = AuthService.login_for_access_token(db, form_data.username, form_data.password)
    logger.info(f"Login successful for: {form_data.username}")
    return res

@app.get("/users/me", response_model=schemas.UserInfo)
async def me(current_user: models.User = Depends(get_current_user)):
    return {"username": current_user.username}

@app.get("/users/stats/dashboard")
async def dashboard_stats(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    return conv_service.get_dashboard_stats(db, current_user.id)

# --- CONVERSATION ENDPOINTS ---
@app.get("/conversations", response_model=List[schemas.ConversationInfo])
async def list_conversations(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    return conv_service.get_user_conversations(db, current_user.id)

@app.get("/conversations/{conversation_id}", response_model=schemas.ConversationHistory)
async def get_history(conversation_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    conv, messages = conv_service.get_conversation_history(db, conversation_id, current_user.id)
    if not conv: raise HTTPException(404)
    return {
        "id": conv.id,
        "title": conv.title,
        "vector_store_id": conv.vector_store_id,
        "messages": messages
    }

@app.put("/conversations/{conversation_id}")
async def rename_convo(conversation_id: int, update: schemas.ConversationUpdate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    c = conv_service.rename_conversation(db, conversation_id, current_user.id, update.title)
    if not c: raise HTTPException(404)
    return c

@app.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    if conv_service.delete_conversation(db, conversation_id, current_user.id):
        return Response(status_code=204)
    raise HTTPException(404)

@app.get("/conversations/search/{query}")
async def search(query: str, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    results = conv_service.search_conversations(db, current_user.id, query)
    return {"query": query, "results": results}

@app.get("/conversations/{conversation_id}/tags")
async def get_tags(conversation_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    tags = conv_service.get_tags(db, conversation_id, current_user.id)
    return {"tags": tags}

@app.post("/conversations/{conversation_id}/tags")
async def add_tag(conversation_id: int, req: schemas.TagCreate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    tag = conv_service.add_tag(db, conversation_id, current_user.id, req.tag)
    if not tag: raise HTTPException(404)
    return {"status": "success", "tag": req.tag}

@app.delete("/conversations/{conversation_id}/tags/{tag}")
async def remove_tag(conversation_id: int, tag: str, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    if conv_service.remove_tag(db, conversation_id, current_user.id, tag):
        return {"status": "success"}
    raise HTTPException(404)

# --- BATCH OPERATIONS ---
@app.post("/conversations/batch/delete")
async def batch_delete(req: schemas.BatchDeleteRequest, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    count = conv_service.batch_delete(db, current_user.id, req.conversation_ids)
    return {"deleted_count": count}

@app.post("/conversations/batch/tags")
async def batch_tag(req: schemas.BatchTagRequest, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    count = conv_service.batch_tag(db, current_user.id, req.conversation_ids, req.tag)
    return {"updated_count": count}

# --- DOCUMENT PROCESSING ---
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), conversation_id: Optional[int] = Form(None), db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    logger.info(f"Uploading file: {file.filename} (User: {current_user.username})")
    content = await file.read()
    f_hash = hashlib.sha256(content).hexdigest()
    
    vs_id = str(uuid.uuid4())
    is_append = False
    conv = None
    
    if conversation_id:
        conv = db.query(models.Conversation).filter(models.Conversation.id == conversation_id, models.Conversation.user_id == current_user.id).first()
        if conv:
            vs_id = conv.vector_store_id
            hashes = conv.get_file_hashes()
            if f_hash in hashes.values():
                logger.info(f"Duplicate file detected: {file.filename}")
                return {"status": "duplicate", "message": "File already exists", "conversation_id": conv.id}
            is_append = True

    path = os.path.join(UPLOAD_DIR, f"{vs_id}_{file.filename}")
    logger.info(f"Saving file to: {path}")
    async with aiofiles.open(path, "wb") as f: await f.write(content)
    
    logger.info(f"Processing file for RAG (vs_id: {vs_id})")
    res = await rag_service.process_file(path, vs_id, is_append)
    if not res: 
        logger.error(f"RAG processing failed for {file.filename}")
        raise HTTPException(500)
    
    title = await rag_service.generate_one_liner_summary(res["preview"])
    logger.info(f"Generated title: {title}")
    
    if not conv:
        labels = {file.filename: f_hash}
        conv = models.Conversation(title=title, user_id=current_user.id, vector_store_id=vs_id)
        conv.set_file_hashes(labels)
        db.add(conv)
    else:
        hashes = conv.get_file_hashes()
        hashes[file.filename] = f_hash
        conv.set_file_hashes(hashes)
        if len(hashes) > 1: conv.title = f"Multiple: {title}"
    
    db.commit()
    db.refresh(conv)
    return {"status": "success", "conversation_id": conv.id, "title": conv.title}

@app.get("/preview/{vector_store_id}/{filename}/{page_num}")
async def preview_page(vector_store_id: str, filename: str, page_num: int, current_user: models.User = Depends(get_current_user)):
    text = await rag_service.get_page_preview(vector_store_id, filename, page_num)
    if text is None: raise HTTPException(404)
    return {"text": text}

# --- CHAT & RAG ENDPOINTS ---
@app.post("/ask")
async def ask(data: dict, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    q, cid = data.get("question"), data.get("conversation_id")
    if not q or not cid: raise HTTPException(400)
    logger.info(f"Chat message: '{q[:50]}...' (Conv: {cid}, User: {current_user.username})")
    
    conv = db.query(models.Conversation).filter(models.Conversation.id == cid, models.Conversation.user_id == current_user.id).first()
    if not conv: 
        logger.warning(f"Chat {cid} not found for user {current_user.username}")
        raise HTTPException(404)
    
    msg_service.add_message(db, cid, "user", q)
    msgs = db.query(models.Message).filter(models.Message.conversation_id == cid).all()
    history = [{"role": m.role, "content": m.content} for m in msgs]
    
    async def stream_output():
        full = ""
        async for chunk in rag_service.answer_question_stream(q, conv.vector_store_id, history):
            full += chunk
            yield chunk
        if full:
            m = msg_service.add_message(db, cid, "assistant", full)
            yield f"\n\n[MSID:{m.id}]"

    return StreamingResponse(stream_output(), media_type="text/plain")

@app.post("/conversations/{conversation_id}/summarize-stream")
async def summarize(conversation_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    msgs = db.query(models.Message).filter(models.Message.conversation_id == conversation_id).all()
    transcript = "\n".join([f"{m.role}: {m.content}" for m in msgs])
    return StreamingResponse(rag_service.generate_summary_stream(transcript), media_type="text/plain")

@app.get("/conversations/{conversation_id}/stats")
async def convo_stats(conversation_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    stats = conv_service.get_stats(db, conversation_id, current_user.id)
    if not stats: raise HTTPException(404)
    return stats

@app.post("/messages/{message_id}/feedback")
async def feedback(message_id: int, req: schemas.MessageFeedbackCreate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    msg_service.add_feedback(db, message_id, current_user.id, req.rating)
    return {"status": "success"}

@app.get("/messages/{message_id}/feedback")
async def get_feedback(message_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    return msg_service.get_feedback_stats(db, message_id)

@app.post("/conversations/{conversation_id}/share")
async def share(conversation_id: int, req: schemas.ShareCreateRequest, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    c = db.query(models.Conversation).filter(models.Conversation.id == conversation_id, models.Conversation.user_id == current_user.id).first()
    if not c: raise HTTPException(404)
    token = secrets.token_urlsafe(32)
    exp = datetime.utcnow() + timedelta(days=req.expires_in_days) if req.expires_in_days else None
    s = models.PublicShare(conversation_id=conversation_id, share_token=token, expires_at=exp)
    db.add(s)
    db.commit()
    # Note: In a real app, you'd use a real domain
    return {"share_token": token, "share_url": f"http://localhost:8000/share/{token}"}

@app.get("/share/{token}")
async def get_share(token: str, db: Session = Depends(get_db)):
    s = db.query(models.PublicShare).filter(models.PublicShare.share_token == token, models.PublicShare.is_active == 1).first()
    if not s or (s.expires_at and datetime.utcnow() > s.expires_at): raise HTTPException(404)
    c = s.conversation
    msgs = db.query(models.Message).filter(models.Message.conversation_id == c.id).order_by(models.Message.timestamp.asc()).all()
    return {"title": c.title, "messages": [{"role": m.role, "content": m.content} for m in msgs]}

@app.get("/conversations/{conversation_id}/export")
async def export(conversation_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)