from fastapi import (
    FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import secrets
from datetime import datetime, timedelta

import os
import logging
import uuid
import aiofiles
import shutil
import fitz 
import hashlib

import models
import auth
import rag_logic
from models import SessionLocal, engine, User, Conversation, Message, MessageFeedback, ConversationTag, PublicShare, ConversationStats
from auth import (
    get_password_hash, verify_password, create_access_token,
    SECRET_KEY, ALGORITHM
)
from jose import JWTError, jwt

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("server.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("🚀 Starting FastAPI Server Application...")

# --- App & DB ---
app = FastAPI()
models.Base.metadata.create_all(bind=engine)
logger.info("✅ Database tables checked/created.")

# --- Config ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_stores")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
logger.info(f"📂 Directories ready: {UPLOAD_DIR}, {VECTOR_STORE_DIR}")

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# --- Models ---
class Token(BaseModel):
    access_token: str
    token_type: str
class UserCreate(BaseModel):
    username: str
    password: str
class UserInfo(BaseModel):
    username: str
class ConversationInfo(BaseModel):
    id: int
    title: str
class MessageInfo(BaseModel):
    id: int  # Add message ID for feedback/ratings
    role: str
    content: str
class ConversationHistory(BaseModel):
    title: str
    vector_store_id: str
    messages: List[MessageInfo]
class SummaryResponse(BaseModel):
    generated_summary: str
    messages: List[MessageInfo]
class ConversationUpdate(BaseModel):
    title: str

# New models for features
class TagCreate(BaseModel):
    tag: str

class MessageFeedbackCreate(BaseModel):
    rating: int  # 1-5 or 1/-1

class ShareCreateRequest(BaseModel):
    expires_in_days: Optional[int] = None  # None = never expires

class BatchDeleteRequest(BaseModel):
    conversation_ids: List[int]

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

# --- Dependencies ---
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise HTTPException(status_code=401)
    except JWTError: 
        logger.warning("❌ Invalid JWT Token used.")
        raise HTTPException(status_code=401)
    
    user = db.query(User).filter(User.username == username).first()
    if not user: 
        logger.warning(f"❌ Token valid but user '{username}' not found in DB.")
        raise HTTPException(status_code=401)
    return user

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- UTILITY FUNCTIONS ---
async def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculate SHA256 hash of file content"""
    return hashlib.sha256(file_bytes).hexdigest()

# --- AUTH ---
@app.post("/register", status_code=201)
async def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    logger.info(f"👤 Registration attempt for: {user_in.username}")
    if db.query(User).filter(User.username == user_in.username).first():
        logger.warning(f"❌ Registration failed: Username '{user_in.username}' taken.")
        raise HTTPException(status_code=400, detail="Username taken")
    
    hashed_password = get_password_hash(user_in.password)
    db.add(User(username=user_in.username, hashed_password=hashed_password))
    db.commit()
    logger.info(f"✅ User registered successfully: {user_in.username}")
    return {"message": "Created"}

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    logger.info(f"🔑 Login attempt for: {form_data.username}")
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"❌ Login failed for {form_data.username}: Incorrect credentials")
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    
    logger.info(f"✅ Login success: {form_data.username}")
    return {"access_token": create_access_token(data={"sub": user.username}), "token_type": "bearer"}

@app.get("/users/me", response_model=UserInfo)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username}

# --- PDF & CHAT ---
@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    conversation_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    logger.info(f"📤 Upload Request received | User: {current_user.username} | File: {file.filename}")
    
    # Read file content and calculate hash
    file_content = await file.read()
    file_hash = await calculate_file_hash(file_content)
    logger.info(f"  📋 File hash: {file_hash}")
    
    vector_store_id = str(uuid.uuid4())
    is_append = False
    conversation = None
    existing_pdfs = {}

    if conversation_id:
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
        if conversation:
            vector_store_id = conversation.vector_store_id
            existing_pdfs = conversation.get_file_hashes()
            
            # Check if the same file was already uploaded to this conversation
            if file_hash in existing_pdfs.values():
                logger.info(f"  ⚡ Duplicate file detected in this conversation! Skipping RAG processing.")
                return {
                    "status": "duplicate", 
                    "conversation_id": conversation.id, 
                    "title": conversation.title,
                    "message": f"File '{file.filename}' already uploaded to this conversation.",
                    "pdf_count": len(existing_pdfs)
                }
            
            # New file in same conversation - append to existing vector store
            is_append = True
            logger.info(f"  ➡ Appending new PDF to Conversation ID: {conversation_id} | Existing PDFs: {len(existing_pdfs)}")
    
    safe_filename = f"{vector_store_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        # Write file to disk
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        logger.info(f"  ✅ File saved to disk. Starting RAG processing...")
        await rag_logic.process_pdf_for_rag(file_path, vector_store_id, is_append=is_append)

        if not conversation:
            # Create new conversation with first PDF
            existing_pdfs[file.filename] = file_hash
            conversation = Conversation(
                title=file.filename, 
                user_id=current_user.id, 
                vector_store_id=vector_store_id
            )
            conversation.set_file_hashes(existing_pdfs)
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            logger.info(f"  ✅ New Conversation created (ID: {conversation.id}) with 1 PDF")
        else:
            # Add to existing conversation
            existing_pdfs[file.filename] = file_hash
            conversation.set_file_hashes(existing_pdfs)
            # Update title to show multiple files if needed
            if len(existing_pdfs) == 1:
                conversation.title = file.filename
            else:
                conversation.title = f"Multiple PDFs ({len(existing_pdfs)} files)"
            db.commit()
            logger.info(f"  ✅ PDF added to Conversation | Total PDFs: {len(existing_pdfs)}")

        return {
            "status": "success", 
            "conversation_id": conversation.id, 
            "title": conversation.title,
            "pdf_count": len(existing_pdfs),
            "message": f"Successfully uploaded '{file.filename}'. Total PDFs in chat: {len(existing_pdfs)}"
        }
    except Exception as e:
        logger.exception("❌ Upload failed unexpectedly")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question_streaming(data: dict, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    question = data.get("question")
    conversation_id = data.get("conversation_id")
    
    logger.info(f"💬 Question received | User: {current_user.username} | ChatID: {conversation_id}")
    logger.info(f"  Query: {question}")

    if not question or not conversation_id: raise HTTPException(400)

    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation: raise HTTPException(404)

    db.add(Message(role="user", content=question, conversation_id=conversation_id))
    db.commit()

    history = [{"role": m.role, "content": m.content} for m in db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp.asc()).all()]

    async def response_generator():
        full_answer = ""
        try:
            logger.info("  ➡ Starting response stream...")
            async for chunk in rag_logic.answer_question_stream(question, conversation.vector_store_id, history):
                full_answer += chunk
                yield chunk
            
            logger.info("  ✅ Stream finished. Saving bot message.")
            if full_answer:
                with SessionLocal() as db2:
                    db2.add(Message(role="assistant", content=full_answer, conversation_id=conversation_id))
                    db2.commit()
        except Exception as e:
            logger.error(f"❌ Streaming failed: {e}")
            yield f"Error: {str(e)}"

    return StreamingResponse(response_generator(), media_type="text/plain")

@app.get("/preview/{vector_store_id}/{page_num}")
async def get_page_preview(vector_store_id: str, page_num: int, current_user: User = Depends(get_current_user)):
    logger.info(f"👀 Preview Request | Store: {vector_store_id} | Page: {page_num}")
    target_file = None
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(vector_store_id):
            target_file = os.path.join(UPLOAD_DIR, f)
            break
            
    if not target_file: 
        logger.error("❌ Source file for preview not found.")
        raise HTTPException(404)

    try:
        doc = fitz.open(target_file)
        if 0 <= page_num - 1 < len(doc):
            return {"text": doc[page_num - 1].get_text()}
        return {"text": "Page out of range."}
    except Exception as e:
        logger.error(f"❌ Preview failed: {e}")
        raise HTTPException(500)

# --- HISTORY ---
@app.get("/conversations", response_model=List[ConversationInfo])
async def get_convos(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Conversation).filter(Conversation.user_id == current_user.id).order_by(Conversation.created_at.desc()).all()

@app.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_history(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not c: raise HTTPException(404)
    msgs = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp.asc()).all()
    return ConversationHistory(title=c.title, vector_store_id=c.vector_store_id, messages=[{"id": m.id, "role": m.role, "content": m.content} for m in msgs])

@app.get("/conversations/{conversation_id}/pdfs")
async def get_conversation_pdfs(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get list of all PDFs uploaded to this conversation"""
    c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not c: raise HTTPException(404, detail="Conversation not found")
    
    file_hashes = c.get_file_hashes()
    return {
        "conversation_id": conversation_id,
        "title": c.title,
        "pdf_count": len(file_hashes),
        "pdfs": [{"filename": name, "hash": hash_val} for name, hash_val in file_hashes.items()]
    }

@app.post("/conversations/{conversation_id}/summarize-stream")
async def summarize_streaming(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Stream the full conversation summary"""
    logger.info(f"📝 Summary stream requested for ChatID: {conversation_id}")
    c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not c: raise HTTPException(404)
    
    msgs = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp.asc()).all()
    transcript = "\n".join([f"{m.role}: {m.content}" for m in msgs])
    
    async def summary_generator():
        try:
            async for chunk in rag_logic.generate_summary_stream(transcript):
                yield chunk
        except Exception as e:
            logger.error(f"❌ Summary streaming failed: {e}")
            yield f"Error: {str(e)}"
    
    return StreamingResponse(summary_generator(), media_type="text/plain")

@app.get("/conversations/{conversation_id}/summarize", response_model=SummaryResponse)
async def summarize(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    logger.info(f"📝 Summary requested for ChatID: {conversation_id}")
    c = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    msgs = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp.asc()).all()
    transcript = "\n".join([f"{m.role}: {m.content}" for m in msgs])
    summary = await rag_logic.generate_summary(transcript)
    return SummaryResponse(generated_summary=summary, messages=[{"role": m.role, "content": m.content} for m in msgs])

@app.put("/conversations/{conversation_id}")
async def rename_convo(conversation_id: int, update: ConversationUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if c:
        c.title = update.title
        db.commit()
        return c
    raise HTTPException(404)

@app.get("/conversations/{conversation_id}/export")
async def export_conversation(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Export conversation as structured JSON for client-side export"""
    logger.info(f"📤 Export requested for Conversation {conversation_id}")
    c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not c: raise HTTPException(404, detail="Conversation not found")
    
    msgs = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp.asc()).all()
    
    return {
        "title": c.title,
        "created_at": c.created_at.isoformat(),
        "user": current_user.username,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat()
            } for m in msgs
        ]
    }

# --- NEW FEATURE: CONVERSATION TAGS ---
@app.post("/conversations/{conversation_id}/tags")
async def add_tag(conversation_id: int, tag_req: TagCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Add a tag to conversation"""
    c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not c: raise HTTPException(404)
    
    existing_tag = db.query(ConversationTag).filter(ConversationTag.conversation_id == conversation_id, ConversationTag.tag == tag_req.tag.lower()).first()
    if existing_tag: raise HTTPException(400, detail="Tag already exists")
    
    tag = ConversationTag(conversation_id=conversation_id, tag=tag_req.tag.lower())
    db.add(tag)
    db.commit()
    logger.info(f"✅ Tag '{tag_req.tag}' added to conversation {conversation_id}")
    return {"status": "success", "tag": tag_req.tag}

@app.get("/conversations/{conversation_id}/tags")
async def get_tags(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get all tags for a conversation"""
    c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not c: raise HTTPException(404)
    
    tags = db.query(ConversationTag).filter(ConversationTag.conversation_id == conversation_id).all()
    return {"tags": [t.tag for t in tags]}

@app.delete("/conversations/{conversation_id}/tags/{tag}")
async def remove_tag(conversation_id: int, tag: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Remove a tag from conversation"""
    c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not c: raise HTTPException(404)
    
    tag_obj = db.query(ConversationTag).filter(ConversationTag.conversation_id == conversation_id, ConversationTag.tag == tag.lower()).first()
    if not tag_obj: raise HTTPException(404, detail="Tag not found")
    
    db.delete(tag_obj)
    db.commit()
    return {"status": "success"}

@app.get("/conversations/by-tag/{tag}")
async def get_conversations_by_tag(tag: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get all conversations with a specific tag"""
    conversations = db.query(Conversation).join(ConversationTag).filter(
        Conversation.user_id == current_user.id,
        ConversationTag.tag == tag.lower()
    ).all()
    return [{"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()} for c in conversations]

# --- NEW FEATURE: SEARCH CONVERSATIONS ---
@app.get("/conversations/search/{query}")
async def search_conversations(query: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Search conversations by keywords in messages"""
    query_lower = query.lower()
    conversations = db.query(Conversation).filter(Conversation.user_id == current_user.id).all()
    
    results = []
    for conv in conversations:
        messages = db.query(Message).filter(Message.conversation_id == conv.id).all()
        matching_messages = [m for m in messages if query_lower in m.content.lower()]
        
        if matching_messages or query_lower in conv.title.lower():
            results.append({
                "conversation_id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "matching_messages_count": len(matching_messages),
                "preview": matching_messages[0].content[:100] if matching_messages else ""
            })
    
    logger.info(f"🔍 Search for '{query}' returned {len(results)} results for user {current_user.username}")
    return {"query": query, "results": results}

# --- NEW FEATURE: MESSAGE REACTIONS/FEEDBACK ---
@app.post("/messages/{message_id}/feedback")
async def add_message_feedback(message_id: int, feedback: MessageFeedbackCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Add feedback (rating) to a message"""
    msg = db.query(Message).filter(Message.id == message_id).first()
    if not msg: raise HTTPException(404, detail="Message not found")
    
    # Check user owns the conversation
    conv = db.query(Conversation).filter(Conversation.id == msg.conversation_id, Conversation.user_id == current_user.id).first()
    if not conv: raise HTTPException(403, detail="Unauthorized")
    
    # Check if user already rated this message
    existing = db.query(MessageFeedback).filter(
        MessageFeedback.message_id == message_id,
        MessageFeedback.user_id == current_user.id
    ).first()
    
    if existing:
        existing.rating = feedback.rating
        db.commit()
        return {"status": "updated"}
    
    fb = MessageFeedback(message_id=message_id, user_id=current_user.id, rating=feedback.rating)
    db.add(fb)
    db.commit()
    logger.info(f"👍 Feedback added to message {message_id} by user {current_user.username}")
    return {"status": "created"}

@app.get("/messages/{message_id}/feedback")
async def get_message_feedback(message_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get feedback stats for a message"""
    feedbacks = db.query(MessageFeedback).filter(MessageFeedback.message_id == message_id).all()
    
    ratings = [f.rating for f in feedbacks]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    thumbs_up = len([r for r in ratings if r == 1])
    thumbs_down = len([r for r in ratings if r == -1])
    
    return {
        "message_id": message_id,
        "total_feedbacks": len(feedbacks),
        "avg_rating": avg_rating,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down
    }

# --- NEW FEATURE: CONVERSATION SHARING ---
@app.post("/conversations/{conversation_id}/share")
async def create_share_link(conversation_id: int, req: ShareCreateRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Create a shareable link for a conversation"""
    c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not c: raise HTTPException(404)
    
    share_token = secrets.token_urlsafe(32)
    expires_at = None
    if req.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=req.expires_in_days)
    
    share = PublicShare(conversation_id=conversation_id, share_token=share_token, expires_at=expires_at)
    db.add(share)
    db.commit()
    
    logger.info(f"🔗 Share link created for conversation {conversation_id}")
    return {
        "share_token": share_token,
        "share_url": f"http://127.0.0.1:8000/share/{share_token}",
        "expires_at": expires_at.isoformat() if expires_at else None
    }

@app.get("/share/{share_token}")
async def get_shared_conversation(share_token: str, db: Session = Depends(get_db)):
    """Get a publicly shared conversation (read-only)"""
    share = db.query(PublicShare).filter(PublicShare.share_token == share_token, PublicShare.is_active == 1).first()
    if not share: raise HTTPException(404, detail="Share not found or expired")
    
    if share.expires_at and datetime.utcnow() > share.expires_at:
        raise HTTPException(403, detail="Share link expired")
    
    conv = db.query(Conversation).filter(Conversation.id == share.conversation_id).first()
    if not conv: raise HTTPException(404)
    
    messages = db.query(Message).filter(Message.conversation_id == conv.id).order_by(Message.timestamp.asc()).all()
    return {
        "title": conv.title,
        "created_at": conv.created_at.isoformat(),
        "messages": [{"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()} for m in messages]
    }

@app.get("/conversations/{conversation_id}/shares")
async def list_shares(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """List all share links for a conversation"""
    c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not c: raise HTTPException(404)
    
    shares = db.query(PublicShare).filter(PublicShare.conversation_id == conversation_id).all()
    return {"shares": [{"share_token": s.share_token, "created_at": s.created_at.isoformat(), "expires_at": s.expires_at.isoformat() if s.expires_at else None, "is_active": bool(s.is_active)} for s in shares]}

@app.delete("/share/{share_token}")
async def revoke_share(share_token: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Revoke a share link"""
    share = db.query(PublicShare).filter(PublicShare.share_token == share_token).first()
    if not share: raise HTTPException(404)
    
    # Verify ownership
    conv = db.query(Conversation).filter(Conversation.id == share.conversation_id, Conversation.user_id == current_user.id).first()
    if not conv: raise HTTPException(403)
    
    share.is_active = 0
    db.commit()
    logger.info(f"🔗 Share link revoked: {share_token}")
    return {"status": "revoked"}

# --- NEW FEATURE: BATCH OPERATIONS ---
@app.post("/conversations/batch/delete")
async def batch_delete_conversations(req: BatchDeleteRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Delete multiple conversations at once"""
    convos = db.query(Conversation).filter(
        Conversation.id.in_(req.conversation_ids),
        Conversation.user_id == current_user.id
    ).all()
    
    for c in convos:
        db.delete(c)
    
    db.commit()
    logger.info(f"🗑️ Batch deleted {len(convos)} conversations for user {current_user.username}")
    return {"deleted_count": len(convos)}

@app.post("/conversations/batch/tags")
async def batch_add_tags(conversation_ids: List[int], tag: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Add a tag to multiple conversations"""
    convos = db.query(Conversation).filter(
        Conversation.id.in_(conversation_ids),
        Conversation.user_id == current_user.id
    ).all()
    
    added_count = 0
    for c in convos:
        existing = db.query(ConversationTag).filter(ConversationTag.conversation_id == c.id, ConversationTag.tag == tag.lower()).first()
        if not existing:
            db.add(ConversationTag(conversation_id=c.id, tag=tag.lower()))
            added_count += 1
    
    db.commit()
    logger.info(f"✅ Added tag '{tag}' to {added_count} conversations")
    return {"updated_count": added_count}

# --- NEW FEATURE: CONVERSATION STATISTICS ---
@app.get("/conversations/{conversation_id}/stats")
async def get_conversation_stats(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get statistics for a conversation"""
    c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not c: raise HTTPException(404)
    
    messages = db.query(Message).filter(Message.conversation_id == conversation_id).all()
    
    user_messages = [m for m in messages if m.role == "user"]
    bot_messages = [m for m in messages if m.role == "assistant"]
    
    avg_question_length = sum(len(m.content) for m in user_messages) / len(user_messages) if user_messages else 0
    avg_response_length = sum(len(m.content) for m in bot_messages) / len(bot_messages) if bot_messages else 0
    
    session_duration = 0
    if messages:
        session_duration = (messages[-1].timestamp - messages[0].timestamp).total_seconds()
    
    # Get or create stats record
    stats = db.query(ConversationStats).filter(ConversationStats.conversation_id == conversation_id).first()
    if not stats:
        stats = ConversationStats(
            conversation_id=conversation_id,
            total_questions=len(user_messages),
            total_messages=len(messages),
            avg_response_length=int(avg_response_length),
            avg_question_length=int(avg_question_length),
            session_duration=int(session_duration)
        )
        db.add(stats)
        db.commit()
    
    return {
        "conversation_id": conversation_id,
        "total_messages": len(messages),
        "total_questions": len(user_messages),
        "total_responses": len(bot_messages),
        "avg_question_length": int(avg_question_length),
        "avg_response_length": int(avg_response_length),
        "session_duration_seconds": int(session_duration),
        "created_at": c.created_at.isoformat()
    }

@app.get("/users/stats/dashboard")
async def get_user_stats_dashboard(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get overall user statistics"""
    convos = db.query(Conversation).filter(Conversation.user_id == current_user.id).all()
    all_messages = db.query(Message).join(Conversation).filter(Conversation.user_id == current_user.id).all()
    
    total_questions = len([m for m in all_messages if m.role == "user"])
    total_conversations = len(convos)
    avg_messages_per_conv = len(all_messages) / total_conversations if total_conversations > 0 else 0
    
    pdfs_uploaded = sum(len(c.get_file_hashes()) for c in convos)
    
    return {
        "total_conversations": total_conversations,
        "total_questions_asked": total_questions,
        "total_pdfs_uploaded": pdfs_uploaded,
        "avg_messages_per_conversation": round(avg_messages_per_conv, 2),
        "total_api_calls": len(all_messages)
    }

@app.delete("/conversations/{conversation_id}", status_code=204)
async def delete_convo(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    logger.info(f"🗑️ Deleting Conversation {conversation_id} for user {current_user.username}")
    c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if c:
        db.delete(c)
        db.commit()
        return Response(status_code=204)
    raise HTTPException(404)