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

import os
import logging
import uuid
import aiofiles
import shutil
import fitz 

import models
import auth
import rag_logic
from models import SessionLocal, engine, User, Conversation, Message
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
    vector_store_id = str(uuid.uuid4())
    is_append = False
    conversation = None

    if conversation_id:
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
        if conversation:
            vector_store_id = conversation.vector_store_id
            is_append = True
            logger.info(f"  ➡ Appending to Conversation ID: {conversation_id}")
    
    safe_filename = f"{vector_store_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(await file.read())
        
        logger.info(f"  ✅ File saved to disk. Starting RAG processing...")
        await rag_logic.process_pdf_for_rag(file_path, vector_store_id, is_append=is_append)

        if not conversation:
            conversation = Conversation(title=file.filename, user_id=current_user.id, vector_store_id=vector_store_id)
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            logger.info(f"  ✅ New Conversation created (ID: {conversation.id})")

        return {"status": "success", "conversation_id": conversation.id, "title": conversation.title}
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
    return ConversationHistory(title=c.title, vector_store_id=c.vector_store_id, messages=[{"role": m.role, "content": m.content} for m in msgs])

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

@app.delete("/conversations/{conversation_id}", status_code=204)
async def delete_convo(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    logger.info(f"🗑️ Deleting Conversation {conversation_id} for user {current_user.username}")
    c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if c:
        db.delete(c)
        db.commit()
        return Response(status_code=204)
    raise HTTPException(404)