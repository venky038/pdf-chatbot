from fastapi import (
    FastAPI, UploadFile, File, Depends, HTTPException, status, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional

import os
import logging
import uuid
import aiofiles
import shutil

# Import project modules
import models
import auth
import rag_logic
from models import SessionLocal, engine, User, Conversation, Message
from auth import (
    get_password_hash, verify_password, create_access_token,
    SECRET_KEY, ALGORITHM
)
from jose import JWTError, jwt

# --- App & DB Setup ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models.Base.metadata.create_all(bind=engine)

# --- Path Configuration ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_stores")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# --- MOUNT STATIC FILES (CRITICAL FOR CITATIONS) ---
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# --- Pydantic Models ---
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
    try:
        yield db
    finally:
        db.close()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None: raise credentials_exception
    return user

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# --- AUTH ENDPOINTS ---
# ==========================

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == user_in.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user_in.password)
    db_user = User(username=user_in.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"message": f"User {db_user.username} registered successfully"}

@app.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserInfo)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username}

# ==========================
# --- CHAT & PDF ENDPOINTS ---
# ==========================

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    vector_store_id = str(uuid.uuid4())
    safe_filename = f"{vector_store_id}.pdf"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    new_conversation = None

    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            if not content: raise HTTPException(status_code=400, detail="Empty file")
            await f.write(content)

        extracted_text = await rag_logic.process_pdf_for_rag(file_path, vector_store_id)
        if not extracted_text: raise HTTPException(status_code=500, detail="Failed to process PDF")

        generated_title = await rag_logic.generate_conversation_title(extracted_text)

        new_conversation = Conversation(title=generated_title, user_id=current_user.id, vector_store_id=vector_store_id)
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)

        return {"status": "success", "conversation_id": new_conversation.id, "title": generated_title}

    except Exception as e:
        if new_conversation: db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(data: dict, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    question = data.get("question")
    conversation_id = data.get("conversation_id")

    conversation = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not conversation: raise HTTPException(status_code=404, detail="Conversation not found")

    user_message = Message(role="user", content=question, conversation_id=conversation_id)
    db.add(user_message)
    db.commit()

    db_messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp.asc()).all()
    history = [{"role": msg.role, "content": msg.content} for msg in db_messages]

    full_response = await rag_logic.answer_question(question, conversation.vector_store_id, history)

    bot_message = Message(role="assistant", content=full_response, conversation_id=conversation_id)
    db.add(bot_message)
    db.commit()

    return {"answer": full_response}

# ==========================
# --- HISTORY ENDPOINTS ---
# ==========================

@app.get("/conversations", response_model=List[ConversationInfo])
async def get_user_conversations(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Conversation).filter(Conversation.user_id == current_user.id).order_by(Conversation.created_at.desc()).all()

@app.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not conversation: raise HTTPException(status_code=404, detail="Not found")

    db_messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp.asc()).all()
    messages = [{"role": msg.role, "content": msg.content} for msg in db_messages]
    
    return ConversationHistory(title=conversation.title, vector_store_id=conversation.vector_store_id, messages=messages)

@app.get("/conversations/{conversation_id}/summarize", response_model=SummaryResponse)
async def get_conversation_summary(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not conversation: raise HTTPException(status_code=404, detail="Not found")
    
    db_messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp.asc()).all()
    
    # Simple formatting for the transcript
    transcript = "\n".join([f"{'User' if msg.role == 'user' else 'AI'}: {msg.content}" for msg in db_messages])
    
    try:
        # We call generate_summary from rag_logic
        generated_summary = await rag_logic.generate_summary(transcript)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate summary")

    messages_for_response = [{"role": msg.role, "content": msg.content} for msg in db_messages]
    return SummaryResponse(generated_summary=generated_summary, messages=messages_for_response)

@app.put("/conversations/{conversation_id}")
async def update_conversation_title(conversation_id: int, conversation_update: ConversationUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if conversation:
        conversation.title = conversation_update.title
        db.commit()
        return conversation
    raise HTTPException(status_code=404, detail="Not found")

@app.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if conversation:
        # Cleanup files (vector store and PDF) logic omitted for brevity, essentially just removing files
        db.delete(conversation)
        db.commit()
        return Response(status_code=204)
    raise HTTPException(status_code=404, detail="Not found")