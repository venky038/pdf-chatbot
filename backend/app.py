from fastapi import (
    FastAPI, UploadFile, File, Depends, HTTPException, status, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List

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

# Create database tables if they don't exist
models.Base.metadata.create_all(bind=engine)

# --- Pydantic Models (Data Shapes) ---
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
    messages: List[MessageInfo]

class SummaryResponse(BaseModel):
    generated_summary: str
    messages: List[MessageInfo]

class ConversationUpdate(BaseModel):
    title: str

# --- Dependencies ---
def get_db():
    """Dependency: Get a new SQLAlchemy DB session for each request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# OAuth2 scheme using password flow, pointing to the /login endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """Dependency: Decode JWT token, find user in DB, handle errors."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# --- Path Configuration ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_stores")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development ease
    allow_credentials=True,
    allow_methods=["*"], # Allow all standard methods
    allow_headers=["*"], # Allow all headers
)

# ===============================================
# --- AUTHENTICATION ENDPOINTS ---
# ===============================================

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    """Registers a new user."""
    existing_user = db.query(User).filter(User.username == user_in.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    hashed_password = get_password_hash(user_in.password)
    db_user = User(username=user_in.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"User '{db_user.username}' registered successfully.")
    return {"message": f"User {db_user.username} registered successfully"}


@app.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    """Logs in a user and returns a JWT access token."""
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.username})
    logger.info(f"User '{user.username}' logged in successfully.")
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=UserInfo)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Gets the username of the currently logged-in user."""
    return {"username": current_user.username}


# ===============================================
# --- CHAT & PDF ENDPOINTS ---
# ===============================================

@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Handles PDF upload: saves file, processes for RAG, generates title,
    creates conversation record in DB.
    Includes cleanup on failure.
    """
    vector_store_id = str(uuid.uuid4())
    safe_filename = f"{vector_store_id}.pdf"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    new_conversation = None # Keep track for potential rollback/cleanup

    try:
        # 1. Save uploaded file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            await f.write(content)
        logger.info(f"Saved uploaded PDF to: {file_path}")

        # 2. Process PDF (create vector store, get text)
        try:
            extracted_text = await rag_logic.process_pdf_for_rag(file_path, vector_store_id)
        except rag_logic.NoSearchableTextError as e:
            logger.warning(f"No searchable text in uploaded PDF: {e}")
            # Cleanup file and any partially created vector store
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up uploaded file: {file_path}")
                except OSError:
                    logger.error(f"Could not remove partially uploaded file: {file_path}")

            vector_store_path = os.path.join(VECTOR_STORE_DIR, vector_store_id)
            if os.path.exists(vector_store_path):
                try:
                    shutil.rmtree(vector_store_path, ignore_errors=True)
                    logger.info(f"Cleaned up vector store directory: {vector_store_path}")
                except OSError:
                    logger.error(f"Could not remove partially created vector store: {vector_store_path}")

            # Inform client the PDF has no searchable text (image-only PDF)
            raise HTTPException(status_code=400, detail=str(e))

        if not extracted_text:
            logger.error(f"Failed to process PDF content for {file_path}")
            raise HTTPException(status_code=500, detail="Failed to process PDF text content.")

        # 3. Generate title from text
        try:
            generated_title = await rag_logic.generate_conversation_title(extracted_text)
        except Exception as title_e:
            logger.warning(f"Failed title generation, using filename: {title_e}")
            generated_title = file.filename or "Uploaded PDF" # Ensure a title

        # 4. Create Conversation in DB
        new_conversation = Conversation(
            title=generated_title,
            user_id=current_user.id,
            vector_store_id=vector_store_id
        )
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation) # Get the conversation ID

        logger.info(f"PDF processed successfully for convo {new_conversation.id}, title '{generated_title}'")
        return {
            "status": "success",
            "message": f"PDF '{file.filename or 'file'}' processed.",
            "conversation_id": new_conversation.id,
            "title": generated_title
        }

    except Exception as e:
        logger.exception(f"Error during PDF upload for user {current_user.username}")

        # Rollback DB changes if conversation was added but subsequent steps failed
        if new_conversation and db.object_session(new_conversation):
             logger.warning(f"Rolling back conversation creation attempt.")
             db.rollback()

        # Cleanup created files/folders
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up uploaded file: {file_path}")
            except OSError:
                logger.error(f"Could not remove partially uploaded file: {file_path}")

        vector_store_path = os.path.join(VECTOR_STORE_DIR, vector_store_id)
        if os.path.exists(vector_store_path):
            try:
                shutil.rmtree(vector_store_path, ignore_errors=True)
                logger.info(f"Cleaned up vector store directory: {vector_store_path}")
            except OSError:
                logger.error(f"Could not remove partially created vector store: {vector_store_path}")

        # Re-raise as HTTPException
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=f"An error occurred during upload: {str(e)}")


@app.post("/ask")
async def ask_question(
    data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Handles user question: validates input, saves question to DB,
    gets response from RAG (standard wait), saves response to DB, and returns it.
    """
    question = data.get("question")
    conversation_id = data.get("conversation_id")

    if not question or not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not conversation_id or not isinstance(conversation_id, int):
        raise HTTPException(status_code=400, detail="Valid Conversation ID required.")

    try:
        # Verify conversation belongs to the user
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found or access denied.")

        # Save user message immediately
        user_message = Message(
            role="user",
            content=question,
            conversation_id=conversation_id
        )
        db.add(user_message)
        db.commit()
        logger.info(f"Saved user message for convo {conversation_id}")

        # Fetch history
        db_messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp.asc()).all()
        history = [{"role": msg.role, "content": msg.content} for msg in db_messages]

        # --- CHANGED: Non-Streaming Logic ---
        
        # 1. Get full answer from RAG logic (await it)
        full_response = await rag_logic.answer_question(
            question, conversation.vector_store_id, history
        )

        # 2. Save Bot Message to DB
        bot_message = Message(
            role="assistant",
            content=full_response,
            conversation_id=conversation_id
        )
        db.add(bot_message)
        db.commit()
        logger.info(f"Saved full bot response for convo {conversation_id}")

        # 3. Return JSON Response
        return {"answer": full_response}

    except HTTPException as http_err:
        logger.warning(f"HTTP Exception in /ask: {http_err.detail}")
        raise http_err
    except Exception as e:
        logger.exception(f"Unexpected error in /ask for convo {conversation_id}")
        raise HTTPException(status_code=500, detail="Server error processing your question.")


# ===============================================
# --- CONVERSATION HISTORY ENDPOINTS ---
# ===============================================

@app.get("/conversations", response_model=List[ConversationInfo])
async def get_user_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Fetches list of conversations (ID and title) for the current user."""
    return db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.created_at.desc()).all()


@app.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Fetches title and all messages for a specific conversation."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied.")

    db_messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.timestamp.asc()).all()

    messages = [{"role": msg.role, "content": msg.content} for msg in db_messages]
    return ConversationHistory(title=conversation.title, messages=messages)


@app.get("/conversations/{conversation_id}/summarize", response_model=SummaryResponse)
async def get_conversation_summary(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Fetches messages, generates AI summary, and returns summary + messages."""
    logger.info(f"Requesting summary for conversation: {conversation_id}")
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied.")

    db_messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.timestamp.asc()).all()
    if not db_messages:
        raise HTTPException(status_code=400, detail="Cannot summarize an empty conversation.")

    transcript_lines = [
        f"{'Human' if msg.role == 'user' else 'Assistant'}: {msg.content}"
        for msg in db_messages
    ]
    transcript = "\n".join(transcript_lines)

    try:
        generated_summary = await rag_logic.generate_summary(transcript)
    except Exception as summary_err:
        logger.error(f"Error calling summary generation for convo {conversation_id}: {summary_err}")
        raise HTTPException(status_code=500, detail="Failed to generate summary.")

    messages_for_response = [{"role": msg.role, "content": msg.content} for msg in db_messages]
    return SummaryResponse(generated_summary=generated_summary, messages=messages_for_response)


# ===============================================
# --- CONVERSATION MANAGEMENT ENDPOINTS ---
# ===============================================

@app.put("/conversations/{conversation_id}", response_model=ConversationInfo)
async def update_conversation_title(
    conversation_id: int,
    conversation_update: ConversationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Updates (renames) the title of a specific conversation."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied.")

    new_title = conversation_update.title.strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")

    conversation.title = new_title
    db.commit()
    db.refresh(conversation)
    logger.info(f"Updated title for conversation {conversation_id} to '{conversation.title}'")
    return conversation


@app.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Deletes a conversation, its messages (via DB cascade),
    and associated PDF file and vector store directory.
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied.")

    vector_store_id = conversation.vector_store_id

    # Define paths for cleanup
    vector_store_path = os.path.join(VECTOR_STORE_DIR, vector_store_id)
    pdf_filename = f"{vector_store_id}.pdf"
    pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)

    # --- Delete associated files first ---
    if os.path.exists(vector_store_path):
        try:
            shutil.rmtree(vector_store_path)
            logger.info(f"Deleted vector store directory: {vector_store_path}")
        except OSError as e:
            logger.error(f"Error deleting vector store {vector_store_path}: {e}")

    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
            logger.info(f"Deleted uploaded PDF: {pdf_path}")
        except OSError as e:
            logger.error(f"Error deleting PDF file {pdf_path}: {e}")

    # --- Delete conversation from DB ---
    try:
        db.delete(conversation)
        db.commit()
        logger.info(f"Deleted conversation {conversation_id} from database.")
    except Exception as db_del_err:
         logger.error(f"Error deleting conversation {conversation_id} from DB: {db_del_err}")
         db.rollback() # Rollback if DB delete fails
         raise HTTPException(status_code=500, detail="Failed to delete conversation record from database.")

    # Return standard No Content response
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# --- (Optional: Add a root endpoint for basic check) ---
@app.get("/")
async def read_root():
    return {"message": "PDF Chatbot API is running"}