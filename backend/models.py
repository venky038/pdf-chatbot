import os
import hashlib
import json
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# Get the project root directory (one level up)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
DB_PATH = os.path.join(PROJECT_ROOT, "chat_app.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

# --- THIS IS THE UPDATED CLASS ---
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Vector store ID (shared for all PDFs in this conversation)
    vector_store_id = Column(String, unique=True, nullable=False) 
    
    # Store file hashes as JSON: {"filename": "hash", ...}
    file_hashes = Column(String, default="{}", nullable=False)
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def get_file_hashes(self):
        """Parse file hashes JSON"""
        try:
            return json.loads(self.file_hashes) if self.file_hashes else {}
        except:
            return {}
    
    def set_file_hashes(self, hashes_dict):
        """Store file hashes as JSON"""
        self.file_hashes = json.dumps(hashes_dict)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    role = Column(String, nullable=False) # "user" or "assistant"
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    
    conversation = relationship("Conversation", back_populates="messages")
    feedbacks = relationship("MessageFeedback", back_populates="message", cascade="all, delete-orphan")

class MessageFeedback(Base):
    """Message reactions/ratings for feedback"""
    __tablename__ = "message_feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5 or thumbs: 1=up, -1=down
    created_at = Column(DateTime, default=datetime.utcnow)
    
    message = relationship("Message", back_populates="feedbacks")
    user = relationship("User")

class ConversationTag(Base):
    """Tags for organizing conversations"""
    __tablename__ = "conversation_tags"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    tag = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="tags")

# Add relationship to Conversation model
Conversation.tags = relationship("ConversationTag", back_populates="conversation", cascade="all, delete-orphan")

class PublicShare(Base):
    """Shareable conversation links"""
    __tablename__ = "public_shares"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    share_token = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # None = never expires
    is_active = Column(Integer, default=1)  # 1=active, 0=disabled
    
    conversation = relationship("Conversation")

class ConversationStats(Base):
    """Analytics for conversations"""
    __tablename__ = "conversation_stats"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, unique=True)
    total_questions = Column(Integer, default=0)
    total_messages = Column(Integer, default=0)
    avg_response_length = Column(Integer, default=0)
    avg_question_length = Column(Integer, default=0)
    session_duration = Column(Integer, default=0)  # in seconds
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation")