import os
import hashlib
import json
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from database import Base

"""
QueryMate Database Schema (SQLAlchemy Models)
This file defines the table structure for our application.
"""

class User(Base):
    """Represents a registered portal user."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    # One-to-many relationship with conversations
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    """
    Represents a research session/chat room.
    A single conversation can hold multiple documents (PDFs/Images).
    """
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # ID linking this chat to a folder in 'vector_stores' (Optional for fresh chats)
    vector_store_id = Column(String, nullable=True) 
    
    # JSON column for tracking which files are indexed here: {"policy_v1.pdf": "sha256_hash", ...}
    file_hashes = Column(String, default="{}", nullable=False)
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    tags = relationship("ConversationTag", back_populates="conversation", cascade="all, delete-orphan")
    
    def get_file_hashes(self):
        """Helper to deserialize the JSON hash string into a Python dictionary."""
        try:
            return json.loads(self.file_hashes) if self.file_hashes else {}
        except:
            return {}
    
    def set_file_hashes(self, hashes_dict):
        """Helper to serialize a Python dictionary into a JSON string for DB storage."""
        self.file_hashes = json.dumps(hashes_dict)

class Message(Base):
    """Represents a specific chat bubble within a conversation."""
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    role = Column(String, nullable=False) # Roles: 'user' or 'assistant'
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    
    conversation = relationship("Conversation", back_populates="messages")
    feedbacks = relationship("MessageFeedback", back_populates="message", cascade="all, delete-orphan")

class MessageFeedback(Base):
    """Stores user sentiment (Ratings/Thumbs) for assistant responses."""
    __tablename__ = "message_feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    rating = Column(Integer, nullable=False)  # 1 = Thumbs Up, -1 = Thumbs Down
    created_at = Column(DateTime, default=datetime.utcnow)
    
    message = relationship("Message", back_populates="feedbacks")
    user = relationship("User")

class ConversationTag(Base):
    """Custom organizational labels attached to chat sessions."""
    __tablename__ = "conversation_tags"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    tag = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="tags")

class PublicShare(Base):
    """Tokens that allow non-users to view specific chat transcripts."""
    __tablename__ = "public_shares"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    share_token = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # Null means the link is permanent
    is_active = Column(Integer, default=1)        # Toggle: 1=active, 0=disabled
    
    conversation = relationship("Conversation")

class ConversationStats(Base):
    """Pre-computed analytics used for the Statistics 📊 feature."""
    __tablename__ = "conversation_stats"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, unique=True)
    total_questions = Column(Integer, default=0)
    total_messages = Column(Integer, default=0)
    avg_response_length = Column(Integer, default=0)
    avg_question_length = Column(Integer, default=0)
    session_duration = Column(Integer, default=0)  # Total activity time in seconds
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation")