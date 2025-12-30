from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

"""
Data Validation Schemas (Pydantic Models)
These classes define the expected Data Structures for API requests and responses.
They ensure that incoming data is clean and outgoing data is correctly formatted.
"""

# --- AUTH SCHEMAS ---

class Token(BaseModel):
    """The structure of an issued JWT access token."""
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    """Payload for registering a new account."""
    username: str
    password: str

class UserInfo(BaseModel):
    """Public profile information."""
    username: str

# --- CONVERSATION SCHEMAS ---

class ConversationInfo(BaseModel):
    """High-level metadata for a chat session (used in sidebar lists)."""
    id: int
    title: str
    created_at: datetime

    class Config:
        from_attributes = True # Allows Pydantic to read data directly from SQLAlchemy objects

class MessageInfo(BaseModel):
    """Details of a single chat message."""
    id: int
    role: str
    content: str
    timestamp: datetime

    class Config:
        from_attributes = True

class ConversationHistory(BaseModel):
    """Full detail view including every message in a specific chat."""
    id: int
    title: str
    vector_store_id: str
    messages: List[MessageInfo]

class SummaryResponse(BaseModel):
    """Structure for the high-level research summary report."""
    generated_summary: str
    messages: List[MessageInfo]

class ConversationUpdate(BaseModel):
    """Request structure for renaming a chat."""
    title: str

class TagCreate(BaseModel):
    """Request structure for adding a new label."""
    tag: str

# --- ANALYTICS & FEEDBACK ---

class MessageFeedbackCreate(BaseModel):
    """User reaction (Thumbs Up/Down)."""
    rating: int

class ShareCreateRequest(BaseModel):
    """Configuration for sharing a chat."""
    expires_in_days: Optional[int] = None

# --- BATCH OPERATIONS ---

class BatchDeleteRequest(BaseModel):
    """List of IDs to be deleted at once."""
    conversation_ids: List[int]

class BatchTagRequest(BaseModel):
    """Instruction to add a tag to multiple chats simultaneously."""
    conversation_ids: List[int]
    tag: str

# --- SEARCH SCHEMAS ---

class SearchResult(BaseModel):
    """A single hit found during global keyword search."""
    conversation_id: int
    title: str
    created_at: datetime
    matching_messages_count: int
    preview: str

class SearchResponse(BaseModel):
    """Full results set for a search query."""
    query: str
    results: List[SearchResult]
