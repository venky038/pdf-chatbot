from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

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
    created_at: datetime

    class Config:
        from_attributes = True

class MessageInfo(BaseModel):
    id: int
    role: str
    content: str
    timestamp: datetime

    class Config:
        from_attributes = True

class ConversationHistory(BaseModel):
    id: int
    title: str
    vector_store_id: str
    messages: List[MessageInfo]

class SummaryResponse(BaseModel):
    generated_summary: str
    messages: List[MessageInfo]

class ConversationUpdate(BaseModel):
    title: str

class TagCreate(BaseModel):
    tag: str

class MessageFeedbackCreate(BaseModel):
    rating: int

class ShareCreateRequest(BaseModel):
    expires_in_days: Optional[int] = None

class BatchDeleteRequest(BaseModel):
    conversation_ids: List[int]

class BatchTagRequest(BaseModel):
    conversation_ids: List[int]
    tag: str

class SearchResult(BaseModel):
    conversation_id: int
    title: str
    created_at: datetime
    matching_messages_count: int
    preview: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
