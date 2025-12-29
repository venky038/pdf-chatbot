import logging
from sqlalchemy.orm import Session
from models import Message, MessageFeedback, Conversation
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class MessageService:
    @staticmethod
    def add_message(db: Session, conversation_id: int, role: str, content: str):
        msg = Message(conversation_id=conversation_id, role=role, content=content)
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg

    @staticmethod
    def add_feedback(db: Session, message_id: int, user_id: int, rating: int):
        # Check message exists and is accessible
        msg = db.query(Message).filter(Message.id == message_id).first()
        if not msg: raise HTTPException(404, detail="Message not found")
        
        # Ownership check
        conv = db.query(Conversation).filter(Conversation.id == msg.conversation_id, Conversation.user_id == user_id).first()
        if not conv: raise HTTPException(403, detail="Forbidden")

        existing = db.query(MessageFeedback).filter(
            MessageFeedback.message_id == message_id,
            MessageFeedback.user_id == user_id
        ).first()

        if existing:
            existing.rating = rating
        else:
            fb = MessageFeedback(message_id=message_id, user_id=user_id, rating=rating)
            db.add(fb)
        
        db.commit()
        return True

    @staticmethod
    def get_feedback_stats(db: Session, message_id: int):
        feedbacks = db.query(MessageFeedback).filter(MessageFeedback.message_id == message_id).all()
        ratings = [f.rating for f in feedbacks]
        return {
            "total": len(feedbacks),
            "thumbs_up": len([r for r in ratings if r == 1]),
            "thumbs_down": len([r for r in ratings if r == -1]),
            "avg": sum(ratings) / len(ratings) if ratings else 0
        }
