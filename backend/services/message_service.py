import logging
from sqlalchemy.orm import Session
from models import Message, MessageFeedback, Conversation
from fastapi import HTTPException

# Logger for message-related auditing and feedback tracking
logger = logging.getLogger("QueryMate.Message")

class MessageService:
    """
    Handles the persistence of chat messages and user feedback ratings.
    """

    @staticmethod
    def add_message(db: Session, conversation_id: int, role: str, content: str):
        """
        Saves a single message (either from 'user' or 'assistant') to the database.
        """
        msg = Message(conversation_id=conversation_id, role=role, content=content)
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg

    @staticmethod
    def add_feedback(db: Session, message_id: int, user_id: int, rating: int):
        """
        Stores or updates a user's reaction (Thumbs Up = 1, Thumbs Down = -1) to a message.
        Includes security checks to ensure users can only rate messages in their own chats.
        """
        # 1. Validation: Ensure the message actually exists
        msg = db.query(Message).filter(Message.id == message_id).first()
        if not msg: 
            logger.warning(f"Feedback attempt on non-existent message ID: {message_id}")
            raise HTTPException(404, detail="Message not found")
        
        # 2. Security: Ensure the user owns the conversation this message belongs to
        conv = db.query(Conversation).filter(Conversation.id == msg.conversation_id, Conversation.user_id == user_id).first()
        if not conv: 
            logger.error(f"Unauthorized feedback attempt: User {user_id} tried to rate message {message_id}")
            raise HTTPException(403, detail="Forbidden")

        # 3. Upsert Logic: If they already rated it, update the rating. Otherwise, create new.
        existing = db.query(MessageFeedback).filter(
            MessageFeedback.message_id == message_id,
            MessageFeedback.user_id == user_id
        ).first()

        if existing:
            logger.info(f"Updating feedback for msg {message_id} to: {rating}")
            existing.rating = rating
        else:
            logger.info(f"New feedback for msg {message_id}: {rating}")
            fb = MessageFeedback(message_id=message_id, user_id=user_id, rating=rating)
            db.add(fb)
        
        db.commit()
        return True

    @staticmethod
    def get_feedback_stats(db: Session, message_id: int):
        """
        Aggregates feedback for a specific message. 
        Useful for analytical dashboards or showing reaction counts in UI.
        """
        feedbacks = db.query(MessageFeedback).filter(MessageFeedback.message_id == message_id).all()
        ratings = [f.rating for f in feedbacks]
        
        up_count = len([r for r in ratings if r == 1])
        down_count = len([r for r in ratings if r == -1])
        
        return {
            "total": len(feedbacks),
            "thumbs_up": up_count,
            "thumbs_down": down_count,
            "avg": sum(ratings) / len(ratings) if ratings else 0
        }
