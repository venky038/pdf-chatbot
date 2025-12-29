import os
import shutil
import logging
from sqlalchemy.orm import Session
from models import Conversation, Message, MessageFeedback, ConversationTag, ConversationStats, PublicShare
from schemas import ConversationUpdate
from datetime import datetime

logger = logging.getLogger(__name__)

class ConversationService:
    def __init__(self, upload_dir: str, vector_store_dir: str):
        self.upload_dir = upload_dir
        self.vector_store_dir = vector_store_dir

    def get_user_conversations(self, db: Session, user_id: int):
        logger.info(f"Retrieving conversations for user_id: {user_id}")
        return db.query(Conversation).filter(Conversation.user_id == user_id).order_by(Conversation.created_at.desc()).all()

    def get_conversation_history(self, db: Session, conversation_id: int, user_id: int):
        logger.info(f"Fetching history for conversation {conversation_id} (user {user_id})")
        c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id).first()
        if not c:
            logger.warning(f"Conversation {conversation_id} not found for user {user_id}")
            return None, None
        messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp.asc()).all()
        return c, messages

    def rename_conversation(self, db: Session, conversation_id: int, user_id: int, title: str):
        logger.info(f"Renaming conversation {conversation_id} to '{title}'")
        c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id).first()
        if c:
            c.title = title
            db.commit()
            db.refresh(c)
        return c

    def delete_conversation(self, db: Session, conversation_id: int, user_id: int):
        logger.info(f"Deleting conversation {conversation_id} (user {user_id})")
        c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id).first()
        if not c:
            logger.warning(f"Delete failed: conversation {conversation_id} not found")
            return False

        vs_id = c.vector_store_id
        self._cleanup_files(vs_id)

        db.delete(c)
        db.commit()
        logger.info(f"Successfully deleted conversation {conversation_id}")
        return True

    def _cleanup_files(self, vs_id: str):
        logger.info(f"Cleaning up physical files for vector store {vs_id}")
        store_path = os.path.join(self.vector_store_dir, vs_id)
        if os.path.exists(store_path):
            try: 
                shutil.rmtree(store_path)
                logger.info(f"Deleted directory: {store_path}")
            except Exception as e: logger.error(f"Failed to delete store {vs_id}: {e}")

        bm25_path = os.path.join(self.vector_store_dir, f"{vs_id}_bm25.pkl")
        if os.path.exists(bm25_path):
            try: 
                os.remove(bm25_path)
                logger.info(f"Deleted BM25 file: {bm25_path}")
            except Exception as e: logger.error(f"Failed to delete BM25 {vs_id}: {e}")

        try:
            removed_uploads = 0
            for f in os.listdir(self.upload_dir):
                if f.startswith(f"{vs_id}_"):
                    os.remove(os.path.join(self.upload_dir, f))
                    removed_uploads += 1
            if removed_uploads > 0:
                logger.info(f"Removed {removed_uploads} upload files for {vs_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup uploads for {vs_id}: {e}")

    def add_tag(self, db: Session, conversation_id: int, user_id: int, tag: str):
        logger.info(f"Adding tag '{tag}' to conversation {conversation_id}")
        c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id).first()
        if not c: return None
        
        tag_lower = tag.lower()
        existing = db.query(ConversationTag).filter(ConversationTag.conversation_id == conversation_id, ConversationTag.tag == tag_lower).first()
        if existing: 
            logger.info(f"Tag '{tag}' already exists for chat {conversation_id}")
            return existing
        
        new_tag = ConversationTag(conversation_id=conversation_id, tag=tag_lower)
        db.add(new_tag)
        db.commit()
        return new_tag

    def get_tags(self, db: Session, conversation_id: int, user_id: int):
        c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id).first()
        if not c: return []
        return [t.tag for t in c.tags]

    def remove_tag(self, db: Session, conversation_id: int, user_id: int, tag: str):
        logger.info(f"Removing tag '{tag}' from conversation {conversation_id}")
        c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id).first()
        if not c: return False
        t = db.query(ConversationTag).filter(ConversationTag.conversation_id == conversation_id, ConversationTag.tag == tag.lower()).first()
        if t:
            db.delete(t)
            db.commit()
            return True
        return False

    def get_dashboard_stats(self, db: Session, user_id: int):
        logger.info(f"Calculating dashboard stats for user {user_id}")
        convs = db.query(Conversation).filter(Conversation.user_id == user_id).all()
        all_messages = db.query(Message).join(Conversation).filter(Conversation.user_id == user_id).all()
        total_questions = len([m for m in all_messages if m.role == "user"])
        pdfs_count = sum(len(c.get_file_hashes()) for c in convs)
        
        return {
            "total_conversations": len(convs),
            "total_questions": total_questions,
            "total_pdfs": pdfs_count,
            "total_messages": len(all_messages)
        }

    def get_stats(self, db: Session, conversation_id: int, user_id: int):
        """Get analytics for a specific conversation"""
        logger.info(f"Fetching analytics for conversation {conversation_id}")
        stats = db.query(ConversationStats).filter(ConversationStats.conversation_id == conversation_id).first()
        if not stats:
            # Create default stats if not found
            logger.info(f"No stats record found for {conversation_id}, calculating on-the-fly")
            msgs = db.query(Message).filter(Message.conversation_id == conversation_id).all()
            q_msgs = [m for m in msgs if m.role == "user"]
            a_msgs = [m for m in msgs if m.role == "assistant"]
            
            return {
                "total_messages": len(msgs),
                "total_questions": len(q_msgs),
                "total_responses": len(a_msgs),
                "avg_question_length": int(sum(len(m.content) for m in q_msgs)/len(q_msgs)) if q_msgs else 0,
                "avg_response_length": int(sum(len(m.content) for m in a_msgs)/len(a_msgs)) if a_msgs else 0,
                "session_duration_seconds": 0
            }
        
        return {
            "total_messages": stats.total_messages,
            "total_questions": stats.total_questions,
            "total_responses": stats.total_messages - stats.total_questions,
            "avg_question_length": stats.avg_question_length,
            "avg_response_length": stats.avg_response_length,
            "session_duration_seconds": stats.session_duration
        }

    def search_conversations(self, db: Session, user_id: int, query: str):
        logger.info(f"Searching chats for: '{query}' (user {user_id})")
        query_lower = query.lower()
        convs = db.query(Conversation).filter(Conversation.user_id == user_id).all()
        results = []
        for c in convs:
            msgs = db.query(Message).filter(Message.conversation_id == c.id).all()
            matches = [m for m in msgs if query_lower in m.content.lower()]
            if matches or query_lower in c.title.lower():
                results.append({
                    "conversation_id": c.id,
                    "title": c.title,
                    "created_at": c.created_at,
                    "matches_count": len(matches),
                    "preview": matches[0].content[:100] if matches else ""
                })
        return results

    def batch_delete(self, db: Session, user_id: int, conversation_ids: list):
        logger.info(f"Batch deleting {len(conversation_ids)} conversations")
        count = 0
        for cid in conversation_ids:
            if self.delete_conversation(db, cid, user_id):
                count += 1
        return count

    def batch_tag(self, db: Session, user_id: int, conversation_ids: list, tag: str):
        logger.info(f"Batch tagging {len(conversation_ids)} conversations with '{tag}'")
        count = 0
        for cid in conversation_ids:
            if self.add_tag(db, cid, user_id, tag):
                count += 1
        return count
