import os
import shutil
import logging
from sqlalchemy.orm import Session
from models import Conversation, Message, MessageFeedback, ConversationTag, ConversationStats, PublicShare
from schemas import ConversationUpdate
from datetime import datetime

# Local logger for database and file management operations
logger = logging.getLogger("QueryMate.Conversation")

class ConversationService:
    """
    Handles all business logic related to conversations:
    - Metadata management (titles, tags, stats)
    - Data retrieval (histories, search)
    - File system hygiene (deleting local PDFs and vector indices)
    """

    def __init__(self, upload_dir: str, vector_store_dir: str):
        self.upload_dir = upload_dir
        self.vector_store_dir = vector_store_dir

    def get_user_conversations(self, db: Session, user_id: int):
        """Returns all chat rooms belonging to a specific user, newest first."""
        logger.info(f"Listing conversations for user: {user_id}")
        return db.query(Conversation).filter(Conversation.user_id == user_id).order_by(Conversation.created_at.desc()).all()

    def get_conversation_history(self, db: Session, conversation_id: int, user_id: int):
        """Fetches a specific conversation and all its associated messages."""
        logger.info(f"Opening chat: {conversation_id}")
        c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id).first()
        if not c:
            logger.warning(f"Unauthorized or invalid chat access: {conversation_id} by user {user_id}")
            return None, None
        
        messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp.asc()).all()
        return c, messages

    def rename_conversation(self, db: Session, conversation_id: int, user_id: int, title: str):
        """Updates the display title of a conversation."""
        c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id).first()
        if c:
            logger.info(f"Manual rename: Chat {conversation_id} -> '{title}'")
            c.title = title
            db.commit()
            db.refresh(c)
        return c

    def delete_conversation(self, db: Session, conversation_id: int, user_id: int):
        """
        Permanent deletion:
        1. Removes DB records (Cascade deletes messages).
        2. Tries to delete the physical PDF files from 'uploads'.
        3. Tries to delete the FAISS/BM25 indices from 'vector_stores'.
        """
        logger.info(f"Triggering full deletion for chat: {conversation_id}")
        c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id).first()
        if not c:
            logger.warning(f"Delete blocked: Chat {conversation_id} not found.")
            return False

        vs_id = c.vector_store_id
        # Clean up physical storage before removing DB record
        self._cleanup_files(vs_id)

        db.delete(c)
        db.commit()
        logger.info(f"Chat {conversation_id} and associated files successfully purged.")
        return True

    def _cleanup_files(self, vs_id: str):
        """Helper to find and remove all local files associated with a vector store ID."""
        logger.info(f"Starting disk cleanup for vs_id: {vs_id}")
        
        # 1. Delete FAISS directory
        store_path = os.path.join(self.vector_store_dir, vs_id)
        if os.path.exists(store_path):
            try: 
                shutil.rmtree(store_path)
                logger.info(f"Removed Vector Store: {store_path}")
            except Exception as e: logger.error(f"Failed to delete index dir: {e}")

        # 2. Delete BM25 pickle file
        bm25_path = os.path.join(self.vector_store_dir, f"{vs_id}_bm25.pkl")
        if os.path.exists(bm25_path):
            try: 
                os.remove(bm25_path)
                logger.info(f"Removed Keyword Index: {bm25_path}")
            except Exception as e: logger.error(f"Failed to delete BM25 file: {e}")

        # 3. Delete all related PDF/Image uploads
        try:
            removed_count = 0
            for f in os.listdir(self.upload_dir):
                if f.startswith(f"{vs_id}_"):
                    os.remove(os.path.join(self.upload_dir, f))
                    removed_count += 1
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} original document files.")
        except Exception as e:
            logger.error(f"Upload cleanup failed: {e}")

    def add_tag(self, db: Session, conversation_id: int, user_id: int, tag: str):
        """Adds a categorizing tag to a conversation for easier filtering."""
        c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id).first()
        if not c: return None
        
        tag_lower = tag.strip().lower()
        if not tag_lower: return None

        existing = db.query(ConversationTag).filter(ConversationTag.conversation_id == conversation_id, ConversationTag.tag == tag_lower).first()
        if existing: return existing
        
        new_tag = ConversationTag(conversation_id=conversation_id, tag=tag_lower)
        db.add(new_tag)
        db.commit()
        logger.info(f"Tag added: '{tag_lower}' to chat {conversation_id}")
        return new_tag

    def get_tags(self, db: Session, conversation_id: int, user_id: int):
        """Retrieves all tags for a specific chat."""
        c = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id).first()
        return [t.tag for t in c.tags] if c else []

    def remove_tag(self, db: Session, conversation_id: int, user_id: int, tag: str):
        """Removes a specific tag."""
        t = db.query(ConversationTag).filter(ConversationTag.conversation_id == conversation_id, ConversationTag.tag == tag.lower()).first()
        if t:
            db.delete(t)
            db.commit()
            logger.info(f"Tag removed: '{tag}' from chat {conversation_id}")
            return True
        return False

    def get_dashboard_stats(self, db: Session, user_id: int):
        """Calculates global analytics for the user's dashboard view."""
        logger.info(f"Compiling dashboard metrics for user: {user_id}")
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
        """Retrieves detailed per-chat analytics (message counts, average lengths)."""
        logger.info(f"Analyzing metrics for chat: {conversation_id}")
        stats = db.query(ConversationStats).filter(ConversationStats.conversation_id == conversation_id).first()
        
        if not stats:
            # Fallback for chats that haven't had stats compiled yet
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
        """Global search across all chat titles and message contents."""
        logger.info(f"Global keyword search: '{query}'")
        q_lower = query.lower()
        convs = db.query(Conversation).filter(Conversation.user_id == user_id).all()
        results = []
        for c in convs:
            msgs = db.query(Message).filter(Message.conversation_id == c.id).all()
            matches = [m for m in msgs if q_lower in m.content.lower()]
            if matches or q_lower in c.title.lower():
                results.append({
                    "conversation_id": c.id,
                    "title": c.title,
                    "created_at": c.created_at,
                    "matches_count": len(matches),
                    "preview": matches[0].content[:120] if matches else "Title Match"
                })
        return results

    def batch_delete(self, db: Session, user_id: int, conversation_ids: list):
        """Bulk deletion utility."""
        logger.info(f"Batch Delete started for {len(conversation_ids)} items.")
        count = 0
        for cid in conversation_ids:
            if self.delete_conversation(db, cid, user_id):
                count += 1
        return count

    def batch_tag(self, db: Session, user_id: int, conversation_ids: list, tag: str):
        """Bulk tagging utility."""
        logger.info(f"Batch Tag started: '{tag}' for {len(conversation_ids)} items.")
        count = 0
        for cid in conversation_ids:
            if self.add_tag(db, cid, user_id, tag):
                count += 1
        return count
