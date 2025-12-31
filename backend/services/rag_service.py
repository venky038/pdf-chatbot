import os
import re
import logging
import json
import pickle
import io
import base64
import asyncio
import httpx
import numpy as np
import fitz  # PyMuPDF
import shutil
from typing import AsyncGenerator, List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

try:
    from markitdown import MarkItDown
    markitdown = MarkItDown()
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

# Set up a dedicated logger for RAG operations to keep logs organized
logger = logging.getLogger("QueryMate.RAG")

# --- OCR ENGINE INITIALIZATION ---
# Scanned PDFs contain images, not text. We use RapidOCR to 'read' these pixels.
try:
    from rapidocr_onnxruntime import RapidOCR
    ocr_engine = RapidOCR()
    OCR_AVAILABLE = True
    logger.info("RapidOCR engine initialized. Scanned PDF support enabled.")
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("RapidOCR not found. System will only process 'selectable' text PDFs.")

class RAGService:
    """
    RAGService handles the entire 'Retrieval-Augmented Generation' lifecycle:
    1. Parsing documents (PDF/Images).
    2. Chunking and indexing into Vector/Keyword stores.
    3. Hybrid Search (Vector + BM25) with Re-ranking.
    4. AI-driven response generation with citations.
    """

    def __init__(self, api_key: str, vector_store_dir: str, upload_dir: str):
        # Sanitize API key: handles accidental 'Bearer' prefix or extra whitespace
        self.api_key = api_key.replace("Bearer ", "").strip() if api_key else ""
        self.vector_store_dir = vector_store_dir
        self.upload_dir = upload_dir
        
        self._embeddings = None
        self._reranker = None
        
        # Intelligent splitter to keep related sentences together
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        self.prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
        logger.info(f"RAG Service Initialized. Models will load on first use.")

    def _get_prompt(self, name: str, default: str) -> str:
        """Loads a prompt template from the prompts folder."""
        path = os.path.join(self.prompts_dir, f"{name}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return default

    @property
    def embeddings(self):
        if self._embeddings is None:
            logger.info("Loading Embeddings Model (all-MiniLM-L6-v2)...")
            self._embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return self._embeddings

    @property
    def reranker(self):
        if self._reranker is None:
            logger.info("Loading Re-ranker Model (ms-marco-MiniLM-L-6-v2)...")
            self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return self._reranker
        
        if not self.api_key:
            logger.error("PERPLEXITY_API_KEY is missing! Reasoning features will be disabled.")

    def get_global_vs_id(self, user_id: int) -> str:
        """Standardized ID for a user's master document collection."""
        return f"user_{user_id}_global"

    def has_documents(self, user_id: int) -> bool:
        """Checks if the user has ever successfully indexed a document."""
        global_vs_id = self.get_global_vs_id(user_id)
        store_path = os.path.join(self.vector_store_dir, global_vs_id)
        bm25_path = os.path.join(self.vector_store_dir, f"{global_vs_id}_bm25.pkl")
        return os.path.exists(os.path.join(store_path, "index.faiss")) and os.path.exists(bm25_path)

    async def generate_one_liner_summary(self, text: str) -> str:
        """Asks the AI to generate a ultra-concise title for a chat session based on document content."""
        if not text: return "Document Chat"
        raw_prompt = self._get_prompt("one_liner_summary", "You are a professional assistant. Provide a title.\n---\nProvide a title for: {text}")
        if "---" in raw_prompt:
            system_msg, user_template = raw_prompt.split("---", 1)
            prompt = user_template.strip().format(text=text[:1200])
            return await self._call_perplexity_simple(prompt, system_msg.strip())
        return await self._call_perplexity_simple(raw_prompt.format(text=text[:1200]))

    async def generate_tags(self, text: str) -> List[str]:
        """Generates 3 relevant keywords for a document."""
        if not text: return ["General"]
        raw_prompt = self._get_prompt("tags_generation", "Provide 3 keywords.\n---\nExtract 3 keywords for: {text}")
        try:
            if "---" in raw_prompt:
                system_msg, user_template = raw_prompt.split("---", 1)
                prompt = user_template.strip().format(text=text[:1000])
                res = await self._call_perplexity_simple(prompt, system_msg.strip())
            else:
                res = await self._call_perplexity_simple(raw_prompt.format(text=text[:1000]))
            
            # Remove quotes, brackets, and extra punctuation before splitting
            clean_res = res.replace('"', '').replace('\'', '').replace('[', '').replace(']', '').replace('*', '')
            return [t.strip() for t in clean_res.split(",") if t.strip()][:3]
        except: return ["Document"]

    async def _call_perplexity_simple(self, prompt: str, system_msg: str = "You are a helpful assistant.") -> str:
        """Helper for standard non-streaming completions."""
        url = "https://api.perplexity.ai/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "sonar", 
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                res.raise_for_status()
                content = res.json()["choices"][0]["message"]["content"].strip()
                
                # Robust cleanup: remove markdown blocks if present
                if "```" in content:
                    content = content.replace("```json", "").replace("```", "").strip()
                
                return content
        except Exception as e:
            logger.error(f"Perplexity simple call failed: {e}")
            return "Document Chat"

    async def process_file(self, file_path: str, vs_id: str, is_append: bool = False):
        """
        Processes a single file using batch-optimized extraction and indexing.
        """
        logger.info(f"Batch Ingestion Initiated: {os.path.basename(file_path)}")
        ext = os.path.splitext(file_path)[1].lower()
        docs = []

        if ext == ".pdf":
            docs = await self._load_pdf_batch(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
            docs = await self._load_image(file_path)
        elif ext in [".docx", ".xlsx", ".pptx", ".md", ".html", ".txt"]:
            docs = await self._load_office_document(file_path)
        
        if not docs:
            logger.warning(f"File yielded no indexable content: {file_path}")
            return None

        # Bulk Chunking
        logger.info(f"Batch Processing: {len(docs)} pages -> Generating chunks...")
        chunks = self.splitter.split_documents(docs)
        
        # Bulk Embedding & Indexing (Per-Chat Store)
        await self._save_chunks(chunks, vs_id, is_append)
        
        preview = " ".join([d.page_content for d in docs[:2]])
        return {"chunks": chunks, "preview": preview[:4000]}

    async def _load_pdf_batch(self, path: str):
        """Extracts text from PDF in a batch-optimized manner with high-speed OCR fallback."""
        loop = asyncio.get_event_loop()
        
        def process_pdf():
            results = []
            try:
                doc = fitz.open(path)
                fname = os.path.basename(path)
                
                for i, page in enumerate(doc):
                    text = page.get_text()
                    
                    # If text is sparse, attempt OCR (if available)
                    if len(text.strip()) < 50 and OCR_AVAILABLE:
                        try:
                            pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
                            res, _ = ocr_engine(pix.tobytes("png"))
                            ocr_text = "\n".join([line[1] for line in res]) if res else ""
                            if len(ocr_text) > len(text):
                                text = ocr_text
                        except Exception as e:
                            logger.error(f"OCR failed for page {i+1}: {e}")
                    
                    if text.strip():
                        results.append(Document(
                            page_content=text, 
                            metadata={"page": i+1, "source": fname}
                        ))
                doc.close()
            except Exception as e:
                logger.error(f"PDF processing error: {e}")
            return results

        # Run the entire blocking operation in a thread to avoid blocking the event loop
        return await loop.run_in_executor(None, process_pdf)

    async def _load_image(self, path: str):
        """Uses Perplexity's vision features to analyze and transcribe images."""
        fname = os.path.basename(path)
        logger.info(f"Analyzing image via AI: {fname}")
        
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        
        url = "https://api.perplexity.ai/chat/completions"
        user_prompt = self._get_prompt("image_description", "Describe this image in detail. Extract any text or data visible exactly.")
        payload = {
            "model": "sonar",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}]
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 200:
                    text = res.json()["choices"][0]["message"]["content"]
                    return [Document(page_content=text, metadata={"page": 1, "source": fname})]
        except Exception as e:
            logger.error(f"AI image analysis failed: {e}")
        return []

    async def _load_office_document(self, path: str):
        """Uses MarkItDown to extract structured text from Office (Docx, Xlsx, Pptx) files."""
        if not MARKITDOWN_AVAILABLE:
            logger.error("MarkItDown not installed. Cannot process Office formats.")
            return []
        
        fname = os.path.basename(path)
        try:
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(None, markitdown.convert, path)
            text = res.text_content
            if text.strip():
                return [Document(page_content=text, metadata={"page": 1, "source": fname})]
        except Exception as e:
            logger.error(f"Office conversion failed for {fname}: {e}")
        return []

    async def _save_chunks(self, chunks, vs_id: str, is_append: bool):
        """Saves documents to both a FAISS vector index and a BM25 keyword index."""
        store_path = os.path.join(self.vector_store_dir, vs_id)
        bm25_path = os.path.join(self.vector_store_dir, f"{vs_id}_bm25.pkl")
        loop = asyncio.get_event_loop()

        def sync_save():
            # 1. Update/Create FAISS Vector Store
            if is_append and os.path.exists(store_path):
                logger.info(f"Appending {len(chunks)} chunks to existing FAISS index: {vs_id}")
                kb = FAISS.load_local(store_path, self.embeddings, allow_dangerous_deserialization=True)
                kb.add_documents(chunks)
                kb.save_local(store_path)
            else:
                logger.info(f"Creating new FAISS index: {vs_id}")
                FAISS.from_documents(chunks, self.embeddings).save_local(store_path)
            
            # 2. Update/Create BM25 Keyword Store
            all_chunks = chunks
            if is_append and os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f: 
                    data = pickle.load(f)
                    all_chunks = data["chunks"] + chunks
            
            logger.info(f"Updating BM25 Keyword Index (Total Chunks: {len(all_chunks)})")
            bm25 = BM25Okapi([c.page_content.lower().split() for c in all_chunks])
            with open(bm25_path, "wb") as f: 
                pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)

        await loop.run_in_executor(None, sync_save)
        logger.info(f"Physical indices finalized for vs_id: {vs_id}")

    async def answer_question_stream(self, question: str, vs_id: str, history: List[Dict], user_id: int = None) -> AsyncGenerator[str, None]:
        """
        The Main Reasoning Loop:
        1. Context Search (Hybrid).
        2. Prompt Engineering (Injection of facts).
        3. Streaming Output from LLM.
        """
        loop = asyncio.get_event_loop()
        
        # 1. Route to the correct knowledge base
        search_id = vs_id
        is_global = False
        
        # If no specific context, try the user's global knowledge base
        if (not vs_id or vs_id == "null") and user_id:
            search_id = self.get_global_vs_id(user_id)
            is_global = True
            logger.info(f"🌐 Querying User {user_id}'s Global Knowledge Base...")

        # 2. QUERY EXPANSION (Multi-Query Retrieval)
        # We generate 3 variations of the user's question to improve retrieval coverage
        queries = await self._generate_expanded_queries(question)
        logger.info(f"Expanded Queries: {queries}")

        # 3. Retrieve the most relevant paragraphs using expanded queries
        context = await loop.run_in_executor(None, self._multi_search, queries, search_id)
        
        if not context or len(context.strip()) < 10:
            logger.info("Search returned no relevant snippets.")
            if is_global:
                yield "🔍 I've searched through all your uploaded documents, but I couldn't find a specific answer to that. Could you try rephrasing or uploading a new file with more details?"
                return
            context = "No relevant document excerpts found."

        sys_prompt = self._get_prompt("chat_system", "You are a professional research assistant. Include citations.")
        
        # Prepare structured message list for the AI
        messages = [{"role": "system", "content": sys_prompt}]
        
        # Clean up conversation history to avoid 'user-user' consecutive roles
        clean_history = []
        for i, m in enumerate(history):
            if i == len(history) - 1 and m["role"] == "user": continue
            
            # Clean content: Use regex to remove [FOLLOW_UPS] block and everything after it
            content = m.get("content", "")
            content = re.sub(r"\[FOLLOW_UPS\].*?(\[MSID:\d+\]|$)", "", content, flags=re.DOTALL)
            
            clean_history.append({"role": m["role"], "content": content.strip()})
        
        messages.extend(clean_history)
        messages.append({"role": "user", "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION: {question}"})

        logger.info("Streaming response from AI...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": "sonar", "messages": messages, "stream": True}
            ) as res:
                if res.status_code != 200:
                    logger.error(f"AI Stream Error: {res.status_code}")
                    yield f"Error: The AI service returned {res.status_code}"
                    return

                async for line in res.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            token = chunk["choices"][0]["delta"].get("content", "")
                            if token: yield token
                        except: pass
        
        # 4. Generate Smart Follow-up suggestions
        suggestions = await self._generate_follow_ups(question, context)
        if suggestions:
            yield f"\n\n[FOLLOW_UPS]{json.dumps(suggestions)}"

    async def _generate_expanded_queries(self, query: str) -> List[str]:
        """Technique: Multi-Query Retrieval. Generates variations to find more relevant snippets."""
        prompt_template = self._get_prompt("query_expansion", "Given query: '{query}', generate 3 variations. Return JSON.")
        prompt = prompt_template.format(query=query)
        try:
            res_json = await self._call_perplexity_simple(prompt, "You are a research query optimizer. Return only JSON.")
            queries = json.loads(res_json)
            if isinstance(queries, list): return [query] + queries[:3]
        except: pass
        return [query]

    async def _generate_follow_ups(self, query: str, context: str) -> List[str]:
        """Generates 3 logical next questions based on the current context."""
        logger.info("Generating Smart Follow-ups...")
        prompt_template = self._get_prompt("follow_ups", "Generate 3 follow-ups for: {context}. Return JSON.")
        prompt = prompt_template.format(context=context[:1500])
        try:
            res_json = await self._call_perplexity_simple(prompt, "You are a research assistant. Return only JSON.")
            logger.info(f"Follow-up Raw Response: {res_json}")
            questions = json.loads(res_json)
            if isinstance(questions, list):
                logger.info(f"Successfully generated {len(questions)} follow-ups.")
                return questions[:3]
        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}")
        return []

    def _multi_search(self, queries: List[str], vs_id: str) -> str:
        """Runs search for multiple query variations and aggregates results."""
        all_candidates = []
        for q in queries:
            results = self._hybrid_search(q, vs_id, top_k=5, raw_result=True)
            if isinstance(results, list):
                all_candidates.extend(results)
        
        if not all_candidates: return ""

        # Deduplicate and Re-rank entire pool
        unique = {d.page_content: d for d in all_candidates}
        candidates = list(unique.values())
        
        # Use first query for re-ranking (the primary intent)
        scores = self.reranker.predict([[queries[0], d.page_content] for d in candidates])
        top = [c for c, s in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:8]]
        
        context_blocks = []
        for d in top:
            fname = os.path.basename(d.metadata.get("source", "Unknown PDF"))
            # Strip 36-char UUID prefix if present
            if len(fname) > 37 and fname[36] == "_":
                fname = fname[37:]
            page = d.metadata.get("page", "?")
            context_blocks.append(f"[Source: {fname} - Page {page}] {d.page_content}")
            
        return "\n\n".join(context_blocks)

    def _hybrid_search(self, query: str, vs_id: str, top_k: int = 6, raw_result: bool = False):
        """Combines Vector (Semantic) and BM25 (Keyword) search, then reranks with a Cross-Encoder."""
        logger.info(f"Retrieval Research: '{query}'")
        store_path = os.path.join(self.vector_store_dir, vs_id)
        bm25_path = os.path.join(self.vector_store_dir, f"{vs_id}_bm25.pkl")
        
        if not os.path.exists(store_path) or not os.path.exists(bm25_path):
            logger.warning(f"Files for vector store {vs_id} are missing.")
            return [] if raw_result else ""

        # Step 1: Semantic Retrieval (finds meaning)
        kb = FAISS.load_local(store_path, self.embeddings, allow_dangerous_deserialization=True)
        v_docs = kb.similarity_search(query, k=15)
        
        # Step 2: Keyword Retrieval (finds exact words)
        with open(bm25_path, "rb") as f: 
            data = pickle.load(f)
        b_docs = data["bm25"].get_top_n(query.lower().split(), data["chunks"], n=15)
        
        # Merge and deduplicate candidates
        unique = {d.page_content: d for d in (v_docs + b_docs)}
        candidates = list(unique.values())
        logger.info(f"Candidates gathered: {len(v_docs)} semantic + {len(b_docs)} keyword. Unique: {len(candidates)}")
        
        if not candidates: return [] if raw_result else ""

        # Step 3: Re-Ranking (The Judge)
        # We send the question and all 30 snippets to a model that decides which ones ARE ACTUALLY USEFUL.
        scores = self.reranker.predict([[query, d.page_content] for d in candidates])
        
        # Return raw Document objects if requested (for _multi_search)
        if raw_result:
            return [c for c, s in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:10]]

        # Pick the top k most relevant snippets to feed the AI
        top = [c for c, s in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]]
        
        context_blocks = []
        for d in top:
            fname = os.path.basename(d.metadata.get("source", "Unknown PDF"))
            page = d.metadata.get("page", "?")
            context_blocks.append(f"[Source: {fname} - Page {page}] {d.page_content}")
            
        return "\n\n".join(context_blocks)

    async def deep_search(self, query: str, user_id: int) -> List[Dict]:
        """Searches across the user's entire historical knowledge base for specific snippets."""
        global_vs_id = self.get_global_vs_id(user_id)
        loop = asyncio.get_event_loop()
        
        # We reuse the hybrid search logic but return it as structured data for the UI
        # We request raw Document objects to extract metadata
        docs = await loop.run_in_executor(None, self._hybrid_search, query, global_vs_id, 10, True)
        
        results = []
        if isinstance(docs, list):
            for d in docs:
                raw_filename = os.path.basename(d.metadata.get("source", "Unknown"))
                # Clean filename for display: strip the UUID prefix (the first underscore)
                display_name = raw_filename.split("_", 1)[-1] if "_" in raw_filename else raw_filename
                
                results.append({
                    "content": d.page_content,
                    "source": display_name,
                    "raw_source": raw_filename,
                    "page": d.metadata.get("page", "?")
                })
        return results

    async def get_library_themes(self, user_id: int) -> Dict:
        """Analyzes the user's master index to extract a conceptual map of their research."""
        global_vs_id = self.get_global_vs_id(user_id)
        loop = asyncio.get_event_loop()
        
        # 1. Get a representative sample of snippets
        docs = await loop.run_in_executor(None, self._hybrid_search, "General Overview Topics", global_vs_id, 20, True)
        if not docs: return {"themes": [], "links": []}

        # 2. Ask AI to synthesize themes and links
        text_sample = "\n".join([d.page_content[:300] for d in docs])
        prompt = self._get_prompt("knowledge_map", "Analyze fragments and build a graph. Return JSON.")
        try:
            res_json = await self._call_perplexity_simple(prompt, "You are a data analyst. Return only JSON Knowledge Graph data.")
            return json.loads(res_json)
        except:
            return {"themes": ["Documents"], "links": []}

    async def get_page_preview(self, vs_id: str, filename: str, page_num: int):
        """Locates a specific page and returns raw text. Now handles 'document' alias for better UX."""
        logger.info(f"Preview Request: {filename} (Pg {page_num}) for VS: {vs_id}")
        
        # Check if requested filename is the generic 'document' alias
        is_generic = filename.lower() == "document"
        target = None

        if not is_generic:
            # Try finding the specific file
            possible_path = os.path.join(self.upload_dir, f"{vs_id}_{filename}")
            if os.path.exists(possible_path):
                target = possible_path

        if not target:
            # Broad search fallback: find ANY file associated with this VS ID
            for f in os.listdir(self.upload_dir):
                if f.startswith(vs_id):
                    # If is_generic (from [Page X]), we take the first PDF match
                    # If specific filename given but path was weird, we fuzzy match
                    if is_generic or filename in f:
                        target = os.path.join(self.upload_dir, f)
                        break
        
        if not target or not os.path.exists(target): 
            logger.warning(f"Preview Target NOT FOUND: {filename}")
            return None
        
        try:
            doc = fitz.open(target)
            if 0 <= page_num - 1 < len(doc):
                text = doc[page_num - 1].get_text()
                doc.close()
                return text
            doc.close()
        except Exception as e:
            logger.error(f"Preview extraction error: {e}")
        return None

    async def generate_summary_stream(self, transcript: str) -> AsyncGenerator[str, None]:
        """Provides a total overview with citations mandated for credibility."""
        prompt_template = self._get_prompt("summary_report", "Summarize this: {transcript}")
        prompt = prompt_template.format(transcript=transcript[:18000])
        messages = [
            {"role": "system", "content": "You are a research assistant who prioritizes data provenance and citations."}, 
            {"role": "user", "content": prompt}
        ]
        
        logger.info("Generating analytical summary with citations...")
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": "sonar", "messages": messages, "stream": True}
            ) as res:
                async for line in res.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            token = chunk["choices"][0]["delta"].get("content", "")
                            if token: yield token
                        except: pass

    async def save_to_global_store(self, chunks: List[Document], user_id: int):
        """Appends new chunks to the user's master index for cross-chat research."""
        global_vs_id = self.get_global_vs_id(user_id)
        logger.info(f"🌎 Mirroring {len(chunks)} chunks to Global Store for User {user_id}")
        await self._save_chunks(chunks, global_vs_id, is_append=True)
