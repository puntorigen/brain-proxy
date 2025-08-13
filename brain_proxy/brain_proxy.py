"""
brain_proxy.py  —  FastAPI / ASGI router with LangMem + Chroma

pip install fastapi openai langchain-chroma langmem tiktoken
"""

from __future__ import annotations
import asyncio, base64, hashlib, json, time, re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union
from .tools import get_registry
from .__version__ import __version__

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from litellm import acompletion, embedding
from langchain.embeddings.base import Embeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from .temporal_utils import extract_timerange
from .upstash_adapter import upstash_vec_factory, UpstashAsyncWrapper as UpstashVectorStore
from .chroma_adapter import chroma_vec_factory, ChromaAsyncWrapper
#import litellm
#litellm._turn_on_debug()

# For creating proper Memory objects
class Memory(BaseModel):
    content: str

# LangMem primitives (functions, not classes)
from langmem import create_memory_manager

# -------------------------------------------------------------------
# Session Memory Manager for ephemeral sessions
# -------------------------------------------------------------------
class SessionMemoryManager:
    """Manages ephemeral session memories with intelligent summarization."""
    
    def __init__(
        self, 
        tenant_id: str,
        memory_model: str,
        max_recent: int = 30,
        summarize_after: int = 50,
        max_memory_mb: float = 10.0
    ):
        self.tenant_id = tenant_id
        self.memory_model = memory_model
        self.max_recent = max_recent
        self.summarize_after = summarize_after
        self.max_memory_mb = max_memory_mb
        
        self.memories: List[Dict[str, Any]] = []
        self.summaries: List[Dict[str, Any]] = []
        self.created_at = datetime.now(timezone.utc)
        self.last_accessed = datetime.now(timezone.utc)
        self.message_count = 0
        
    def update_access_time(self):
        """Update the last accessed time for TTL management."""
        self.last_accessed = datetime.now(timezone.utc)
        
    async def add_memory(self, content: str, role: str = "user") -> None:
        """Add a new memory and trigger summarization if needed."""
        self.memories.append({
            "content": content,
            "role": role,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        self.message_count += 1
        self.update_access_time()
        
        # Trigger summarization if we exceed the threshold
        if len(self.memories) > self.summarize_after:
            await self._summarize_old_memories()
            
    async def _summarize_old_memories(self) -> None:
        """Summarize older memories to prevent overflow."""
        if len(self.memories) <= self.max_recent:
            return
            
        # Get memories to summarize (all except the most recent ones)
        to_summarize = self.memories[:-self.max_recent]
        
        # Group by hour for summarization
        hourly_groups = {}
        for mem in to_summarize:
            timestamp = datetime.fromisoformat(mem["timestamp"])
            hour_key = timestamp.strftime("%Y-%m-%d %H:00")
            
            if hour_key not in hourly_groups:
                hourly_groups[hour_key] = []
            hourly_groups[hour_key].append(mem)
            
        # Create summaries using the memory model
        from litellm import acompletion
        
        for hour_key, memories in hourly_groups.items():
            # Build conversation for summarization
            messages_text = "\n".join([
                f"{m['role']}: {m['content']}" for m in memories
            ])
            
            summary_prompt = f"""Summarize this conversation segment concisely, preserving key facts, decisions, and context:

{messages_text}

Provide a brief summary (2-3 sentences) capturing the essential information."""

            try:
                response = await _safe_acompletion(
                    model=self.memory_model,
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=200
                )
                
                summary_content = response.choices[0].message.content
                
                self.summaries.append({
                    "summary": summary_content,
                    "period": hour_key,
                    "message_count": len(memories),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                # Fall back to simple concatenation if summarization fails
                self.summaries.append({
                    "summary": f"[{len(memories)} messages from {hour_key}]",
                    "period": hour_key,
                    "message_count": len(memories),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        # Keep only recent memories
        self.memories = self.memories[-self.max_recent:]
        
    def get_all_memories(self) -> List[str]:
        """Get all memories including summaries for retrieval."""
        result = []
        
        # Add summaries first (older context)
        for summary in self.summaries:
            result.append(f"[Summary from {summary['period']}]: {summary['summary']}")
            
        # Add recent memories
        for mem in self.memories:
            result.append(f"{mem['content']}")
            
        return result
        
    def get_session_data(self) -> Dict[str, Any]:
        """Get all session data for the on_session_end callback."""
        return {
            "tenant_id": self.tenant_id,
            "messages": self.memories.copy(),
            "summaries": self.summaries.copy(),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "message_count": self.message_count
        }
        
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        import sys
        total_size = sys.getsizeof(self.memories) + sys.getsizeof(self.summaries)
        return total_size / (1024 * 1024)


_MEMORY_INSTRUCTIONS = """
You are a long-term memory manager that maintains semantic, procedural, and episodic memories
for a life-long learning agent.

--------------------------------------------------------------------------------
0. ⛔️  FILTER  ⛔️
Before doing anything else, IGNORE and DO NOT STORE:
  • Transient errors, one-off failures, “I don't have access to X”, or logging noise.
  • Ephemeral operational states of the system (latency, rate limits, debug traces).
  • Polite fillers, apologies, or meta-comments that do not change future behaviour.
  • Messages that merely repeat existing memories without adding new facts.
  • Messages that are related to a tool that's going to be executed or called.
  • Messages that are not in the format of a tool call.

--------------------------------------------------------------------------------
1. 📥  EXTRACT & CONTEXTUALISE
  • Capture stable facts, user preferences, goals, constraints, and relationships.
  • When uncertain, tag with a confidence score (p(x)=…).
  • Quote supporting snippets only when strictly necessary.
  • Always keep the timestamps (date/time) of the messages.
  • Always respond in english.

--------------------------------------------------------------------------------
2. 🔄  COMPARE & UPDATE
  • Detect novelty vs existing store; merge or supersede as needed.
  • Compress or discard redundant memories to keep the store dense.
  • Remove information proven false or obsolete.

--------------------------------------------------------------------------------
3. 🧠  SYNTHESISE & REASON
  • Infer patterns, habits, or higher-level rules that will guide future actions.
  • Generalise when possible and annotate with probabilistic confidence.

--------------------------------------------------------------------------------
4. 📝  WRITE
Store each memory exactly as you would like to recall it when deciding how to act.
Prioritise:
  • Surprising deviations from prior patterns.
  • Persistent facts repeatedly reinforced.
  • Information that will affect long-term strategy or user satisfaction.

Do **NOT** store anything that violates step 0.  Favour dense, declarative sentences
over raw chat fragments.  Use the agent’s first-person voice when relevant (“I…”).
"""



# -------------------------------------------------------------------
# Pydantic schemas (OpenAI spec + file‑data part)
# -------------------------------------------------------------------
class FileData(BaseModel):
    name: str
    mime: str
    data: str  # base‑64 bytes


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None
    file_data: Optional[FileData] = Field(None, alias="file_data")


class ChatMessage(BaseModel):
    role: str
    content: str | List[ContentPart]
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None  # OpenAI-compatible tools format


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


async def _maybe(fn, *a, **k):
    return await fn(*a, **k) if asyncio.iscoroutinefunction(fn) else fn(*a, **k)


async def _safe_acompletion(**kwargs):
    """
    Wrapper for acompletion with basic retry logic for production stability.
    Only retries on clearly transient errors.
    """
    max_retries = 2  # Conservative retry count
    
    for attempt in range(max_retries + 1):
        try:
            return await acompletion(**kwargs)
        except Exception as e:
            error_str = str(e).lower()
            
            # Only retry on clearly transient errors
            is_retryable = any(pattern in error_str for pattern in [
                'rate limit', 'timeout', 'connection', 'server error', 
                '429', '500', '502', '503', '504'
            ])
            
            # Don't retry on permanent errors
            is_permanent = any(pattern in error_str for pattern in [
                'invalid_request_error', 'authentication_error', 
                'permission_denied', 'invalid_api_key', 'model_not_found'
            ])
            
            if is_permanent or attempt >= max_retries:
                raise e
                
            if is_retryable and attempt < max_retries:
                delay = 1.0 * (2 ** attempt)  # 1s, 2s
                print(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                raise e


# -------------------------------------------------------------------
# Vector store factories
# -------------------------------------------------------------------
def default_vector_store_factory(tenant, embeddings, max_workers: int = 10):
    return chroma_vec_factory(f"vec_{tenant}", embeddings, max_workers=max_workers)


# -------------------------------------------------------------------
# Utility classes
# -------------------------------------------------------------------
class LiteLLMEmbeddings(Embeddings):
    """Embeddings provider that uses litellm's synchronous embedding function.
    This enables support for any provider supported by litellm.
    """
    
    def __init__(self, model: str):
        """Initialize with model in litellm format (e.g., 'openai/text-embedding-3-small')"""
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple documents"""
        results = []
        # Process each text individually to handle potential rate limits
        for text in texts:
            response = embedding(
                model=self.model,
                input=text
            )
            # Handle the response format properly
            if hasattr(response, 'data') and response.data:
                # OpenAI-like format with data.embedding
                if hasattr(response.data[0], 'embedding'):
                    results.append(response.data[0].embedding)
                # Dict format with data[0]['embedding']
                elif isinstance(response.data[0], dict) and 'embedding' in response.data[0]:
                    results.append(response.data[0]['embedding'])
            # Direct embedding array format
            elif isinstance(response, list) and len(response) > 0:
                results.append(response[0])
            # Fallback
            else:
                print(f"Warning: Unexpected embedding response format: {type(response)}")
                if isinstance(response, dict) and 'embedding' in response:
                    results.append(response['embedding'])
                elif isinstance(response, dict) and 'data' in response:
                    data = response['data']
                    if isinstance(data, list) and len(data) > 0:
                        if isinstance(data[0], dict) and 'embedding' in data[0]:
                            results.append(data[0]['embedding'])
        
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """Get embeddings for a single query"""
        response = embedding(
            model=self.model,
            input=text
        )
        
        # Handle the response format properly
        if hasattr(response, 'data') and response.data:
            # OpenAI-like format with data.embedding
            if hasattr(response.data[0], 'embedding'):
                return response.data[0].embedding
            # Dict format with data[0]['embedding']
            elif isinstance(response.data[0], dict) and 'embedding' in response.data[0]:
                return response.data[0]['embedding']
        # Direct embedding array format
        elif isinstance(response, list) and len(response) > 0:
            return response[0]
        # Dictionary format
        elif isinstance(response, dict):
            if 'data' in response:
                data = response['data']
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict) and 'embedding' in data[0]:
                        return data[0]['embedding']
            elif 'embedding' in response:
                return response['embedding']
        
        # If we get here, print the response type for debugging
        print(f"Warning: Unexpected embedding response format: {type(response)}")
        print(f"Response content: {response}")
        
        # Return empty list as fallback (should not happen)
        return []


# -------------------------------------------------------------------
# BrainProxy
# -------------------------------------------------------------------
class BrainProxy:
    """Drop‑in OpenAI‑compatible proxy with Chroma + LangMem memory"""

    def __init__(
        self,
        *,
        vector_store_factory: Callable[[str, Any, int], ChromaAsyncWrapper | UpstashVectorStore] = default_vector_store_factory,
        # memory settings
        enable_memory: bool = True,
        memory_model: str = "openai/gpt-4o-mini",  # litellm format e.g. "azure/gpt-35-turbo",
        # tools settings
        tools: Optional[List[Dict[str, Any]]] = None,
        use_registry_tools: bool = True,
        embedding_model: str = "openai/text-embedding-3-small",  # litellm format e.g. "azure/ada-002"
        tool_filtering_model: Optional[str] = None,  # optional fast model to filter available tools (improves quality), e.g. "azure/gpt-35-turbo"
        mem_top_k: int = 6,
        mem_working_max: int = 12,
        enable_global_memory: bool = False, # enables _global tenant access from all tenants
        # misc
        default_model: str = "openai/gpt-4o-mini",  # litellm format e.g. "azure/gpt-4"
        storage_dir: str | Path = "tenants",
        extract_text: Callable[[Path, str], str | List[Document]] | None = None,
        manager_fn: Callable[..., Any] | None = None,  # multi‑agent hook
        # auth hooks
        auth_hook: Optional[Callable[[Request, str], Any]] = None,
        usage_hook: Optional[Callable[[str, int, float], Any]] = None,
        local_tools_handler: Optional[Callable[[str, str, Dict[str, Any]], Any]] = None,
        on_thinking: Optional[Callable[[str, str], Any]] = None,  # Callback (tenant_id, state) for 'thinking'/'ready' states
        max_upload_mb: int = 20,
        temporal_awareness: bool = True, # enable temporal awareness (time tracking of knowledge)
        system_prompt: Optional[str] = None,
        debug: bool = False,
        # Upstash settings
        upstash_rest_url: Optional[str] = None,
        upstash_rest_token: Optional[str] = None,
        max_workers: int = 10,
        # Session management settings
        enable_session_memory: bool = True,
        session_ttl_hours: int = 24,
        session_max_messages: int = 100,
        session_summarize_after: int = 50,
        session_memory_max_mb: float = 10.0,
        on_session_end: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ):
        # Initialize basic attributes first
        self.storage_dir = Path(storage_dir)
        self.memory_model = memory_model
        self.enable_memory = enable_memory
        
        # Initialize tools
        self.tools = []
        if use_registry_tools:
            registry = get_registry()
            self.tools.extend(registry.get_tools())
            # Add implementations to instance
            for tool_def in registry.get_tools():
                name = tool_def["function"]["name"]
                if impl := registry.get_implementation(name):
                    setattr(self, name, impl)
        if tools:
            self.tools.extend(tools)
        self.memory_model = memory_model
        self.mem_top_k = mem_top_k
        self.mem_working_max = mem_working_max
        self.enable_global_memory = enable_global_memory
        self.default_model = default_model
        self.embedding_model = embedding_model
        self.tool_filtering_model = tool_filtering_model
        self.extract_text = extract_text or (
            lambda p, m: p.read_text("utf-8", "ignore")
        )
        self.manager_fn = manager_fn
        self.auth_hook = auth_hook
        self.usage_hook = usage_hook
        self.local_tools_handler = local_tools_handler
        self.on_thinking = on_thinking
        self.max_upload_bytes = max_upload_mb * 1024 * 1024
        self._mem_managers: Dict[str, Any] = {}
        self._tenant_tools: Dict[str, Any] = {}
        self.temporal_awareness = temporal_awareness
        self.system_prompt = system_prompt
        self.debug = debug
        self.upstash_rest_url = upstash_rest_url
        self.upstash_rest_token = upstash_rest_token
        
        # Initialize session management attributes
        self.enable_session_memory = enable_session_memory
        self.session_ttl_hours = session_ttl_hours
        self.session_max_messages = session_max_messages
        self.session_summarize_after = session_summarize_after
        self.session_memory_max_mb = session_memory_max_mb
        self.on_session_end = on_session_end
        self._session_memories: Dict[str, SessionMemoryManager] = {}
        self._session_ttl: Dict[str, datetime] = {}

        # Initialize embeddings using litellm's synchronous embedding function
        underlying_embeddings = LiteLLMEmbeddings(model=self.embedding_model)
        fs = LocalFileStore(f"{self.storage_dir}/embeddings_cache")
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=underlying_embeddings,
            document_embedding_cache=fs,
            namespace=self.embedding_model
        )
        
        # Set up vector store factory
        if upstash_rest_url and upstash_rest_token:
            # Use Upstash if credentials are provided
            self.vec_factory = lambda tenant: upstash_vec_factory(
                tenant,
                self.embeddings,
                upstash_rest_url,
                upstash_rest_token,
                max_workers=max_workers
            )
        else:
            # Otherwise use ChromaDB
            self.vec_factory = lambda tenant: vector_store_factory(tenant, self.embeddings, max_workers=max_workers)
        
        self.router = APIRouter()
        self._mount()

    def _log(self, message: str, *args) -> None:
        """Log debug messages only when debug is enabled."""
        if self.debug:
            print(message, *args)

    def _maybe_prefix(self, text: str) -> str:
        """Return [timestamp] text if temporal_awareness on; else plain text."""
        if self.temporal_awareness:
            return f"[{datetime.now(timezone.utc).isoformat()}] {text}"
        return text

    # ----------------------------------------------------------------
    # Memory helpers
    # ----------------------------------------------------------------
    def _parse_tenant_session(self, tenant: str) -> Tuple[str, Optional[str]]:
        """Parse tenant ID into base tenant and optional session ID.
        
        Args:
            tenant: Full tenant ID (e.g., "tenant1" or "tenant1:session_id")
            
        Returns:
            Tuple of (base_tenant, session_id) where session_id is None if no session
        """
        if ':' in tenant and self.enable_session_memory:
            parts = tenant.split(':', 1)
            base_tenant = parts[0]
            session_id = parts[1]
            
            # Validate session ID to prevent injection
            if not re.match(r'^[\w\+\-\.\@]+$', session_id):
                raise ValueError(f"Invalid session ID format: {session_id}")
                
            return base_tenant, session_id
        return tenant, None
    
    # ----------------------------------------------------------------
    async def _get_or_create_session(self, full_tenant_id: str) -> SessionMemoryManager:
        """Get existing session or create new one, with TTL refresh."""
        now = datetime.now(timezone.utc)
        
        # Only cleanup expired sessions occasionally to avoid blocking every request
        if not hasattr(self, '_last_cleanup') or (now - self._last_cleanup).total_seconds() > 300:  # 5 minutes
            self._last_cleanup = now
            asyncio.create_task(self._cleanup_expired_sessions())
        
        if full_tenant_id in self._session_memories:
            # Session exists - refresh TTL
            self._session_ttl[full_tenant_id] = now
            session = self._session_memories[full_tenant_id]
            session.update_access_time()
            return session
        
        # Create new session
        session = SessionMemoryManager(
            tenant_id=full_tenant_id,
            memory_model=self.memory_model,
            max_recent=self.session_max_messages // 3,  # Keep 1/3 as recent
            summarize_after=self.session_summarize_after,
            max_memory_mb=self.session_memory_max_mb
        )
        
        self._session_memories[full_tenant_id] = session
        self._session_ttl[full_tenant_id] = now
        return session
    
    # ----------------------------------------------------------------
    async def _cleanup_expired_sessions(self):
        """Clean up sessions that have exceeded TTL."""
        now = datetime.now(timezone.utc)
        ttl_delta = timedelta(hours=self.session_ttl_hours)
        
        expired_sessions = []
        for tenant_id, last_access in self._session_ttl.items():
            if now - last_access > ttl_delta:
                expired_sessions.append(tenant_id)
        
        for tenant_id in expired_sessions:
            session = self._session_memories.get(tenant_id)
            
            # Clean up the session immediately
            if tenant_id in self._session_memories:
                del self._session_memories[tenant_id]
            if tenant_id in self._session_ttl:
                del self._session_ttl[tenant_id]
                
            self._log(f"Cleaned up expired session: {tenant_id}")
            
            # Call the on_session_end callback in background to avoid blocking
            if session and self.on_session_end:
                asyncio.create_task(self._call_session_end_callback(tenant_id, session))
    
    async def _call_session_end_callback(self, tenant_id: str, session):
        """Call the on_session_end callback in background."""
        try:
            await _maybe(self.on_session_end, tenant_id, session.get_session_data())
        except Exception as e:
            self._log(f"Error in on_session_end callback: {e}")
    
    # ----------------------------------------------------------------
    def _get_mem_manager(self, tenant: str):
        """Get or create memory manager for tenant"""
        # For sessions, we need to use the base tenant's memory manager
        base_tenant, session_id = self._parse_tenant_session(tenant)
        
        # Use base tenant for persistent memories
        mem_key = base_tenant
        
        if mem_key in self._mem_managers:
            return self._mem_managers[mem_key]

        # use the base tenant's chroma collection for memory as well
        vec = self.vec_factory(f"{mem_key}_memory")
        async def _search_mem(query: str, k: int):
            docs = await vec.similarity_search(query, k=k)
            return [d.page_content for d in docs]

        async def _store_mem(memories: List[Any]):
            """Store memories in the vector database."""
            docs = []
            for m in memories:
                try:
                    # Convert any memory format to a string and store it
                    if hasattr(m, 'content'):
                        content = str(m.content)
                    elif isinstance(m, dict) and 'content=' in m:
                        content = str(m['content='])
                    elif isinstance(m, dict) and 'content' in m:
                        content = str(m['content'])
                    elif isinstance(m, str):
                        content = m
                    else:
                        content = str(m)
                    
                    now_iso = datetime.now(timezone.utc).isoformat()
                    docs.append(
                        Document(
                            page_content=self._maybe_prefix(content),
                            metadata={
                                "timestamp": now_iso
                            }
                        )
                    )
                except Exception as e:
                    self._log(f"Error processing memory: {e}")
            
            if docs:
                self._log(f"Storing {len(docs)} memories for tenant {mem_key}")
                await vec.add_documents(docs)
                self._log(f"Successfully stored memories")

        # Use langchain_litellm's ChatLiteLLM for memory manager directly
        # No wrapper to avoid potential deadlocks
        manager = create_memory_manager(ChatLiteLLM(model=self.memory_model), instructions=_MEMORY_INSTRUCTIONS)
        
        self._mem_managers[mem_key] = (manager, _search_mem, _store_mem)
        return self._mem_managers[mem_key]

    async def _retrieve_memories(self, tenant: str, user_text: str) -> str:
        """Return a '\n'-joined block of relevant memories (filtered by time if possible)."""
        if not self.enable_memory:
            self._log(f"Memory disabled for tenant {tenant}")
            return ""

        # Parse tenant to check for session
        base_tenant, session_id = self._parse_tenant_session(tenant)
        
        # Get base tenant memories
        mgr, search, _ = self._get_mem_manager(tenant)  # Uses base_tenant internally
        if not mgr:
            self._log(f"No memory manager found for tenant {base_tenant}")
            return ""

        # 1️⃣  broad search in parallel
        raw: List[str] = []
        search_tasks = [search(user_text, k=self.mem_top_k * 3)]

        # Get global memories
        global_mgr, global_search, _ = self._get_mem_manager('_global')

        if self.enable_global_memory and global_mgr:
            search_tasks.append(global_search(user_text, k=self.mem_top_k * 3))
        
        # Gather results from all searches
        results = await asyncio.gather(*search_tasks)
        raw.extend(results[0])  # Base tenant memories
        if self.enable_global_memory and global_mgr:
            raw.extend(results[1])  # Global memories
        
        # Add session memories if we have a session
        if session_id and self.enable_session_memory:
            session = await self._get_or_create_session(tenant)
            session_memories = session.get_all_memories()
            
            # Add session memories with priority (they're more recent/relevant)
            for mem in session_memories:
                # Add a marker to distinguish session memories
                raw.insert(0, f"[SESSION] {mem}")
            
            self._log(f"Added {len(session_memories)} session memories for {tenant}")

        # 2️⃣  try to detect a date / relative phrase
        timerange = extract_timerange(user_text) if self.temporal_awareness else None
        if timerange:
            start, end = timerange
            filtered = []
            for mem in raw:
                # we stored ISO timestamps in the memory doc’s metadata and also
                # prefixed them in text like “[2025-06-20T14:03:00+00:00] …”
                m = re.match(r"\[(\d{4}-\d{2}-\d{2}T[^]]+)\]", mem)
                ts = m.group(1) if m else ""
                if ts and start.isoformat() <= ts <= end.isoformat():
                    filtered.append(mem)
            memories = filtered or raw
        else:
            memories = raw

        # Sort memories by timestamp (oldest first) if they have timestamps
        def extract_timestamp(memory):
            m = re.match(r"\[(\d{4}-\d{2}-\d{2}T[^]]+)\]", memory)
            return m.group(1) if m else "0" # Default to oldest if no timestamp

        memories.sort(key=extract_timestamp)  # Oldest first
        # Take the last k memories (most recent ones)
        memories = memories[-self.mem_top_k:]
        return "\\n".join(memories)  # Return the last k memories

    async def _write_memories(
        self, tenant: str, conversation: List[Dict[str, Any]]
    ):
        """Extract and store memories from the conversation."""
        if not self.enable_memory:
            return
        # Create a background task instead of processing immediately
        asyncio.create_task(self._process_memories_background(tenant, conversation))
        
    async def _process_memories_background(
        self, tenant: str, conversation: List[Dict[str, Any]]
    ):
        """Process and store memories in the background."""
        # Parse tenant to check for session
        base_tenant, session_id = self._parse_tenant_session(tenant)
        
        # Handle session memories if we have a session
        if session_id and self.enable_session_memory:
            try:
                session = await self._get_or_create_session(tenant)
                
                # Store recent conversation in session memory
                for msg in conversation:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if content:
                            await session.add_memory(content, role)
                
                self._log(f"Stored {len(conversation)} messages in session memory for {tenant}")
                
                # Check if session memory usage is too high
                if session.estimate_memory_usage() > self.session_memory_max_mb:
                    self._log(f"Session memory usage high for {tenant}, triggering summarization")
                    await session._summarize_old_memories()
                    
            except Exception as e:
                self._log(f"Error storing session memories: {e}")
        
        # Also process persistent memories for base tenant (not for pure sessions)
        # Only store significant information in persistent memory
        if not session_id or len(conversation) > 5:  # Threshold for significance
            manager_tuple = self._get_mem_manager(tenant)  # Uses base_tenant internally
            if not manager_tuple:
                return
            manager, _, store = manager_tuple
        
        try:
            # Get memories from the manager
            self._log(f"Extracting memories for tenant {tenant}")
            raw_memories = await manager(conversation)
            
            # Debug logging to understand the format
            self._log(f"Raw memory count: {len(raw_memories) if raw_memories else 0}")
            if raw_memories and self.debug:
                for i, mem in enumerate(raw_memories):
                    self._log(f"Raw memory {i+1} type: {type(mem)}")
                    if hasattr(mem, 'id') and hasattr(mem, 'content'):
                        self._log(f"  String representation: {str(mem)[:50]}")
            
            # Convert ExtractedMemory objects to proper format
            if raw_memories:
                # Create a list to hold properly formatted memories
                proper_memories = []
                
                for mem in raw_memories:
                    try:
                        # Extract the content properly based on the object type
                        
                        # Case 1: ExtractedMemory named tuple (id, content)
                        now_iso = datetime.now(timezone.utc).isoformat()
                        if hasattr(mem, 'id') and hasattr(mem, 'content'):
                            if hasattr(mem.content, 'content'):
                                # Extract content from the BaseModel
                                content = mem.content.content
                                formatted_mem = {"content": self._maybe_prefix(content)}
                                proper_memories.append(formatted_mem)
                            elif hasattr(mem.content, 'model_dump'):
                                # Extract content using model_dump method
                                model_data = mem.content.model_dump()
                                if 'content' in model_data:
                                    formatted_mem = {"content": self._maybe_prefix(model_data['content'])}
                                    proper_memories.append(formatted_mem)
                                else:
                                    # If no content field, use the whole model data as string
                                    formatted_mem = {"content": self._maybe_prefix(str(model_data))}
                                    proper_memories.append(formatted_mem)
                            elif isinstance(mem.content, dict) and 'content' in mem.content:
                                # Content is a dict with content field
                                formatted_mem = {"content": self._maybe_prefix(mem.content['content'])}
                                proper_memories.append(formatted_mem)
                            else:
                                # Fallback for other types
                                formatted_mem = {"content": self._maybe_prefix(str(mem.content))}
                                proper_memories.append(formatted_mem)
                                
                        # Case 2: Dictionary with 'content' key
                        elif isinstance(mem, dict) and 'content' in mem:
                            formatted_mem = {"content": self._maybe_prefix(str(mem['content']))}
                            proper_memories.append(formatted_mem)
                            
                        # Case 3: Malformed dictionaries with format {'content=': val, 'text': val}
                        elif isinstance(mem, dict) and 'content=' in mem:
                            # Find text fields (longer string keys)
                            text_keys = [k for k in mem.keys() 
                                       if k != 'content=' and isinstance(k, str) and len(k) > 10]
                            
                            if text_keys:
                                # Use the text key with actual content
                                longest_key = max(text_keys, key=len)
                                formatted_mem = {"content": self._maybe_prefix(longest_key)}
                                proper_memories.append(formatted_mem)
                                self._log(f"  Fixed complex memory format: {longest_key[:30]}...")
                            else:
                                # Fallback: concatenate all string values
                                content_parts = []
                                for k, v in mem.items():
                                    if isinstance(v, str) and len(v) > 2:
                                        content_parts.append(v)
                                    elif isinstance(k, str) and len(k) > 10 and k != 'content=':
                                        content_parts.append(k)
                                        
                                if content_parts:
                                    content = " ".join(content_parts)
                                    formatted_mem = {"content": self._maybe_prefix(content)}
                                    proper_memories.append(formatted_mem)
                                else:
                                    # Last resort: use content= value
                                    formatted_mem = {"content": self._maybe_prefix(str(mem['content=']))}
                                    proper_memories.append(formatted_mem)
                            
                        # Case 4: String value
                        elif isinstance(mem, str):
                            formatted_mem = {"content": self._maybe_prefix(mem)}
                            proper_memories.append(formatted_mem)
                            
                        # Case 5: Any other object with __dict__ attribute
                        elif hasattr(mem, '__dict__'):
                            mem_dict = mem.__dict__
                            if 'content' in mem_dict:
                                formatted_mem = {"content": self._maybe_prefix(str(mem_dict['content']))}
                                proper_memories.append(formatted_mem)
                            else:
                                # Use the entire object representation
                                formatted_mem = {"content": self._maybe_prefix(str(mem))}
                                proper_memories.append(formatted_mem)
                        
                        # If nothing worked, skip this memory
                        else:
                            self._log(f"  Could not extract content from memory: {type(mem)}")
                            
                    except Exception as e:
                        self._log(f"  Error formatting memory: {e}")
                        continue
                
                self._log(f"Formatted {len(proper_memories)} memories properly")
                
                if proper_memories:
                    # Store the properly formatted memories
                    self._log(f"Storing {len(proper_memories)} memories for tenant {tenant}")
                    await store(proper_memories)
                    self._log(f"Successfully stored memories")
                    self._log(f"Memory storage complete")
            else:
                self._log("No memories to store")
                
        except Exception as e:
            self._log(f"Error in memory processing: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            # Continue with the request even if memory fails

    # ----------------------------------------------------------------
    # File upload handling for RAG
    # ----------------------------------------------------------------
    def _split_files(
        self, msgs: List[ChatMessage]
    ) -> tuple[List[Dict[str, Any]], List[FileData]]:
        """Return messages with file data removed, plus list of file data"""
        conv_msgs: List[Dict[str, Any]] = []
        files = []

        for msg in msgs:
            # simple text-only message, no parts
            if isinstance(msg.content, str):
                conv_msgs.append({"role": msg.role, "content": msg.content})
                continue

            # one or more parts
            text_parts = []
            for part in msg.content:
                if part.type == "text":
                    text_parts.append(part.text or "")
                elif part.file_data:
                    try:
                        if len(base64.b64decode(part.file_data.data)) > self.max_upload_bytes:
                            raise ValueError(f"File too large: {part.file_data.name}")
                        files.append(part.file_data)
                    except Exception as e:
                        self._log(f"Error decoding file: {e}")

            if text_parts:
                conv_msgs.append({"role": msg.role, "content": "\n".join(text_parts)})

        return conv_msgs, files

    async def _ingest_files(self, files: List[FileData], tenant: str):
        """Ingest files into vector store. Handles both raw text and pre-processed Documents."""
        if not files:
            return
            
        # Check if this is an ephemeral session
        base_tenant, session_id = self._parse_tenant_session(tenant)
        
        if session_id is not None:
            # Block file uploads for ephemeral sessions
            raise HTTPException(
                status_code=400,
                detail="File uploads are not allowed for ephemeral sessions. Please use the base tenant endpoint for file uploads."
            )
        
        docs = []
        
        # Create tenant directory if it doesn't exist (use base_tenant for safety)
        tenant_dir = Path(f"{self.storage_dir}/{base_tenant}/files")
        tenant_dir.mkdir(exist_ok=True, parents=True)
        
        for file in files:
            self._log(f"Ingesting file: {file.name} ({file.mime})")
            try:
                name = file.name.replace(" ", "_")
                path = tenant_dir / name
                path.write_bytes(base64.b64decode(file.data))
                
                # Extract content using provided function
                content = self.extract_text(path, file.mime)
                
                # Handle both string and Document list returns
                if isinstance(content, str):
                    # Split text into chunks if it's a string
                    if content.strip():
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        chunks = text_splitter.split_text(content)
                        timestamp = datetime.now(timezone.utc).isoformat()
                        docs.extend([
                            Document(
                                page_content=chunk,
                                metadata={
                                    "name": file.name,
                                    "timestamp": timestamp,
                                    "chunk": i
                                }
                            ) for i, chunk in enumerate(chunks)
                        ])
                elif isinstance(content, list) and all(isinstance(d, Document) for d in content):
                    # If we got pre-processed Documents, just add timestamp if not present
                    timestamp = datetime.now(timezone.utc).isoformat()
                    for doc in content:
                        if "timestamp" not in doc.metadata:
                            doc.metadata["timestamp"] = timestamp
                        docs.append(doc)
                else:
                    self._log(f"Warning: extract_text returned invalid type for {file.name}")
                    
            except Exception as e:
                self._log(f"Error ingesting file: {e}")

        if docs:
            vec = self.vec_factory(base_tenant)  # Use base_tenant for file storage
            await vec.add_documents(docs)

    # ----------------------------------------------------------------
    # RAG
    # ----------------------------------------------------------------
    async def _rag(self, msgs: List[Dict[str, Any]], tenant: str, k: int = 4):
        """Retrieve info from vector store and inject it into the conversation"""
        if len(msgs) == 0:
            return msgs
        
        # Use base tenant for document retrieval
        base_tenant, _ = self._parse_tenant_session(tenant)
        vec = self.vec_factory(base_tenant)

        # get query from last message
        query = msgs[-1]["content"] if isinstance(msgs[-1]["content"], str) else ""
        if not query:
            return msgs

        docs = await vec.similarity_search(query, k=k)
        if not docs:
            return msgs

        context_str = "\n\n".join([d.page_content for d in docs])
        msgs = msgs[:-1] + [
            {
                "role": "system",
                "content": "Relevant context from documents:\n\n" + context_str,
            },
            msgs[-1],
        ]
        return msgs

    # ----------------------------------------------------------------
    # Upstream dispatch
    # ----------------------------------------------------------------
    def _validate_messages(self, messages):
        """
        Ensure messages conform to OpenAI's requirements for tool messages:
        - Each 'tool' message must be a response to a preceding assistant message with 'tool_calls'
        - Each 'tool' message must have a 'tool_call_id' that matches one in the assistant's 'tool_calls'
        """
        valid_msgs = []
        tool_call_ids = set()  # Track valid tool_call_ids from assistant messages
        
        for msg in messages:
            if msg.get("role") == "assistant":
                # Keep all assistant messages
                valid_msgs.append(msg)
                # If this assistant has tool_calls, add their IDs to our tracking set
                if msg.get("tool_calls"):
                    for tc in msg.get("tool_calls", []):
                        if tc.get("id"):
                            tool_call_ids.add(tc.get("id"))
            elif msg.get("role") == "tool":
                # Only keep tool messages that have a valid tool_call_id
                if msg.get("tool_call_id") in tool_call_ids:
                    valid_msgs.append(msg)
            else:
                # Keep all other messages (user, system, etc.)
                valid_msgs.append(msg)
        
        return valid_msgs

    async def _filter_tools_via_llm(self, msgs: list[dict], tool_defs: List[dict]) -> List[dict]:
        # 1. Get last user message content
        user_prompt = next(
            (msg["content"] for msg in reversed(msgs) if msg["role"] == "user" and isinstance(msg["content"], str)),
            None
        )
        if not user_prompt:
            return tool_defs  # fallback: return all if no user message found

        # 2. Format available tools list
        tools_str = "\n".join(
            f"- {tool['function']['name']} ({tool['function'].get('description', 'No description')})"
            for tool in tool_defs
        )

        # 3. Build prompt
        prompt = f"""You are a helpful assistant that selects the most relevant tools for a given user message.

    The user wrote:
    ```{user_prompt}```

    Available tools:
    ``` {tools_str}````

    # Try to always return the 2-5 most similar or related tools to the user message.
    # Reply strictly in JSON format like:
    {{"selected_tools": ["tool_name1", "tool_name2"]}}"""

        # 4. Run LLM call
        response = await _safe_acompletion(
            model=self.tool_filtering_model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You only reply with a JSON object listing selected tools."},
                {"role": "user", "content": prompt}
            ]
        )

        # 5. Parse the response
        try:
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from markdown code blocks if present
            if content.startswith('```'):
                # Look for content between ```json and ``` or between ``` and ```
                lines = content.split('\n')
                json_lines = []
                in_code_block = False
                
                for line in lines:
                    if line.strip().startswith('```'):
                        if in_code_block:
                            break  # End of code block
                        else:
                            in_code_block = True  # Start of code block
                            continue
                    elif in_code_block:
                        json_lines.append(line)
                
                if json_lines:
                    content = '\n'.join(json_lines).strip()
            
            # Parse the JSON content
            parsed = json.loads(content)
            selected = parsed.get("selected_tools", [])
            return [tool for tool in tool_defs if tool["function"]["name"] in selected]
        except Exception as e:
            self._log(f"[ToolFilter] Error parsing LLM response: {e}", response)
            return tool_defs  # fallback: return all

    async def _dispatch(self, msgs, model: str, *, stream: bool, tools: Optional[List[Dict[str, Any]]] = None, tenant: Optional[str] = None, temperature: Optional[float] = None):
        """Dispatch to litellm API with tools support"""
        kwargs = {
            "model": model,
            "messages": msgs,
            "stream": stream
        }
        if temperature:
            kwargs["temperature"] = temperature
        
        # Combine server-defined tools with request tools if any
        final_tools = []
        local_tools = []
        if self.tools:
            final_tools.extend(self.tools)
        if tools: # tools specified in request (should be handled by self.local_tools_handler instead of _execute_tool)
            final_tools.extend(tools)
            local_tools.extend(tools)
        if tenant in self._tenant_tools:
            final_tools.extend(self._tenant_tools[tenant])
            local_tools.extend(self._tenant_tools[tenant])
        
        by_name = {}
        for tool in final_tools:
            by_name[tool['function']['name']] = tool
        final_tools = list(by_name.values())
        filtered_tools = await self._filter_tools_via_llm(msgs, final_tools)
        if filtered_tools:
            self._log(f"➡️  Enviando {len(filtered_tools)} of {len(final_tools)} tools: {[t['function']['name'] for t in filtered_tools]}")
            kwargs["tools"] = filtered_tools
            kwargs["tool_choice"] = "auto"  # Let the model decide when to use tools
                    
        msgs = self._validate_messages(msgs)

        kwargs["messages"] = msgs
        self._log(f"➡️  Enviando kwargs: {kwargs}")
        response = await _safe_acompletion(**kwargs)
        
        # Process tool calls if present
        if not stream and hasattr(response.choices[0], "message") and hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls
            available_tools = {t["function"]["name"]: t["function"] for t in final_tools}
            local_tools = {t["function"]["name"]: t["function"] for t in local_tools}
                
            # Execute each tool call
            tool_results = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                # first try to execute local tool if in there
                local_tool_failed = False
                if function_name in local_tools:
                    try:
                        # Execute the tool with the self.local_tools_handler and get result
                        function_args = json.loads(tool_call.function.arguments)
                        tool_result = await self.local_tools_handler(tenant, function_name, function_args)
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": str(tool_result)
                        })
                    except Exception as e:
                        # if local tool fails, try to execute server tool
                        self._log(f"Error executing local tool {function_name}: {str(e)}")
                        self._log(f"(fallback) Attempting to execute it as a server tool {function_name}")
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error executing tool: {str(e)}"
                        })
                        local_tool_failed = True
                
                if function_name in available_tools and not local_tool_failed:
                    try:
                        # Execute the tool and get result
                        function_args = json.loads(tool_call.function.arguments)
                        tool_result = await self._execute_tool(function_name, function_args)
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": str(tool_result)
                        })
                    except Exception as e:
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error executing tool: {str(e)}"
                        })
            
            # If we have tool results, make a follow-up call with the results
            if tool_results:
                # Add tool results to messages
                new_msgs = self._prune_msgs_for_tool_followup(msgs) + [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in tool_calls
                        ]
                    },
                    *tool_results  # Tool results
                ]
                
                # Make follow-up call without tools to get final response
                kwargs["messages"] = new_msgs
                kwargs.pop("tools", None)  # Remove tools to prevent infinite loops
                kwargs.pop("tool_choice", None)
                response = await _safe_acompletion(**kwargs)
                
        return response
        
    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Execute a tool and return its result"""
        # First check registry
        registry = get_registry()
        if impl := registry.get_implementation(tool_name):
            if asyncio.iscoroutinefunction(impl):
                return await impl(**tool_args)
            return impl(**tool_args)
            
        # Then check instance methods
        if hasattr(self, tool_name):
            method = getattr(self, tool_name)
            if asyncio.iscoroutinefunction(method):
                return await method(**tool_args)
            return method(**tool_args)
            
        raise ValueError(f"Tool {tool_name} not found or not implemented")
        
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Return the JSON schema for available tools"""
        return self.tools or []
        
    def _prune_msgs_for_tool_followup(self, msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remueve cualquier bloque tool/tool_calls anteriores para dejar lista la secuencia."""
        pruned = list(msgs)

        # Recorre hacia atrás buscando el último bloque tool_call + tools
        for i in range(len(pruned) - 1, -1, -1):
            msg = pruned[i]
            if msg["role"] == "tool":
                # Si no hay un tool_calls justo antes, este tool es inválido
                if i == 0 or pruned[i - 1].get("tool_calls") is None:
                    pruned.pop(i)
            elif msg.get("tool_calls"):
                # Si encontramos un bloque de tool_calls seguido de tools, eliminamos ese bloque completo
                return pruned[:i]
        return pruned

    # ----------------------------------------------------------------
    # FastAPI route
    # ----------------------------------------------------------------
    def _mount(self):
        @self.router.post("/{tenant}/tools")
        async def set_tools(request: Request, tenant: str):
            # Special handling auth
            if self.auth_hook:
                await _maybe(self.auth_hook, request, tenant)

            body = await request.json()
            if not isinstance(body, list):
                raise HTTPException(status_code=400, detail="Expected array of tools")
                
            # Validate each tool has required fields
            for tool in body:
                if not isinstance(tool, dict) or 'type' not in tool or 'function' not in tool:
                    raise HTTPException(status_code=400, detail="Invalid tool schema")
            
            self._tenant_tools[tenant] = body
            return {"status": "success", "count": len(body)}

        @self.router.post("/{tenant}/chat/completions")
        async def chat(request: Request, tenant: str):
            self._log(f"Brain-Proxy - Version {__version__}")
            # Special handling auth
            if self.auth_hook:
                await _maybe(self.auth_hook, request, tenant)

            body = await request.json()
            #self._log(f"Preprocess Chat request for tenant {tenant}", body)
            req = ChatRequest(**body)
            msgs, files = self._split_files(req.messages)

            if files:
                self._log(f"Ingesting {len(files)} files for tenant {tenant}")
                await self._ingest_files(files, tenant)

            # Add global system prompt at the beginning if provided
            if self.system_prompt:
                self._log(f"Adding global system prompt: '{self.system_prompt[:30]}...'")
                # Check if the first message is already a system message
                if msgs and msgs[0].get("role") == "system":
                    # Augment existing system message
                    msgs[0]["content"] = f"{self.system_prompt}\n\n{msgs[0]['content']}"
                else:
                    # Add new system message at the beginning
                    msgs = [{"role": "system", "content": self.system_prompt}] + msgs

            # ── inject current UTC time so the model understands “hoy”, “ayer”… ──
            if self.temporal_awareness:
                now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
                msgs = (
                    [{"role": "system", "content": f"Current UTC time is {now_iso}."}]
                    + msgs
                )

            # LangMem retrieve
            if self.enable_memory:
                self._log(f"Memory enabled for tenant {tenant}, processing message")
                user_text = (
                    msgs[-1]["content"]
                    if isinstance(msgs[-1]["content"], str)
                    else next(
                        p["text"] for p in msgs[-1]["content"] if p["type"] == "text"
                    )
                )
                self._log(f"Extracting user text: '{user_text[:30]}...'")
                
                # Trigger on_thinking callback with 'thinking' state
                if self.on_thinking:
                    try:
                        await _maybe(self.on_thinking, tenant, 'thinking')
                        self._log(f"on_thinking callback triggered with 'thinking' state for tenant {tenant}")
                    except Exception as e:
                        self._log(f"Error in on_thinking callback: {e}")
                
                mem_block = await self._retrieve_memories(tenant, user_text)
                if mem_block:
                    self._log(f"Adding memory block to conversation: {len(mem_block)} chars")
                    msgs = msgs[:-1] + [
                        {
                            "role": "system",
                            "content": "Relevant memories:\n" + mem_block,
                        },
                        msgs[-1],
                    ]
                else:
                    self._log("No memory block to add")
            else:
                self._log(f"Memory disabled for tenant {tenant}")
            
            msgs = await self._rag(msgs, tenant)
            msgs = self._prune_msgs_for_tool_followup(msgs)
            original_msgs = list(msgs)  # copia para evitar mutaciones posteriores

            # set temperature only if we're assigned tools throght the endpoint for this tenant
            def get_temperature(tool_count: int) -> float:
                return max(0.1, 1.0 - 0.1 * tool_count)

            temperature_ = None
            if tenant in self._tenant_tools:
                # TODO: make this dynamic based on the number of tools assigned
                tool_count = len(self._tenant_tools[tenant])
                temperature_ = get_temperature(tool_count)
                self._log(f"Setting temperature to {temperature_} for tenant {tenant}")

            upstream_iter = await self._dispatch(
                msgs, 
                req.model or self.default_model, 
                stream=req.stream,
                tools=req.tools,
                tenant=tenant,
                temperature=temperature_
            )
            t0 = time.time()

            if not req.stream:
                # Trigger on_thinking callback with 'ready' state before sending response
                if self.on_thinking:
                    try:
                        await _maybe(self.on_thinking, tenant, 'ready')
                        self._log(f"on_thinking callback triggered with 'ready' state for tenant {tenant}")
                    except Exception as e:
                        self._log(f"Error in on_thinking callback (ready state): {e}")
                
                # No need to await here since _dispatch already returns the complete response
                response_data = upstream_iter.model_dump()
                await self._write_memories(
                    tenant, 
                    msgs 
                    + [
                        {
                            "role": "assistant",
                            "content": self._maybe_prefix(
                                upstream_iter.choices[0].message.content
                            ),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    ]
                )
                if self.usage_hook and upstream_iter.usage:
                    await _maybe(
                        self.usage_hook,
                        tenant,
                        upstream_iter.usage.total_tokens,
                        time.time() - t0,
                    )
                return JSONResponse(response_data)

            async def _process_chunk_payload(chunk) -> dict:
                """Process a chunk into a payload."""
                try:
                    return json.loads(chunk.model_dump_json())
                except Exception:
                    return chunk

            async def _handle_content_delta(delta: dict, buf: List[str], tokens: int, payload: dict) -> tuple[List[str], int, str]:
                """Handle content delta updates."""
                if "content" in delta and delta["content"] is not None:
                    buf.append(delta["content"])
                    tokens += len(delta["content"])
                    return buf, tokens, f"data: {json.dumps(payload)}\n\n"
                return buf, tokens, ""

            async def _process_tool_call(tc: dict, tool_call_parts: dict, current_call_idx: Optional[int]) -> tuple[dict, Optional[int]]:
                """Process a single tool call and update the accumulator."""
                idx = tc.get("index", current_call_idx)
                if idx is None:
                    return tool_call_parts, current_call_idx

                accum = tool_call_parts.get(idx, {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {"name": None, "arguments": ""},
                    "index": idx
                })

                fn = accum["function"]
                upd_fn = tc.get("function", {})
                if "name" in upd_fn and not fn["name"]:
                    fn["name"] = upd_fn["name"]
                if "arguments" in upd_fn:
                    fn["arguments"] += upd_fn["arguments"]

                if tc.get("id"):
                    accum["id"] = tc["id"]

                tool_call_parts[idx] = accum
                return tool_call_parts, idx

            async def _prepare_tool_calls(tool_call_parts: dict) -> List[dict]:
                """Prepare and validate tool calls for execution."""
                tool_calls = []
                for i, (_, tc_partial) in enumerate(tool_call_parts.items()):
                    tc = tc_partial.copy()
                    tc["id"] = tc.get("id") or f"call_{i}"
                    if not isinstance(tc["id"], str):
                        tc["id"] = str(tc["id"])

                    function = tc.get("function")
                    if not isinstance(function, dict):
                        continue

                    name = function.get("name")
                    if not isinstance(name, str) or not name:
                        continue

                    if "arguments" not in function or not isinstance(function["arguments"], str):
                        function["arguments"] = "{}"

                    tc["function"] = function
                    tool_calls.append(tc)
                return tool_calls

            async def _get_final_tools(tenant: str, req) -> tuple[List[dict], dict, dict]:
                """Get final tools list and create tool mappings."""
                final_tools = self.tools or []
                local_tools = req.tools or []
                if req.tools:
                    final_tools += req.tools
                    local_tools += req.tools
                if tenant in self._tenant_tools:
                    final_tools += self._tenant_tools[tenant]
                    local_tools += self._tenant_tools[tenant]

                available_tools = {t["function"]["name"]: t["function"] for t in final_tools}
                local_tools_dict = {t["function"]["name"]: t["function"] for t in local_tools or []}
                return final_tools, available_tools, local_tools_dict

            # streaming path
            async def event_stream() -> AsyncIterator[str]:
                # Trigger on_thinking callback with 'ready' state before streaming
                if self.on_thinking:
                    try:
                        await _maybe(self.on_thinking, tenant, 'ready')
                        self._log(f"on_thinking callback triggered with 'ready' state for tenant {tenant} (streaming)")
                    except Exception as e:
                        self._log(f"Error in on_thinking callback (ready state, streaming): {e}")
                
                tokens = 0
                buf: List[str] = []
                tool_call_parts: dict[str, dict] = {}
                tool_calls_detected = False
                current_call_idx = None

                async for chunk in upstream_iter:
                    payload = await _process_chunk_payload(chunk)
                    choice = payload["choices"][0]
                    delta = choice.get("delta", {})

                    # Handle content delta
                    buf, tokens, content_response = await _handle_content_delta(delta, buf, tokens, payload)
                    if content_response:
                        yield content_response

                    # Handle tool calls
                    for tc in (delta.get("tool_calls", []) or []):
                        tool_calls_detected = True
                        tool_call_parts, current_call_idx = await _process_tool_call(tc, tool_call_parts, current_call_idx)
                        yield f"data: {json.dumps(payload)}\n\n"

                    # Check for tool calls completion
                    if choice.get("finish_reason") == "tool_calls":
                        self._log(f"TOOL CALLS FINISHED! Found {len(tool_call_parts)} calls: {tool_call_parts}")
                        break

                if not tool_calls_detected:
                    yield "data: [DONE]\n\n"
                    await self._write_memories(tenant, msgs + [{
                        "role": "assistant",
                        "content": self._maybe_prefix("".join(buf)),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }])
                    if self.usage_hook:
                        await _maybe(self.usage_hook, tenant, tokens, time.time() - t0)
                    return

                # Process tool calls
                tool_calls = await _prepare_tool_calls(tool_call_parts)
                final_tools, available_tools, local_tools = await _get_final_tools(tenant, req)
                tool_results = []

                for tool_call in tool_calls:
                    name = tool_call["function"]["name"]
                    args_str = tool_call["function"].get("arguments", "")
                    args = {}
                    if args_str.strip():
                        try:
                            args = json.loads(args_str)
                        except Exception as e:
                            self._log(f"❌ Tool {name} args JSON error: {e}")
                    local_tool_failed = False

                    # Add debug logging BEFORE executing the tool
                    self._log(f"⚙️ EXECUTING TOOL: {name} with args: {args}")

                    if name in local_tools:
                        try:
                            self._log(f"⚙️ Calling local tool handler for: {name}")
                            result = await self.local_tools_handler(tenant, name, args)
                            self._log(f"⚙️ Local tool {name} result: {result}")
                            tool_results.append({
                                "tool_call_id": tool_call["id"],
                                "role": "tool",
                                "name": name,
                                "content": str(result),
                            })
                            continue
                        except Exception as e:
                            self._log(f"❌ Local tool {name} error: {e}")
                            self._log(f"❌ Exception type: {type(e)}")
                            import traceback
                            self._log(f"❌ Traceback: {traceback.format_exc()}")
                            tool_results.append({
                                "tool_call_id": tool_call["id"],
                                "role": "tool",
                                "name": name,
                                "content": f"Error: {str(e)}"
                            })
                            local_tool_failed = True

                    if name in available_tools and not local_tool_failed:
                        try:
                            self._log(f"⚙️ Calling remote tool handler for: {name}")
                            result = await self._execute_tool(name, args)
                            self._log(f"⚙️ Remote tool {name} result: {result}")
                            tool_results.append({
                                "tool_call_id": tool_call["id"],
                                "role": "tool",
                                "name": name,
                                "content": str(result),
                            })
                        except Exception as e:
                            tool_results.append({
                                "tool_call_id": tool_call["id"],
                                "role": "tool",
                                "name": name,
                                "content": f"Error executing tool: {str(e)}"
                            })

                followup_msgs = self._prune_msgs_for_tool_followup(original_msgs) + [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    *tool_results
                ]

                # TODO: send the tools here as well if they're defined; or refactor to make the calls in a loop
                # Recursive follow-up loop after first tool call
                while True:
                    filtered_tools = await self._filter_tools_via_llm(followup_msgs, final_tools)
                    if filtered_tools:
                        self._log(f"➡️  Enviando {len(filtered_tools)} of {len(final_tools)} tools ...")
                    followup_iter = await _safe_acompletion(
                        model=req.model or self.default_model,
                        messages=followup_msgs,
                        stream=True,
                        tools=filtered_tools,
                        tool_choice="auto"
                    )

                    tool_calls_detected = False
                    tool_call_parts = {}
                    current_call_idx = None

                    async for chunk in followup_iter:
                        payload = json.loads(chunk.model_dump_json())
                        choice = payload["choices"][0]
                        delta = choice.get("delta", {})

                        # Handle streaming content
                        if "content" in delta:
                            content = delta.get("content", "")
                            buf.append(content or "")
                            tokens += len(content or "")
                        yield f"data: {json.dumps(payload)}\n\n"

                        # Detect additional tool calls
                        for tc in (delta.get("tool_calls", []) or []):
                            tool_calls_detected = True
                            idx = tc.get("index", current_call_idx)
                            current_call_idx = idx
                            if idx is None:
                                continue

                            accum = tool_call_parts.get(idx, {
                                "id": tc.get("id"),
                                "type": "function",
                                "function": {"name": None, "arguments": ""},
                                "index": idx
                            })

                            fn = accum["function"]
                            upd_fn = tc.get("function", {})
                            if "name" in upd_fn and not fn["name"]:
                                fn["name"] = upd_fn["name"]
                            if "arguments" in upd_fn:
                                fn["arguments"] += upd_fn["arguments"]
                            if tc.get("id"):
                                accum["id"] = tc["id"]
                            tool_call_parts[idx] = accum
                            yield f"data: {json.dumps(payload)}\n\n"

                        if choice.get("finish_reason") == "tool_calls":
                            break

                    if not tool_calls_detected:
                        break

                    # Execute new tool calls
                    new_tool_calls = await _prepare_tool_calls(tool_call_parts)
                    new_tool_results = []
                    for tool_call in new_tool_calls:
                        name = tool_call["function"]["name"]
                        args_str = tool_call["function"].get("arguments", "")
                        args = {}
                        if args_str.strip():
                            try:
                                args = json.loads(args_str)
                            except Exception as e:
                                self._log(f"❌ Tool {name} args JSON error: {e}")
                        local_tool_failed = False
                        if name in local_tools:
                            try:
                                result = await self.local_tools_handler(tenant, name, args)
                                new_tool_results.append({
                                    "tool_call_id": tool_call["id"],
                                    "role": "tool",
                                    "name": name,
                                    "content": str(result),
                                })
                                continue
                            except Exception as e:
                                self._log(f"❌ Local tool {name} error: {e}")
                                new_tool_results.append({
                                    "tool_call_id": tool_call["id"],
                                    "role": "tool",
                                    "name": name,
                                    "content": f"Error: {str(e)}"
                                })
                                local_tool_failed = True
                        if name in available_tools and not local_tool_failed:
                            try:
                                result = await self._execute_tool(name, args)
                                new_tool_results.append({
                                    "tool_call_id": tool_call["id"],
                                    "role": "tool",
                                    "name": name,
                                    "content": str(result),
                                })
                            except Exception as e:
                                new_tool_results.append({
                                    "tool_call_id": tool_call["id"],
                                    "role": "tool",
                                    "name": name,
                                    "content": f"Error executing tool: {str(e)}"
                                })

                    followup_msgs = self._prune_msgs_for_tool_followup(followup_msgs) + [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": new_tool_calls,
                        },
                        *new_tool_results
                    ]

                # TODO: make this run in other thread or background
                await self._write_memories(tenant, msgs + [{
                    "role": "assistant",
                    "content": self._maybe_prefix("".join(buf)),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }])

                if self.usage_hook:
                    await _maybe(self.usage_hook, tenant, tokens, time.time() - t0)

                # Clear and yield done
                buf.clear()
                tool_call_parts.clear()   # 🔴 limpia para un posible 2.º ciclo
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")


# -------------------------------------------------------------------
# Example Chroma factories
# -------------------------------------------------------------------
"""
# Usage
from fastapi import FastAPI
from brain_proxy import BrainProxy

proxy = BrainProxy()

app = FastAPI()
app.include_router(proxy.router, prefix="/v1")

# Point any OpenAI SDK at
# http://localhost:8000/v1/<tenant>/chat/completions
# Upload files via messages[].content[].file_data
# Enjoy RAG + LangMem without extra DBs or infra
"""