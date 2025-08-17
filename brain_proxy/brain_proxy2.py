"""
brain_proxy2.py — Clean, modular OpenAI-compatible proxy with memory and RAG

A refactored version of BrainProxy with improved architecture:
- Separation of concerns with dedicated service classes
- Cleaner error handling and logging
- Simplified streaming logic
- Better type safety
- Reduced code duplication
"""

from __future__ import annotations
import asyncio
import base64
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import (
    Any, AsyncIterator, Callable, Dict, List, Optional, 
    Tuple, TypedDict, Union
)

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
from langmem import create_memory_manager

from .tools import get_registry
from .__version__ import __version__
from .temporal_utils import extract_timerange
from .upstash_adapter import upstash_vec_factory, UpstashAsyncWrapper as UpstashVectorStore
from .chroma_adapter import chroma_vec_factory, ChromaAsyncWrapper


# ==============================================================================
# Type Definitions
# ==============================================================================

class MessageDict(TypedDict, total=False):
    role: str
    content: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]
    tool_call_id: Optional[str]
    name: Optional[str]
    timestamp: Optional[str]


class ToolCall(TypedDict):
    id: str
    type: str
    function: Dict[str, Any]


class FileData(BaseModel):
    name: str
    mime: str
    data: str  # base64 encoded


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None
    file_data: Optional[FileData] = Field(None, alias="file_data")


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class BrainProxyConfig:
    """Configuration for BrainProxy2"""
    # Memory settings
    enable_memory: bool = True
    memory_model: str = "openai/gpt-4o-mini"
    mem_top_k: int = 6
    mem_working_max: int = 12
    enable_global_memory: bool = False
    
    # Session settings
    enable_session_memory: bool = True
    session_ttl_hours: int = 24
    session_max_messages: int = 100
    session_summarize_after: int = 50
    session_memory_max_mb: float = 10.0
    
    # Tool settings
    use_registry_tools: bool = True
    tool_filtering_model: Optional[str] = None
    
    # Model settings
    default_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "openai/text-embedding-3-small"
    
    # Storage settings
    storage_dir: Union[str, Path] = "tenants"
    max_upload_mb: int = 20
    
    # Feature flags
    temporal_awareness: bool = True
    system_prompt: Optional[str] = None
    debug: bool = False
    
    # Vector store settings
    upstash_rest_url: Optional[str] = None
    upstash_rest_token: Optional[str] = None
    max_workers: int = 10
    
    # Callbacks
    auth_hook: Optional[Callable] = None
    usage_hook: Optional[Callable] = None
    local_tools_handler: Optional[Callable] = None
    on_thinking: Optional[Callable] = None
    on_session_end: Optional[Callable] = None
    extract_text: Optional[Callable] = None
    manager_fn: Optional[Callable] = None
    vector_store_factory: Optional[Callable] = None


# ==============================================================================
# Constants
# ==============================================================================

MEMORY_INSTRUCTIONS = """
You are a long-term memory manager that maintains semantic, procedural, and episodic memories
for a life-long learning agent.

--------------------------------------------------------------------------------
0. ⛔️  FILTER  ⛔️
Before doing anything else, IGNORE and DO NOT STORE:
  • Transient errors, one-off failures, "I don't have access to X", or logging noise.
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

Do **NOT** store anything that violates step 0. Favour dense, declarative sentences
over raw chat fragments. Use the agent's first-person voice when relevant ("I…").
"""


# ==============================================================================
# Helper Functions
# ==============================================================================

def sha256(data: bytes) -> str:
    """Generate SHA256 hash of bytes"""
    return hashlib.sha256(data).hexdigest()


async def maybe_await(fn, *args, **kwargs):
    """Call function, awaiting if it's async"""
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return fn(*args, **kwargs)


async def safe_llm_call(**kwargs) -> Any:
    """LLM call with retry logic for transient errors"""
    max_retries = 2
    
    # Add logprobs for OpenAI models to fix response format issues
    if 'logprobs' not in kwargs and 'model' in kwargs:
        model = kwargs.get('model', '')
        if model.startswith('openai/') or model.startswith('gpt-'):
            kwargs['logprobs'] = True
    
    for attempt in range(max_retries + 1):
        try:
            return await acompletion(**kwargs)
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if error is retryable
            is_retryable = any(pattern in error_str for pattern in [
                'rate limit', 'timeout', 'connection', 'server error',
                '429', '500', '502', '503', '504'
            ])
            
            is_permanent = any(pattern in error_str for pattern in [
                'invalid_request_error', 'authentication_error',
                'permission_denied', 'invalid_api_key', 'model_not_found'
            ])
            
            if is_permanent or attempt >= max_retries:
                raise
            
            if is_retryable and attempt < max_retries:
                delay = 1.0 * (2 ** attempt)
                await asyncio.sleep(delay)
            else:
                raise


# ==============================================================================
# Service Classes
# ==============================================================================

class SafeChatLiteLLM(ChatLiteLLM):
    """ChatLiteLLM wrapper that handles response format issues gracefully"""
    
    def __init__(self, *args, **kwargs):
        # Add logprobs to fix litellm response format issues
        if 'logprobs' not in kwargs and 'model' in kwargs:
            model = kwargs.get('model', '')
            if model.startswith('openai/') or model.startswith('gpt-'):
                kwargs['logprobs'] = True
        super().__init__(*args, **kwargs)
    
    def _create_chat_result(self, response):
        try:
            return super()._create_chat_result(response)
        except KeyError as e:
            if 'choices' in str(e):
                # Handle missing 'choices' field
                from langchain_core.outputs import ChatResult, ChatGeneration
                from langchain_core.messages import AIMessage
                
                content = "Memory extraction failed due to LLM response format issue."
                if isinstance(response, dict):
                    content = response.get('error', {}).get('message', content) if 'error' in response else content
                    content = response.get('message', content)
                    content = response.get('content', content)
                
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content=content),
                            generation_info=response if isinstance(response, dict) else {}
                        )
                    ],
                    llm_output=response if isinstance(response, dict) else {}
                )
            raise


class LiteLLMEmbeddings(Embeddings):
    """Embeddings provider using litellm"""
    
    def __init__(self, model: str):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            response = embedding(model=self.model, input=text)
            results.append(self._extract_embedding(response))
        return results
    
    def embed_query(self, text: str) -> List[float]:
        response = embedding(model=self.model, input=text)
        return self._extract_embedding(response)
    
    def _extract_embedding(self, response) -> List[float]:
        """Extract embedding from various response formats"""
        if hasattr(response, 'data') and response.data:
            if hasattr(response.data[0], 'embedding'):
                return response.data[0].embedding
            elif isinstance(response.data[0], dict) and 'embedding' in response.data[0]:
                return response.data[0]['embedding']
        elif isinstance(response, list) and len(response) > 0:
            return response[0]
        elif isinstance(response, dict):
            if 'data' in response:
                data = response['data']
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict) and 'embedding' in data[0]:
                        return data[0]['embedding']
            elif 'embedding' in response:
                return response['embedding']
        return []


# ==============================================================================
# Session Management
# ==============================================================================

class SessionMemoryManager:
    """Manages ephemeral session memories with intelligent summarization"""
    
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
        self.last_accessed = datetime.now(timezone.utc)
    
    async def add_memory(self, content: str, role: str = "user") -> None:
        """Add a new memory and trigger summarization if needed"""
        self.memories.append({
            "content": content,
            "role": role,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        self.message_count += 1
        self.update_access_time()
        
        if len(self.memories) > self.summarize_after:
            await self._summarize_old_memories()
    
    async def _summarize_old_memories(self) -> None:
        """Summarize older memories to prevent overflow"""
        if len(self.memories) <= self.max_recent:
            return
        
        to_summarize = self.memories[:-self.max_recent]
        
        # Group by hour for summarization
        hourly_groups = {}
        for mem in to_summarize:
            timestamp = datetime.fromisoformat(mem["timestamp"])
            hour_key = timestamp.strftime("%Y-%m-%d %H:00")
            
            if hour_key not in hourly_groups:
                hourly_groups[hour_key] = []
            hourly_groups[hour_key].append(mem)
        
        # Create summaries
        for hour_key, memories in hourly_groups.items():
            messages_text = "\n".join([
                f"{m['role']}: {m['content']}" for m in memories
            ])
            
            summary_prompt = f"""Summarize this conversation segment concisely, preserving key facts, decisions, and context:

{messages_text}

Provide a brief summary (2-3 sentences) capturing the essential information."""

            try:
                response = await safe_llm_call(
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
            except Exception:
                self.summaries.append({
                    "summary": f"[{len(memories)} messages from {hour_key}]",
                    "period": hour_key,
                    "message_count": len(memories),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        self.memories = self.memories[-self.max_recent:]
    
    def get_all_memories(self) -> List[str]:
        """Get all memories including summaries for retrieval"""
        result = []
        
        for summary in self.summaries:
            result.append(f"[Summary from {summary['period']}]: {summary['summary']}")
        
        for mem in self.memories:
            result.append(f"{mem['content']}")
        
        return result
    
    def get_session_data(self) -> Dict[str, Any]:
        """Get all session data for callbacks"""
        return {
            "tenant_id": self.tenant_id,
            "messages": self.memories.copy(),
            "summaries": self.summaries.copy(),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "message_count": self.message_count
        }
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        import sys
        total_size = sys.getsizeof(self.memories) + sys.getsizeof(self.summaries)
        return total_size / (1024 * 1024)


class SessionService:
    """Service for managing ephemeral sessions"""
    
    def __init__(self, config: BrainProxyConfig):
        self.config = config
        self._sessions: Dict[str, SessionMemoryManager] = {}
        self._session_ttl: Dict[str, datetime] = {}
        self._last_cleanup = datetime.now(timezone.utc)
    
    def parse_tenant_session(self, tenant: str) -> Tuple[str, Optional[str]]:
        """Parse tenant ID into base tenant and optional session ID"""
        if ':' in tenant and self.config.enable_session_memory:
            parts = tenant.split(':', 1)
            base_tenant = parts[0]
            session_id = parts[1]
            
            if not re.match(r'^[\w\+\-\.\@]+$', session_id):
                raise ValueError(f"Invalid session ID format: {session_id}")
            
            return base_tenant, session_id
        return tenant, None
    
    async def get_or_create(self, full_tenant_id: str) -> SessionMemoryManager:
        """Get existing session or create new one"""
        now = datetime.now(timezone.utc)
        
        # Periodic cleanup
        if (now - self._last_cleanup).total_seconds() > 300:
            self._last_cleanup = now
            asyncio.create_task(self.cleanup_expired())
        
        if full_tenant_id in self._sessions:
            self._session_ttl[full_tenant_id] = now
            session = self._sessions[full_tenant_id]
            session.update_access_time()
            return session
        
        # Create new session
        session = SessionMemoryManager(
            tenant_id=full_tenant_id,
            memory_model=self.config.memory_model,
            max_recent=self.config.session_max_messages // 3,
            summarize_after=self.config.session_summarize_after,
            max_memory_mb=self.config.session_memory_max_mb
        )
        
        self._sessions[full_tenant_id] = session
        self._session_ttl[full_tenant_id] = now
        return session
    
    async def cleanup_expired(self):
        """Clean up expired sessions"""
        now = datetime.now(timezone.utc)
        ttl_delta = timedelta(hours=self.config.session_ttl_hours)
        
        expired = [
            tenant_id for tenant_id, last_access in self._session_ttl.items()
            if now - last_access > ttl_delta
        ]
        
        for tenant_id in expired:
            session = self._sessions.pop(tenant_id, None)
            self._session_ttl.pop(tenant_id, None)
            
            if session and self.config.on_session_end:
                asyncio.create_task(
                    maybe_await(self.config.on_session_end, tenant_id, session.get_session_data())
                )


# ==============================================================================
# Memory Service
# ==============================================================================

class MemoryService:
    """Service for managing long-term memories"""
    
    def __init__(self, config: BrainProxyConfig, vector_factory: Callable):
        self.config = config
        self.vector_factory = vector_factory
        self._managers: Dict[str, Tuple[Any, Callable, Callable]] = {}
    
    def get_manager(self, tenant: str) -> Tuple[Any, Callable, Callable]:
        """Get or create memory manager for tenant"""
        # Parse session to get base tenant
        if ':' in tenant:
            base_tenant = tenant.split(':', 1)[0]
        else:
            base_tenant = tenant
        
        if base_tenant in self._managers:
            return self._managers[base_tenant]
        
        vec = self.vector_factory(f"{base_tenant}_memory")
        
        async def search_mem(query: str, k: int):
            docs = await vec.similarity_search(query, k=k)
            return [d.page_content for d in docs]
        
        async def store_mem(memories: List[Any]):
            docs = []
            for m in memories:
                try:
                    content = self._extract_memory_content(m)
                    if content:
                        now_iso = datetime.now(timezone.utc).isoformat()
                        page_content = content
                        if self.config.temporal_awareness:
                            page_content = f"[{now_iso}] {content}"
                        
                        docs.append(Document(
                            page_content=page_content,
                            metadata={"timestamp": now_iso}
                        ))
                except Exception:
                    pass
            
            if docs:
                await vec.add_documents(docs)
        
        manager = create_memory_manager(
            SafeChatLiteLLM(model=self.config.memory_model),
            instructions=MEMORY_INSTRUCTIONS
        )
        
        self._managers[base_tenant] = (manager, search_mem, store_mem)
        return self._managers[base_tenant]
    
    def _extract_memory_content(self, memory: Any) -> Optional[str]:
        """Extract content from various memory formats"""
        if hasattr(memory, 'content'):
            if hasattr(memory.content, 'content'):
                return str(memory.content.content)
            return str(memory.content)
        elif isinstance(memory, dict):
            if 'content' in memory:
                return str(memory['content'])
            # Handle malformed dicts
            if 'content=' in memory:
                text_keys = [k for k in memory.keys() 
                           if k != 'content=' and isinstance(k, str) and len(k) > 10]
                if text_keys:
                    return max(text_keys, key=len)
        elif isinstance(memory, str):
            return memory
        return None
    
    async def retrieve(self, tenant: str, query: str, session_service: SessionService) -> str:
        """Retrieve relevant memories"""
        if not self.config.enable_memory:
            return ""
        
        base_tenant, session_id = session_service.parse_tenant_session(tenant)
        
        # Get base tenant memories
        mgr, search, _ = self.get_manager(tenant)
        if not mgr:
            return ""
        
        # Search in parallel
        search_tasks = [search(query, k=self.config.mem_top_k * 3)]
        
        # Add global memories if enabled
        if self.config.enable_global_memory:
            global_mgr, global_search, _ = self.get_manager('_global')
            if global_mgr:
                search_tasks.append(global_search(query, k=self.config.mem_top_k * 3))
        
        results = await asyncio.gather(*search_tasks)
        raw = []
        raw.extend(results[0])
        if self.config.enable_global_memory and len(results) > 1:
            raw.extend(results[1])
        
        # Add session memories
        if session_id and self.config.enable_session_memory:
            session = await session_service.get_or_create(tenant)
            session_memories = session.get_all_memories()
            for mem in session_memories:
                raw.insert(0, f"[SESSION] {mem}")
        
        # Apply temporal filtering if enabled
        if self.config.temporal_awareness:
            timerange = extract_timerange(query)
            if timerange:
                start, end = timerange
                filtered = []
                for mem in raw:
                    match = re.match(r"\[(\d{4}-\d{2}-\d{2}T[^]]+)\]", mem)
                    if match:
                        ts = match.group(1)
                        if start.isoformat() <= ts <= end.isoformat():
                            filtered.append(mem)
                raw = filtered or raw
        
        # Sort by timestamp and take most recent
        def extract_timestamp(memory):
            match = re.match(r"\[(\d{4}-\d{2}-\d{2}T[^]]+)\]", memory)
            return match.group(1) if match else "0"
        
        raw.sort(key=extract_timestamp)
        memories = raw[-self.config.mem_top_k:]
        return "\n".join(memories)
    
    async def store(self, tenant: str, conversation: List[Dict[str, Any]], session_service: SessionService):
        """Store memories from conversation"""
        if not self.config.enable_memory:
            return
        
        asyncio.create_task(self._process_memories(tenant, conversation, session_service))
    
    async def _process_memories(self, tenant: str, conversation: List[Dict[str, Any]], session_service: SessionService):
        """Process and store memories in background"""
        base_tenant, session_id = session_service.parse_tenant_session(tenant)
        
        # Handle session memories
        if session_id and self.config.enable_session_memory:
            try:
                session = await session_service.get_or_create(tenant)
                for msg in conversation:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if content:
                            await session.add_memory(content, role)
                
                if session.estimate_memory_usage() > self.config.session_memory_max_mb:
                    await session._summarize_old_memories()
            except Exception:
                pass
        
        # Store persistent memories for significant conversations
        if not session_id or len(conversation) > 5:
            manager_tuple = self.get_manager(tenant)
            if not manager_tuple:
                return
            
            manager, _, store = manager_tuple
            
            try:
                raw_memories = await manager(conversation)
                if raw_memories:
                    await store(raw_memories)
            except Exception:
                pass


# ==============================================================================
# Document Service
# ==============================================================================

class DocumentService:
    """Service for handling document ingestion and RAG"""
    
    def __init__(self, config: BrainProxyConfig, vector_factory: Callable):
        self.config = config
        self.vector_factory = vector_factory
        self.storage_dir = Path(config.storage_dir)
        self.max_upload_bytes = config.max_upload_mb * 1024 * 1024
        self.extract_text = config.extract_text or self._default_extract
    
    def _default_extract(self, path: Path, mime: str) -> str:
        return path.read_text("utf-8", "ignore")
    
    async def ingest_files(self, files: List[FileData], tenant: str):
        """Ingest files into vector store"""
        if not files:
            return
        
        # Check for session - don't allow file uploads in sessions
        if ':' in tenant:
            base_tenant = tenant.split(':', 1)[0]
            raise HTTPException(
                status_code=400,
                detail="File uploads are not allowed for ephemeral sessions"
            )
        else:
            base_tenant = tenant
        
        docs = []
        tenant_dir = self.storage_dir / base_tenant / "files"
        tenant_dir.mkdir(exist_ok=True, parents=True)
        
        for file in files:
            try:
                # Validate file size
                if len(base64.b64decode(file.data)) > self.max_upload_bytes:
                    raise ValueError(f"File too large: {file.name}")
                
                # Save file
                name = file.name.replace(" ", "_")
                path = tenant_dir / name
                path.write_bytes(base64.b64decode(file.data))
                
                # Extract content
                content = self.extract_text(path, file.mime)
                
                if isinstance(content, str) and content.strip():
                    # Split into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = splitter.split_text(content)
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
                    # Pre-processed documents
                    timestamp = datetime.now(timezone.utc).isoformat()
                    for doc in content:
                        if "timestamp" not in doc.metadata:
                            doc.metadata["timestamp"] = timestamp
                        docs.append(doc)
            except Exception:
                pass
        
        if docs:
            vec = self.vector_factory(base_tenant)
            await vec.add_documents(docs)
    
    async def search(self, query: str, tenant: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        # Use base tenant for document retrieval
        if ':' in tenant:
            base_tenant = tenant.split(':', 1)[0]
        else:
            base_tenant = tenant
        
        vec = self.vector_factory(base_tenant)
        docs = await vec.similarity_search(query, k=k)
        
        if not docs:
            return []
        
        context_str = "\n\n".join([d.page_content for d in docs])
        return [{
            "role": "system",
            "content": "Relevant context from documents:\n\n" + context_str
        }]


# ==============================================================================
# Tool Service
# ==============================================================================

class ToolService:
    """Service for handling tool filtering and execution"""
    
    def __init__(self, config: BrainProxyConfig):
        self.config = config
        self.registry = get_registry()
        self.tools = []
        self._tenant_tools: Dict[str, List[Dict]] = {}
        
        if config.use_registry_tools:
            self.tools.extend(self.registry.get_tools())
    
    def set_tenant_tools(self, tenant: str, tools: List[Dict[str, Any]]):
        """Set tools for a specific tenant"""
        self._tenant_tools[tenant] = tools
    
    def get_tools_for_tenant(self, tenant: str, request_tools: Optional[List[Dict]] = None) -> List[Dict]:
        """Get all available tools for a tenant"""
        final_tools = []
        
        if self.tools:
            final_tools.extend(self.tools)
        if request_tools:
            final_tools.extend(request_tools)
        if tenant in self._tenant_tools:
            final_tools.extend(self._tenant_tools[tenant])
        
        # Deduplicate by name
        by_name = {}
        for tool in final_tools:
            by_name[tool['function']['name']] = tool
        
        return list(by_name.values())
    
    async def filter_tools(self, messages: List[Dict], tools: List[Dict]) -> List[Dict]:
        """Filter tools based on relevance to the conversation"""
        if not self.config.tool_filtering_model or not tools:
            return tools
        
        # Get last user message
        user_prompt = next(
            (msg["content"] for msg in reversed(messages) 
             if msg["role"] == "user" and isinstance(msg["content"], str)),
            None
        )
        if not user_prompt:
            return tools
        
        # Format tools list
        tools_str = "\n".join(
            f"- {tool['function']['name']} ({tool['function'].get('description', 'No description')})"
            for tool in tools
        )
        
        prompt = f"""You are a helpful assistant that selects the most relevant tools for a given user message.

The user wrote:
```{user_prompt}```

Available tools:
```{tools_str}```

# Try to always return the 2-5 most similar or related tools to the user message.
# Reply strictly in JSON format like:
{{"selected_tools": ["tool_name1", "tool_name2"]}}"""
        
        try:
            response = await safe_llm_call(
                model=self.config.tool_filtering_model,
                messages=[
                    {"role": "system", "content": "You only reply with a JSON object listing selected tools."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from markdown if present
            if content.startswith('```'):
                lines = content.split('\n')
                json_lines = []
                in_code_block = False
                
                for line in lines:
                    if line.strip().startswith('```'):
                        if in_code_block:
                            break
                        else:
                            in_code_block = True
                            continue
                    elif in_code_block:
                        json_lines.append(line)
                
                if json_lines:
                    content = '\n'.join(json_lines).strip()
            
            parsed = json.loads(content)
            selected = parsed.get("selected_tools", [])
            return [tool for tool in tools if tool["function"]["name"] in selected]
        except Exception:
            return tools
    
    async def execute(self, name: str, args: Dict[str, Any], local_handler: Optional[Callable] = None, tenant: Optional[str] = None) -> Any:
        """Execute a tool and return its result"""
        # Note: local_handler is not used here since we handle it in the calling code
        # This method only handles registry tools
        
        # Check registry
        if impl := self.registry.get_implementation(name):
            return await maybe_await(impl, **args)
        
        raise ValueError(f"Tool {name} not found or not implemented")


# ==============================================================================
# Streaming Service  
# ==============================================================================

class StreamingService:
    """Service for handling streaming responses"""
    
    def __init__(self, config: BrainProxyConfig):
        self.config = config
    
    async def process_chunk(self, chunk) -> Dict:
        """Process a streaming chunk into a clean payload"""
        try:
            # Extract fields directly without Pydantic serialization
            result = {
                "choices": [],
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "unknown"
            }
            
            if hasattr(chunk, 'model'):
                result["model"] = str(chunk.model) if chunk.model else "unknown"
            
            if hasattr(chunk, 'created'):
                result["created"] = int(chunk.created) if chunk.created else int(time.time())
            
            if hasattr(chunk, 'choices') and chunk.choices:
                for choice in chunk.choices:
                    choice_data = {
                        "index": 0,
                        "delta": {},
                        "finish_reason": None
                    }
                    
                    if hasattr(choice, 'index'):
                        choice_data["index"] = int(choice.index) if choice.index is not None else 0
                    
                    if hasattr(choice, 'finish_reason'):
                        choice_data["finish_reason"] = str(choice.finish_reason) if choice.finish_reason else None
                    
                    if hasattr(choice, 'delta') and choice.delta:
                        delta = choice.delta
                        
                        if hasattr(delta, 'content') and delta.content is not None:
                            choice_data["delta"]["content"] = str(delta.content)
                        
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            tool_calls_list = []
                            for tc in delta.tool_calls:
                                tc_data = {
                                    "index": 0,
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                }
                                
                                if hasattr(tc, 'index') and tc.index is not None:
                                    tc_data["index"] = int(tc.index)
                                if hasattr(tc, 'id') and tc.id:
                                    tc_data["id"] = str(tc.id)
                                if hasattr(tc, 'type') and tc.type:
                                    tc_data["type"] = str(tc.type)
                                
                                if hasattr(tc, 'function') and tc.function:
                                    fn = tc.function
                                    if hasattr(fn, 'name') and fn.name:
                                        tc_data["function"]["name"] = str(fn.name)
                                    if hasattr(fn, 'arguments') and fn.arguments is not None:
                                        tc_data["function"]["arguments"] = str(fn.arguments)
                                
                                tool_calls_list.append(tc_data)
                            
                            if tool_calls_list:
                                choice_data["delta"]["tool_calls"] = tool_calls_list
                    
                    result["choices"].append(choice_data)
            
            if not result["choices"]:
                result["choices"] = [{"index": 0, "delta": {}, "finish_reason": None}]
            
            return result
            
        except Exception:
            return {
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "unknown"
            }


# ==============================================================================
# Main BrainProxy2 Class
# ==============================================================================

class BrainProxy2:
    """Clean, modular OpenAI-compatible proxy with memory and RAG"""
    
    def __init__(self, config: Optional[BrainProxyConfig] = None, **kwargs):
        # Merge config with kwargs
        if config:
            self.config = config
        else:
            self.config = BrainProxyConfig(**kwargs)
        
        # Initialize storage
        self.storage_dir = Path(self.config.storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize embeddings
        underlying_embeddings = LiteLLMEmbeddings(model=self.config.embedding_model)
        fs = LocalFileStore(f"{self.storage_dir}/embeddings_cache")
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=underlying_embeddings,
            document_embedding_cache=fs,
            namespace=self.config.embedding_model
        )
        
        # Set up vector store factory
        if self.config.upstash_rest_url and self.config.upstash_rest_token:
            self.vector_factory = lambda tenant: upstash_vec_factory(
                tenant,
                self.embeddings,
                self.config.upstash_rest_url,
                self.config.upstash_rest_token,
                max_workers=self.config.max_workers
            )
        elif self.config.vector_store_factory:
            self.vector_factory = lambda tenant: self.config.vector_store_factory(
                tenant, self.embeddings, self.config.max_workers
            )
        else:
            self.vector_factory = lambda tenant: chroma_vec_factory(
                f"vec_{tenant}", self.embeddings, max_workers=self.config.max_workers
            )
        
        # Initialize services
        self.session_service = SessionService(self.config)
        self.memory_service = MemoryService(self.config, self.vector_factory)
        self.document_service = DocumentService(self.config, self.vector_factory)
        self.tool_service = ToolService(self.config)
        self.streaming_service = StreamingService(self.config)
        
        # Set up router
        self.router = APIRouter()
        self._setup_routes()
    
    def _log(self, message: str, *args):
        """Log debug messages"""
        if self.config.debug:
            print(f"[BrainProxy2] {message}", *args)
    
    def _validate_messages(self, messages: List[Dict]) -> List[Dict]:
        """Validate and clean messages for OpenAI API"""
        valid_msgs = []
        tool_call_ids = set()
        
        for msg in messages:
            if msg.get("role") == "assistant":
                valid_msgs.append(msg)
                if msg.get("tool_calls"):
                    for tc in msg.get("tool_calls", []):
                        if tc.get("id"):
                            tool_call_ids.add(tc.get("id"))
            elif msg.get("role") == "tool":
                if msg.get("tool_call_id") in tool_call_ids:
                    valid_msgs.append(msg)
            else:
                valid_msgs.append(msg)
        
        return valid_msgs
    
    def _split_files(self, messages: List[ChatMessage]) -> Tuple[List[Dict], List[FileData]]:
        """Extract file data from messages"""
        conv_msgs = []
        files = []
        
        for msg in messages:
            if isinstance(msg.content, str):
                conv_msgs.append({"role": msg.role, "content": msg.content})
                continue
            
            text_parts = []
            for part in msg.content:
                if part.type == "text":
                    text_parts.append(part.text or "")
                elif part.file_data:
                    files.append(part.file_data)
            
            if text_parts:
                conv_msgs.append({"role": msg.role, "content": "\n".join(text_parts)})
        
        return conv_msgs, files
    
    async def _prepare_messages(
        self, 
        messages: List[Dict], 
        tenant: str
    ) -> List[Dict]:
        """Prepare messages with memory, context, and system prompts"""
        result = messages.copy()
        
        # Add system prompt
        if self.config.system_prompt:
            if result and result[0].get("role") == "system":
                result[0]["content"] = f"{self.config.system_prompt}\n\n{result[0]['content']}"
            else:
                result = [{"role": "system", "content": self.config.system_prompt}] + result
        
        # Add temporal awareness
        if self.config.temporal_awareness:
            now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
            result = [{"role": "system", "content": f"Current UTC time is {now_iso}."}] + result
        
        # Add memories
        if self.config.enable_memory and result:
            user_text = result[-1].get("content", "")
            if isinstance(user_text, str) and user_text:
                
                # Trigger thinking callback
                if self.config.on_thinking:
                    try:
                        await maybe_await(self.config.on_thinking, tenant, 'thinking')
                    except Exception:
                        pass
                
                mem_block = await self.memory_service.retrieve(tenant, user_text, self.session_service)
                if mem_block:
                    result = result[:-1] + [
                        {"role": "system", "content": "Relevant memories:\n" + mem_block},
                        result[-1]
                    ]
        
        # Add document context
        if result:
            query = result[-1].get("content", "")
            if isinstance(query, str) and query:
                doc_context = await self.document_service.search(query, tenant)
                if doc_context:
                    result = result[:-1] + doc_context + [result[-1]]
        
        return result
    
    async def _dispatch(
        self,
        messages: List[Dict],
        model: str,
        stream: bool,
        tools: Optional[List[Dict]] = None,
        tenant: Optional[str] = None
    ):
        """Dispatch to LLM with tools support"""
        kwargs = {
            "model": model,
            "messages": self._validate_messages(messages),
            "stream": stream
        }
        
        # Get and filter tools
        all_tools = self.tool_service.get_tools_for_tenant(tenant, tools)
        if all_tools:
            filtered_tools = await self.tool_service.filter_tools(messages, all_tools)
            if filtered_tools:
                kwargs["tools"] = filtered_tools
                kwargs["tool_choice"] = "auto"
                self._log(f"Using {len(filtered_tools)} of {len(all_tools)} tools")
        
        response = await safe_llm_call(**kwargs)
        
        # Handle tool calls in non-streaming mode
        if not stream and hasattr(response.choices[0], "message"):
            msg = response.choices[0].message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_results = await self._execute_tool_calls(
                    msg.tool_calls, tenant, tools
                )
                
                if tool_results:
                    # Make follow-up call
                    new_msgs = messages + [
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
                                } for tc in msg.tool_calls
                            ]
                        },
                        *tool_results
                    ]
                    
                    kwargs["messages"] = self._validate_messages(new_msgs)
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                    response = await safe_llm_call(**kwargs)
        
        return response
    
    async def _execute_tool_calls(
        self,
        tool_calls,
        tenant: str,
        request_tools: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Execute tool calls and return results"""
        results = []
        
        # Get all local tools (from request + tenant tools)
        local_tool_names = set()
        if request_tools:
            local_tool_names.update(t["function"]["name"] for t in request_tools)
        tenant_tools = self.tool_service._tenant_tools.get(tenant, [])
        if tenant_tools:
            local_tool_names.update(t["function"]["name"] for t in tenant_tools)
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            
            try:
                function_args = json.loads(tool_call.function.arguments)
            except Exception:
                function_args = {}
            
            self._log(f"Executing tool: {function_name} with args: {function_args}")
            
            try:
                # Check if this tool should be handled by local_tools_handler
                is_local = function_name in local_tool_names
                
                if is_local and self.config.local_tools_handler:
                    self._log(f"Calling local_tools_handler for: {function_name}")
                    result = await maybe_await(
                        self.config.local_tools_handler,
                        tenant, function_name, function_args
                    )
                else:
                    self._log(f"Calling tool_service.execute for: {function_name}")
                    result = await self.tool_service.execute(
                        function_name, function_args
                    )
                
                self._log(f"Tool {function_name} result: {str(result)[:100]}...")
                
                results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(result)
                })
            except Exception as e:
                self._log(f"Error executing tool {function_name}: {e}")
                results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"Error executing tool: {str(e)}"
                })
        
        return results
    
    async def _handle_streaming(
        self,
        upstream_iter,
        messages: List[Dict],
        tenant: str,
        request
    ) -> AsyncIterator[str]:
        """Handle streaming responses with tool support"""
        
        # Trigger ready callback
        if self.config.on_thinking:
            try:
                await maybe_await(self.config.on_thinking, tenant, 'ready')
            except Exception:
                pass
        
        tokens = 0
        buf = []
        tool_call_parts = {}
        tool_calls_detected = False
        
        # Get all local tools (from request + tenant tools)
        local_tool_names = set()
        if request.tools:
            local_tool_names.update(t["function"]["name"] for t in request.tools)
        tenant_tools = self.tool_service._tenant_tools.get(tenant, [])
        if tenant_tools:
            local_tool_names.update(t["function"]["name"] for t in tenant_tools)
        
        # Process initial stream
        async for chunk in upstream_iter:
            payload = await self.streaming_service.process_chunk(chunk)
            choice = payload["choices"][0]
            delta = choice.get("delta", {})
            
            # Handle content
            if "content" in delta and delta["content"] is not None:
                buf.append(delta["content"])
                tokens += len(delta["content"])
                yield f"data: {json.dumps(payload)}\n\n"
            
            # Handle tool calls
            for tc in (delta.get("tool_calls", []) or []):
                tool_calls_detected = True
                idx = tc.get("index", 0)
                
                if idx not in tool_call_parts:
                    tool_call_parts[idx] = {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {"name": None, "arguments": ""},
                        "index": idx
                    }
                
                accum = tool_call_parts[idx]
                fn = accum["function"]
                upd_fn = tc.get("function", {})
                
                if "name" in upd_fn and not fn["name"]:
                    fn["name"] = upd_fn["name"]
                if "arguments" in upd_fn:
                    fn["arguments"] += upd_fn["arguments"]
                if tc.get("id"):
                    accum["id"] = tc["id"]
                
                yield f"data: {json.dumps(payload)}\n\n"
            
            # Check for completion
            if choice.get("finish_reason") == "tool_calls":
                self._log(f"Tool calls detected for tenant {tenant}: {len(tool_call_parts)} calls")
                break
        
        # If no tool calls, we're done
        if not tool_calls_detected:
            yield "data: [DONE]\n\n"
            
            # Store memories
            await self.memory_service.store(tenant, messages + [{
                "role": "assistant",
                "content": "".join(buf),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }], self.session_service)
            
            return
        
        # Execute tool calls and stream follow-up
        tool_calls = list(tool_call_parts.values())
        tool_results = []
        
        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            args_str = tool_call["function"].get("arguments", "")
            args = {}
            
            if args_str.strip():
                try:
                    args = json.loads(args_str)
                except Exception as e:
                    self._log(f"Error parsing tool args for {name}: {e}")
            
            self._log(f"Executing tool: {name} with args: {args}")
            
            try:
                # Check if this tool should be handled by local_tools_handler
                is_local = name in local_tool_names
                
                if is_local and self.config.local_tools_handler:
                    self._log(f"Calling local_tools_handler for: {name}")
                    result = await maybe_await(
                        self.config.local_tools_handler,
                        tenant, name, args
                    )
                else:
                    self._log(f"Calling tool_service.execute for: {name}")
                    result = await self.tool_service.execute(name, args)
                
                self._log(f"Tool {name} result: {str(result)[:100]}...")
                
                tool_results.append({
                    "tool_call_id": tool_call["id"] or f"call_{len(tool_results)}",
                    "role": "tool",
                    "name": name,
                    "content": str(result)
                })
            except Exception as e:
                self._log(f"Error executing tool {name}: {e}")
                tool_results.append({
                    "tool_call_id": tool_call["id"] or f"call_{len(tool_results)}",
                    "role": "tool",
                    "name": name,
                    "content": f"Error: {str(e)}"
                })
        
        # Stream follow-up response with recursive tool support
        followup_msgs = messages + [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls
            },
            *tool_results
        ]
        
        # Recursive loop for handling additional tool calls in follow-up
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            self._log(f"Making follow-up call after tool execution for tenant {tenant} (iteration {iteration})")
            
            try:
                followup_iter = await safe_llm_call(
                    model=request.model or self.config.default_model,
                    messages=self._validate_messages(followup_msgs),
                    stream=True
                )
            except Exception as e:
                self._log(f"Error in follow-up call: {e}")
                yield f"data: {json.dumps({'choices': [{'delta': {'content': f'Error: {str(e)}'}, 'index': 0}]})}\n\n"
                break
            
            additional_tool_calls = {}
            content_streamed = False
            
            try:
                async for chunk in followup_iter:
                    payload = await self.streaming_service.process_chunk(chunk)
                    choice = payload["choices"][0]
                    delta = choice.get("delta", {})
                    
                    # Handle content streaming
                    if "content" in delta and delta["content"]:
                        buf.append(delta["content"])
                        tokens += len(delta["content"])
                        content_streamed = True
                        yield f"data: {json.dumps(payload)}\n\n"
                    
                    # Check for additional tool calls
                    for tc in (delta.get("tool_calls", []) or []):
                        idx = tc.get("index", 0)
                        if idx not in additional_tool_calls:
                            additional_tool_calls[idx] = {
                                "id": tc.get("id"),
                                "type": "function",
                                "function": {"name": None, "arguments": ""},
                                "index": idx
                            }
                        
                        accum = additional_tool_calls[idx]
                        fn = accum["function"]
                        upd_fn = tc.get("function", {})
                        
                        if "name" in upd_fn and not fn["name"]:
                            fn["name"] = upd_fn["name"]
                        if "arguments" in upd_fn:
                            fn["arguments"] += upd_fn["arguments"]
                        if tc.get("id"):
                            accum["id"] = tc["id"]
                        
                        # Don't yield tool call chunks to client
                        # yield f"data: {json.dumps(payload)}\n\n"
                    
                    # Check for finish reason
                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        if finish_reason == "tool_calls":
                            self._log(f"Additional tool calls detected in follow-up")
                            break
                        elif finish_reason in ["stop", "length"]:
                            # Normal completion
                            if not content_streamed:
                                # If no content was streamed, send an empty message
                                yield f"data: {json.dumps({'choices': [{'delta': {'content': ''}, 'index': 0, 'finish_reason': finish_reason}]})}\n\n"
                            break
            except Exception as e:
                self._log(f"Error streaming follow-up response: {e}")
                yield f"data: {json.dumps({'choices': [{'delta': {'content': f'Error: {str(e)}'}, 'index': 0}]})}\n\n"
                break
            
            # If no additional tool calls, we're done
            if not additional_tool_calls:
                break
            
            # Execute additional tool calls
            self._log(f"Executing {len(additional_tool_calls)} additional tool calls")
            new_tool_calls = list(additional_tool_calls.values())
            new_tool_results = []
            
            for tool_call in new_tool_calls:
                name = tool_call["function"]["name"]
                args_str = tool_call["function"].get("arguments", "")
                args = {}
                
                if args_str.strip():
                    try:
                        args = json.loads(args_str)
                    except Exception as e:
                        self._log(f"Error parsing additional tool args for {name}: {e}")
                
                try:
                    is_local = name in local_tool_names
                    
                    if is_local and self.config.local_tools_handler:
                        result = await maybe_await(
                            self.config.local_tools_handler,
                            tenant, name, args
                        )
                    else:
                        result = await self.tool_service.execute(name, args)
                    
                    new_tool_results.append({
                        "tool_call_id": tool_call["id"] or f"call_{len(new_tool_results)}",
                        "role": "tool",
                        "name": name,
                        "content": str(result)
                    })
                except Exception as e:
                    new_tool_results.append({
                        "tool_call_id": tool_call["id"] or f"call_{len(new_tool_results)}",
                        "role": "tool",
                        "name": name,
                        "content": f"Error: {str(e)}"
                    })
            
            # Update messages for next iteration
            followup_msgs = followup_msgs + [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": new_tool_calls
                },
                *new_tool_results
            ]
        
        yield "data: [DONE]\n\n"
        
        # Store memories
        await self.memory_service.store(tenant, messages + [{
            "role": "assistant",
            "content": "".join(buf),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }], self.session_service)
    
    def _setup_routes(self):
        """Set up FastAPI routes"""
        
        @self.router.post("/{tenant}/tools")
        async def set_tools(request: Request, tenant: str):
            if self.config.auth_hook:
                await maybe_await(self.config.auth_hook, request, tenant)
            
            body = await request.json()
            if not isinstance(body, list):
                raise HTTPException(status_code=400, detail="Expected array of tools")
            
            for tool in body:
                if not isinstance(tool, dict) or 'type' not in tool or 'function' not in tool:
                    raise HTTPException(status_code=400, detail="Invalid tool schema")
            
            self.tool_service.set_tenant_tools(tenant, body)
            return {"status": "success", "count": len(body)}
        
        @self.router.post("/{tenant}/chat/completions")
        async def chat(request: Request, tenant: str):
            self._log(f"Request for tenant {tenant}, version {__version__}")
            
            # Auth check
            if self.config.auth_hook:
                await maybe_await(self.config.auth_hook, request, tenant)
            
            # Parse request
            body = await request.json()
            req = ChatRequest(**body)
            
            # Extract files
            msgs, files = self._split_files(req.messages)
            
            # Ingest files if present
            if files:
                self._log(f"Ingesting {len(files)} files")
                await self.document_service.ingest_files(files, tenant)
            
            # Prepare messages
            msgs = await self._prepare_messages(msgs, tenant)
            
            # Dispatch to LLM
            t0 = time.time()
            response = await self._dispatch(
                msgs,
                req.model or self.config.default_model,
                req.stream,
                req.tools,
                tenant
            )
            
            # Handle response
            if not req.stream:
                # Trigger ready callback
                if self.config.on_thinking:
                    try:
                        await maybe_await(self.config.on_thinking, tenant, 'ready')
                    except Exception:
                        pass
                
                # Store memories
                await self.memory_service.store(
                    tenant,
                    msgs + [{
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }],
                    self.session_service
                )
                
                # Track usage
                if self.config.usage_hook and response.usage:
                    await maybe_await(
                        self.config.usage_hook,
                        tenant,
                        response.usage.total_tokens,
                        time.time() - t0
                    )
                
                return JSONResponse(response.model_dump())
            
            # Stream response
            return StreamingResponse(
                self._handle_streaming(response, msgs, tenant, req),
                media_type="text/event-stream"
            )


# ==============================================================================
# Module exports
# ==============================================================================

__all__ = [
    'BrainProxy2',
    'BrainProxyConfig',
    'SessionMemoryManager',
    'SessionService',
    'MemoryService',
    'DocumentService',
    'ToolService',
    'StreamingService'
]
