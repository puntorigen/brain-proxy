"""
brain_proxy.py  —  FastAPI / ASGI router with LangMem + Chroma

pip install fastapi openai langchain-chroma langmem tiktoken
"""

from __future__ import annotations
import asyncio, base64, hashlib, json, time, re
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from litellm import acompletion, embedding
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
import anyio
import os
import re
from pydantic import BaseModel

# For creating proper Memory objects
class Memory(BaseModel):
    content: str

# LangMem primitives (functions, not classes)
from langmem import create_memory_manager

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


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


async def _maybe(fn, *a, **k):
    return await fn(*a, **k) if asyncio.iscoroutinefunction(fn) else fn(*a, **k)


# -------------------------------------------------------------------
# Vector store factories
# -------------------------------------------------------------------
def chroma_vec_factory(collection_name: str, embeddings) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=f".chroma/{collection_name}",
        embedding_function=embeddings,
    )

def default_vector_store_factory(tenant, embeddings):
    return chroma_vec_factory(f"vec_{tenant}", embeddings)


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
        vector_store_factory: Callable[[str, Any], Chroma] = default_vector_store_factory,
        # memory settings
        enable_memory: bool = True,
        memory_model: str = "openai/gpt-4o-mini",  # litellm format e.g. "azure/gpt-35-turbo"
        embedding_model: str = "openai/text-embedding-3-small",  # litellm format e.g. "azure/ada-002"
        mem_top_k: int = 6,
        mem_working_max: int = 12,
        # misc
        default_model: str = "openai/gpt-4o-mini",  # litellm format e.g. "azure/gpt-4"
        storage_dir: str | Path = "tenants",
        extract_text: Callable[[Path, str], str] | None = None,
        manager_fn: Callable[..., Any] | None = None,  # multi‑agent hook
        auth_hook: Callable[[Request, str], Any] | None = None,
        usage_hook: Callable[[str, int, float], Any] | None = None,
        max_upload_mb: int = 20,
        debug: bool = False,
    ):
        # Initialize basic attributes first
        self.storage_dir = Path(storage_dir)
        self.embedding_model = embedding_model
        self.enable_memory = enable_memory
        self.memory_model = memory_model
        self.mem_top_k = mem_top_k
        self.mem_working_max = mem_working_max
        self.default_model = default_model
        self.extract_text = extract_text or (
            lambda p, m: p.read_text("utf-8", "ignore")
        )
        self.manager_fn = manager_fn
        self.auth_hook = auth_hook
        self.usage_hook = usage_hook
        self.max_upload_bytes = max_upload_mb * 1024 * 1024
        self._mem_managers: Dict[str, Any] = {}
        self.debug = debug

        # Initialize embeddings using litellm's synchronous embedding function
        underlying_embeddings = LiteLLMEmbeddings(model=self.embedding_model)
        fs = LocalFileStore(f"{self.storage_dir}/embeddings_cache")
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=underlying_embeddings,
            document_embedding_cache=fs,
            namespace=self.embedding_model
        )
        
        self.vec_factory = lambda tenant: vector_store_factory(tenant, self.embeddings)
        self.router = APIRouter()
        self._mount()

    def _log(self, message: str) -> None:
        """Log debug messages only when debug is enabled."""
        if self.debug:
            print(message)

    # ----------------------------------------------------------------
    # Memory helpers
    # ----------------------------------------------------------------
    def _get_mem_manager(self, tenant: str):
        """Get or create memory manager for tenant"""
        if tenant in self._mem_managers:
            return self._mem_managers[tenant]

        # use the tenant's chroma collection for memory as well
        vec = self.vec_factory(f"{tenant}_memory")
        async def _search_mem(query: str, k: int):
            docs = vec.similarity_search(query, k=k)
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
                    
                    docs.append(Document(page_content=content))
                except Exception as e:
                    self._log(f"Error processing memory: {e}")
            
            if docs:
                self._log(f"Storing {len(docs)} memories for tenant {tenant}")
                vec.add_documents(docs)
                self._log(f"Successfully stored memories")

        # Use langchain_litellm's ChatLiteLLM for memory manager directly
        # No wrapper to avoid potential deadlocks
        manager = create_memory_manager(ChatLiteLLM(model=self.memory_model))
        
        self._mem_managers[tenant] = (manager, _search_mem, _store_mem)
        return self._mem_managers[tenant]

    async def _retrieve_memories(self, tenant: str, user_text: str) -> str:
        """Retrieve relevant memories for the given user text."""
        if not self.enable_memory:
            self._log(f"Memory disabled for tenant {tenant}")
            return ""
            
        self._log(f"Retrieving memories for tenant {tenant} with query: '{user_text[:30]}...'")
        manager_tuple = self._get_mem_manager(tenant)
        if not manager_tuple:
            self._log(f"No memory manager found for tenant {tenant}")
            return ""
            
        _, search, _ = manager_tuple
        try:
            self._log(f"Searching for memories with k={self.mem_top_k}")
            memories = await search(user_text, k=self.mem_top_k)
            self._log(f"Found {len(memories)} memories")
            
            if memories:
                # Log first few characters of each memory for debugging
                for i, memory in enumerate(memories):
                    preview = memory[:50] + "..." if len(memory) > 50 else memory
                    self._log(f"Memory {i+1}: {preview}")
                    
                memory_block = "\n".join(memories)
                return memory_block
            else:
                self._log("No memories found")
                return ""
        except Exception as e:
            self._log(f"Error retrieving memories: {e}")
            return ""

    async def _write_memories(
        self, tenant: str, conversation: List[Dict[str, Any]]
    ):
        """Extract and store memories from the conversation."""
        if not self.enable_memory:
            return
        manager_tuple = self._get_mem_manager(tenant)
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
                        if hasattr(mem, 'id') and hasattr(mem, 'content'):
                            if hasattr(mem.content, 'content'):
                                # Extract content from the BaseModel
                                content = mem.content.content
                                formatted_mem = {"content": content}
                                proper_memories.append(formatted_mem)
                            elif hasattr(mem.content, 'model_dump'):
                                # Extract content using model_dump method
                                model_data = mem.content.model_dump()
                                if 'content' in model_data:
                                    formatted_mem = {"content": model_data['content']}
                                    proper_memories.append(formatted_mem)
                                else:
                                    # If no content field, use the whole model data as string
                                    formatted_mem = {"content": str(model_data)}
                                    proper_memories.append(formatted_mem)
                            elif isinstance(mem.content, dict) and 'content' in mem.content:
                                # Content is a dict with content field
                                formatted_mem = {"content": mem.content['content']}
                                proper_memories.append(formatted_mem)
                            else:
                                # Fallback for other types
                                formatted_mem = {"content": str(mem.content)}
                                proper_memories.append(formatted_mem)
                                
                        # Case 2: Dictionary with 'content' key
                        elif isinstance(mem, dict) and 'content' in mem:
                            formatted_mem = {"content": str(mem['content'])}
                            proper_memories.append(formatted_mem)
                            
                        # Case 3: Malformed dictionaries with format {'content=': val, 'text': val}
                        elif isinstance(mem, dict) and 'content=' in mem:
                            # Find text fields (longer string keys)
                            text_keys = [k for k in mem.keys() 
                                       if k != 'content=' and isinstance(k, str) and len(k) > 10]
                            
                            if text_keys:
                                # Use the text key with actual content
                                longest_key = max(text_keys, key=len)
                                formatted_mem = {"content": longest_key}
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
                                    formatted_mem = {"content": content}
                                    proper_memories.append(formatted_mem)
                                else:
                                    # Last resort: use content= value
                                    formatted_mem = {"content": str(mem['content='])}
                                    proper_memories.append(formatted_mem)
                            
                        # Case 4: String value
                        elif isinstance(mem, str):
                            formatted_mem = {"content": mem}
                            proper_memories.append(formatted_mem)
                            
                        # Case 5: Any other object with __dict__ attribute
                        elif hasattr(mem, '__dict__'):
                            mem_dict = mem.__dict__
                            if 'content' in mem_dict:
                                formatted_mem = {"content": str(mem_dict['content'])}
                                proper_memories.append(formatted_mem)
                            else:
                                # Use the entire object representation
                                formatted_mem = {"content": str(mem)}
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
        """Ingest files into vector store"""
        if not files:
            return
        docs = []
        
        # Create tenant directory if it doesn't exist
        tenant_dir = Path(f"{self.storage_dir}/{tenant}/files")
        tenant_dir.mkdir(exist_ok=True, parents=True)
        
        for file in files:
            self._log(f"Ingesting file: {file.name} ({file.mime})")
            try:
                name = file.name.replace(" ", "_")
                data = base64.b64decode(file.data)
                # Store file in tenant-specific folder
                path = tenant_dir / f"{_sha(data)[:8]}_{name}"
                path.write_bytes(data)
                text = self.extract_text(path, file.mime)
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"name": file.name}))
            except Exception as e:
                self._log(f"Error ingesting file: {e}")

        if docs:
            vec = self.vec_factory(tenant)
            vec.add_documents(docs)

    # ----------------------------------------------------------------
    # RAG
    # ----------------------------------------------------------------
    async def _rag(self, msgs: List[Dict[str, Any]], tenant: str, k: int = 4):
        """Retrieve info from vector store and inject it into the conversation"""
        if len(msgs) == 0:
            return msgs
        vec = self.vec_factory(tenant)

        # get query from last message
        query = msgs[-1]["content"] if isinstance(msgs[-1]["content"], str) else ""
        if not query:
            return msgs

        docs = vec.similarity_search(query, k=k)
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
    async def _dispatch(self, msgs, model: str, *, stream: bool):
        """Dispatch to litellm API"""
        if stream:
            return await acompletion(
                model=model, messages=msgs, stream=stream
            )
        else:
            # For non-streaming responses, we need to await the response directly
            return await acompletion(
                model=model, messages=msgs, stream=False
            )


    # ----------------------------------------------------------------
    # FastAPI route
    # ----------------------------------------------------------------
    def _mount(self):
        @self.router.post("/{tenant}/chat/completions")
        async def chat(request: Request, tenant: str):
            if self.auth_hook:
                await _maybe(self.auth_hook, request, tenant)

            body = await request.json()
            req = ChatRequest(**body)
            msgs, files = self._split_files(req.messages)

            if files:
                self._log(f"Ingesting {len(files)} files for tenant {tenant}")
                await self._ingest_files(files, tenant)

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

            upstream_iter = await self._dispatch(
                msgs, req.model or self.default_model, stream=req.stream
            )
            t0 = time.time()

            if not req.stream:
                # No need to await here since _dispatch already returns the complete response
                response_data = upstream_iter.model_dump()
                await self._write_memories(tenant, msgs + [upstream_iter.choices[0].message.model_dump()])
                if self.usage_hook and upstream_iter.usage:
                    await _maybe(
                        self.usage_hook,
                        tenant,
                        upstream_iter.usage.total_tokens,
                        time.time() - t0,
                    )
                return JSONResponse(response_data)

            # streaming path
            async def event_stream() -> AsyncIterator[str]:
                tokens = 0
                buf: List[str] = []
                async for chunk in upstream_iter:
                    payload = json.loads(chunk.model_dump_json())
                    delta = payload["choices"][0].get("delta", {}).get("content", "")
                    if delta is None:
                        delta = ""
                    tokens += len(delta)
                    buf.append(delta)
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
                await self._write_memories(
                    tenant, msgs + [{"role": "assistant", "content": "".join(buf)}]
                )
                if self.usage_hook:
                    await _maybe(self.usage_hook, tenant, tokens, time.time() - t0)

            return StreamingResponse(event_stream(), media_type="text/event-stream")


# -------------------------------------------------------------------
# Example Chroma factories
# -------------------------------------------------------------------
"""
# Usage
from fastapi import FastAPI
from brain_proxy import BrainProxy

proxy = BrainProxy(
    #openai_api_key="sk-…",
)

app = FastAPI()
app.include_router(proxy.router, prefix="/v1")

# Point any OpenAI SDK at
# http://localhost:8000/v1/<tenant>/chat/completions
# Upload files via messages[].content[].file_data
# Enjoy RAG + LangMem without extra DBs or infra
"""