"""
brain_proxy.py  —  FastAPI / ASGI router with LangMem + Chroma

pip install fastapi openai langchain-chroma langmem tiktoken
"""

from __future__ import annotations
import asyncio, base64, hashlib, json, time
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
        memory_model: str = "openai/o4-mini-2025-04-16",  # litellm format e.g. "azure/gpt-35-turbo"
        embedding_model: str = "openai/text-embedding-3-small",  # litellm format e.g. "azure/ada-002"
        mem_top_k: int = 6,
        mem_working_max: int = 12,
        # misc
        default_model: str = "openai/gpt-4o",  # litellm format e.g. "azure/gpt-4"
        storage_dir: str | Path = "tenants",
        extract_text: Callable[[Path, str], str] | None = None,
        manager_fn: Callable[..., Any] | None = None,  # multi‑agent hook
        auth_hook: Callable[[Request, str], Any] | None = None,
        usage_hook: Callable[[str, int, float], Any] | None = None,
        max_upload_mb: int = 20,
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
            # Accepts a list of Memory objects or strings, store as documents
            docs = []
            for m in memories:
                if hasattr(m, 'content'):
                    # It's a Memory object, extract content as string
                    docs.append(Document(page_content=str(m.content)))
                elif isinstance(m, str):
                    # It's already a string
                    docs.append(Document(page_content=m))
                else:
                    # Try to convert to string as fallback
                    docs.append(Document(page_content=str(m)))
            
            if docs:
                vec.add_documents(docs)

        # Use langchain_litellm's ChatLiteLLM for memory manager
        manager = create_memory_manager(ChatLiteLLM(model=self.memory_model))

        self._mem_managers[tenant] = (manager, _search_mem, _store_mem)
        return self._mem_managers[tenant]

    async def _retrieve_memories(self, tenant: str, user_text: str) -> str:
        if not self.enable_memory:
            return ""
        manager_tuple = self._get_mem_manager(tenant)
        if not manager_tuple:
            return ""
        _, search, _ = manager_tuple
        memories = await search(user_text, k=self.mem_top_k)
        return "\n".join(memories)

    async def _write_memories(
        self, tenant: str, conversation: List[Dict[str, Any]]
    ):
        if not self.enable_memory:
            return
        manager_tuple = self._get_mem_manager(tenant)
        if not manager_tuple:
            return
        manager, _, store = manager_tuple
        manager_fn = self.manager_fn
        memories = []

        if manager_fn:
            memories_or_manager = await _maybe(manager_fn, tenant, conversation)
            if not memories_or_manager:
                return
            if isinstance(memories_or_manager, Callable):
                manager = memories_or_manager
            else:
                memories = memories_or_manager
        if not memories:  # could be using manager but didn't find any memories
            memories = await manager(conversation)
        if not memories:
            return

        # TODO: prune/filter memories

        try:
            await store(memories)
        except Exception as e:
            # sometimes mem can be too large
            print(f"Error storing memories: {e}")

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
                        print(f"Error decoding file: {e}")

            if text_parts:
                conv_msgs.append({"role": msg.role, "content": "\n".join(text_parts)})

        return conv_msgs, files

    async def _ingest_files(self, files: List[FileData], tenant: str):
        """Ingest files into vector store"""
        if not files:
            return
        docs = []
        for file in files:
            print(f"Ingesting file: {file.name} ({file.mime})")
            try:
                name = file.name.replace(" ", "_")
                data = base64.b64decode(file.data)
                path = Path(f"{self.storage_dir}/{tenant}_{_sha(data)[:8]}_{name}")
                path.parent.mkdir(exist_ok=True, parents=True)
                path.write_bytes(data)
                text = self.extract_text(path, file.mime)
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"name": file.name}))
            except Exception as e:
                print(f"Error ingesting file: {e}")

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
                await self._ingest_files(files, tenant)

            # LangMem retrieve
            if self.enable_memory:
                user_text = (
                    msgs[-1]["content"]
                    if isinstance(msgs[-1]["content"], str)
                    else next(
                        p["text"] for p in msgs[-1]["content"] if p["type"] == "text"
                    )
                )
                mem_block = await self._retrieve_memories(tenant, user_text)
                if mem_block:
                    msgs = msgs[:-1] + [
                        {
                            "role": "system",
                            "content": "Relevant memories:\n" + mem_block,
                        },
                        msgs[-1],
                    ]

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